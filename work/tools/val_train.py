"""Train the win-probability value net and check it is actually calibrated.

The bar is NOT "low loss". A net that predicts 0.5 everywhere has respectable
loss and is useless as a re-ranker. What matters is whether it separates winning
from losing positions, so this reports:

  * AUC                 -- can it rank a winning state above a losing one?
  * accuracy vs a prize-difference baseline -- does it beat the single obvious
    hand-written feature? If not, there is nothing learned worth shipping.
  * calibration by decile -- are its 0.8s really 80%?

Exports pure-Python weights so inference in the agent is a dot product with no
sklearn dependency.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "out", "val_data.npz")

d = np.load(DATA, allow_pickle=True)
X, Y = d["X"], d["Y"]
names = [str(x) for x in d["names"]]
print(f"rows={len(X)}  features={X.shape[1]}  label mean={Y.mean():.4f}")

# Split by GAME, never by row. Rows from one game share a label and correlated
# features; a random row split puts the same game on both sides and inflates
# everything. A first run split by row and reported AUC 0.9825 -- pure leakage.
G = d["G"]
games = np.unique(G)
rng = np.random.default_rng(0)
rng.shuffle(games)
cut_g = int(0.8 * len(games))
tr_games = set(games[:cut_g].tolist())
mask_tr = np.array([g in tr_games for g in G])
tr = np.where(mask_tr)[0]
te = np.where(~mask_tr)[0]
print(f"split by game: {len(tr_games)} train games / {len(games)-len(tr_games)} test games")

ybin_te = (Y[te] > 0.5).astype(int)
print(f"train={len(tr)}  test={len(te)}  test win-share={ybin_te.mean():.4f}")

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6

lin = Ridge(alpha=1.0)
lin.fit((X[tr] - mu) / sd, Y[tr])
p_lin = lin.predict((X[te] - mu) / sd)

gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                   l2_regularization=1.0, early_stopping=True,
                                   validation_fraction=0.1, random_state=0)
gb.fit(X[tr], Y[tr])
p_gb = gb.predict(X[te])

# the obvious hand-written baseline this must beat to be worth anything
pd_i = names.index("prize_diff")
p_base = X[te][:, pd_i]

print("\n%-22s %8s %10s" % ("model", "AUC", "acc@0.5"))
print("-" * 44)
for nm, p in (("prize_diff baseline", p_base),
              ("ridge (ships)", p_lin),
              ("gradient boosting", p_gb)):
    try:
        auc = roc_auc_score(ybin_te, p)
    except Exception:
        auc = float("nan")
    thr = np.median(p) if nm == "prize_diff baseline" else 0.5
    acc = ((p > thr).astype(int) == ybin_te).mean()
    print("%-22s %8.4f %10.4f" % (nm, auc, acc))

print("\ncalibration of the ridge model (what ships):")
q = np.quantile(p_lin, np.linspace(0, 1, 11))
for i in range(10):
    m = (p_lin >= q[i]) & (p_lin <= q[i + 1])
    if m.sum() > 30:
        print(f"  pred {p_lin[m].mean():.3f}  ->  actual {ybin_te[m].mean():.3f}"
              f"   (n={m.sum()})")

w = lin.coef_ / sd
b = float(lin.intercept_ - (lin.coef_ * mu / sd).sum())
print("\nstrongest signals:")
order = np.argsort(np.abs(w))[::-1]
for i in order[:10]:
    print(f"  {w[i]:+9.4f}  {names[i]}")

out = os.path.join(WORK, "lib", "valnet.py")
with open(out, "w") as f:
    f.write('"""Win-probability value net (linear, exported as plain weights).\n\n'
            'Trained on both points of view of real games between ~1085-rated\n'
            'agents, labels discounted toward 0.5 by distance from the end.\n'
            'Pure Python so the agent needs no sklearn at inference.\n"""\n')
    f.write("W = %r\n" % [round(float(x), 6) for x in w])
    f.write("B = %r\n" % round(b, 6))
    f.write("NAMES = %r\n" % names)
    f.write('''

def score(feats):
    """feats: list[float] in NAMES order -> win probability in (0,1)."""
    z = B
    for i, v in enumerate(feats):
        z += W[i] * v
    if z < 0.0:
        return 0.0
    if z > 1.0:
        return 1.0
    return z
''')
print(f"\nwrote {out}")
