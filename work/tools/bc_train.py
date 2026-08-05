"""Behavioural cloning, step 2: train a scorer and measure top-1 accuracy.

Evaluation is deliberately the SAME quantity our hand-written policy is measured
on: for each decision, does the model's argmax option match the winner's choice?
That makes 24.0% (our best rule-based agent on same-archetype winners) the bar.

Split is BY EPISODE-GROUP, never by row: rows from one decision are perfectly
correlated, so a random row split would leak the answer and inflate accuracy.

Two models:
  * logistic regression -> ships as a pure-Python dot product, zero dependencies
  * gradient boosting   -> stronger, needs sklearn at inference
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "out", "bc_lucario.npz")

d = np.load(DATA, allow_pickle=True)
X, Y, G = d["X"], d["Y"], d["G"]
names = [str(x) for x in d["names"]]
print(f"rows={len(X)}  decisions={len(np.unique(G))}  features={X.shape[1]}  "
      f"positive rate={Y.mean():.4f}")

# split by decision id so no decision straddles train/test
uniq = np.unique(G)
rng = np.random.default_rng(0)
rng.shuffle(uniq)
cut = int(0.8 * len(uniq))
tr_ids, te_ids = set(uniq[:cut].tolist()), set(uniq[cut:].tolist())
tr = np.array([g in tr_ids for g in G])
te = ~tr
print(f"train decisions={len(tr_ids)}  test decisions={len(te_ids)}")


def top1(scores, Gte, Yte):
    """Fraction of decisions where argmax score is the winner's chosen option."""
    ok = tot = 0
    order = np.argsort(Gte, kind="stable")
    Gs, Ss, Ys = Gte[order], scores[order], Yte[order]
    i = 0
    n = len(Gs)
    while i < n:
        j = i
        while j < n and Gs[j] == Gs[i]:
            j += 1
        blk = slice(i, j)
        if Ys[blk].sum() >= 1:
            tot += 1
            if Ys[blk][np.argmax(Ss[blk])] == 1:
                ok += 1
        i = j
    return ok / max(tot, 1), tot


from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

mu = X[tr].mean(0)
sd = X[tr].std(0) + 1e-6

print("\n--- logistic regression (ships as pure-python weights) ---")
lr = LogisticRegression(max_iter=2000, C=1.0)
lr.fit((X[tr] - mu) / sd, Y[tr])
s_lr = lr.decision_function((X[te] - mu) / sd)
a_lr, n_dec = top1(s_lr, G[te], Y[te])
print(f"top-1 accuracy: {a_lr:.4f}  over {n_dec} held-out decisions")

print("\n--- gradient boosting ---")
gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                    max_depth=None, l2_regularization=1.0,
                                    early_stopping=True, validation_fraction=0.1,
                                    random_state=0)
gb.fit(X[tr], Y[tr])
s_gb = gb.predict_proba(X[te])[:, 1]
a_gb, _ = top1(s_gb, G[te], Y[te])
print(f"top-1 accuracy: {a_gb:.4f}  over {n_dec} held-out decisions")

# baselines that must be beaten to mean anything
rngs = np.random.default_rng(1)
a_rand, _ = top1(rngs.random(te.sum()), G[te], Y[te])
a_first, _ = top1(-np.arange(te.sum(), dtype=np.float64), G[te], Y[te])
print(f"\nbaselines -> random: {a_rand:.4f}   always-first-option: {a_first:.4f}")
print("our hand-written policy on same-archetype winners: 0.2402")

print("\ntop positive weights (linear model):")
w = lr.coef_[0] / sd
idx = np.argsort(w)[::-1]
for i in idx[:12]:
    print(f"  {w[i]:+8.3f}  {names[i]}")
print("top negative:")
for i in idx[-8:]:
    print(f"  {w[i]:+8.3f}  {names[i]}")

out = os.path.join(WORK, "out", "bc_model.npz")
np.savez(out, w=lr.coef_[0], b=lr.intercept_, mu=mu, sd=sd,
         names=np.array(names))
print(f"\nwrote linear weights to {out}")
