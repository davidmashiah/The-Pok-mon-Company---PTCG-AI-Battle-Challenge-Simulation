"""Does the cloned model know anything a constant does not?

`bc_train.py` reports top-1 accuracy over every decision, and on this dataset the
trivial "always take option 0" policy scores 0.4637 while the trained gradient
boosting scores 0.4983. A +0.035 edge over a constant is not obviously skill --
top-1 over all frames is dominated by forced and near-forced choices, where every
method looks identical and nothing is being learned.

So split the held-out set the only way that discriminates:

  EASY   the pilot took option 0     -- a constant already gets these right
  HARD   the pilot took another one  -- a constant gets these ALL wrong, by
                                        construction, so any accuracy here is
                                        real information the model extracted

A model whose entire advantage lives in EASY has learned the option ordering, not
the game, and would ship as an expensive way to return [0]. That is a failure
mode this repo has already met: `default_audit.py` found we answer
DAMAGE_COUNTER_ANY with index 0 in 115 of 115 decisions -- a decision not being
made. Cloning our way into more of that is worse than useless.

Splitting by episode-group, never by row, exactly as bc_train.py does: rows from
one decision are perfectly correlated and a row split leaks the answer.

  python work/tools/bc_hard_frames.py work/out/bc_grimm.npz
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)

DATA = (sys.argv[1] if len(sys.argv) > 1
        else os.path.join(WORK, "out", "bc_grimm.npz"))

d = np.load(DATA, allow_pickle=True)
X, Y, G = d["X"], d["Y"], d["G"]
print(f"rows={len(X)}  decisions={len(np.unique(G))}  features={X.shape[1]}")

uniq = np.unique(G)
rng = np.random.default_rng(0)
rng.shuffle(uniq)
cut = int(0.8 * len(uniq))
tr_ids = set(uniq[:cut].tolist())
tr = np.array([g in tr_ids for g in G])
te = ~tr

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                       random_state=0)
model.fit(X[tr], Y[tr])
scores = model.predict_proba(X[te])[:, 1]

Gte, Yte = G[te], Y[te]
order = np.argsort(Gte, kind="stable")
Gs, Ys, Ss = Gte[order], Yte[order], scores[order]

easy_tot = easy_hit = hard_tot = hard_hit = 0
hard_const_hit = 0
start = 0
for i in range(1, len(Gs) + 1):
    if i == len(Gs) or Gs[i] != Gs[start]:
        y = Ys[start:i]
        s = Ss[start:i]
        if y.sum() >= 1:
            truth = int(np.argmax(y))          # first chosen option
            pred = int(np.argmax(s))
            if truth == 0:
                easy_tot += 1
                easy_hit += (pred == 0)
            else:
                hard_tot += 1
                hard_hit += (pred == truth)
                hard_const_hit += (0 == truth)   # always 0, by definition never
        start = i

tot = easy_tot + hard_tot
print(f"\nheld-out decisions: {tot}   "
      f"EASY (pilot took option 0): {easy_tot} ({easy_tot/max(1,tot):.3f})   "
      f"HARD: {hard_tot} ({hard_tot/max(1,tot):.3f})")
print(f"\n{'split':6} {'n':>6} {'model':>8} {'always-0':>10}")
print("-" * 34)
print(f"{'EASY':6} {easy_tot:6d} {easy_hit/max(1,easy_tot):8.4f} "
      f"{1.0:10.4f}")
print(f"{'HARD':6} {hard_tot:6d} {hard_hit/max(1,hard_tot):8.4f} "
      f"{hard_const_hit/max(1,hard_tot):10.4f}")
overall = (easy_hit + hard_hit) / max(1, tot)
const = easy_tot / max(1, tot)
print(f"{'ALL':6} {tot:6d} {overall:8.4f} {const:10.4f}")

print("\nHARD is the whole question. A constant scores exactly 0.0000 there by\n"
      "construction, so the model's HARD accuracy is the only number that is\n"
      "information rather than option ordering.")
