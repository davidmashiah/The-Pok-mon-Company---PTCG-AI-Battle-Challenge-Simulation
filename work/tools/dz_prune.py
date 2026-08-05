"""Choose what to do with the learned id embeddings before shipping.

Context, measured: the OPPONENT plays field decks the net has seen thousands of
times (Grimmsnarl, Ogerpon, Fezandipiti), so memorised ids are worth keeping
for those. WE play Mega Lucario ex, which the net has seen ~12 times, so its id
embedding is untrained noise at N(0,1) scale -- actively harmful, since it is
LARGER than the trained vectors it sits beside.

So the candidates are:
  keep all      -- status quo, noise on our own cards
  zero unseen   -- keep memorised knowledge where it exists, delete it where it
                   was never learned
  zero all      -- pure attribute model, identical behaviour on every deck

Pick on measurement, not on which sounds tidiest. Test split is by GAME.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))
sys.path.insert(0, HERE)
from dz_unseen import (  # noqa: E402
    AC, G, HC, HT, M, S, A, Y, Net, te_i, top1, tr_g,
)

MIN_COUNT = 20

sd = torch.load(os.path.join(WORK, "out", "dz_model2.pt"), map_location="cpu")

# frequency of each card id in the TRAIN split only
tr_mask = np.array([g in tr_g for g in G])
cnt = np.zeros(sd["card.weight"].shape[0], dtype=np.int64)
for row_ac, row_m in zip(AC[tr_mask], M[tr_mask]):
    for cid, mk in zip(row_ac, row_m):
        if mk > 0.5:
            cnt[int(cid)] += 1
for row in HC[tr_mask]:
    for cid in row:
        cnt[int(cid)] += 1

rare = cnt < MIN_COUNT
rare[0] = False                      # id 0 is the padding row, leave it alone
print(f"card rows: {len(cnt)}   seen >= {MIN_COUNT} times in train: "
      f"{int((~rare).sum())}   rare/unseen: {int(rare.sum())}")

DECK = [673, 674, 675, 676, 677, 678, 1102, 1123, 1141, 1142,
        1152, 1159, 1182, 1192, 1227, 1252, 6]
ours_rare = [c for c in DECK if rare[c + 1]]
print(f"OUR deck's cards that are rare/unseen: {ours_rare}")

base = (Y[te_i] == 0).mean()
print(f"\ntest decisions {len(te_i)}   always-first baseline {base:.4f}\n")
print(f"{'variant':<28}{'top-1':>9}{'vs baseline':>13}")
print("-" * 52)

results = {}
for tag in ("keep all", "zero unseen", "zero all"):
    w = sd["card.weight"].clone()
    if tag == "zero unseen":
        w[torch.from_numpy(rare)] = 0.0
    elif tag == "zero all":
        w.zero_()
    m = Net(True)
    s2 = dict(sd)
    s2["card.weight"] = w
    m.load_state_dict(s2, strict=False)
    acc = top1(m, te_i)
    results[tag] = (acc, w)
    print(f"{tag:<28}{acc:>9.4f}{acc - base:>+13.4f}")

best = max(results, key=lambda k: results[k][0])
print(f"\nbest: {best}")
out = os.path.join(WORK, "out", "dz_model2_pruned.pt")
sd["card.weight"] = results[best][1]
torch.save(sd, out)
print(f"wrote {out}  (card.weight = '{best}')")
