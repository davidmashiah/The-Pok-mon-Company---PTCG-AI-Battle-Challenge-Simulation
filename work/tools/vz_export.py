"""Export the value net to numpy and PROVE the port matches torch.

Same contract as the action-scorer export: nothing ships until the numpy
inference in work/lib/vznp.py reproduces torch's own win probabilities on real
positions. The descriptor table is recomputed from the bundled engine on both
sides rather than exported, so the two cannot disagree.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))
import dzfeat  # noqa: E402
import vznp  # noqa: E402

DATA = os.path.join(WORK, "out", "vz_data.npz")
CKPT = os.path.join(WORK, "out", "vz_model.pt")
OUT = os.path.join(WORK, "lib", "vz_weights.npz")

d = np.load(DATA)
S, HT, HC, Y = d["S"], d["HT"], d["HC"], d["Y"]
STATE_NF = S.shape[1]
DESC = torch.from_numpy(dzfeat.card_desc_table())
DESC_D = DESC.shape[1]
CARD_VOCAB, N_EVT = dzfeat.CARD_VOCAB, 32
sd = torch.load(CKPT, map_location="cpu")


class VNet(nn.Module):
    """Must match vz_train.VNet exactly; load_state_dict enforces the shapes."""

    def __init__(self, emb=32, hid=128):
        super().__init__()
        self.card = nn.Embedding(CARD_VOCAB, emb, padding_idx=0)
        self.desc = nn.Linear(DESC_D, emb, bias=False)
        self.register_buffer("dtab", DESC)
        self.evt = nn.Embedding(N_EVT, 8)
        self.gru = nn.GRU(emb + 8, 64, batch_first=True)
        self.state = nn.Sequential(nn.Linear(STATE_NF, hid), nn.ReLU(),
                                   nn.Linear(hid, 64), nn.ReLU())
        # Dropout is inference-inert but it OCCUPIES AN INDEX in Sequential,
        # so the exported names shift (head.3 / head.5). Mirror the training
        # module exactly or load_state_dict silently mismatches.
        self.head = nn.Sequential(nn.Linear(64 + 64, hid), nn.ReLU(),
                                  nn.Dropout(0.0),
                                  nn.Linear(hid, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def card_rep(self, ids):
        return self.card(ids) + self.desc(self.dtab[ids])

    def forward(self, s, ht, hc):
        h = torch.cat([self.card_rep(hc), self.evt(ht.clamp(0, N_EVT - 1))], -1)
        _, hn = self.gru(h)
        z = torch.cat([self.state(s), hn[-1]], -1)
        return torch.sigmoid(self.head(z)).squeeze(-1)


tm = VNet()
missing, unexpected = tm.load_state_dict(sd, strict=False)
assert not unexpected, f"unexpected keys: {unexpected}"
assert set(missing) <= {"dtab"}, f"missing weights: {missing}"
tm.eval()

np.savez_compressed(OUT, **{k: v.numpy() for k, v in sd.items()})
print(f"exported {len(sd)} tensors -> {OUT} "
      f"({os.path.getsize(OUT)/1e6:.2f} MB)")

vznp._W = None
vznp._DTAB = None
vznp._TRIED = False
assert vznp.load(), "numpy side failed to load the exported weights"

rng = np.random.default_rng(0)
idx = rng.choice(len(S), 60, replace=False)
worst = 0.0
for i in idx:
    with torch.no_grad():
        t = float(tm(torch.tensor(S[i])[None],
                     torch.tensor(HT[i]).long()[None],
                     torch.tensor(HC[i]).long()[None])[0])
    n = vznp.win_prob(S[i], HT[i], HC[i])
    worst = max(worst, abs(t - n))

print(f"\nverification on 60 real positions:")
print(f"  max |torch - numpy| win probability : {worst:.3e}")
assert worst < 1e-4, "numpy port does not match torch"
print("  PORT VERIFIED -- numpy inference is exact, torch not needed at runtime")
