"""Export the descriptor model to numpy weights and PROVE the port is exact.

Same contract as dz_export.py: nothing ships until the numpy inference in
work/lib/dznp.py reproduces torch's own logits on real decisions. The extra
piece here is the attribute projection (desc.weight); the descriptor TABLE
itself is not exported -- both sides recompute it from the bundled engine via
dzfeat, so they cannot disagree.
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
import dznp  # noqa: E402

DATA = os.path.join(WORK, "out", sys.argv[2] if len(sys.argv) > 2 else "dz_all2.npz")
CKPT = os.path.join(WORK, "out", sys.argv[1] if len(sys.argv) > 1 else "dz_model2.pt")
OUT = os.path.join(WORK, "lib", "dz_weights.npz")

d = np.load(DATA)
STATE_NF = d["S"].shape[1]
ACT_NF = d["A"].shape[2]
sd = torch.load(CKPT, map_location="cpu")
DESC = torch.from_numpy(dzfeat.card_desc_table())
DESC_D = DESC.shape[1]
CARD_VOCAB, N_EVT = dzfeat.CARD_VOCAB, 32


class Net2(nn.Module):
    """Must match dz_train2.Net2 exactly; strict load_state_dict enforces it."""

    def __init__(self, emb=32, hid=128):
        super().__init__()
        self.card = nn.Embedding(CARD_VOCAB, emb, padding_idx=0)
        self.desc = nn.Linear(DESC_D, emb, bias=False)
        self.register_buffer("dtab", DESC)
        self.evt = nn.Embedding(N_EVT, 8)
        self.gru = nn.GRU(emb + 8, 64, batch_first=True)
        self.state = nn.Sequential(nn.Linear(STATE_NF, hid), nn.ReLU(),
                                   nn.Linear(hid, 64), nn.ReLU())
        self.score = nn.Sequential(
            nn.Linear(64 + 64 + ACT_NF + emb, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1))

    def card_rep(self, ids):
        return self.card(ids) + self.desc(self.dtab[ids])

    def forward(self, s, a, ac, m, ht, hc):
        B, K, _ = a.shape
        h = torch.cat([self.card_rep(hc), self.evt(ht.clamp(0, N_EVT - 1))], -1)
        _, hn = self.gru(h)
        hn = hn[-1]
        st = self.state(s)
        ctx = torch.cat([st, hn], -1).unsqueeze(1).expand(B, K, 128)
        z = torch.cat([ctx, a, self.card_rep(ac)], -1)
        return self.score(z).squeeze(-1).masked_fill(m < 0.5, -1e9)


tm = Net2()
missing, unexpected = tm.load_state_dict(sd, strict=False)
assert not unexpected, f"checkpoint has unexpected keys: {unexpected}"
assert set(missing) <= {"dtab"}, f"checkpoint is missing weights: {missing}"
tm.eval()

np.savez_compressed(OUT, **{k: v.numpy() for k, v in sd.items()},
                    state_nf=np.array([STATE_NF]))
print(f"exported {len(sd)} tensors -> {OUT} "
      f"({os.path.getsize(OUT)/1e6:.2f} MB)   desc_d={DESC_D}")

# force a clean reload of the numpy side so we test what will actually ship
dznp._W = None
dznp._DTAB = None
dznp._LOAD_TRIED = False
assert dznp.load(), "numpy side failed to load the exported weights"
assert dznp.state_nf() == STATE_NF
assert "desc.weight" in dznp._W, "descriptor projection missing from export"

rng = np.random.default_rng(0)
idx = rng.choice(len(d["S"]), 60, replace=False)
worst = 0.0
agree = 0
for i in idx:
    s, a, ac, m, ht, hc = (d["S"][i], d["A"][i], d["AC"][i], d["M"][i],
                           d["HT"][i], d["HC"][i])
    with torch.no_grad():
        tl = tm(torch.tensor(s)[None], torch.tensor(a)[None],
                torch.tensor(ac).long()[None], torch.tensor(m)[None],
                torch.tensor(ht).long()[None],
                torch.tensor(hc).long()[None])[0].numpy()
    nl = dznp.logits(s, a, ac, m, ht, hc)
    live = m > 0.5
    worst = max(worst, float(np.abs(tl[live] - nl[live]).max()))
    agree += int(np.argmax(tl) == np.argmax(nl))

print("\nverification on 60 real decisions:")
print(f"  max |torch - numpy| on live options : {worst:.3e}")
print(f"  argmax agreement                    : {agree}/60")
assert worst < 1e-3, "numpy port does not match torch"
assert agree == 60, "numpy port picks a different option"
print("  PORT VERIFIED -- numpy inference is exact, torch not needed at runtime")
