"""How well does the net play cards it has NEVER SEEN? -- which is our case.

Measured: our deck's Pokemon are ~0 of 264,093 option observations in the
field's games. So in-distribution top-1 is the wrong yardstick for us; it
rewards memorising the field's card ids, which tells us nothing about how the
net handles Mega Lucario ex.

This simulates our situation directly by ZEROING the learned per-id embedding
at evaluation time and keeping only what transfers -- the attribute descriptors
and the generic action features. A net that has merely memorised ids collapses
to the always-first baseline. A net that learned to reason about cards by
ATTRIBUTE keeps most of its edge, and that surviving edge is the part that will
still be there when it picks up our deck.

Usage: python work/tools/dz_unseen.py
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

DATA = os.path.join(WORK, "out", "dz_all2.npz")
d = np.load(DATA)
S, A, AC, M, HT, HC, Y, G = (d["S"], d["A"], d["AC"], d["M"],
                             d["HT"], d["HC"], d["Y"], d["G"])
STATE_NF, ACT_NF = S.shape[1], A.shape[2]
DESC = torch.from_numpy(dzfeat.card_desc_table())
DESC_D = DESC.shape[1]
CARD_VOCAB, N_EVT = dzfeat.CARD_VOCAB, 32

games = np.unique(G)
rng = np.random.default_rng(0)
rng.shuffle(games)
tr_g = set(games[:int(0.8 * len(games))].tolist())
te_i = np.where(np.array([g not in tr_g for g in G]))[0]


class Net(nn.Module):
    """Covers both checkpoints: `desc` is absent in v1, present in v2."""

    def __init__(self, use_desc, emb=32, hid=128):
        super().__init__()
        self.use_desc = use_desc
        self.card = nn.Embedding(CARD_VOCAB, emb, padding_idx=0)
        if use_desc:
            self.desc = nn.Linear(DESC_D, emb, bias=False)
            self.register_buffer("dtab", DESC)
        self.evt = nn.Embedding(N_EVT, 8)
        self.gru = nn.GRU(emb + 8, 64, batch_first=True)
        self.state = nn.Sequential(nn.Linear(STATE_NF, hid), nn.ReLU(),
                                   nn.Linear(hid, 64), nn.ReLU())
        self.score = nn.Sequential(
            nn.Linear(64 + 64 + ACT_NF + emb, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.kill_ids = False

    def card_rep(self, ids):
        e = torch.zeros_like(self.card(ids)) if self.kill_ids else self.card(ids)
        if self.use_desc:
            e = e + self.desc(self.dtab[ids])
        return e

    def forward(self, s, a, ac, m, ht, hc):
        B, K, _ = a.shape
        h = torch.cat([self.card_rep(hc), self.evt(ht.clamp(0, N_EVT - 1))], -1)
        _, hn = self.gru(h)
        st = self.state(s)
        ctx = torch.cat([st, hn[-1]], -1).unsqueeze(1).expand(B, K, 128)
        z = torch.cat([ctx, a, self.card_rep(ac)], -1)
        return self.score(z).squeeze(-1).masked_fill(m < 0.5, -1e9)


def top1(model, idx):
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            j = idx[i:i + 512]
            lg = model(torch.from_numpy(S[j]), torch.from_numpy(A[j]),
                       torch.from_numpy(AC[j]).long(), torch.from_numpy(M[j]),
                       torch.from_numpy(HT[j]).long(),
                       torch.from_numpy(HC[j]).long())
            ok += (lg.argmax(1) == torch.from_numpy(Y[j])).sum().item()
            tot += len(j)
    return ok / max(tot, 1)


base = (Y[te_i] == 0).mean()
print(f"test decisions {len(te_i)}   always-first baseline {base:.4f}\n")
print(f"{'model':<34}{'normal':>9}{'ids ZEROED':>13}{'edge kept':>11}")
print("-" * 68)
for tag, ckpt, use_desc in [
        ("v1  id embedding only", "dz_model.pt", False),
        ("v2  + attribute descriptors", "dz_model2.pt", True)]:
    p = os.path.join(WORK, "out", ckpt)
    if not os.path.exists(p):
        print(f"{tag:<34}{'(missing)':>9}")
        continue
    m = Net(use_desc)
    m.load_state_dict(torch.load(p, map_location="cpu"), strict=False)
    m.kill_ids = False
    a = top1(m, te_i)
    m.kill_ids = True
    b = top1(m, te_i)
    kept = (b - base) / (a - base) if a > base else 0.0
    print(f"{tag:<34}{a:>9.4f}{b:>13.4f}{kept:>10.0%}")
print("\n'ids ZEROED' is the honest proxy for playing OUR deck: every card the")
print("net must reason about arrives with no memorised identity, exactly as")
print("Mega Lucario ex does. 'edge kept' is the share of its advantage over the")
print("always-first baseline that survives.")
