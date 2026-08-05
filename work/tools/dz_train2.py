"""DouZero scorer v2: card ATTRIBUTE descriptors + id-embedding dropout.

Why this exists, measured not assumed: our deck's Pokemon appear ~0 times in
the field's games (0 of 264,093 option observations for Makuhita/Hariyama/Dusk
Ball; 12 for Mega Lucario ex). A per-id learned embedding is untrained noise
for exactly the cards we play, so v1 could only reason about our deck through
the handful of generic trainers it had seen.

Two changes:
  1. Every card id also carries a fixed ATTRIBUTE descriptor (hp, stage, ex /
     mega flags, typing, retreat, attack costs, best damage, damage-per-energy,
     has-ability) computed from the engine table. Unseen cards still arrive
     described.
  2. ID-EMBEDDING DROPOUT during training: the learned per-id vector is zeroed
     at random, so the net cannot lean on memorised ids and must use the
     descriptors. This is what makes the representation transfer to cards it
     has never seen -- which is the whole point.

Reported metrics are unchanged so the number is comparable to v1's 0.5162:
top-1, the always-first baseline, and top-1 under candidate SHUFFLING.
Split is by GAME.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))
import dzfeat  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "out", "dz_all2.npz")
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 22
ID_DROP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
OUT = os.path.join(WORK, "out", "dz_model2.pt")

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

d = np.load(DATA)
S, A, AC, M, HT, HC, Y, G = (d["S"], d["A"], d["AC"], d["M"],
                             d["HT"], d["HC"], d["Y"], d["G"])
N, K, ACT_NF = A.shape
STATE_NF = S.shape[1]
DESC = torch.from_numpy(dzfeat.card_desc_table())
DESC_D = DESC.shape[1]
print(f"decisions={N} candidates<={K} state_nf={STATE_NF} act_nf={ACT_NF} "
      f"desc_d={DESC_D} id_drop={ID_DROP}")

games = np.unique(G)
rng = np.random.default_rng(0)
rng.shuffle(games)
cut = int(0.8 * len(games))
tr_g = set(games[:cut].tolist())
tr = np.array([g in tr_g for g in G])
te = ~tr
tr_i, te_i = np.where(tr)[0], np.where(te)[0]
print(f"split by game: {cut} train / {len(games)-cut} test  "
      f"({tr.sum()} / {te.sum()} decisions)")

CARD_VOCAB = dzfeat.CARD_VOCAB
N_EVT = 32


class Net2(nn.Module):
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
        """id embedding (dropped at random in training) + attribute projection."""
        e = self.card(ids)
        if self.training and ID_DROP > 0:
            keep = (torch.rand(ids.shape + (1,), device=e.device) >= ID_DROP)
            e = e * keep.float()
        return e + self.desc(self.dtab[ids])

    def forward(self, s, a, ac, m, ht, hc):
        B, Kc, _ = a.shape
        h = torch.cat([self.card_rep(hc), self.evt(ht.clamp(0, N_EVT - 1))], -1)
        _, hn = self.gru(h)
        hn = hn[-1]
        st = self.state(s)
        ctx = torch.cat([st, hn], -1).unsqueeze(1).expand(B, Kc, 128)
        z = torch.cat([ctx, a, self.card_rep(ac)], -1)
        return self.score(z).squeeze(-1).masked_fill(m < 0.5, -1e9)


def batches(idx, bs):
    order = np.random.permutation(idx)
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        yield (torch.from_numpy(S[j]), torch.from_numpy(A[j]),
               torch.from_numpy(AC[j]).long(), torch.from_numpy(M[j]),
               torch.from_numpy(HT[j]).long(), torch.from_numpy(HC[j]).long(),
               torch.from_numpy(Y[j]))


def evaluate(model, idx, shuffle_candidates=False):
    model.eval()
    ok = tot = 0
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            j = idx[i:i + 512]
            s, a, ac, m, ht, hc, y = (torch.from_numpy(S[j]), torch.from_numpy(A[j]),
                                      torch.from_numpy(AC[j]).long(),
                                      torch.from_numpy(M[j]),
                                      torch.from_numpy(HT[j]).long(),
                                      torch.from_numpy(HC[j]).long(),
                                      torch.from_numpy(Y[j]))
            if shuffle_candidates:
                B = a.shape[0]
                perm = torch.stack([torch.randperm(K, generator=g) for _ in range(B)])
                a = torch.gather(a, 1, perm.unsqueeze(-1).expand(-1, -1, a.shape[2]))
                ac = torch.gather(ac, 1, perm)
                m = torch.gather(m, 1, perm)
                y = (perm == y.unsqueeze(1)).float().argmax(1)
            logits = model(s, a, ac, m, ht, hc)
            ok += (logits.argmax(1) == y).sum().item()
            tot += len(j)
    return ok / max(tot, 1)


first_baseline = (Y[te_i] == 0).mean()
print(f"\nbaseline 'always pick option 0' on this split: {first_baseline:.4f}")

model = Net2()
print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss()

best = 0.0
for ep in range(EPOCHS):
    model.train()
    t0 = time.time()
    tot = 0.0
    nb = 0
    for s, a, ac, m, ht, hc, y in batches(tr_i, 256):
        opt.zero_grad()
        loss = lossf(model(s, a, ac, m, ht, hc), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += loss.item()
        nb += 1
    acc = evaluate(model, te_i)
    sh = evaluate(model, te_i, shuffle_candidates=True)
    flag = ""
    if acc > best:
        best = acc
        sd = {k: v for k, v in model.state_dict().items() if k != "dtab"}
        torch.save(sd, OUT)
        flag = "  <- saved"
    print(f"epoch {ep+1:>2}  loss {tot/max(nb,1):.4f}  top1 {acc:.4f}  "
          f"shuffled {sh:.4f}  ({time.time()-t0:.0f}s){flag}")

print(f"\nBEST top-1 {best:.4f}   vs always-first {first_baseline:.4f}   "
      f"vs v1 (id-embedding only) 0.5162")
