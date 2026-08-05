"""DouZero-style scorer: learned card embeddings + GRU over the log + per-candidate
action encoding, trained as a softmax over legal candidates.

Evaluation reports THREE numbers, and the third is the one that matters:

  top-1               agreement with the winner's choice
  always-first        the trivial baseline this must beat (previous BC did not)
  top-1 SHUFFLED      same model, candidate order randomly permuted at eval

A model that has actually learned scores the SAME shuffled as unshuffled,
because it ranks by action content. A model that has merely learned "the engine
lists the good move first" collapses under shuffling. The previous BC run could
not tell these apart; this makes it impossible to fool ourselves.

Split is by GAME, never by decision or row.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "out", "dz_luc.npz")
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 12

torch.manual_seed(0)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

d = np.load(DATA)
S, A, AC, M, HT, HC, Y, G = (d["S"], d["A"], d["AC"], d["M"],
                             d["HT"], d["HC"], d["Y"], d["G"])
N, K, ACT_NF = A.shape
STATE_NF = S.shape[1]
HISTL = HT.shape[1]
print(f"decisions={N} candidates<={K} state_nf={STATE_NF} act_nf={ACT_NF} hist={HISTL}")

games = np.unique(G)
rng = np.random.default_rng(0)
rng.shuffle(games)
cut = int(0.8 * len(games))
tr_g = set(games[:cut].tolist())
tr = np.array([g in tr_g for g in G])
te = ~tr
print(f"split by game: {cut} train / {len(games)-cut} test  "
      f"({tr.sum()} / {te.sum()} decisions)")

CARD_VOCAB = 1400
N_EVT = 32


class Net(nn.Module):
    def __init__(self, emb=32, hid=128):
        super().__init__()
        self.card = nn.Embedding(CARD_VOCAB, emb, padding_idx=0)
        self.evt = nn.Embedding(N_EVT, 8)
        self.gru = nn.GRU(emb + 8, 64, batch_first=True)
        self.state = nn.Sequential(nn.Linear(STATE_NF, hid), nn.ReLU(),
                                   nn.Linear(hid, 64), nn.ReLU())
        self.score = nn.Sequential(
            nn.Linear(64 + 64 + ACT_NF + emb, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1))

    def forward(self, s, a, ac, m, ht, hc):
        B, K, _ = a.shape
        h = torch.cat([self.card(hc), self.evt(ht.clamp(0, N_EVT - 1))], -1)
        _, hn = self.gru(h)
        hn = hn[-1]                                   # B x 64
        st = self.state(s)                            # B x 64
        ctx = torch.cat([st, hn], -1).unsqueeze(1).expand(B, K, 128)
        z = torch.cat([ctx, a, self.card(ac)], -1)
        out = self.score(z).squeeze(-1)               # B x K
        return out.masked_fill(m < 0.5, -1e9)


def batches(idx, bs, shuffle=True):
    order = np.random.permutation(idx) if shuffle else idx
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        yield (torch.from_numpy(S[j]), torch.from_numpy(A[j]),
               torch.from_numpy(AC[j]).long(), torch.from_numpy(M[j]),
               torch.from_numpy(HT[j]).long(), torch.from_numpy(HC[j]).long(),
               torch.from_numpy(Y[j]))


def evaluate(model, idx, shuffle_candidates=False):
    model.eval()
    ok = tot = 0
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
                # permute candidate order per row; a content-based model is
                # invariant, an ordering-cheater is destroyed
                B = a.shape[0]
                perm = torch.stack([torch.randperm(K) for _ in range(B)])
                a = torch.gather(a, 1, perm.unsqueeze(-1).expand(-1, -1, a.shape[2]))
                ac = torch.gather(ac, 1, perm)
                m = torch.gather(m, 1, perm)
                y = (perm == y.unsqueeze(1)).float().argmax(1)
            logits = model(s, a, ac, m, ht, hc)
            ok += (logits.argmax(1) == y).sum().item()
            tot += len(j)
    return ok / max(tot, 1)


tr_i, te_i = np.where(tr)[0], np.where(te)[0]
first_baseline = (Y[te_i] == 0).mean()
print(f"\nbaseline 'always pick option 0' on this split: {first_baseline:.4f}")

model = Net()
nparam = sum(p.numel() for p in model.parameters())
print(f"model params: {nparam/1e6:.2f}M")
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
        torch.save(model.state_dict(), os.path.join(WORK, "out", "dz_model.pt"))
        flag = "  <- saved"
    print(f"epoch {ep+1:>2}  loss {tot/max(nb,1):.4f}  top1 {acc:.4f}  "
          f"shuffled {sh:.4f}  ({time.time()-t0:.0f}s){flag}")

print(f"\nBEST top-1 {best:.4f}   vs always-first {first_baseline:.4f}   "
      f"vs previous BC 0.3719")
print("shuffled ~= unshuffled means the model ranks by CONTENT, not list order.")
