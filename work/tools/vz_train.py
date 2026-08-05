"""Train the DouZero-encoder value net, against the only bar that matters.

GO/NO-GO: it must beat the `prize_diff` baseline by a clear margin on AUC.
The current shipped value net does NOT -- ridge 0.7121 vs prize_diff 0.7122 --
which is why swapping it into a search that already weights prizes x1000 would
change nothing. If this net cannot clear that bar it does not ship either, and
we say so instead of shipping a tie dressed up as progress.

Split is by GAME. Both POVs of every game are present, so a game contributes one
winning and one losing trajectory; keeping them on the same side of the split
prevents the net from seeing a game's outcome at training time and being scored
on its mirror.
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

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "out", "vz_data.npz")
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
OUT = os.path.join(WORK, "out", "vz_model.pt")

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

d = np.load(DATA)
S, HT, HC, Y, G = d["S"], d["HT"], d["HC"], d["Y"], d["G"]
STATE_NF = S.shape[1]
DESC = torch.from_numpy(dzfeat.card_desc_table())
DESC_D = DESC.shape[1]
CARD_VOCAB, N_EVT = dzfeat.CARD_VOCAB, 32
print(f"rows={len(S)} state_nf={STATE_NF} hist={HT.shape[1]} "
      f"label mean={Y.mean():.4f}")

games = np.unique(G)
rng = np.random.default_rng(0)
rng.shuffle(games)
cut = int(0.8 * len(games))
tr_g = set(games[:cut].tolist())
tr = np.array([g in tr_g for g in G])
tr_i, te_i = np.where(tr)[0], np.where(~tr)[0]
print(f"split by game: {cut} train / {len(games)-cut} test "
      f"({len(tr_i)} / {len(te_i)} rows)")


def auc(y, p):
    """Rank-based AUC: can it put a winning state above a losing one?"""
    y = np.asarray(y) > 0.5
    p = np.asarray(p)
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return 0.5
    return (ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


# the bar: one hand-written feature
PD = dzfeat.FEATNAMES.index("prize_diff")
base_auc = auc(Y[te_i], S[te_i, PD])
print(f"\nBASELINE  prize_diff alone : AUC {base_auc:.4f}   <-- the bar to beat")


class VNet(nn.Module):
    def __init__(self, emb=32, hid=128, drop=0.30):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.card = nn.Embedding(CARD_VOCAB, emb, padding_idx=0)
        self.desc = nn.Linear(DESC_D, emb, bias=False)
        self.register_buffer("dtab", DESC)
        self.evt = nn.Embedding(N_EVT, 8)
        self.gru = nn.GRU(emb + 8, 64, batch_first=True)
        self.state = nn.Sequential(nn.Linear(STATE_NF, hid), nn.ReLU(),
                                   nn.Linear(hid, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64 + 64, hid), nn.ReLU(),
                                  nn.Dropout(drop),
                                  nn.Linear(hid, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def card_rep(self, ids):
        return self.card(ids) + self.desc(self.dtab[ids])

    def forward(self, s, ht, hc):
        h = torch.cat([self.card_rep(hc), self.evt(ht.clamp(0, N_EVT - 1))], -1)
        _, hn = self.gru(h)
        z = self.drop(torch.cat([self.state(s), hn[-1]], -1))
        return torch.sigmoid(self.head(z)).squeeze(-1)


def evaluate(model, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 1024):
            j = idx[i:i + 1024]
            out.append(model(torch.from_numpy(S[j]),
                             torch.from_numpy(HT[j]).long(),
                             torch.from_numpy(HC[j]).long()).numpy())
    return np.concatenate(out) if out else np.zeros(0)


model = VNet()
print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=3e-4)
lossf = nn.BCELoss()

best = 0.0
for ep in range(EPOCHS):
    model.train()
    order = np.random.permutation(tr_i)
    tot = nb = 0
    t0 = time.time()
    for i in range(0, len(order), 256):
        j = order[i:i + 256]
        opt.zero_grad()
        p = model(torch.from_numpy(S[j]), torch.from_numpy(HT[j]).long(),
                  torch.from_numpy(HC[j]).long())
        loss = lossf(p, torch.from_numpy(Y[j]))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += loss.item()
        nb += 1
    a = auc(Y[te_i], evaluate(model, te_i))
    flag = ""
    if a > best:
        best = a
        torch.save({k: v for k, v in model.state_dict().items() if k != "dtab"},
                   OUT)
        flag = "  <- saved"
    print(f"epoch {ep+1:>2}  loss {tot/max(nb,1):.4f}  AUC {a:.4f}  "
          f"({time.time()-t0:.0f}s){flag}")

print(f"\nBEST AUC {best:.4f}   vs prize_diff baseline {base_auc:.4f}   "
      f"vs old linear net 0.7121")
if best < base_auc + 0.02:
    print("VERDICT: does NOT clear the bar -- this must not ship as a leaf "
          "evaluator; it would only re-derive the prize count evaluate() "
          "already has.")
else:
    print("VERDICT: clears the bar -- worth wiring in as the search's leaf "
          "evaluator and testing on the ladder.")
