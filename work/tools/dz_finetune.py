"""Fine-tune the field model on games won with OUR OWN deck.

The hypothesis being tested, stated before the measurement: the net imitates
winners playing OTHER decks (Grimmsnarl, Ogerpon, Fezandipiti), so it has
learned those decks' strategies. Better card generalisation (v2 descriptors,
105% edge retention on unseen cards) improved every offline metric and then
LOST in games -- 0.4867 / 0.5000 against v14 versus v1's 0.5297. That points at
the strategy being wrong for our deck, not the card representation.

This prints the number that settles it before any fine-tuning happens:

    zero-shot top-1 of the field model on MEGA LUCARIO winners' decisions

If the field model already predicts our deck's winners about as well as it
predicts the field's (~0.52), deck mismatch is NOT the problem and more
deck-matched data will not help -- do not spend an hour downloading it. If it
is far worse, the mismatch is real and worth the download.

Usage: python work/tools/dz_finetune.py [data.npz] [epochs] [lr]
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

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    WORK, "out", "dz_luc_matched.npz")
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
BASE = os.path.join(WORK, "out", "dz_model2.pt")
OUT = os.path.join(WORK, "out", "dz_model_luc.pt")

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
CARD_VOCAB, N_EVT = dzfeat.CARD_VOCAB, 32
games = np.unique(G)
print(f"deck-matched decisions={N} from {len(games)} games")
if N < 500:
    print("TOO LITTLE DATA -- not enough deck-matched play to conclude anything")
    sys.exit(2)

rng = np.random.default_rng(0)
rng.shuffle(games)
cut = max(1, int(0.75 * len(games)))
tr_g = set(games[:cut].tolist())
tr = np.array([g in tr_g for g in G])
tr_i, te_i = np.where(tr)[0], np.where(~tr)[0]
print(f"split by game: {cut} train / {len(games)-cut} test "
      f"({len(tr_i)} / {len(te_i)} decisions)")


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
            nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def card_rep(self, ids):
        return self.card(ids) + self.desc(self.dtab[ids])

    def forward(self, s, a, ac, m, ht, hc):
        B, Kc, _ = a.shape
        h = torch.cat([self.card_rep(hc), self.evt(ht.clamp(0, N_EVT - 1))], -1)
        _, hn = self.gru(h)
        st = self.state(s)
        ctx = torch.cat([st, hn[-1]], -1).unsqueeze(1).expand(B, Kc, 128)
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


model = Net2()
model.load_state_dict(torch.load(BASE, map_location="cpu"), strict=False)

base_rate = (Y[te_i] == 0).mean()
zero_shot = top1(model, te_i)
print(f"\nalways-first baseline on OUR deck's decisions : {base_rate:.4f}")
print(f"field model, ZERO-SHOT on OUR deck            : {zero_shot:.4f}")
print(f"field model on the FIELD's own decisions      : 0.5249  (reference)")
if zero_shot >= 0.50:
    print("\n=> the field model already predicts our deck's winners about as well\n"
          "   as it predicts the field's. Deck mismatch is NOT the bottleneck;\n"
          "   downloading more deck-matched days would not pay for itself.")
else:
    print(f"\n=> {0.5249 - zero_shot:+.4f} worse on our deck than on the field's.\n"
          "   Deck mismatch is real; more deck-matched data is worth downloading.")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss()
best = zero_shot
print(f"\nfine-tuning (lr={LR}) -- must beat zero-shot {zero_shot:.4f} to be kept")
for ep in range(EPOCHS):
    model.train()
    order = np.random.permutation(tr_i)
    tot = 0.0
    nb = 0
    for i in range(0, len(order), 128):
        j = order[i:i + 128]
        opt.zero_grad()
        out = model(torch.from_numpy(S[j]), torch.from_numpy(A[j]),
                    torch.from_numpy(AC[j]).long(), torch.from_numpy(M[j]),
                    torch.from_numpy(HT[j]).long(), torch.from_numpy(HC[j]).long())
        loss = lossf(out, torch.from_numpy(Y[j]))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += loss.item()
        nb += 1
    acc = top1(model, te_i)
    flag = ""
    if acc > best:
        best = acc
        sd = {k: v for k, v in model.state_dict().items() if k != "dtab"}
        torch.save(sd, OUT)
        flag = "  <- saved"
    print(f"epoch {ep+1:>2}  loss {tot/max(nb,1):.4f}  top1 {acc:.4f}{flag}")

print(f"\nBEST {best:.4f}  vs zero-shot {zero_shot:.4f}  "
      f"vs always-first {base_rate:.4f}")
if best <= zero_shot:
    print("fine-tuning did NOT beat the zero-shot field model -- nothing saved.")
