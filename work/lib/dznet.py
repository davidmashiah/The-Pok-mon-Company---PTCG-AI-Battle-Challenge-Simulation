"""Inference for the DouZero-style scorer inside the agent.

Design constraints learned the hard way tonight:
  * NEVER raise. Any failure returns None and the tuned policy keeps its choice.
  * NEVER override without evidence: if the model's top choice is not clearly
    ahead of the policy's, defer. Every disaster today came from replacing a
    tuned policy with something noisier.
  * No sklearn, no file paths resolved from __file__ (the harness exec()s
    main.py, so __file__ does not exist).

The model ranks candidates by ACTION CONTENT (learned card embedding + action
features + board state + GRU over the log), which is why it is invariant to the
order the engine lists options in -- verified: top-1 0.4956 unshuffled vs
0.4995 shuffled, against a 0.4305 always-first baseline.
"""
import os
import sys

_KAGGLE = "/kaggle_simulations/agent"
_OK = False
_MODEL = None
_TORCH = None

try:
    import torch  # noqa: E402
    import torch.nn as nn  # noqa: E402
    _TORCH = torch
    _OK = True
except Exception:
    _OK = False

CARD_VOCAB = 1400
N_EVT = 32
MAX_CAND = 24
HIST = 24
ACT_NF = 10


if _OK:
    class _Net(nn.Module):
        def __init__(self, state_nf, emb=32, hid=128):
            super().__init__()
            self.card = nn.Embedding(CARD_VOCAB, emb, padding_idx=0)
            self.evt = nn.Embedding(N_EVT, 8)
            self.gru = nn.GRU(emb + 8, 64, batch_first=True)
            self.state = nn.Sequential(nn.Linear(state_nf, hid), nn.ReLU(),
                                       nn.Linear(hid, 64), nn.ReLU())
            self.score = nn.Sequential(
                nn.Linear(64 + 64 + ACT_NF + emb, hid), nn.ReLU(),
                nn.Linear(hid, hid), nn.ReLU(),
                nn.Linear(hid, 1))

        def forward(self, s, a, ac, m, ht, hc):
            B, K, _ = a.shape
            h = torch.cat([self.card(hc), self.evt(ht.clamp(0, N_EVT - 1))], -1)
            _, hn = self.gru(h)
            hn = hn[-1]
            st = self.state(s)
            ctx = torch.cat([st, hn], -1).unsqueeze(1).expand(B, K, 128)
            z = torch.cat([ctx, a, self.card(ac)], -1)
            return self.score(z).squeeze(-1).masked_fill(m < 0.5, -1e9)


def load(state_nf):
    """Load weights from beside the agent. Returns True if the model is live."""
    global _MODEL
    if not _OK:
        return False
    if _MODEL is not None:
        return True
    for cand in ("dz_model.pt", os.path.join(_KAGGLE, "dz_model.pt")):
        try:
            if os.path.exists(cand):
                m = _Net(state_nf)
                m.load_state_dict(_TORCH.load(cand, map_location="cpu"))
                m.eval()
                _TORCH.set_num_threads(1)      # agent shares a small box
                _MODEL = m
                return True
        except Exception:
            continue
    return False


def rank(state_vec, act_feats, act_cards, hist_types, hist_cards, n_opts,
         margin=0.35):
    """Return the model's preferred option index, or None to defer.

    `margin` is a confidence floor in logit space. Below it the model is not
    clearly preferring anything and the tuned policy -- worth ~700 rating
    points -- keeps its own choice.
    """
    if _MODEL is None or n_opts < 2 or n_opts > MAX_CAND:
        return None
    try:
        t = _TORCH
        with t.no_grad():
            s = t.tensor(state_vec, dtype=t.float32).unsqueeze(0)
            a = t.zeros(1, MAX_CAND, ACT_NF)
            ac = t.zeros(1, MAX_CAND, dtype=t.long)
            m = t.zeros(1, MAX_CAND)
            for i in range(n_opts):
                a[0, i] = t.tensor(act_feats[i], dtype=t.float32)
                ac[0, i] = int(act_cards[i])
                m[0, i] = 1.0
            ht = t.tensor(hist_types, dtype=t.long).unsqueeze(0)
            hc = t.tensor(hist_cards, dtype=t.long).unsqueeze(0)
            logits = _MODEL(s, a, ac, m, ht, hc)[0][:n_opts]
            top2 = t.topk(logits, min(2, n_opts))
            best = int(top2.indices[0])
            if n_opts >= 2 and float(top2.values[0] - top2.values[1]) < margin:
                return None                    # not confident enough to overrule
            return best
    except Exception:
        return None
