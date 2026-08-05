"""Export the trained net to numpy-only weights, and PROVE the port is exact.

Why not just ship torch: it is 518 MB installed against a ~197 MB tarball
limit. Why not rely on torch being present in the Kaggle image: if it is
absent the agent silently falls back to the tuned policy and nothing reports
it -- the exact failure that has already appeared five times in this project
(dead search_begin, unbundled meta_decks, starved best_action twice, an
unrunnable PIMC budget).

So the port is numpy-only, and this script asserts it matches torch's own
output on real inputs before anything ships.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "tools"))

DATA = os.path.join(WORK, "out", "dz_all2.npz")
CKPT = os.path.join(WORK, "out", "dz_model.pt")
OUT = os.path.join(WORK, "lib", "dz_weights.npz")

d = np.load(DATA)
STATE_NF = d["S"].shape[1]
sd = torch.load(CKPT, map_location="cpu")

np.savez_compressed(OUT, **{k: v.numpy() for k, v in sd.items()},
                    state_nf=np.array([STATE_NF]))
print(f"exported {len(sd)} tensors -> {OUT} "
      f"({os.path.getsize(OUT)/1e6:.2f} MB)")
for k, v in sd.items():
    print(f"   {k:<28} {tuple(v.shape)}")

# ---- numpy re-implementation, verified against torch below ----------------
W = dict(np.load(OUT))


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def np_forward(s, a, ac, m, ht, hc):
    """s:(NF,) a:(K,ACT) ac:(K,) m:(K,) ht:(H,) hc:(H,) -> logits (K,)"""
    emb_c = W["card.weight"][hc]                     # H x emb
    emb_e = W["evt.weight"][np.clip(ht, 0, W["evt.weight"].shape[0] - 1)]
    x = np.concatenate([emb_c, emb_e], -1)           # H x (emb+8)

    # single-layer GRU, batch_first, hidden 64
    Wi, Wh = W["gru.weight_ih_l0"], W["gru.weight_hh_l0"]
    bi, bh = W["gru.bias_ih_l0"], W["gru.bias_hh_l0"]
    H = Wh.shape[1]
    h = np.zeros(H, dtype=np.float32)
    for t in range(x.shape[0]):
        gi = Wi @ x[t] + bi
        gh = Wh @ h + bh
        ir, iz, in_ = gi[:H], gi[H:2 * H], gi[2 * H:]
        hr, hz, hn = gh[:H], gh[H:2 * H], gh[2 * H:]
        r = _sigmoid(ir + hr)
        z = _sigmoid(iz + hz)
        n = np.tanh(in_ + r * hn)
        h = (1 - z) * n + z * h

    st = _relu(W["state.0.weight"] @ s + W["state.0.bias"])
    st = _relu(W["state.2.weight"] @ st + W["state.2.bias"])
    ctx = np.concatenate([st, h])                    # 128

    K = a.shape[0]
    z1 = np.concatenate([np.repeat(ctx[None, :], K, 0), a, W["card.weight"][ac]], -1)
    y = _relu(z1 @ W["score.0.weight"].T + W["score.0.bias"])
    y = _relu(y @ W["score.2.weight"].T + W["score.2.bias"])
    y = (y @ W["score.4.weight"].T + W["score.4.bias"]).ravel()
    return np.where(m < 0.5, -1e9, y)


# ---- verification against torch ------------------------------------------
sys.path.insert(0, os.path.join(WORK, "lib"))
import dznet  # noqa: E402
# build the reference model directly from the checkpoint: dznet.load() resolves
# paths relative to the agent directory, which is not where this script runs.
tm = dznet._Net(STATE_NF)
tm.load_state_dict(sd)
tm.eval()

rng = np.random.default_rng(0)
idx = rng.choice(len(d["S"]), 40, replace=False)
worst = 0.0
agree = 0
for i in idx:
    s, a, ac, m, ht, hc = (d["S"][i], d["A"][i], d["AC"][i], d["M"][i],
                           d["HT"][i], d["HC"][i])
    with torch.no_grad():
        tl = tm(torch.tensor(s)[None], torch.tensor(a)[None],
                torch.tensor(ac).long()[None], torch.tensor(m)[None],
                torch.tensor(ht).long()[None], torch.tensor(hc).long()[None])[0].numpy()
    nl = np_forward(s, a, ac, m, ht, hc)
    live = m > 0.5
    worst = max(worst, float(np.abs(tl[live] - nl[live]).max()))
    agree += int(np.argmax(tl) == np.argmax(nl))

print(f"\nverification on 40 real decisions:")
print(f"  max |torch - numpy| on live options : {worst:.3e}")
print(f"  argmax agreement                    : {agree}/40")
assert worst < 1e-3, "numpy port does not match torch"
assert agree == 40, "numpy port picks a different option"
print("  PORT VERIFIED -- numpy inference is exact, torch not needed at runtime")
