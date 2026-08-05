"""Numpy-only inference for the DouZero-style scorer. No torch at runtime.

Torch is 518 MB installed against a ~197 MB tarball limit, and depending on it
being present in the Kaggle image would mean a SILENT fallback to the tuned
policy if it were not -- the exact failure mode that has already killed five
components in this project. The weights are 0.44 MB of numpy, and
work/tools/dz_export.py asserts this file reproduces torch's own logits on real
decisions before anything ships.

Contract, learned the hard way:
  * NEVER raise. Any failure returns None and the tuned policy keeps its pick.
  * NEVER overrule without evidence: below `margin` logits of separation the
    model is not clearly preferring anything, so the policy -- worth ~700
    rating -- keeps its own choice.
  * No __file__ (the harness exec()s main.py, so it does not exist).
"""
import os

import numpy as np

_KAGGLE = "/kaggle_simulations/agent"
_W = None
_DTAB = None
_LOAD_TRIED = False
_STATE_NF = 0

MAX_CAND = 24
ACT_NF = 10


def loaded():
    return _W is not None


def state_nf():
    return _STATE_NF


def load():
    """Load weights from beside the agent. Returns True if the model is live."""
    global _W, _LOAD_TRIED, _STATE_NF
    if _W is not None:
        return True
    if _LOAD_TRIED:
        return False
    _LOAD_TRIED = True
    for cand in ("dz_weights.npz",
                 os.path.join(_KAGGLE, "dz_weights.npz"),
                 os.path.join("work", "lib", "dz_weights.npz")):
        try:
            if os.path.exists(cand):
                w = dict(np.load(cand))
                _STATE_NF = int(w["state_nf"][0])
                _W = {k: v.astype(np.float32) if v.dtype != np.int64 else v
                      for k, v in w.items()}
                return True
        except Exception:
            continue
    return False


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _card_rep(ids):
    """Learned id embedding plus the fixed attribute projection.

    The descriptor table is RECOMPUTED from the bundled engine rather than
    shipped in the weights, so the table the net trained against and the table
    it serves against cannot silently disagree. Models trained before
    descriptors existed simply have no 'desc.weight' and fall back to the plain
    embedding.
    """
    e = _W["card.weight"][ids]
    dw = _W.get("desc.weight")
    if dw is None:
        return e
    global _DTAB
    if _DTAB is None:
        import dzfeat
        _DTAB = dzfeat.card_desc_table()
    return e + _DTAB[ids] @ dw.T


def _gru(hc, ht):
    """Single-layer GRU over the history window -> final hidden (64,)."""
    W = _W
    emb_c = _card_rep(hc)
    ne = W["evt.weight"].shape[0]
    emb_e = W["evt.weight"][np.clip(ht, 0, ne - 1)]
    x = np.concatenate([emb_c, emb_e], -1)

    Wi, Wh = W["gru.weight_ih_l0"], W["gru.weight_hh_l0"]
    bi, bh = W["gru.bias_ih_l0"], W["gru.bias_hh_l0"]
    H = Wh.shape[1]
    gi_all = x @ Wi.T + bi                     # T x 3H, precomputed in one gemm
    h = np.zeros(H, dtype=np.float32)
    for t in range(x.shape[0]):
        gi = gi_all[t]
        gh = Wh @ h + bh
        r = _sigmoid(gi[:H] + gh[:H])
        z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        h = (1 - z) * n + z * h
    return h


def logits(state_vec, act_feats, act_cards, mask, hist_types, hist_cards):
    """-> (K,) logits, masked options at -1e9. Mirrors the torch forward()."""
    W = _W
    h = _gru(hist_cards, hist_types)
    st = np.maximum(W["state.0.weight"] @ state_vec + W["state.0.bias"], 0.0)
    st = np.maximum(W["state.2.weight"] @ st + W["state.2.bias"], 0.0)
    ctx = np.concatenate([st, h])

    K = act_feats.shape[0]
    z = np.concatenate([np.repeat(ctx[None, :], K, 0), act_feats,
                        _card_rep(act_cards)], -1)
    y = np.maximum(z @ W["score.0.weight"].T + W["score.0.bias"], 0.0)
    y = np.maximum(y @ W["score.2.weight"].T + W["score.2.bias"], 0.0)
    y = (y @ W["score.4.weight"].T + W["score.4.bias"]).ravel()
    return np.where(mask < 0.5, -1e9, y)


def rank(state_vec, act_feats, act_cards, mask, hist_types, hist_cards,
         n_opts, margin=0.35):
    """Model's preferred option index, or None to defer to the tuned policy."""
    if _W is None or n_opts < 2 or n_opts > MAX_CAND:
        return None
    try:
        lg = logits(state_vec, act_feats, act_cards, mask,
                    hist_types, hist_cards)[:n_opts]
        order = np.argsort(-lg)
        if float(lg[order[0]] - lg[order[1]]) < margin:
            return None                       # not confident enough to overrule
        return int(order[0])
    except Exception:
        return None


def rank_all(state_vec, act_feats, act_cards, mask, hist_types, hist_cards,
             n_opts):
    """Full preference order over legal options, or None. No margin gate."""
    if _W is None or n_opts < 1 or n_opts > MAX_CAND:
        return None
    try:
        lg = logits(state_vec, act_feats, act_cards, mask,
                    hist_types, hist_cards)[:n_opts]
        return [int(i) for i in np.argsort(-lg)]
    except Exception:
        return None
