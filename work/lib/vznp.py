"""Numpy-only win-probability value net, used as the SEARCH's leaf evaluator.

Why this exists: both search attempts failed because of the leaf function, not
the search. v30 (1 ply) scored 0.3500 and v31 (2 ply + PIMC) scored 0.0530
against v14 -- deeper lookahead made it WORSE, which is the signature of
optimising harder against a wrong target. The hand-written evaluate() rewards
invested energy and surviving HP, exactly what a resource-dump maximises.

This replaces it with a net trained on real outcomes. Measured on a game-split
holdout: AUC 0.7986 against a 0.7039 prize-difference baseline, versus the old
linear value net's 0.7121 (which was statistically indistinguishable from just
counting prizes, and therefore useless to a search that already weights prizes
x1000).

Contract, as everywhere else here: never raise, and if the weights are missing
return None so the caller keeps its own ordering rather than silently scoring
every position identically.
"""
import os

import numpy as np

_KAGGLE = "/kaggle_simulations/agent"
_W = None
_DTAB = None
_TRIED = False


def loaded():
    return _W is not None


def load():
    global _W, _TRIED
    if _W is not None:
        return True
    if _TRIED:
        return False
    _TRIED = True
    for cand in ("vz_weights.npz",
                 os.path.join(_KAGGLE, "vz_weights.npz"),
                 os.path.join("work", "lib", "vz_weights.npz")):
        try:
            if os.path.exists(cand):
                _W = {k: v.astype(np.float32)
                      for k, v in dict(np.load(cand)).items()}
                return True
        except Exception:
            continue
    return False


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _card_rep(ids):
    global _DTAB
    e = _W["card.weight"][ids]
    dw = _W.get("desc.weight")
    if dw is None:
        return e
    if _DTAB is None:
        import dzfeat
        _DTAB = dzfeat.card_desc_table()
    return e + _DTAB[ids] @ dw.T


def _gru(hc, ht):
    W = _W
    ne = W["evt.weight"].shape[0]
    x = np.concatenate([_card_rep(hc),
                        W["evt.weight"][np.clip(ht, 0, ne - 1)]], -1)
    Wi, Wh = W["gru.weight_ih_l0"], W["gru.weight_hh_l0"]
    bi, bh = W["gru.bias_ih_l0"], W["gru.bias_hh_l0"]
    H = Wh.shape[1]
    gi_all = x @ Wi.T + bi
    h = np.zeros(H, dtype=np.float32)
    for t in range(x.shape[0]):
        gi = gi_all[t]
        gh = Wh @ h + bh
        r = _sigmoid(gi[:H] + gh[:H])
        z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        h = (1 - z) * n + z * h
    return h


def win_prob(state_vec, hist_types, hist_cards):
    """-> P(we win) in (0,1), or None if the net is unavailable."""
    if _W is None:
        return None
    try:
        W = _W
        h = _gru(hist_cards, hist_types)
        st = np.maximum(W["state.0.weight"] @ state_vec + W["state.0.bias"], 0.0)
        st = np.maximum(W["state.2.weight"] @ st + W["state.2.bias"], 0.0)
        z = np.concatenate([st, h])
        y = np.maximum(W["head.0.weight"] @ z + W["head.0.bias"], 0.0)
        # Dropout occupies index 2 in the trained Sequential, so the second and
        # third Linear layers are head.3 and head.5. Accept either layout so a
        # model trained before dropout was added still loads.
        k2 = "head.3.weight" if "head.3.weight" in W else "head.2.weight"
        k3 = "head.5.weight" if "head.5.weight" in W else "head.4.weight"
        b2, b3 = k2.replace("weight", "bias"), k3.replace("weight", "bias")
        y = np.maximum(W[k2] @ y + W[b2], 0.0)
        out = (W[k3] @ y + W[b3]).ravel()[0]
        return float(_sigmoid(out))
    except Exception:
        return None
