"""Clone of a strong human-built pilot: their deck AND their policy.

Everything before this imitated the whole field, which taught the net other
decks' strategies. Measured consequence: on a deck the net had not learned, it
picked correctly 0.3555 of the time against a 0.4088 always-first baseline --
worse than trivial, so the margin gate was the only thing keeping the agent
near even.

This is a different target. One competitor accounts for 1,379 published games
at a 0.606 win rate against 119 distinct opponents, all with one coherent
Mega Lopunny ex list. That is 64,947 decisions from a single policy that is
measurably far stronger than ours, playing the deck bundled here. The net is
trained on those decisions, and it drives.

Fallback order, strongest evidence first:
  1. PROVEN lethal  -- the engine itself played the line out to a win under our
     determinization. Nothing heuristic beats a proof.
  2. THE CLONE      -- ranks the legal options; this is the policy we are here
     to reproduce, so it decides by default.
  3. GENERIC POLICY -- archetype-agnostic, never raises, always legal. This is
     policy.py, NOT the Mega Lucario logic from v14: that logic reasons about
     Riolu/Hariyama lines this deck does not contain and would mislead it.

No __file__ anywhere: kaggle_environments exec()s this file, so __file__ is
undefined there. That bug already killed one submission.
"""
from __future__ import annotations

import os

from cg.api import Observation, to_observation_class

# --- deck ------------------------------------------------------------------
# Their exact 60, seen 315 times across their games.
DECK = [
    11, 11, 11, 11, 13, 14, 14, 14, 66, 66,
    66, 66, 174, 305, 305, 305, 305, 848, 848, 848,
    848, 849, 849, 849, 1086, 1086, 1086, 1086, 1121, 1121,
    1121, 1121, 1122, 1122, 1122, 1122, 1152, 1152, 1152, 1152,
    1174, 1174, 1174, 1174, 1182, 1182, 1182, 1197, 1225, 1225,
    1225, 1225, 1227, 1227, 1227, 1227, 1229, 1229, 1229, 1229,
]


def _load_deck():
    """deck.csv is the contract with the harness; the literal above is a fallback."""
    for p in ("deck.csv", "/kaggle_simulations/agent/deck.csv"):
        try:
            if os.path.exists(p):
                with open(p) as fh:
                    ids = [int(x) for x in fh.read().split() if x.strip()]
                if len(ids) == 60:
                    return ids
        except Exception:
            pass
    return list(DECK)


my_deck = _load_deck()

# --- generic fallback policy ----------------------------------------------
_POLICY_OK = False
try:
    import policy as _policy
    _POLICY_OK = True
except Exception:
    _POLICY_OK = False

# --- proven-lethal search --------------------------------------------------
_FS_OK = False
try:
    import fsearch as _fs
    _FS_OK = True
except Exception:
    _FS_OK = False

# --- the clone -------------------------------------------------------------
_DZ_OK = False
try:
    import dzfeat as _dzf
    import dznp as _dznp
    _DZ_OK = _dznp.load()
except Exception:
    _DZ_OK = False

DZ_MARGIN = 0.0          # the clone IS the policy here, so it decides by default
_DZ_HIST = None
_DZ_STATS = {"calls": 0, "fired": 0, "changed": 0}
_prev_turn = -1


def _dz_reset():
    global _DZ_HIST
    if _DZ_OK:
        try:
            _DZ_HIST = _dzf.History()
        except Exception:
            pass


def _dz_observe(obs_dict):
    """Feed every frame's logs. The extractor pushed logs on every ACTIVE frame
    and this agent is invoked on exactly those, so the two streams match."""
    if not _DZ_OK:
        return
    try:
        if _DZ_HIST is None:
            _dz_reset()
        _DZ_HIST.push(obs_dict)
    except Exception:
        pass


def _dz_order(obs_dict, n):
    """Full preference order over legal options, or None."""
    if not _DZ_OK or _DZ_HIST is None:
        return None
    try:
        sel = obs_dict.get("select") or {}
        opts = sel.get("option") or []
        if len(opts) < 2 or len(opts) > _dzf.MAX_CAND:
            return None
        me = (obs_dict.get("current") or {}).get("yourIndex", 0)
        sv = _dzf.featurize(obs_dict, me)
        if sv is None or sv.shape[0] != _dznp.state_nf():
            return None
        af, ac, mk = _dzf.encode_options(obs_dict, opts, me)
        ht, hc = _DZ_HIST.arrays()
        _DZ_STATS["calls"] += 1
        order = _dznp.rank_all(sv, af, ac, mk, ht, hc, min(n, len(opts)))
        if order:
            _DZ_STATS["fired"] += 1
        return order
    except Exception:
        return None


def _lethal(obs_dict, obs: Observation):
    if not _FS_OK:
        return None
    try:
        if not _fs.lethal_plausible(obs):
            return None
        det = _fs.Determinizer(obs, my_deck)
        return _fs.find_lethal(obs, det, time_budget=0.8)
    except Exception:
        return None


def _fallback(obs, obs_dict):
    if _POLICY_OK:
        try:
            r = _policy.choose(obs, my_deck)
            if r:
                return r
        except Exception:
            pass
    return None


def agent(obs_dict: dict) -> list:
    _dz_observe(obs_dict)          # before any early return: history must not skip frames
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return my_deck if obs_dict.get("select") is None else [0]
    if obs.select is None:
        return my_deck

    global _prev_turn
    try:
        t = obs.current.turn
        if t < _prev_turn:         # new episode: history must not leak across games
            _dz_reset()
        _prev_turn = t
    except Exception:
        pass

    n = len(obs.select.option)
    lo = max(1, obs.select.minCount)
    hi = max(min(obs.select.maxCount, n), min(lo, n))

    try:
        # 1. a proven win beats anything either policy prefers
        won = _lethal(obs_dict, obs)
        if won:
            won = [i for i in won if 0 <= i < n]
            if won and obs.select.minCount <= len(won) <= obs.select.maxCount:
                return won

        # 2. the clone
        ordered = _dz_order(obs_dict, n)

        # 3. generic fallback, and as a tail so the count can always be filled
        tail = _fallback(obs, obs_dict) or list(range(n))
        tail = [i for i in tail if 0 <= i < n]
        if ordered:
            if tail and tail[0] != ordered[0]:
                _DZ_STATS["changed"] += 1
            seen = set(ordered)
            ordered = list(ordered) + [i for i in tail if i not in seen]
        else:
            ordered = tail
        ordered = [i for i in ordered if 0 <= i < n]
        if not ordered:
            return list(range(min(lo, n)))
        return ordered[:hi]
    except Exception:
        return list(range(min(lo, n)))
