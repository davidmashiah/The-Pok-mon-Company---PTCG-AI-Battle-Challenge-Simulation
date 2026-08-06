"""Determinized search over the policy's own top-K candidates.

Knobs come from the environment so a tuner never has to rewrite a file inside
the bundle -- rewriting bundle config is what silently invalidated an entire
router search in this project (`from X import Y` is a no-op once X is in
sys.modules). On Kaggle none of these are set and the defaults apply.
"""
import os
import random
import time

_OK = True
try:
    from cg.api import (OptionType, SelectContext, search_begin, search_end,
                        search_release, search_step, to_observation_class)
except Exception:
    _OK = False


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _i(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


BUDGET_S = _f("PTCG_SEARCH_BUDGET", 3.0)
DETERMINIZATIONS = _i("PTCG_SEARCH_DET", 8)
MAX_CANDIDATES = _i("PTCG_SEARCH_CANDS", 4)
MARGIN = _f("PTCG_SEARCH_MARGIN", 500.0)
ROLLOUT_STEPS = _i("PTCG_SEARCH_STEPS", 30)
ENABLED = _i("PTCG_SEARCH_ON", 1)
# Only search when the policy's own top two options are within GATE of each
# other. 0 disables the gate (search every eligible decision).
GATE = _f("PTCG_SEARCH_GATE", 0.0)

# A reference 60 for guessing the opponent's hidden cards. Marnie's Grimmsnarl
# is 32% of the top 50 and is also our own list, so it is both the single most
# likely opponent deck and exactly right in the mirror -- which is the matchup
# carrying ~47% of the panel weight.
_OPP_REF = None
_BASIC_FOR_FACEDOWN = 646          # Marnie's Impidimp, a Basic

_RANK = {"order": None, "scores": None, "token": 0}
_STATS = {
    "decisions": 0, "eligible": 0, "ran": 0, "overrode": 0,
    "agreed": 0, "errors": 0, "playouts": 0, "time": 0.0,
    "starved": 0, "gapsum": 0.0, "gapmax": 0.0, "gaphits": 0,
}
# Observed (best - base) leaf gaps, so MARGIN is set from the distribution
# rather than guessed. A margin below the noise floor of the leaf turns the
# search into a random-move generator.
GAPS = []


def get_stats():
    return dict(_STATS)


def reset_stats():
    for k in _STATS:
        _STATS[k] = 0.0 if k in ("time", "gapsum", "gapmax") else 0
    del GAPS[:]


def reset_decision():
    _RANK["order"] = None
    _RANK["scores"] = None
    _RANK["token"] += 1


def note_rank(order, scores):
    """Called by main._model_action with its full ranking of the options."""
    _RANK["order"] = list(order)
    _RANK["scores"] = list(scores)


# ------------------------------------------------------------------ leaf
def _side(p):
    out = []
    if p.active and p.active[0] is not None:
        out.append(p.active[0])
    for b in (p.bench or []):
        if b is not None:
            out.append(b)
    return out


def _evaluate(o, me_idx):
    """Prizes dominate; damage is the tiebreak. Deliberately arithmetic."""
    st = o.current
    if st is None:
        return 0.0
    if st.result is not None and st.result != -1:
        return 1e6 if st.result == me_idx else -1e6
    me = st.players[me_idx]
    opp = st.players[1 - me_idx]
    # prize LISTS shrink as their owner takes prizes
    val = 1000.0 * ((6 - len(me.prize)) - (6 - len(opp.prize)))
    for p in _side(opp):
        val += (p.maxHp - p.hp) * 1.0
    for p in _side(me):
        val -= (p.maxHp - p.hp) * 0.8
    val += 12.0 * len([b for b in (me.bench or []) if b is not None])
    val += 2.0 * int(getattr(me, "handCount", 0) or 0)
    return val


# --------------------------------------------------------------- rollout
_PREF = {}
if _OK:
    _PREF = {
        int(OptionType.ABILITY): 6,
        int(OptionType.EVOLVE): 5,
        int(OptionType.PLAY): 4,
        int(OptionType.ATTACH): 3,
        int(OptionType.ATTACK): 2,
        int(OptionType.RETREAT): 1,
        int(OptionType.END): 0,
    }


def _greedy(o, ability_used):
    """Cheap turn completion. Development first, then attack, then end --
    attacking ends the turn, so an attack must be the LAST thing done."""
    sel = o.select
    opts = sel.option or []
    if not opts:
        return None
    k = max(1, int(sel.minCount or 1))
    if sel.context != SelectContext.MAIN:
        return list(range(min(k, len(opts))))
    best_i, best_s = 0, -1e9
    for i, opt in enumerate(opts):
        t = int(opt.type)
        s = _PREF.get(t, 1)
        # An unbounded Ability loop is a known hang in this engine; damp it.
        if t == int(OptionType.ABILITY) and ability_used[0] >= 5:
            s = -1
        if s > best_s:
            best_i, best_s = i, s
    if int(opts[best_i].type) == int(OptionType.ABILITY):
        ability_used[0] += 1
    return [best_i]


def _rollout(sid, o, me_idx):
    """Finish our turn, let the opponent answer once, then score."""
    start_turn = o.current.turn
    ability_used = [0]
    steps = 0
    while steps < ROLLOUT_STEPS:
        st = o.current
        if st is None or (st.result is not None and st.result != -1):
            break
        if st.turn > start_turn + 1:
            break
        if o.select is None or not (o.select.option or []):
            break
        pick = _greedy(o, ability_used)
        if not pick:
            break
        nxt = search_step(sid, pick)
        if nxt is None or nxt.observation is None:
            break
        o, sid = nxt.observation, nxt.searchId
        steps += 1
    return _evaluate(o, me_idx), sid


def _determinize(o, me_idx, my_deck):
    me = o.current.players[me_idx]
    opp = o.current.players[1 - me_idx]
    mine = list(my_deck)
    random.shuffle(mine)
    need_d = int(me.deckCount or 0)
    need_p = len(me.prize or [])
    your_deck = mine[:need_d]
    your_prize = mine[need_d:need_d + need_p]
    while len(your_prize) < need_p:            # deck+prize can exceed our 60
        your_prize.append(mine[0])             # only when the engine says so
    ref = list(_OPP_REF or my_deck)
    random.shuffle(ref)
    od, op_, oh = int(opp.deckCount or 0), len(opp.prize or []), int(opp.handCount or 0)
    need = od + op_ + oh
    while len(ref) < need:
        ref = ref + ref
    return (your_deck, your_prize, ref[:od], ref[od:od + op_],
            ref[od + op_:od + op_ + oh])


def _playout(o, me_idx, det, pick):
    """One candidate, one FIXED determinization -> leaf value."""
    sid = None
    try:
        yd, yp, od, op_, oh = det
        st = search_begin(o, list(yd), list(yp), list(od), list(op_),
                          list(oh), [_BASIC_FOR_FACEDOWN], True)
        sid = st.searchId
        nxt = search_step(sid, list(pick))
        if nxt is None or nxt.observation is None:
            return None
        val, sid = _rollout(nxt.searchId, nxt.observation, me_idx)
        _STATS["playouts"] += 1
        return val
    except Exception:
        _STATS["errors"] += 1
        return None
    finally:
        if sid is not None:
            # Release as each playout ends. Not releasing made one move in
            # this engine take 1089 s in an earlier build.
            try:
                search_release(sid)
            except Exception:
                pass


def _score_all(o, me_idx, my_deck, cands, dets, deadline):
    """PAIRED comparison: every candidate is judged on the SAME hidden-info
    samples, so the difference between two candidates is not contaminated by
    the difference between two shuffles.

    Unpaired sampling made the leaf gaps unreadable -- one determinization
    drawing into a prize is worth 1000, which swamped every real distinction at
    the 2-3 samples we can afford. With `manual_coin=True` and a deterministic
    rollout policy, a fixed determinization makes a playout reproducible, so
    the paired difference is close to pure signal.

    Returns (mean value per candidate, wins per candidate, samples taken).
    """
    total = dict((c, 0.0) for c in cands)
    wins = dict((c, 0) for c in cands)
    n = 0
    for _ in range(dets):
        if time.time() > deadline:
            break
        det = _determinize(o, me_idx, my_deck)
        vals = {}
        for c in cands:
            v = _playout(o, me_idx, det, [c])
            if v is not None:
                vals[c] = v
        if len(vals) < len(cands):
            continue                      # partial sample proves nothing
        best_c = max(vals, key=lambda k: vals[k])
        wins[best_c] += 1
        for c in cands:
            total[c] += vals[c]
        n += 1
    if not n:
        return None, None, 0
    return dict((c, total[c] / n) for c in cands), wins, n


def validate(obs_dict, chosen, my_deck):
    """Return a better single-option selection, or None to keep `chosen`."""
    global _OPP_REF
    _STATS["decisions"] += 1
    if not (_OK and ENABLED):
        return None
    order = _RANK["order"]
    if order is None or len(chosen) != 1:
        return None
    if _OPP_REF is None:
        _OPP_REF = list(my_deck)
    t0 = time.time()
    try:
        o = to_observation_class(obs_dict)
    except Exception:
        return None
    if o.select is None or o.select.context != SelectContext.MAIN:
        return None
    opts = o.select.option or []
    if len(opts) < 2 or o.current is None:
        return None
    if getattr(o, "search_begin_input", None) is None:
        return None

    cands = []
    for i in list(chosen) + list(order):
        if i not in cands and 0 <= i < len(opts):
            cands.append(i)
        if len(cands) >= MAX_CANDIDATES:
            break
    if len(cands) < 2:
        return None

    # Spend the budget where a decision is actually in doubt. When the policy
    # ranks its top option far above the rest, searching it only burns time we
    # then cannot spend on the close calls -- which is exactly the spot the
    # rank-88 team reports needing help with.
    sc = _RANK["scores"]
    if GATE > 0 and sc is not None and len(cands) >= 2:
        top = sc[cands[0]]
        rival = max(sc[c] for c in cands[1:])
        if (top - rival) > GATE:
            return None

    _STATS["eligible"] += 1
    me_idx = o.current.yourIndex
    deadline = t0 + BUDGET_S
    scored, wins, n = _score_all(o, me_idx, my_deck, cands,
                                 DETERMINIZATIONS, deadline)
    try:
        search_end()
    except Exception:
        pass
    _STATS["time"] += time.time() - t0
    if n < DETERMINIZATIONS:
        # Ran out of budget before every determinization completed. Counted,
        # never quietly used as if it were a full comparison.
        _STATS["starved"] += 1
    if not scored or n == 0:
        return None
    _STATS["ran"] += 1

    base = chosen[0]
    if base not in scored:
        return None
    best = max(scored, key=lambda k: scored[k])
    gap = scored[best] - scored[base]
    if gap > 0:
        GAPS.append(gap)
        _STATS["gapsum"] += gap
        _STATS["gaphits"] += 1
        if gap > _STATS["gapmax"]:
            _STATS["gapmax"] = gap
    # Two independent conditions, both required. The mean must clear a real
    # margin AND the challenger must actually beat the policy's pick on most
    # of the sampled worlds -- a single lucky determinization should never be
    # able to move a decision on its own.
    majority = wins.get(best, 0) * 2 > n
    if best == base or gap <= MARGIN or not majority:
        _STATS["agreed"] += 1
        return None
    _STATS["overrode"] += 1
    return [best]
