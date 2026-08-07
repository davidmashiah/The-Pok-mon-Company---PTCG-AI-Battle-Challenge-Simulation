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

try:
    from cg.api import all_attack, all_card_data
    _ATK = {a.attackId: a for a in all_attack()}
    _CARD = {c.cardId: c for c in all_card_data()}
except Exception:
    _ATK, _CARD = {}, {}


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


BUDGET_S = _f("PTCG_SEARCH_BUDGET", 0.9)
DETERMINIZATIONS = _i("PTCG_SEARCH_DET", 3)
MAX_CANDIDATES = _i("PTCG_SEARCH_CANDS", 3)
MARGIN = _f("PTCG_SEARCH_MARGIN", 1000.0)
ROLLOUT_STEPS = _i("PTCG_SEARCH_STEPS", 30)
ENABLED = _i("PTCG_SEARCH_ON", 1)
# Only search when the policy's own top two options are within GATE of each
# other. 0 disables the gate (search every eligible decision).
GATE = _f("PTCG_SEARCH_GATE", 0.0)
# Search non-MAIN prompts too (Munkidori targeting, Boss target, promote).
SEARCH_NONMAIN = _i("PTCG_SEARCH_NONMAIN", 1)

# Real 60-card lists read verbatim out of the top-50 teams' own replays. The
# opponent's hidden cards are dealt from whichever of these best explains the
# cards we have actually SEEN them play.
#
# This is the difference between a determinization and a fantasy. An earlier
# build here determinized most of the field as a deck that could not exist --
# the library had zero lists for archetypes it faced constantly -- and every
# playout was a game against an opponent nobody plays. Guessing Grimmsnarl for
# all of them is the same mistake in a politer form: it is right for 32% of the
# field and wrong for the rest.
try:
    from opp_library import DECKS as _LIB
except Exception:
    _LIB = []
_OPP_REF = None                    # our own 60, the fallback prior
_BASIC_FOR_FACEDOWN = 646          # Marnie's Impidimp, a Basic
_MIN_MATCH = 4                     # below this the evidence is too thin to act

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


def _damage(o, opt):
    """What this attack would actually do to their Active, with Weakness.

    Without this the rollout chose among attacks by option TYPE alone, i.e.
    effectively at random -- so a simulated turn would pass up an available
    knockout. The leaf is prize-dominated, so missing knockouts in the rollout
    biases every candidate's score in the same direction as the mistake it is
    supposed to detect.
    """
    a = _ATK.get(int(getattr(opt, "attackId", 0) or 0))
    if a is None:
        return 0.0, False
    dmg = float(a.damage or 0)
    if dmg <= 0:
        return 0.0, False
    try:
        st = o.current
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        mine = me.active[0] if (me.active and me.active[0]) else None
        tgt = opp.active[0] if (opp.active and opp.active[0]) else None
        if mine is not None and tgt is not None:
            tc = _CARD.get(tgt.id)
            mc = _CARD.get(mine.id)
            if (tc is not None and mc is not None
                    and getattr(tc, "weakness", None) is not None
                    and int(tc.weakness) == int(mc.energyType)):
                dmg *= 2.0
            return dmg, dmg >= float(tgt.hp)
    except Exception:
        pass
    return dmg, False


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
        s = float(_PREF.get(t, 1))
        if t == int(OptionType.ATTACK):
            dmg, ko = _damage(o, opt)
            # A knockout is worth a prize and outranks any development; among
            # non-lethal attacks, more damage is still strictly better.
            s = 20.0 + dmg / 1000.0 if ko else 2.0 + dmg / 100000.0
        # An unbounded Ability loop is a known hang in this engine; damp it.
        if t == int(OptionType.ABILITY) and ability_used[0] >= 5:
            s = -1.0
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


def _seen_opponent(o, opp_idx):
    """Every opponent card we have actually observed, as a multiset."""
    seen = []
    try:
        p = o.current.players[opp_idx]
        for mon in _side(p):
            seen.append(mon.id)
            for c in (mon.energyCards or []):
                seen.append(c.id)
            for c in (mon.tools or []):
                seen.append(c.id)
            for c in (mon.preEvolution or []):
                seen.append(c.id)
        for c in (p.discard or []):
            seen.append(c.id)
    except Exception:
        pass
    return seen


def _match_opponent(o, opp_idx, my_deck):
    """Pick the real decklist that best explains what we have seen them play.

    Multiset overlap, so a list is only credited for copies it could actually
    supply. Below _MIN_MATCH observed cards we keep the Grimmsnarl prior rather
    than committing to a guess -- on turn one every deck matches equally well
    and a confident wrong answer is worse than a vague right one.
    """
    seen = _seen_opponent(o, opp_idx)
    if not _LIB or len(seen) < _MIN_MATCH:
        return list(_OPP_REF or my_deck)
    want = {}
    for c in seen:
        want[c] = want.get(c, 0) + 1
    best, best_score = None, 0
    for deck in _LIB:
        have = {}
        for c in deck:
            have[c] = have.get(c, 0) + 1
        score = 0
        for cid, n in want.items():
            h = have.get(cid, 0)
            score += n if h >= n else h
        if score > best_score:
            best, best_score = deck, score
    if best is None or best_score < _MIN_MATCH:
        return list(_OPP_REF or my_deck)
    return list(best)


def _opp_pool(o, me_idx, my_deck):
    """The matched list MINUS everything we have already watched them play.

    Built once per decision, not once per playout: the subtraction is O(seen x
    deck) and doing it inside the playout loop cost 12-26 s/episode against
    w30's 5 s for no extra information.
    """
    ref = _match_opponent(o, 1 - me_idx, my_deck)
    for cid in _seen_opponent(o, 1 - me_idx):
        try:
            ref.remove(cid)
        except ValueError:
            pass
    return ref


def _determinize(o, me_idx, my_deck, opp_pool):
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

    ref = list(opp_pool)
    random.shuffle(ref)
    od, op_, oh = int(opp.deckCount or 0), len(opp.prize or []), int(opp.handCount or 0)
    need = od + op_ + oh
    while len(ref) < need:
        ref = ref + (list(opp_pool) or list(my_deck))
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


def _score_all(o, me_idx, my_deck, opp_pool, cands, dets, deadline):
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
        det = _determinize(o, me_idx, my_deck, opp_pool)
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
    if len(chosen) != 1:
        return None
    if order is None and not SEARCH_NONMAIN:
        return None
    if _OPP_REF is None:
        _OPP_REF = list(my_deck)
    t0 = time.time()
    try:
        o = to_observation_class(obs_dict)
    except Exception:
        return None
    if o.select is None:
        return None
    is_main = (o.select.context == SelectContext.MAIN)
    if not (is_main or SEARCH_NONMAIN):
        return None
    opts = o.select.option or []
    if len(opts) < 2 or o.current is None:
        return None
    if getattr(o, "search_begin_input", None) is None:
        return None

    # Non-MAIN selections were never searched, and that is where this deck's
    # most leveraged decision lives: Munkidori's Adrena-Brain moves up to 3
    # damage counters onto one of THEIR Pokemon, four copies, most turns. Which
    # body, which target and how many counters are all non-MAIN prompts, and in
    # a mirror decided by whether a two-hit knockout becomes a one-hit knockout
    # that choice is the game. `order` only exists when the model scored this
    # context, so outside MAIN fall back to the options themselves.
    ranked = order if order else list(range(len(opts)))
    cands = []
    for i in list(chosen) + list(ranked):
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
    if GATE > 0 and is_main and sc is not None and len(cands) >= 2:
        top = sc[cands[0]]
        rival = max(sc[c] for c in cands[1:])
        if (top - rival) > GATE:
            return None

    _STATS["eligible"] += 1
    me_idx = o.current.yourIndex
    deadline = t0 + BUDGET_S
    opp_pool = _opp_pool(o, me_idx, my_deck)
    scored, wins, n = _score_all(o, me_idx, my_deck, opp_pool, cands,
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
