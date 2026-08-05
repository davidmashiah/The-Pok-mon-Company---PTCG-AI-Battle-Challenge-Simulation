"""Determinized Monte Carlo on pivotal decisions, scored by actual wins.

Why this and not the earlier full-turn search: that one scored positions with a
hand-written evaluation and measured -11.2 points, non-overlapping CIs. A
hand-written value function is a worse judge than the tuned heuristic it was
overriding. This never evaluates a position -- it plays the game to the end and
counts wins, which is unbiased by construction.

Budget arithmetic, measured on this machine:
    rollout to terminal   ~144 ms   (7/sec, median depth 85 engine steps)
    episode allowance      600 s    (cabt.json: actTimeout=0, overage 600)
    -> ~2800 rollouts per episode
Spread over ~60 decisions that is 46 each: useless. Concentrated on the ~10
decisions that decide games it is ~280 each, ~70 per candidate, which separates
a 60% line from a 40% one.

So the whole design rests on spending the budget only where it changes the
answer, and never raising the total.
"""
import time

from cg.api import Observation, OptionType, SelectContext

try:
    from cg.api import search_begin, search_end, search_step
    HAVE = True
except Exception:                                    # pragma: no cover
    HAVE = False


def _i(x, d=-1):
    try:
        return int(x)
    except (TypeError, ValueError):
        return d


def is_pivotal(obs: Observation, cheap_scores=None):
    """Is this decision worth real compute?

    Pivotal means the game can turn here: someone is close to taking their last
    prizes, a knockout is live, or the heuristic itself is nearly indifferent
    between its top choices (exactly where its judgement is least reliable).
    """
    try:
        st = obs.current
        if st is None or obs.select is None:
            return False
        if _i(obs.select.context) != _i(SelectContext.MAIN):
            return False
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]

        if len(me.prize or []) <= 3 or len(opp.prize or []) <= 3:
            return True                      # prize race is live
        oact = opp.active[0] if (opp.active and opp.active[0]) else None
        if oact is not None and oact.maxHp and oact.hp <= 0.5 * oact.maxHp:
            return True                      # a knockout is in reach
        if cheap_scores and len(cheap_scores) >= 2:
            s = sorted(cheap_scores, reverse=True)
            if s[0] - s[1] <= max(1.0, abs(s[0]) * 0.02):
                return True                  # heuristic is indifferent
        return False
    except Exception:
        return False


def _playout(state, my_index, rollout_policy, max_steps=400):
    """Play a determinized line to the end. Returns 1 win / 0 loss / 0.5 draw."""
    cur = state
    for _ in range(max_steps):
        sobs = cur.observation
        st = sobs.current
        if st is not None and st.result != -1:
            if st.result == my_index:
                return 1.0
            if st.result == 2:
                return 0.5
            return 0.0
        sel = sobs.select
        if sel is None:
            return 0.5
        try:
            pick = rollout_policy(sobs)
            nxt = search_step(cur.searchId, list(pick))
        except Exception:
            return 0.5
        if nxt is None:
            return 0.5
        cur = nxt
    return 0.5                                # unresolved -> neutral, not a win


def choose(obs: Observation, det, rollout_policy, candidates,
           time_budget=8.0, max_candidates=4, min_rollouts=12):
    """Return the candidate with the best measured win rate, or None.

    Returns None whenever the evidence is too thin to overrule the heuristic --
    the caller keeps its own choice. Refusing to answer is the whole safety
    property: the tuned policy is worth ~700 rating points and a noisy sample
    must not be allowed to override it.
    """
    if not HAVE or obs.select is None or obs.current is None:
        return None
    cand = list(candidates)[:max_candidates]
    if len(cand) < 2:
        return None
    kw = det.build(obs)
    if kw is None:
        return None

    my_index = obs.current.yourIndex
    t0 = time.time()
    wins = {a: 0.0 for a in cand}
    runs = {a: 0 for a in cand}

    try:
        # round-robin so an early timeout still leaves candidates comparable
        while time.time() - t0 < time_budget:
            progressed = False
            for a in cand:
                if time.time() - t0 >= time_budget:
                    break
                try:
                    root = search_begin(obs, manual_coin=False, **kw)
                    if root is None:
                        continue
                    nxt = search_step(root.searchId, [a])
                    if nxt is None:
                        continue
                    wins[a] += _playout(nxt, my_index, rollout_policy)
                    runs[a] += 1
                    progressed = True
                finally:
                    try:
                        search_end()
                    except Exception:
                        pass
            if not progressed:
                break
    except Exception:
        return None

    scored = [(wins[a] / runs[a], runs[a], a) for a in cand if runs[a] >= min_rollouts]
    if len(scored) < 2:
        return None
    scored.sort(reverse=True)
    best_wr, best_n, best_a = scored[0]
    nxt_wr, nxt_n, _ = scored[1]
    # require a margin comfortably outside sampling noise before overruling
    se = (0.25 / best_n + 0.25 / nxt_n) ** 0.5
    if best_wr - nxt_wr < 2.0 * se:
        return None
    return best_a
