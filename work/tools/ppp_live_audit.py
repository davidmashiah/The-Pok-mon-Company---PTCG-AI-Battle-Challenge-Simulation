"""Does the Premium Power Pro +30 survive the frame on which we play the card?

v32's "PPP fix" adds +30 to the attack planner's damage model, gated on

    self.hand_counts[C.PREMIUM_POWER_PRO] > 0

But the printed effect is "DURING THIS TURN, attacks used by your {F} Pokemon
do 30 more damage to your opponent's Active Pokemon". The card is an Item: the
moment we play it, it leaves the hand for the discard, hand_counts goes to 0,
and the planner reverts to the UNFIXED damage model -- for every remaining
decision of the turn, which is exactly the set of decisions the +30 was
supposed to inform:

    _score_play(BOSS_ORDERS)  3200   -- which Pokemon to drag up
    _score_play(SWITCH)       6000   -- which attacker to bring in
    _score_attach                    -- where the turn's energy goes
    _score_option(ATTACK)     1100   -- Mega Brave vs Aura Jab
    _score_option(RETREAT)    2000

PPP itself scores 5000, above all of those, so it is played FIRST and the
window in which the fix is live is the shortest one in the turn.

This audit runs real games and counts, on our own MAIN frames:
  * frames where the PPP effect is live (played earlier this turn)
  * of those, how many are scored with the buff missing
  * of those, how many produce a DIFFERENT attack plan when the buff is
    restored -- and how many flip a target from "no KO" to "KO"

The counterfactual re-runs the agent's own _plan_attack with hand_counts
bumped, so it exercises the shipped code path rather than a re-implementation.

Usage:
  python work/tools/ppp_live_audit.py --games 40 --opp grimmsnarl
"""
import argparse
import os
import sys
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
OUT = os.path.join(WORK, "out")

PPP = 1141


def _load_agent(agent_dir):
    """exec-load exactly as kaggle_environments does: agent dir on sys.path,
    cwd left at the REPO ROOT (handoff gotcha #4)."""
    full = os.path.join(WORK, "agents", agent_dir)
    if full not in sys.path:
        sys.path.insert(0, full)
    with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
        src = fh.read()
    env = {}
    exec(compile(src, "main.py", "exec"), env)
    return env


def _instrument(env, stats):
    """Wrap _plan_attack with a live counterfactual.

    On every MAIN frame we let the shipped planner run, snapshot its plan, then
    re-run it with one extra Premium Power Pro in hand_counts and compare. When
    the effect is genuinely live but the card has left the hand, the second plan
    is the CORRECT one and any difference is a decision the agent got wrong.
    """
    AP = env["AdvancedPolicy"]
    orig = AP._plan_attack

    def snapshot():
        p = env["plan"]
        return (p.attacker, p.target, p.attack_index, p.remain_hp, p.needs_energy)

    def patched(self):
        orig(self)
        base = snapshot()
        live = bool(getattr(env["__builtins__"], "_x", None))  # unused
        ppp_live = PPP_STATE["live"]
        in_hand = self.hand_counts[PPP] > 0

        stats["main_frames"] += 1
        if ppp_live:
            stats["frames_effect_live"] += 1
            if not in_hand:
                stats["frames_effect_live_but_unmodelled"] += 1
                # counterfactual: restore the buff the card actually grants
                self.hand_counts[PPP] += 1
                try:
                    orig(self)
                    fixed = snapshot()
                finally:
                    self.hand_counts[PPP] -= 1
                # remain_hp shifts by 30 on almost every frame, so "the plan
                # tuple differs" is not evidence of anything. What matters is
                # whether the OPTION WE ACTUALLY PLAY changes. Score the real
                # option list under both plans and compare the argmax.
                if fixed != base:
                    stats["plan_changes"] += 1
                    if fixed[1] != base[1]:
                        stats["target_changes"] += 1
                    if fixed[0] != base[0]:
                        stats["attacker_changes"] += 1
                    if fixed[2] != base[2]:
                        stats["attack_changes"] += 1
                    # remain_hp <= 0 means "this plan knocks the target out"
                    if base[3] > 0 >= fixed[3]:
                        stats["no_KO_to_KO"] += 1

                    def top_under(p):
                        env["plan"].attacker, env["plan"].target = p[0], p[1]
                        env["plan"].attack_index, env["plan"].remain_hp = p[2], p[3]
                        env["plan"].needs_energy = p[4]
                        sc = [self._score_option(o) for o in self.select.option]
                        best = max(range(len(sc)), key=lambda i: sc[i])
                        return best, self.select.option[best].type

                    try:
                        b_i, b_t = top_under(base)
                        f_i, f_t = top_under(fixed)
                        if b_i != f_i:
                            stats["ACTION_CHANGES"] += 1
                            stats["act_%s->%s" % (b_t.name, f_t.name)] += 1
                    except Exception:
                        stats["action_cmp_error"] += 1
                # restore the shipped (wrong) plan so play is unaffected
                orig(self)

    AP._plan_attack = patched
    return orig


PPP_STATE = {"live": False, "turn": -999}


def _track_ppp(obs, my_index):
    """Is the PPP effect live for US on this frame?

    Derived from obs.logs, which carries every event since our last selection:
    TURN_START resets, PLAY(1141) by us sets. Mirrors what the fixed agent will
    have to do at runtime, so if this tracker is wrong the fix is wrong too.
    """
    from cg.api import LogType
    st = obs.current
    turn = st.turn if st is not None else 0
    if turn != PPP_STATE["turn"]:
        PPP_STATE["turn"] = turn
        PPP_STATE["live"] = False
    for lg in (obs.logs or []):
        t = int(lg.type)
        if t == int(LogType.TURN_START) or t == int(LogType.TURN_END):
            PPP_STATE["live"] = False
        elif t == int(LogType.PLAY) and lg.cardId == PPP and lg.playerIndex == my_index:
            PPP_STATE["live"] = True


def _worker(job):
    agent_dir, opp_deck, n, seed0 = job
    sys.path.insert(0, os.path.join(WORK, "lib"))
    # PRIVATE cwd per worker. main.py does Path("deck.csv").write_text(DECK) at
    # import; with every worker sharing one cwd they race on that file and a
    # reader sees a truncated deck ("The deck must contain 60 cards."). Keeping
    # sys.path rooted at the repo so dznp/vznp still resolve their weights.
    import tempfile
    os.chdir(tempfile.mkdtemp(prefix="pppaudit_"))
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    stats = Counter()
    env = _load_agent(agent_dir)
    _instrument(env, stats)
    fn = [v for v in env.values() if callable(v)][-1]
    my_deck = list(env["my_deck"])
    opp = list(opp_deck) if opp_deck else list(my_deck)

    # a plain mirror pilot for the opponent seat -- we are counting OUR frames,
    # so the opponent only has to be a legal, non-degenerate player
    env2 = _load_agent(agent_dir)
    fn2 = [v for v in env2.values() if callable(v)][-1]

    wins = 0
    for g in range(n):
        first = ((seed0 + g) % 2 == 0)
        d0, d1 = (my_deck, opp) if first else (opp, my_deck)
        p0, p1 = (fn, fn2) if first else (fn2, fn)
        me_idx = 0 if first else 1
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                if o.current is not None and o.current.result != -1:
                    if o.current.result == me_idx:
                        wins += 1
                    break
                who = o.current.yourIndex if o.current is not None else 0
                if who == me_idx:
                    _track_ppp(o, me_idx)
                sel = (p0 if who == 0 else p1)(obs)
                obs = battle_select(list(sel))
        except Exception:
            pass
        finally:
            battle_finish()
        PPP_STATE["turn"] = -999
    stats["games"] += n
    stats["wins"] += wins
    return dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="v34_stadium")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--opp", default="")
    a = ap.parse_args()

    opp_deck = None
    if a.opp:
        md = json.load(open(os.path.join(OUT, "meta_decks.json")))
        decks = md["decks"] if isinstance(md, dict) else md
        for d in decks:
            cards = d.get("deck") if isinstance(d, dict) else d
            if a.opp == "grimmsnarl" and 648 in cards:
                opp_deck = cards
                break
        if opp_deck is None:
            print("no matching opponent deck found; using mirror")

    per = max(1, a.games // a.workers)
    jobs = [(a.agent, opp_deck, per, i * 1000) for i in range(a.workers)]
    tot = Counter()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(_worker, j) for j in jobs]):
            tot.update(f.result())

    n = tot["main_frames"]
    live = tot["frames_effect_live"]
    blind = tot["frames_effect_live_but_unmodelled"]
    print(f"\n{tot['games']} games in {time.time()-t0:.0f}s "
          f"(our win rate {tot['wins']}/{tot['games']})")
    print(f"  our MAIN frames                           : {n}")
    print(f"  frames where the PPP effect was LIVE      : {live}"
          f"  ({100*live/max(1,n):.1f}% of MAIN)")
    print(f"  ...of those, scored WITHOUT the +30       : {blind}"
          f"  ({100*blind/max(1,live):.1f}% of live frames)")
    print(f"  ...of those, the plan CHANGES when fixed  : {tot['plan_changes']}"
          f"  ({100*tot['plan_changes']/max(1,blind):.1f}%)")
    print(f"       target changes                       : {tot['target_changes']}")
    print(f"       attacker changes                     : {tot['attacker_changes']}")
    print(f"       attack (Mega Brave/Aura Jab) changes : {tot['attack_changes']}")
    print(f"       no-KO -> KO flips                    : {tot['no_KO_to_KO']}")
    print(f"  ==> frames where the PLAYED ACTION changes: {tot['ACTION_CHANGES']}"
          f"  ({tot['ACTION_CHANGES']/max(1,tot['games']):.2f} per game)")
    for k, v in sorted(tot.items()):
        if k.startswith("act_"):
            print(f"       {k[4:]:<34} : {v}")
    if tot.get("action_cmp_error"):
        print(f"       (comparison errors: {tot['action_cmp_error']})")


if __name__ == "__main__":
    main()
