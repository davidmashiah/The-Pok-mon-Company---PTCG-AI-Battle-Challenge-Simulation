"""How often is the attack planner's knockout prediction WRONG?

One-shot rates and win rates are noisy because they count games. This counts
ATTACKS, of which there are an order of magnitude more, and it measures the
thing the v37 changes actually claim to fix: the planner's damage model.

For every attack we launch at the opponent's Active, record what the plan
predicted (`plan.remain_hp <= 0` means "this knocks it out") and then watch what
the engine did. Two error modes, and they are not symmetric:

  MISSED   predicted no KO, engine knocked it out anyway.
           The model understates our damage -- the v14/v32 failure. Costs us
           knockouts we could have aimed at a better target, and through the
           same `prize` variable can score a GAME-WINNING attack as non-lethal.

  PHANTOM  predicted a KO, engine did not deliver one.
           The model overstates our damage. Strictly worse: we tap out the
           board for a knockout that never happens. Every guard added in v37 is
           written to keep this at zero.

A fix that reduces MISSED while pushing PHANTOM above baseline is not a fix.

Usage:
  python work/tools/damage_model_audit.py --agents v32_ppp,v37_combo --games 60
"""
import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
OUT = os.path.join(WORK, "out")
GRIMMSNARL = 648


def _load(agent_dir, deck=None):
    full = os.path.join(WORK, "agents", agent_dir)
    if full not in sys.path:
        sys.path.insert(0, full)
    with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
        src = fh.read()
    env = {}
    exec(compile(src, "main.py", "exec"), env)
    if deck is not None:
        env["my_deck"][:] = list(deck)
    return env


def _worker(job):
    agent_dir, opp_deck, n, seed0 = job
    sys.path.insert(0, os.path.join(WORK, "lib"))
    os.chdir(tempfile.mkdtemp(prefix="dmg_"))
    from cg.api import OptionType, SelectContext, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    env = _load(agent_dir)
    fn = [v for v in env.values() if callable(v)][-1]
    my_deck = list(env["my_deck"])
    env2 = _load(agent_dir, opp_deck)
    fn2 = [v for v in env2.values() if callable(v)][-1]

    st = Counter()
    pending = {}       # set when we launch an attack, checked on the next frame

    def wrapped(obs_dict):
        sel = fn(obs_dict)
        try:
            o = to_observation_class(obs_dict)
            if (o.select is not None and o.current is not None
                    and o.select.context == SelectContext.MAIN and sel):
                opt = o.select.option[sel[0]]
                if opt.type == OptionType.ATTACK:
                    p = env["plan"]
                    # only the Active: a benched target's HP is not what
                    # remain_hp was computed against once Boss's Orders moves it
                    if p.target == 0:
                        opp = o.current.players[1 - o.current.yourIndex]
                        act = (opp.active or [None])[0]
                        if act is not None:
                            pending.clear()
                            pending["serial"] = act.serial
                            pending["pred_ko"] = (p.remain_hp <= 0)
                            pending["hp"] = act.hp
                            pending["id"] = act.id
        except Exception:
            st["instr_err"] += 1
        return sel

    for g in range(n):
        first = ((seed0 + g) % 2 == 0)
        d0, d1 = (my_deck, opp_deck) if first else (opp_deck, my_deck)
        p0, p1 = (wrapped, fn2) if first else (fn2, wrapped)
        me = 0 if first else 1
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        pending.clear()
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                c = o.current
                over = c is not None and c.result != -1
                # resolve a pending prediction as soon as we can see their board
                if pending and c is not None and (over or c.yourIndex == me):
                    opp = c.players[1 - me]
                    still = None
                    gone_to_evo = False
                    for p in list(opp.active or []) + list(opp.bench or []):
                        if p is None:
                            continue
                        if p.serial == pending["serial"]:
                            still = p
                            break
                        # Evolving replaces the Pokemon's serial and pushes the
                        # old card into preEvolution. Without this the survivor
                        # reads as "vanished" and every evolution is scored as
                        # a knockout we did not get -- inflating MISSED for
                        # every agent equally, but making the absolute rate a
                        # measure of the opponent's evolutions, not our model.
                        for c2 in (p.preEvolution or []):
                            if c2 is not None and getattr(c2, "serial", None) == pending["serial"]:
                                still, gone_to_evo = p, True
                                break
                        if gone_to_evo:
                            break
                    got_ko = (still is None) or (still.hp <= 0 and not gone_to_evo)
                    pred = pending["pred_ko"]
                    st["attacks_scored"] += 1
                    if pred and got_ko:
                        st["correct_KO"] += 1
                    elif (not pred) and (not got_ko):
                        st["correct_noKO"] += 1
                    elif pred and not got_ko:
                        st["PHANTOM"] += 1
                        if pending["id"] == GRIMMSNARL:
                            st["PHANTOM_grimm"] += 1
                    else:
                        st["MISSED"] += 1
                        if pending["id"] == GRIMMSNARL:
                            st["MISSED_grimm"] += 1
                    pending.clear()
                if over:
                    st["wins" if c.result == me else "losses"] += 1
                    break
                who = c.yourIndex if c is not None else 0
                obs = battle_select(list((p0 if who == 0 else p1)(obs)))
        except Exception:
            st["err"] += 1
        finally:
            battle_finish()
        st["games"] += 1
    return dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="v32_ppp,v37_combo")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    md = json.load(open(os.path.join(OUT, "meta_decks.json"), encoding="utf-8"))
    grim = next(v["deck"] for _, v in
                sorted(md["teams"].items(), key=lambda kv: -(kv[1].get("score") or 0))
                if GRIMMSNARL in (v.get("deck") or []))

    print(f"attack-level damage-model accuracy, {a.games} games vs a real "
          f"leaderboard Grimmsnarl list\n")
    print(f"{'agent':<16} {'attacks':>8} {'correct':>8} {'MISSED':>7} "
          f"{'PHANTOM':>8} {'err rate':>9}   {'vs Grimmsnarl':>22}")
    print("-" * 92)
    for ag in a.agents.split(","):
        ag = ag.strip()
        per = max(1, a.games // a.workers)
        jobs = [(ag, grim, per, i * 7717) for i in range(a.workers)]
        tot = Counter()
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for f in as_completed([ex.submit(_worker, j) for j in jobs]):
                tot.update(f.result())
        n = max(1, tot["attacks_scored"])
        ok = tot["correct_KO"] + tot["correct_noKO"]
        bad = tot["MISSED"] + tot["PHANTOM"]
        print(f"{ag:<16} {tot['attacks_scored']:>8} {ok:>8} {tot['MISSED']:>7} "
              f"{tot['PHANTOM']:>8} {100*bad/n:>8.1f}%   "
              f"MISSED {tot['MISSED_grimm']:>3} / PHANTOM {tot['PHANTOM_grimm']:>3}"
              f"   ({time.time()-t0:.0f}s)")
    print("\nMISSED  = model understated damage; the engine knocked it out anyway.")
    print("PHANTOM = model overstated damage; we committed to a KO that never came.")
    print("PHANTOM must not rise. It is the error mode that loses games outright.")


if __name__ == "__main__":
    main()
