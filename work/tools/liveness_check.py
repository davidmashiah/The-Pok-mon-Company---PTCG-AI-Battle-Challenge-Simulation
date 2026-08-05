"""Assert every component of the agent ACTUALLY RAN. Single process, no pool.

Five shipped bundles in this project carried components that were dead on
arrival and failed silently -- search_begin called with the wrong signature and
swallowed by a bare except, meta_decks.py never bundled, best_action starved
because choose() truncates to maxCount=1. Each looked fine in a win-rate test,
because a dead component just means "the baseline, but slower".

So: never trust that a new branch executes. Count it.

Reports, for v37's additions:
  ppp_live_frames    Premium Power Pro tracked as already played this turn
  ppp_stack_frames   the planner saw TWO copies available (the 330 line)
  gravity_pending    Gravity Mountain in hand, about to land, target is Stage 2
  lethal_hits        fsearch.find_lethal returned a proven winning line

A zero on any of these means the corresponding fix is not in the game.

Usage: python work/tools/liveness_check.py --agent v37_combo --games 8
"""
import argparse
import json
import os
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
OUT = os.path.join(WORK, "out")
GRIMMSNARL = 648


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="v37_combo")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--opp", default="grimmsnarl")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(WORK, "lib"))
    full = os.path.join(WORK, "agents", a.agent)
    sys.path.insert(0, full)
    md = json.load(open(os.path.join(OUT, "meta_decks.json"), encoding="utf-8"))
    grim = next(v["deck"] for _, v in
                sorted(md["teams"].items(), key=lambda kv: -(kv[1].get("score") or 0))
                if GRIMMSNARL in (v.get("deck") or []))

    os.chdir(tempfile.mkdtemp(prefix="live_"))
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    src = open(os.path.join(full, "main.py"), encoding="utf-8-sig").read()
    env = {}
    exec(compile(src, "main.py", "exec"), env)
    fn = [v for v in env.values() if callable(v)][-1]
    my_deck = list(env["my_deck"])

    env2 = {}
    exec(compile(src, "main.py", "exec"), env2)
    fn2 = [v for v in env2.values() if callable(v)][-1]
    env2["my_deck"][:] = list(grim)

    st = Counter()

    # --- instrument the policy: count which branches were reached ------------
    AP = env["AdvancedPolicy"]
    orig_init = AP.__init__

    def init(self, obs):
        orig_init(self, obs)
        if self.context == env["SelectContext"].MAIN:
            st["main_frames"] += 1
            if getattr(self, "ppp_now", 0) >= 1:
                st["ppp_live_frames"] += 1
            if getattr(self, "ppp_best", 0) >= 2:
                st["ppp_stack_frames"] += 1
            if getattr(self, "gravity_pending", False):
                st["gravity_pending"] += 1
    AP.__init__ = init

    orig_lethal = env["_lethal_line"]

    def lethal(obs_dict, obs):
        r = orig_lethal(obs_dict, obs)
        st["lethal_calls"] += 1
        if r:
            st["lethal_hits"] += 1
        return r
    env["_lethal_line"] = lethal

    for g in range(a.games):
        first = (g % 2 == 0)
        d0, d1 = (my_deck, grim) if first else (grim, my_deck)
        p0, p1 = (fn, fn2) if first else (fn2, fn)
        me = 0 if first else 1
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                if o.current is not None and o.current.result != -1:
                    st["wins" if o.current.result == me else "losses"] += 1
                    break
                who = o.current.yourIndex if o.current is not None else 0
                obs = battle_select(list((p0 if who == 0 else p1)(obs)))
        except Exception as e:
            st["err"] += 1
            print("  game error:", repr(e)[:160])
        finally:
            battle_finish()
        st["games"] += 1

    print(f"\n{a.agent}: {st['games']} games vs Grimmsnarl "
          f"({st['wins']}W/{st['losses']}L)")
    print(f"  MAIN frames                                   : {st['main_frames']}")
    rows = [
        ("PPP already played this turn (tracker fired)", "ppp_live_frames"),
        ("planner saw 2 PPP available (the 330 line)", "ppp_stack_frames"),
        ("Gravity Mountain pending vs a Stage 2", "gravity_pending"),
        ("find_lethal returned a proven win", "lethal_hits"),
    ]
    bad = []
    for label, key in rows:
        v = st[key]
        flag = "DEAD" if v == 0 else "live"
        if v == 0:
            bad.append(label)
        print(f"  [{flag}] {label:<44}: {v}")
    print(f"  (find_lethal was called {st['lethal_calls']} times)")
    if bad:
        print("\n  *** these branches never executed: " + "; ".join(bad))
        sys.exit(1)
    print("\n  all new branches executed at least once.")


if __name__ == "__main__":
    main()
