"""Did the search layer actually RUN, and did it change any decision?

This exists because the alternative is the most expensive bug class in this
project. v57_pimc_full shipped a playout search that raised TypeError on every
call and played 701.8 points of ladder as a pure heuristic; the coalition expert
in w8 is gated behind a profile that never matches and fired 0 times in 933
instrumented decisions. Both looked completely healthy from the outside.

So before any win-rate number is believed, this reports, per game:
  decisions seen / eligible for search / searches completed
  overrode vs agreed, and total playouts
  seconds of search per episode (the ladder allows 600 s)

A layer that reports ran=0 is not "a small gain", it is not installed.

  python work/tools/search_liveness.py --agent w30_search --games 6
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


def load(name):
    full = os.path.join(AGENTS, name)
    if full not in sys.path:
        sys.path.insert(0, full)
    cwd = os.getcwd()
    try:
        os.chdir(full)
        env = {}
        exec(compile(open(os.path.join(full, "main.py"),
                          encoding="utf-8-sig").read(), "main.py", "exec"), env)
        fn = [v for v in env.values() if callable(v)][-1]
        # Some bundles call to_observation_class on the setup frame, which
        # needs `logs` present (w1_alakazam raises TypeError without it).
        try:
            d = fn({"current": None, "select": None, "logs": []})
        except Exception:
            d = None
    finally:
        os.chdir(cwd)
        # Do NOT evict this bundle's modules: unlike the gauntlet we keep a
        # handle on the agent's own search_validator to read its counters. The
        # opponent is loaded second, so only IT could shadow, and the panel
        # opponents ship no module names in common with the w8 family.
    if not (isinstance(d, (list, tuple)) and len(d) == 60):
        d = [int(x) for x in open(os.path.join(full, "deck.csv"),
                                  encoding="utf-8").read().split() if x.strip()]
    return fn, [int(x) for x in d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="w30_search")
    ap.add_argument("--opponent", default="w5_grimmsnarl")
    ap.add_argument("-n", "--games", type=int, default=6)
    a = ap.parse_args()

    fa, da = load(a.agent)
    fb, db = load(a.opponent)
    import search_validator as sv

    print(f"search_validator: budget={sv.BUDGET_S}s det={sv.DETERMINIZATIONS} "
          f"cands={sv.MAX_CANDIDATES} margin={sv.MARGIN} ok={sv._OK}")

    wins = 0
    allgaps = []
    for g in range(a.games):
        sv.reset_stats()
        for f in (fa, fb):
            try:
                f({"current": None, "select": None, "logs": []})
            except Exception:
                pass
        first = (g % 2 == 0)
        p0, p1 = (fa, fb) if first else (fb, fa)
        d0, d1 = (da, db) if first else (db, da)
        me = 0 if first else 1
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        res = None
        t0 = time.time()
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                if o.current is not None and o.current.result != -1:
                    res = o.current.result
                    break
                who = o.current.yourIndex
                obs = battle_select(list((p0 if who == 0 else p1)(obs)))
        except Exception as exc:
            print("   game error:", type(exc).__name__, exc)
        finally:
            battle_finish()
        dt = time.time() - t0
        s = sv.get_stats()
        if res == me:
            wins += 1
        print(f"  game {g+1}: {'WIN ' if res == me else 'loss'}  "
              f"decisions={s['decisions']:4d} eligible={s['eligible']:4d} "
              f"ran={s['ran']:4d} starved={s['starved']:4d} "
              f"overrode={s['overrode']:3d} "
              f"agreed={s['agreed']:4d} playouts={s['playouts']:5d} "
              f"errors={s['errors']:4d} search={s['time']:.2f}s "
              f"episode={dt:.1f}s")
        allgaps.extend(sv.GAPS)

    print(f"\n{wins}/{a.games} wins (n far too small to mean anything -- this "
          f"run is about whether the layer executes)")
    if allgaps:
        allgaps.sort()
        def q(f):
            return allgaps[min(len(allgaps) - 1, int(f * len(allgaps)))]
        print(f"\nleaf gap (best - policy's pick) over {len(allgaps)} searched "
              f"decisions:\n  median {q(0.5):.0f}  p75 {q(0.75):.0f}  "
              f"p90 {q(0.9):.0f}  p99 {q(0.99):.0f}  max {allgaps[-1]:.0f}")
        print("  MARGIN must sit above the leaf's noise, below its real "
              "disagreements.\n  1000 = a whole prize card.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
