"""Does an agent get SLOWER the longer its process lives?

Why this exists: `fsearch.pimc_terminal` once deferred `search_release` to the
end of a call, left tens of thousands of engine search states alive, and the
engine degraded until a single move took 1,089,510 ms. The gate's per-move
timing caught it only because the gate plays several games in one process.

`p1_codex` (jazivxt, LB rank 121) calls `search_begin`/`search_step` and then
only `search_end()` -- it never releases the intermediate search ids. If
`search_end` does not reclaim them, the same degradation applies, and a local
gauntlet would silently measure a crippled agent while the ladder (fresh
process per episode) would not. Either answer changes what we ship, so measure
it rather than assume.

Reports, per game: wall time, agent-only time, worst single move. A healthy
agent's curve is flat.

  python work/tools/agent_drift.py --agent p1_codex --opponent v51_roman_safe -n 12
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))


def load(name):
    """exec-load a bundle exactly as kaggle_environments does."""
    full = os.path.join(WORK, "agents", name)
    if full not in sys.path:
        sys.path.insert(0, full)
    cwd = os.getcwd()
    try:
        os.chdir(full)
        with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
            src = fh.read()
        env = {}
        exec(compile(src, "main.py", "exec"), env)
    finally:
        os.chdir(cwd)
    fns = [v for v in env.values() if callable(v)]
    deck = env.get("DECK") or env.get("my_deck") or env.get("MY_DECK")
    if not deck:
        with open(os.path.join(full, "deck.csv"), encoding="utf-8") as fh:
            deck = [int(x) for x in fh.read().split() if x.strip()]
    return fns[-1], list(deck), env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--opponent", default="v51_roman_safe")
    ap.add_argument("-n", "--games", type=int, default=12)
    args = ap.parse_args()

    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    fa, da, env_a = load(args.agent)
    fb, db, _ = load(args.opponent)
    print(f"{args.agent} vs {args.opponent}: {args.games} games in ONE process\n")
    print(f"{'game':>4} {'wall_s':>8} {'A_time_s':>9} {'worst_move_ms':>14} "
          f"{'moves':>6} {'turns':>6}  result")

    wa = wb = 0
    first_a = last_a = None
    for g in range(args.games):
        a_first = (g % 2 == 0)
        p0, p1 = (fa, fb) if a_first else (fb, fa)
        d0, d1 = (da, db) if a_first else (db, da)
        a_idx = 0 if a_first else 1
        obs, _ = battle_start(list(d0), list(d1))
        t_game = time.time()
        a_time = 0.0
        worst = 0.0
        moves = 0
        turns = 0
        res = None
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                if o.current is not None and o.current.result != -1:
                    res = o.current.result
                    turns = o.current.turn
                    break
                who = o.current.yourIndex if o.current is not None else 0
                t0 = time.time()
                sel = (p0 if who == 0 else p1)(obs)
                dt = time.time() - t0
                if (who == 0) == a_first:
                    a_time += dt
                    worst = max(worst, dt)
                    moves += 1
                obs = battle_select(list(sel))
        except Exception as e:
            print(f"   ERROR {type(e).__name__}: {e}")
        finally:
            battle_finish()
        if res == a_idx:
            wa += 1
        elif res is not None and res != 2:
            wb += 1
        wall = time.time() - t_game
        if first_a is None:
            first_a = a_time
        last_a = a_time
        print(f"{g:>4} {wall:>8.1f} {a_time:>9.1f} {worst*1000:>14.0f} "
              f"{moves:>6} {turns:>6}  {res}")

    print(f"\nscore {wa}-{wb}")
    if first_a and last_a:
        print(f"agent-time drift: game 0 {first_a:.1f}s -> game {args.games-1} "
              f"{last_a:.1f}s  ({last_a/max(first_a,1e-9):.2f}x)")
    # Dump every counter dict the agent exposes. A component that never ran is
    # the recurring failure here -- five shipped components were silently dead
    # -- so print the counters rather than trusting that the code is reachable.
    for name, st in sorted(env_a.items()):
        if not (name.lower().endswith("stats") and isinstance(st, dict)):
            continue
        print(f"\n{name}: {st}")
        if st.get("calls"):
            print(f"  {st['ms']/st['calls']:.0f} ms mean over {st['calls']} calls"
                  if st.get("ms") else "")
        if st.get("playouts"):
            done = st.get("terminal", 0)
            print(f"  playouts {st['playouts']} -> terminal {done} "
                  f"({done/max(st['playouts'],1):.1%}), truncated {st.get('trunc',0)}")
            print(f"  decisions with enough samples {st.get('ran',0)}, "
                  f"overrides {st.get('overrides',0)}/{st.get('considered',0)}, "
                  f"failures {st.get('fail',0)}")
        elif st.get("considered"):
            print(f"  overrides {st.get('overrides',0)}/{st['considered']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
