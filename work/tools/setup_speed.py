"""How fast does the agent get its Stage 2 attacking? Counted per game, not per win.

From 25 live ladder games of 55299973, restricted to the Grimmsnarl MIRROR --
30% of the top-50 field and where we win 0.300:

    our first attack on turn <= 4   ->  2 wins, 0 losses
    our first attack on turn >= 5   ->  1 win,  7 losses

Both sides swing the same Shadow Bullet for 180, so the mirror is a pure race
and the deck that gets there first wins it. There is also a tail: in 6 of 25
games we did not attack until turn 8 or never attacked at all, and went 2-4.

Win rate cannot resolve a change this size at any sample count we can afford --
that is the lesson of damage_model_audit.py, which counted ~500 ATTACKS per 120
games and settled a question the win rate could not. This counts one number per
game, on the same principle: the turn of our first attack. It is a mechanism,
it is nearly deterministic given the policy, and it moves long before the win
rate does.

  python work/tools/setup_speed.py --agent w8_grimm_tuned --opponent w5_grimmsnarl -n 60
"""
import argparse
import os
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")


def _load(full):
    if full not in sys.path:
        sys.path.insert(0, full)
    cwd = os.getcwd()
    try:
        os.chdir(full)
        with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
            src = fh.read()
        env = {}
        exec(compile(src, "main.py", "exec"), env)
        fn = [v for v in env.values() if callable(v)][-1]
        deck = None
        try:
            r = fn({"current": None, "select": None})
            if isinstance(r, (list, tuple)) and len(r) == 60:
                deck = [int(x) for x in r]
        except Exception:
            pass
    finally:
        os.chdir(cwd)
    if deck is None:
        with open(os.path.join(full, "deck.csv"), encoding="utf-8") as fh:
            deck = [int(x) for x in fh.read().split() if x.strip()]
    return fn, deck


def _worker(job):
    agent, opponent, n, seed0 = job
    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import OptionType, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    fa, da = _load(os.path.join(AGENTS, agent))
    fb, db = _load(os.path.join(AGENTS, opponent))

    out = []
    for g in range(n):
        a_first = ((seed0 + g) % 2 == 0)
        p0, p1 = (fa, fb) if a_first else (fb, fa)
        d0, d1 = (da, db) if a_first else (db, da)
        a_idx = 0 if a_first else 1
        for f in (p0, p1):
            try:
                f({"current": None, "select": None})
            except Exception:
                pass
        obs, _sd = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        first_us = first_op = None
        res = None
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                if o.current is not None and o.current.result != -1:
                    res = o.current.result
                    break
                who = o.current.yourIndex if o.current is not None else 0
                sel = list((p0 if who == 0 else p1)(obs))
                # An ATTACK is only an attack if the option we PICKED is one --
                # reading the offered options would count turns where attacking
                # was merely legal, which is the opposite of what we are asking.
                try:
                    s = o.select
                    if s is not None and sel:
                        picked = [s.option[i] for i in sel
                                  if 0 <= i < len(s.option)]
                        if any(p.type == OptionType.ATTACK for p in picked):
                            t = o.current.turn
                            if who == a_idx:
                                first_us = first_us if first_us is not None else t
                            else:
                                first_op = first_op if first_op is not None else t
                except Exception:
                    pass
                obs = battle_select(sel)
        except Exception:
            res = None
        finally:
            battle_finish()
        out.append((first_us, first_op, (res == a_idx) if res is not None else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--opponent", default="w5_grimmsnarl")
    ap.add_argument("-n", "--games", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    per = max(1, args.games // args.workers)
    jobs = [(args.agent, args.opponent, per, w * 7919)
            for w in range(args.workers)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_worker, jobs):
            rows.extend(r)

    us = [t for t, _, _ in rows if t is not None]
    never = sum(1 for t, _, _ in rows if t is None)
    print(f"\n{args.agent} vs {args.opponent}: {len(rows)} games")
    print(f"  our first-attack turn: mean {statistics.mean(us):.2f} "
          f"median {statistics.median(us):.0f}" if us else "  never attacked")
    print(f"  distribution: {dict(sorted(Counter(us).items()))}")
    print(f"  games we NEVER attacked: {never} ({never/max(len(rows),1):.1%})")
    by4 = sum(1 for t in us if t <= 4)
    print(f"  attacking by turn 4: {by4}/{len(rows)} = {by4/max(len(rows),1):.3f}"
          "   <- the mirror splits on this")

    fast = [(w) for t, _, w in rows if t is not None and t <= 4 and w is not None]
    slow = [(w) for t, _, w in rows if (t is None or t >= 5) and w is not None]
    if fast:
        print(f"  win rate when we attack by turn 4: "
              f"{sum(fast)}/{len(fast)} = {sum(fast)/len(fast):.3f}")
    if slow:
        print(f"  win rate when we do not:            "
              f"{sum(slow)}/{len(slow)} = {sum(slow)/len(slow):.3f}")
    wins = [w for _, _, w in rows if w is not None]
    if wins:
        print(f"  overall win rate: {sum(wins)}/{len(wins)} = "
              f"{sum(wins)/len(wins):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
