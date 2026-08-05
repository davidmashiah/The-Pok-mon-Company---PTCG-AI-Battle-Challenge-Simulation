"""Local arena: play N games between two (policy, deck) pairs.

Reports win rate with a Wilson 95% CI so we never read noise as signal.
Sides are swapped every game so first-player advantage cancels.

Usage:
  python work/tools/arena.py --a greedy:decks/abomasnow_v2.csv \
                             --b random:lib/sample_deck.csv --n 200
"""
import argparse
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
import policy  # noqa: E402


def read_deck(path):
    p = path if os.path.isabs(path) else os.path.join(WORK, path)
    with open(p) as f:
        rows = [ln.strip() for ln in f if ln.strip()]
    d = [int(r) for r in rows[:60]]
    if len(d) != 60:
        raise SystemExit(f"{path}: deck has {len(d)} cards, need 60")
    return d


def load_agent_dir(agent_dir):
    """Load agents/<name>/main.py the way kaggle-environments does.

    exec() with no __file__, then take the LAST callable. Also returns the
    deck the agent itself resolved, so we test the real pairing.
    """
    d = agent_dir if os.path.isabs(agent_dir) else os.path.join(WORK, "agents", agent_dir)
    if d not in sys.path:
        sys.path.insert(0, d)
    cwd = os.getcwd()
    try:
        os.chdir(d)
        with open(os.path.join(d, "main.py"), encoding="utf-8-sig") as fh:
            src = fh.read()
        env = {}
        exec(compile(src, "main.py", "exec"), env)
    finally:
        os.chdir(cwd)
    fns = [v for v in env.values() if callable(v)]
    fn = fns[-1]
    if getattr(fn, "__name__", None) != "agent":
        raise SystemExit(f"{d}: last callable is {fn.__name__!r}, not 'agent'")
    deck = env.get("DECK") or env.get("my_deck")
    return fn, list(deck)


def make_policy(kind, deck, rng):
    if kind == "greedy":
        return lambda obs: policy.act(obs, deck)

    if kind == "random":
        def rnd(obs_dict):
            o = to_observation_class(obs_dict)
            if o.select is None:
                return list(deck)
            sd = o.select
            n = len(sd.option)
            k = rng.randint(sd.minCount, min(sd.maxCount, n))
            return rng.sample(range(n), k) if k > 0 else []
        return rnd

    raise SystemExit(f"unknown policy '{kind}'")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def play(pol0, pol1, deck0, deck1, cap=4000):
    """Return 0 / 1 / 2(draw) / None(error)."""
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        return None
    try:
        pols = (pol0, pol1)
        for _ in range(cap):
            o = to_observation_class(obs)
            if o.current is not None and o.current.result != -1:
                return o.current.result
            who = o.current.yourIndex if o.current is not None else 0
            sel = pols[who](obs)
            obs = battle_select(list(sel))
        return None
    except Exception as e:
        print(f"    [error] {type(e).__name__}: {e}")
        return None
    finally:
        battle_finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="policy:deckpath")
    ap.add_argument("--b", required=True, help="policy:deckpath")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    def build(spec):
        kind, ref = spec.split(":", 1)
        if kind == "agent":
            fn, dk = load_agent_dir(ref)
            return fn, dk
        dk = read_deck(ref)
        return make_policy(kind, dk, rng), dk

    pa, deck_a = build(args.a)
    pb, deck_b = build(args.b)

    wa = wb = draw = err = 0
    t0 = time.time()
    for g in range(args.n):
        # swap sides each game: A is player0 on even games
        if g % 2 == 0:
            r = play(pa, pb, deck_a, deck_b)
            a_is = 0
        else:
            r = play(pb, pa, deck_b, deck_a)
            a_is = 1
        if r is None:
            err += 1
        elif r == 2:
            draw += 1
        elif r == a_is:
            wa += 1
        else:
            wb += 1
        if (g + 1) % 50 == 0:
            dec = wa + wb
            p, lo, hi = wilson(wa, dec)
            print(f"  {g+1:>4} games | A {wa} - {wb} B (draw {draw}, err {err}) "
                  f"| A winrate {p:.3f} [{lo:.3f},{hi:.3f}]")

    dt = time.time() - t0
    dec = wa + wb
    p, lo, hi = wilson(wa, dec)
    print()
    print(f"A = {args.a}")
    print(f"B = {args.b}")
    print(f"  games={args.n} decisive={dec} draws={draw} errors={err}  ({dt:.1f}s, "
          f"{args.n/max(dt,1e-9):.1f} g/s)")
    print(f"  A win rate {p:.4f}  Wilson95% [{lo:.4f}, {hi:.4f}]")
    if lo > 0.5:
        print("  => A is better (CI entirely above 50%)")
    elif hi < 0.5:
        print("  => B is better (CI entirely below 50%)")
    else:
        print("  => INCONCLUSIVE at this N. Do not act on this difference.")
    if err:
        print(f"  !! {err} errored games — investigate before submitting")


if __name__ == "__main__":
    main()
