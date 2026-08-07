"""Which prompts does OUR policy answer with a reflex instead of a decision?

Paid for immediately. The divergence audit found we take the MAXIMUM count on
REMOVE_DAMAGE_COUNTER_COUNT in 167 of 182 decisions while a 1100+ pilot takes
the minimum -- and forcing the minimum collapsed our win rate from 0.5523 to
0.1208 over 240 games. So that pilot is defaulting to option 0 on the single
most valuable prompt in the deck, and winning anyway. Disagreement with a
stronger player does not mean we are wrong; it means SOMEBODY is on autopilot.

Which makes the useful question the mirror image: where are WE the one
defaulting? A context we always answer with index 0, or always with the same
value regardless of the board, is a decision we are not making -- and Munkidori
shows one such prompt can be worth 0.4 of win rate.

Reports, per select context, how concentrated our answer is:
  * share taken by our single most common option INDEX
  * whether that index is 0 (the classic "return [0]" fallback)
  * how many distinct options we ever pick

High concentration on a context with many options and many occurrences is the
signature to chase.

  python work/tools/default_audit.py --agent _sub_handwritten_v26
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
sys.path.insert(0, os.path.join(WORK, "lib"))

CTX = {
    0: "MAIN", 1: "SETUP_ACTIVE", 2: "SETUP_BENCH", 3: "SWITCH",
    4: "TO_ACTIVE", 5: "TO_BENCH", 6: "TO_FIELD", 7: "TO_HAND",
    8: "DISCARD", 9: "TO_DECK", 10: "TO_DECK_BOTTOM", 11: "TO_PRIZE",
    35: "ATTACK", 36: "DISABLE_ATTACK", 37: "EVOLVE", 38: "DRAW_COUNT",
    39: "DAMAGE_COUNTER_COUNT", 40: "REMOVE_DAMAGE_COUNTER_COUNT",
    41: "IS_FIRST", 42: "MULLIGAN",
}


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
        try:
            fn({"current": None, "select": None, "logs": []})
        except Exception:
            pass
    finally:
        os.chdir(cwd)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="_sub_handwritten_v26")
    ap.add_argument("--replays", default=os.path.join(WORK, "out",
                                                      "top_replays"))
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()

    fn = load(a.agent)
    picks = collections.defaultdict(collections.Counter)
    nopts = collections.defaultdict(collections.Counter)

    for path in sorted(glob.glob(os.path.join(a.replays, "*.json")))[:a.limit]:
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        for st in (d.get("steps") or []):
            for ag in st:
                obs = ag.get("observation") or {}
                sel = obs.get("select")
                if not sel:
                    continue
                opts = sel.get("option") or []
                if len(opts) < 2:
                    continue
                try:
                    ctx = int(sel.get("context", -1)
                              if sel.get("context") is not None else -1)
                except Exception:
                    ctx = -1
                try:
                    ours = fn(obs)
                except Exception:
                    continue
                if not isinstance(ours, list) or not ours:
                    continue
                if not (0 <= ours[0] < len(opts)):
                    continue
                picks[ctx][ours[0]] += 1
                nopts[ctx][len(opts)] += 1

    print(f"{a.agent}: how concentrated is our answer, per prompt type?\n")
    print(f"{'context':30s} {'n':>6} {'opts':>5} {'top idx':>8} "
          f"{'share':>7}  flag")
    print("-" * 74)
    rows = []
    for ctx, c in picks.items():
        n = sum(c.values())
        idx, cnt = c.most_common(1)[0]
        avg_opts = (sum(k * v for k, v in nopts[ctx].items())
                    / max(1, sum(nopts[ctx].values())))
        rows.append((n, ctx, idx, cnt / n, avg_opts, len(c)))
    rows.sort(key=lambda r: -r[0])
    for n, ctx, idx, share, avg_opts, distinct in rows:
        flag = ""
        if share >= 0.90 and avg_opts >= 2.5:
            flag = "REFLEX" + (" (index 0)" if idx == 0 else "")
        elif share >= 0.75 and avg_opts >= 3:
            flag = "suspicious"
        print(f"{CTX.get(ctx, str(ctx)):30s} {n:6d} {avg_opts:5.1f} "
              f"{idx:8d} {share:7.3f}  {flag}")
    print("\nREFLEX = one option index taken >=90% of the time across a prompt "
          "that\naverages 2.5+ choices. That is a decision not being made.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
