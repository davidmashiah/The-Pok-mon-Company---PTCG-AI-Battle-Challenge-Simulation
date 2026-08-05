"""Validate a tuned genome hard enough to be worth a submission slot, then promote it.

The bar exists because a local head-to-head does NOT convert to ladder points at
anything like face value. Measured, on this ladder, with this repo's own agents:

    v61_codex_safe beat v51_roman_safe 0.8333 over 240 games
    ... and gained roughly +60 ladder points over v51's typical ~700.

So 0.83 head-to-head bought +60. Reading a 0.55 head-to-head as "we are 200
points better" is how this project talked itself into shipping v61 as a +300.
It was not.

What the ladder actually rewards is the win rate against the FIELD YOU GET
MATCHED INTO, and that is measurable from our own replays. Split by opponent
rating:

    v61 (ours, ~760)              vs 800-900 opponents: 0.357
    jazivxt live agent (~960)     vs 800-900 opponents: 0.769
                                  vs 900-1000:          0.562
                                  vs 1000+:             0.400

To score ~1000 an agent needs roughly a coin flip against 900-1000 rated
opposition. That is the target, and it is why beating v61 by a little means
nothing: our strongest local sparring partner IS v61, and it is a ~760 agent.
Beating a 760 agent 0.55 predicts a 780 agent.

Hence two rules here:

  1. PROMOTION IS RELATIVE, AND ITERATED. A genome must beat the CURRENT base at
     n>=240. When it does it becomes the new base and tuning continues against
     it, so the sparring partner climbs with us. That is the only mechanism
     available for building an opponent stronger than anything we can download.
  2. IT MUST NOT REGRESS ON THE PANEL. Tuning against one opponent optimises for
     beating that opponent, and generational self-play amplifies exactly that.
     A genuinely stronger agent holds up against the independent published
     agents too; one that only beats its own parent has overfitted to it.

  python work/tools/promote.py --candidate v80_tuned --base v61_codex_safe
"""
import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
STORE = os.path.join(WORK, "out", "gauntlet.json")

# Independent published agents. None of them is ours and none is derived from
# the candidate, so they cannot be gamed by tuning against the base.
PANEL = ["w1_alakazam", "w2_archaludon", "z_roman950", "w5_grimmsnarl"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - hw), min(1.0, c + hw)


def bundle_hash(agent):
    sys.path.insert(0, HERE)
    import gauntlet
    return gauntlet.bundle_hash(agent)


def cell(a, b):
    try:
        store = json.load(open(STORE))
    except Exception:
        return None
    key = f"{a}@{bundle_hash(a)}|{b}@{bundle_hash(b)}"
    c = store.get(key)
    if not c:
        return None
    n = c["wa"] + c["wb"]
    if n == 0:
        return None
    return (c["wa"], n) + wilson(c["wa"], n)[1:]


def run(a, b, games, workers):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    subprocess.run([sys.executable, "-u", os.path.join(HERE, "gauntlet.py"),
                    "--agents", f"{a},{b}", "--games", str(games),
                    "--workers", str(workers)], cwd=ROOT, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--base", default="v61_codex_safe")
    ap.add_argument("--games", type=int, default=240)
    ap.add_argument("--panel-games", type=int, default=160)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-panel", action="store_true")
    args = ap.parse_args()

    print(f"\n=== head-to-head: {args.candidate} vs {args.base} "
          f"(n={args.games}) ===")
    run(args.candidate, args.base, args.games, args.workers)
    h2h = cell(args.candidate, args.base)
    if not h2h:
        raise SystemExit("no head-to-head cell produced")
    wa, n, lo, hi = h2h
    p = wa / n
    print(f"  {p:.4f} [{lo:.3f},{hi:.3f}]  n={n}")

    beats = lo > 0.5
    print(f"\n  beats the base with 95% confidence: {'YES' if beats else 'NO'}")
    if not beats:
        print("  -> NOT promoted. An inconclusive head-to-head is not an "
              "improvement; that is how 13 false positives got adopted here.")
        return 1

    if args.skip_panel:
        print("  (panel check skipped by request)")
        return 0

    print("\n=== panel: must not regress against independent agents ===")
    regressed = []
    for opp in PANEL:
        if not os.path.isdir(os.path.join(WORK, "agents", opp)):
            continue
        run(args.candidate, opp, args.panel_games, args.workers)
        c = cell(args.candidate, opp)
        b = cell(args.base, opp)
        if not c:
            print(f"  {opp:16s} candidate: no data")
            continue
        cp = c[0] / c[1]
        if b:
            bp = b[0] / b[1]
            delta = cp - bp
            flag = "  REGRESSION" if delta < -0.05 else ""
            print(f"  {opp:16s} candidate {cp:.3f} (n={c[1]})  "
                  f"base {bp:.3f} (n={b[1]})  delta {delta:+.3f}{flag}")
            if delta < -0.05:
                regressed.append(opp)
        else:
            print(f"  {opp:16s} candidate {cp:.3f} (n={c[1]})  "
                  f"base: not measured -- measure it before trusting this")

    if regressed:
        print(f"\n  -> NOT promoted: regressed against {', '.join(regressed)}. "
              "Beating its own parent while losing to independent agents is "
              "overfitting to the sparring partner, not strength.")
        return 1

    print(f"\n  -> PROMOTE. Set BASE = {args.candidate} in tune_weights.py and "
          "continue the search against it.")
    print("     Then gate before any submission:")
    print(f"     python work/tools/build_and_gate.py --agent {args.candidate} --games 10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
