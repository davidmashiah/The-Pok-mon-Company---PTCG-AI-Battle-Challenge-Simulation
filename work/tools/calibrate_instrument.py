"""Does our local field score predict the ladder at all? Correlate them and find out.

The Band-1 rule from the ML-optimisation playbook: "if local CV and the
held-out/leaderboard score diverge, stop everything and fix CV first." That is
our situation exactly -- `field_now.py` ranked `_sub_v28` +15.9 rating above
`w34_koroll`, and across 144 real episodes they are indistinguishable (0.528 vs
0.500). Every agent decision in this project has been graded by an instrument
whose correlation with the ladder has never been measured.

So measure it. For every agent we have BOTH a local field score and a real
ladder result, this reports the pair and their rank correlation. It is the
cheapest high-impact thing available: both numbers already exist on disk.

Reads live scores from the Kaggle submissions API (same KGAT token as
kaggle_submit.py) and local field from the gauntlet store, joining on the
tarball name, which is the agent directory name.

  python work/tools/calibrate_instrument.py
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
sys.path.insert(0, HERE)

import field_now  # noqa: E402


def live_scores():
    """agent name -> list of live public scores, newest first."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    out = subprocess.run(
        [os.path.join(ROOT, ".venv", "Scripts", "kaggle.exe"),
         "competitions", "submissions", "pokemon-tcg-ai-battle", "--csv"],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=180).stdout
    rows = [r for r in out.splitlines() if r.strip()]
    if not rows:
        return {}
    import csv
    import io
    scores = {}
    for r in csv.DictReader(io.StringIO("\n".join(rows))):
        fn = (r.get("fileName") or "").replace(".tar.gz", "")
        try:
            s = float(r.get("publicScore") or "")
        except ValueError:
            continue
        scores.setdefault(fn, []).append(s)
    return scores


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def main():
    store = json.load(open(os.path.join(WORK, "out", "gauntlet.json")))
    live = live_scores()
    print(f"submissions with a live score: {len(live)}\n")

    rows = []
    for agent, scores in sorted(live.items()):
        if not os.path.isdir(os.path.join(WORK, "agents", agent)):
            continue
        f, cells, den = field_now.field(store, agent)
        measured = sum(1 for c in cells if c[2] is not None)
        if f is None or measured < len(field_now.PANEL):
            continue
        rows.append((agent, f, max(scores), sum(scores) / len(scores),
                     len(scores), measured))

    if len(rows) < 3:
        print("not enough agents with BOTH a full local panel and a live score")
        for agent, scores in sorted(live.items()):
            has = os.path.isdir(os.path.join(WORK, "agents", agent))
            print(f"  {agent:22s} live={max(scores):7.1f} "
                  f"bundle={'yes' if has else 'MISSING'}")
        return 0

    rows.sort(key=lambda r: -r[1])
    print(f"{'agent':22s} {'local field':>11} {'live max':>9} "
          f"{'live mean':>10} {'draws':>6}")
    print("-" * 64)
    for a, f, mx, mean, k, _ in rows:
        print(f"{a:22s} {f:11.4f} {mx:9.1f} {mean:10.1f} {k:6d}")

    xs = [r[1] for r in rows]
    for label, idx in (("live MAX", 2), ("live MEAN", 3)):
        ys = [r[idx] for r in rows]
        rho = spearman(xs, ys)
        print(f"\nSpearman(local field, {label}) = "
              + (f"{rho:+.3f}" if rho is not None else "n/a")
              + f"   over {len(rows)} agents")
    print("\nA rank correlation near zero means the local panel has NOT been "
          "ranking\nagents the way the ladder does, and every decision made on "
          "it is unsupported.\nUse live MEAN, not MAX: identical bundles here "
          "have drawn 200 points apart.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
