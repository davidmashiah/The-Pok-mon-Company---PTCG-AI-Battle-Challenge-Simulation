"""Package the tuner's best genome as a real agent bundle, for INDEPENDENT validation.

The hill-climb in `tune_weights.py` runs its own game loop, so validating its
output with that same loop would prove nothing about the loop. This writes the
genome into a normal bundle so it can be measured through `gauntlet.py` and
gated by `build_and_gate.py` -- different code, different harness, same question.

The genome ships as `alak_w.json` beside main.py, which is the hook the base's
author already resolves from `/kaggle_simulations/agent/`, so nothing about the
policy source changes.

  python work/tools/build_tuned.py --name v80_tuned
  python work/tools/gauntlet.py --agents v80_tuned,v61_codex_safe --games 240 --workers 3
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
BASE = "v61_codex_safe"
GENOME = os.path.join(WORK, "out", "alak_w_best.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v80_tuned")
    ap.add_argument("--genome", default=GENOME)
    args = ap.parse_args()

    if not os.path.exists(args.genome):
        raise SystemExit(f"no genome at {args.genome} -- has the tuner accepted anything?")
    genome = json.load(open(args.genome))

    out = os.path.join(AGENTS, args.name)
    os.makedirs(out, exist_ok=True)
    for fn in ("main.py", "deck.csv"):
        shutil.copy(os.path.join(AGENTS, BASE, fn), os.path.join(out, fn))
    with open(os.path.join(out, "alak_w.json"), "w") as fh:
        json.dump(genome, fh, indent=1)

    # Prove the override actually binds, from a FOREIGN cwd, before anything is
    # measured. The base resolves alak_w.json relative to cwd first, so a bundle
    # that only works when you happen to be standing in it is exactly the silent
    # degradation this repo keeps paying for.
    sys.path.insert(0, os.path.join(WORK, "lib"))
    sys.path.insert(0, out)
    cwd = os.getcwd()
    try:
        os.chdir(out)
        env = {}
        with open("main.py", encoding="utf-8") as fh:
            exec(compile(fh.read(), "main.py", "exec"), env)
    finally:
        os.chdir(cwd)
    w = env.get("W") or {}
    differ = [k for k, v in genome.items() if abs(float(w.get(k, 1e18)) - float(v)) > 1e-6]
    base_env_changed = sum(1 for k, v in genome.items()
                           if abs(float(v) - float(w.get(k, v))) > 1e-6)
    if differ:
        raise SystemExit(f"alak_w.json did NOT bind: {len(differ)} weights differ, "
                         f"e.g. {differ[:5]}")
    print(f"built work/agents/{args.name}: {len(genome)} weights bound, "
          f"{base_env_changed} mismatches")
    print("validate with:")
    print(f"  python work/tools/gauntlet.py --agents {args.name},{BASE} "
          f"--games 240 --workers 3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
