"""Un-muzzle the search that is already installed in our best agent.

w34_koroll ships search_validator.py and it WORKS -- search_liveness reports
ran=37/37, 41/41, 33/33, 46/46 with zero errors. It has simply never been
allowed to do anything:

    BUDGET_S=0.6  DET=3  CANDS=3  MARGIN=1000.0

  * 3 determinizations x 3 candidates is NINE playouts per decision. Measured
    end to end that is 12 s of an episode. The ladder allows 600 s. We are
    spending 2% of the compute we are entitled to.
  * MARGIN=1000 means the search may only overrule the policy when it sees a
    full prize card of advantage. The measured leaf-gap distribution over 69
    searched decisions is median 61, p75 666, p90 1528 -- so that threshold
    silences roughly nine tenths of what the search actually found. It overrode
    9 times in 157 searches.

"Search did not help" was never tested. What was tested was a search permitted
to speak 6% of the time on nine playouts.

The two dials must move TOGETHER, and that is the whole design of this tool.
Lowering MARGIN alone just lets nine-playout noise overrule a tuned policy: at
that sample size a leaf gap of 61 is not a signal. Raising playouts alone leaves
the good estimates gagged. So each variant sets both, and the margin is chosen
from the gap distribution rather than as a round number.

Knobs are BAKED INTO THE FILE, never read from the environment at measure time.
gauntlet.py content-hashes the bundle; an env var changes behaviour without
changing the hash, so two different configurations would silently pool their
games into one cell. That is the single failure this repo guards hardest
against, and it has already happened here twice.

  python work/tools/build_deep_search.py --name w60_deep --det 12 --cands 6 \
      --budget 5.0 --margin 250
"""
import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")

# constant name -> (regex over search_validator.py, formatter)
KNOBS = {
    "budget": (r"^BUDGET_S\s*=.*$", "BUDGET_S = {!r}"),
    "det": (r"^DETERMINIZATIONS\s*=.*$", "DETERMINIZATIONS = {!r}"),
    "cands": (r"^MAX_CANDIDATES\s*=.*$", "MAX_CANDIDATES = {!r}"),
    "margin": (r"^MARGIN\s*=.*$", "MARGIN = {!r}"),
    "steps": (r"^ROLLOUT_STEPS\s*=.*$", "ROLLOUT_STEPS = {!r}"),
    "gate": (r"^GATE\s*=.*$", "GATE = {!r}"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="w34_koroll")
    ap.add_argument("--name", required=True)
    ap.add_argument("--budget", type=float)
    ap.add_argument("--det", type=int)
    ap.add_argument("--cands", type=int)
    ap.add_argument("--margin", type=float)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--gate", type=float)
    a = ap.parse_args()

    src_dir = os.path.join(AGENTS, a.base)
    vpath = os.path.join(src_dir, "search_validator.py")
    if not os.path.exists(vpath):
        raise SystemExit(f"{a.base} has no search_validator.py")
    src = open(vpath, encoding="utf-8-sig").read()

    applied = {}
    for key, (pattern, fmt) in KNOBS.items():
        val = getattr(a, key)
        if val is None:
            continue
        new = fmt.format(val)
        src, n = re.subn(pattern, new, src, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"could not rewrite {key!r} -- {n} matches")
        applied[key] = val
    if not applied:
        raise SystemExit("no knobs given; nothing would differ from the base")
    compile(src, "search_validator.py", "exec")

    dst = os.path.join(AGENTS, a.name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src_dir, dst, ignore=shutil.ignore_patterns("__pycache__"))
    with open(os.path.join(dst, "search_validator.py"), "w",
              encoding="utf-8") as f:
        f.write(src)

    print(f"built work/agents/{a.name} from {a.base}")
    for k, v in applied.items():
        print(f"   {k:8s} -> {v}")
    playouts = (applied.get("det", 3) or 3) * (applied.get("cands", 3) or 3)
    print(f"   ~{playouts} playouts per searched decision "
          f"(base is 9; the episode budget is 600 s and the base uses 12 s)")
    print("\nVerify it with search_liveness.py BEFORE trusting any win rate:\n"
          f"  python work/tools/search_liveness.py --agent {a.name} --games 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
