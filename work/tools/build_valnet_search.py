"""Replace the search's LEAF with the trained value net.

The diagnosis this acts on. The search machinery is healthy -- 2225 decisions/s,
`search_liveness.py` shows every requested playout executes, zero errors. What is
wrong is what it believes at the bottom. `_evaluate` is a hand-written
arithmetic: 1000 per prize, plus damage dealt, minus damage taken, small bonuses
for bench and hand. It is BIASED, not merely noisy, so buying more playouts
converges faster onto a wrong number -- which is exactly what happened when the
muzzle came off (mirror 0.4915 -> 0.3025, -4.5 sigma at 72 playouts).

The replacement is measured, not assumed. `valfeat.features` + the ridge net
separate winning from losing positions at **AUC 0.7430**, against the
prize-difference baseline's 0.6308 -- i.e. clearly better than the single
hand-written quantity `_evaluate` leans on hardest -- and its deciles are
monotone from 0.190->0.142 up to 0.806->0.857.

Both files are COPIED INTO the bundle rather than imported from work/lib,
because the ladder runs the agent directory alone and an import that resolves
locally and not on Kaggle is the silent-failure mode this project has already
hit five times.

Scale matters and is easy to get wrong. `_evaluate` returned ~1000 per prize;
the net returns a probability in (0,1). Multiplying by 1000 makes one point of
MARGIN one tenth of a percent of win probability, so MARGIN must be reset -- the
inherited 1000.0 would demand a 100% swing and the search would never speak, and
the inherited 250 would still demand 25%. Terminal states keep their +/-1e6 so a
proven win always outranks any estimate.

The original heuristic stays as the fallback when featurising fails, so a bad
frame degrades to today's behaviour instead of to zero.

  python work/tools/build_valnet_search.py --name w80_val --det 6 --cands 4 \
      --budget 2.0 --margin 40
"""
import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
LIB = os.path.join(WORK, "lib")

KNOBS = {
    "budget": (r"^BUDGET_S\s*=.*$", "BUDGET_S = {!r}"),
    "det": (r"^DETERMINIZATIONS\s*=.*$", "DETERMINIZATIONS = {!r}"),
    "cands": (r"^MAX_CANDIDATES\s*=.*$", "MAX_CANDIDATES = {!r}"),
    "margin": (r"^MARGIN\s*=.*$", "MARGIN = {!r}"),
    "steps": (r"^ROLLOUT_STEPS\s*=.*$", "ROLLOUT_STEPS = {!r}"),
    "gate": (r"^GATE\s*=.*$", "GATE = {!r}"),
}

OLD_EVAL_HEAD = 'def _evaluate(o, me_idx):\n'

NEW_EVAL = '''try:
    import valfeat as _valfeat
    import valnet as _valnet
    _VALNET_OK = True
except Exception:
    _VALNET_OK = False

VALNET_SCALE = 1000.0


def _evaluate(o, me_idx):
    """Trained win probability at the leaf; the old arithmetic as fallback."""
    st = o.current
    if st is None:
        return 0.0
    if st.result is not None and st.result != -1:
        return 1e6 if st.result == me_idx else -1e6
    if _VALNET_OK:
        try:
            f = _valfeat.features(o, me_idx)
            if f is not None:
                # RANK with the raw score. score() squashes, and an earlier
                # build shipped weights whose raw output sat far outside [0,1],
                # so the old clamp returned a constant for 94% of positions and
                # the search ranked leaves with noise.
                return VALNET_SCALE * float(_valnet.score_raw(f))
        except Exception:
            pass
    return _evaluate_heuristic(o, me_idx)


def _evaluate_heuristic(o, me_idx):
'''


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

    if src.count(OLD_EVAL_HEAD) != 1:
        raise SystemExit("could not find exactly one _evaluate definition")
    src = src.replace(OLD_EVAL_HEAD, NEW_EVAL, 1)

    applied = {}
    for key, (pattern, fmt) in KNOBS.items():
        val = getattr(a, key)
        if val is None:
            continue
        src, n = re.subn(pattern, fmt.format(val), src, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"could not rewrite {key!r} -- {n} matches")
        applied[key] = val
    compile(src, "search_validator.py", "exec")

    for need in ("valfeat.py", "valnet.py"):
        if not os.path.exists(os.path.join(LIB, need)):
            raise SystemExit(f"work/lib/{need} missing -- train the net first")

    dst = os.path.join(AGENTS, a.name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src_dir, dst, ignore=shutil.ignore_patterns("__pycache__"))
    with open(os.path.join(dst, "search_validator.py"), "w",
              encoding="utf-8") as f:
        f.write(src)
    for need in ("valfeat.py", "valnet.py"):
        shutil.copy2(os.path.join(LIB, need), os.path.join(dst, need))

    print(f"built work/agents/{a.name} from {a.base}")
    print("   leaf     -> trained value net (AUC 0.7430 vs 0.6308 prize-diff)")
    print("   vendored -> valfeat.py, valnet.py (bundle must stand alone)")
    for k, v in applied.items():
        print(f"   {k:8s} -> {v}")
    if "margin" not in applied:
        print("\n   !! MARGIN not set. The leaf now returns 0-1000 win "
              "probability,\n      so an inherited 1000.0 can never be "
              "exceeded and the search is mute.")
    print("\nVerify it RUNS before believing any win rate:")
    print(f"  python work/tools/search_liveness.py --agent {a.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
