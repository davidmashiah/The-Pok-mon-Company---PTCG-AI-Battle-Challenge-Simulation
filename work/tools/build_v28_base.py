"""Give w34's override stack a better baseline to override.

Two measurements that only make sense together:

  * Ablating w34's layers HURTS -- `off:model` -0.0765, `off:human` -0.1751
    (n~294 each vs an intact 0.4915). Inside w34, that stack is load-bearing.
  * `_sub_v28`, which bypasses the entire stack and calls `policies/v28/main.py`
    directly, still scores HIGHER on the corrected panel: field 0.6455 against
    w34's 0.6243, about +16 rating, on two columns that clear significance
    (Dragapult +0.167 / 3.8 sigma, Mega Lucario +0.124 / 2.5 sigma).

The reading that fits both: w34's `strategic_policy` baseline is weaker than
v28's policy, and the guard stack spends its value climbing back out of a hole
the baseline dug. If that is right the two are additive -- keep the stack, hand
it the better starting action.

So this swaps ONLY the seed. `fallback` becomes v28's action instead of
`strategic_policy`'s, everywhere it is used: as the GBM's fallback, as the
"strategic" entry the human/route layer chooses among, and as the final legality
backstop. Every guard, the GBM, the router and the validator are untouched.

This is a real fork, not a tweak, and it can plausibly be worse: `human_choose`
was tuned against strategic_policy's habits, so a different seed may sit outside
what it expects. That is the experiment.

Both pilots ship in this bundle already -- w34_koroll and _sub_v28 are the same
187 files apart from main.py -- so nothing is vendored and no module can collide.

  python work/tools/build_v28_base.py --name w90_v28base
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")

IMPORT_ANCHOR = "from strategic_policy import agent as strategic_agent"
IMPORT_NEW = ("from strategic_policy import agent as strategic_agent\n"
              "from policies.v28 import main as _v28_base")

# the reset path: keep strategic warm, and prime v28 too
RESET_ANCHOR = """        try:
            strategic_agent(obs)
        except Exception:
            pass
        return list(DECK)"""
RESET_NEW = """        try:
            strategic_agent(obs)
        except Exception:
            pass
        try:
            _v28_base.agent(obs)
        except Exception:
            pass
        return list(DECK)"""

# the seed itself
FALLBACK_ANCHOR = """    try:
        fallback = strategic_agent(obs)
    except Exception:
        fallback = []"""
FALLBACK_NEW = """    # v28's policy as the baseline the stack refines. strategic_agent is still
    # driven so its internal state stays coherent for the guards that read it.
    try:
        strategic_agent(obs)
    except Exception:
        pass
    try:
        fallback = list(_v28_base.agent(obs))
    except Exception:
        try:
            fallback = strategic_agent(obs)
        except Exception:
            fallback = []"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="w34_koroll")
    ap.add_argument("--name", default="w90_v28base")
    a = ap.parse_args()

    src_path = os.path.join(AGENTS, a.base, "main.py")
    src = open(src_path, encoding="utf-8-sig").read()

    for label, old, new in (("import", IMPORT_ANCHOR, IMPORT_NEW),
                            ("reset", RESET_ANCHOR, RESET_NEW),
                            ("fallback", FALLBACK_ANCHOR, FALLBACK_NEW)):
        if src.count(old) != 1:
            raise SystemExit(
                f"anchor {label!r} matched {src.count(old)} times -- refusing "
                f"to guess")
        src = src.replace(old, new, 1)
    compile(src, "main.py", "exec")

    dst = os.path.join(AGENTS, a.name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(AGENTS, a.base), dst,
                    ignore=shutil.ignore_patterns("__pycache__"))
    with open(os.path.join(dst, "main.py"), "w", encoding="utf-8") as f:
        f.write(src)

    print(f"built work/agents/{a.name} from {a.base}")
    print("   fallback seed -> policies/v28  (was strategic_policy)")
    print("   GBM, router, human, advisor, residual, tactical, development,")
    print("   validator: all UNCHANGED")
    print("\nCompare against BOTH parents -- it has to beat the better of them:")
    print(f"  python work/tools/field_now.py --agents "
          f"w34_koroll,_sub_v28,{a.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
