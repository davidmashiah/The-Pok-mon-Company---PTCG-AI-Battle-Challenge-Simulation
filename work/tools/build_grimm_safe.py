"""w7_grimm_safe = w5_grimmsnarl + the one fix that makes it shippable.

Why this exists. Under the corrected top-band field weights, tetsutani's
published Grimmsnarl agent projects 805 against our current submission's 726,
and it is the structurally better base: Grimmsnarl is 30% of the top 50, so
playing it turns the largest slice of the field into the MIRROR at 0.457 rather
than the 0.207 hole v61 has there.

But it does not survive the gate. Its bundle resolves like this:

    if "__file__" in globals():
        BASE = Path(globals()["__file__"]).resolve().parent
    else:
        _base_candidates = (Path("/kaggle_simulations/agent"), Path.cwd())

kaggle_environments exec()s main.py, so `__file__` is undefined; on Kaggle the
absolute path saves it; anywhere else BASE falls back to the CWD and import dies
outright on `models/feature_schema.pkl.gz`. That is the third time this repo has
hit the same shape of bug -- v34 shipped a deck it never played, our model
weights loaded relative to cwd and silently degraded the agent to a plain
heuristic, and v61's own decklist resolved only inside the setup frame.

The fix is to LOOK for the bundle rather than assume the cwd is it. sys.path is
the reliable signal: kaggle_environments inserts the agent directory into it,
and so does every harness in this repo. The Kaggle absolute path stays first, so
behaviour on the ladder is bit-for-bit what its author submitted.

  python work/tools/build_grimm_safe.py
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
BASE_AGENT = "w5_grimmsnarl"
NEW_AGENT = "w7_grimm_safe"

OLD = '''    _base_candidates = (Path("/kaggle_simulations/agent"), Path.cwd())'''

NEW = '''    # Find the bundle instead of assuming the cwd is it. The harness exec()s
    # this file so __file__ is undefined, and the cwd fallback below crashed
    # import outright on models/feature_schema.pkl.gz whenever the process was
    # started from anywhere else. sys.path is the dependable signal:
    # kaggle_environments inserts the agent directory into it, and every local
    # harness does the same. The Kaggle absolute path stays FIRST so play on the
    # ladder is unchanged.
    import sys as _sys
    _base_candidates = tuple(
        [Path("/kaggle_simulations/agent")]
        + [Path(_p) for _p in _sys.path if _p]
        + [Path.cwd()]
    )'''


def main():
    src_dir = os.path.join(AGENTS, BASE_AGENT)
    dst_dir = os.path.join(AGENTS, NEW_AGENT)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir,
                    ignore=shutil.ignore_patterns("__pycache__"))

    mp = os.path.join(dst_dir, "main.py")
    with open(mp, encoding="utf-8") as fh:
        src = fh.read()
    if OLD not in src:
        raise SystemExit("anchor not found -- base changed, re-read main.py")
    if src.count(OLD) != 1:
        raise SystemExit(f"anchor found {src.count(OLD)}x, expected 1")
    src = src.replace(OLD, NEW)
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write(src)
    compile(src, "main.py", "exec")

    # Prove it loads from a FOREIGN cwd, which is the whole point of the change.
    import subprocess
    import tempfile
    probe = (
        "import os,sys,tempfile\n"
        f"sys.path.insert(0, r'{dst_dir}')\n"
        f"sys.path.insert(0, r'{os.path.join(WORK, 'lib')}')\n"
        "os.chdir(tempfile.mkdtemp(prefix='grimmprobe_'))\n"
        f"src=open(r'{mp}',encoding='utf-8').read()\n"
        "env={}\n"
        "exec(compile(src,'main.py','exec'), env)\n"
        "fns=[v for v in env.values() if callable(v)]\n"
        "r=fns[-1]({'current':None,'select':None})\n"
        "print('SETUP_OK', isinstance(r,(list,tuple)) and len(r)==60)\n"
    )
    p = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       cwd=os.path.dirname(WORK))
    print((p.stdout or "").strip() or (p.stderr or "")[-800:])
    if "SETUP_OK True" not in (p.stdout or ""):
        raise SystemExit("still does not load from a foreign cwd")
    print(f"built work/agents/{NEW_AGENT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
