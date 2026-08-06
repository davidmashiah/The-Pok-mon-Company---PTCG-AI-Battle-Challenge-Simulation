"""w9_fastsetup: hand the SETUP turns to the sub-policy that sets up fastest.

The mirror is 30% of the top-50 field and we win it 0.333 over 12 live games.
Turn-by-turn traces of those games show why: both sides swing the identical
Shadow Bullet for 180, so it is a pure race, and

    our first attack on turn <= 4   ->  2 wins, 0 losses
    our first attack on turn >= 5   ->  1 win,  7 losses

Real ladder Grimmsnarl players attack on turn 3-4. We average 4.89 and get
there by turn 4 in only 45% of games.

The coalition is not the fastest thing in its own bundle. Measured standalone
over 60 games each, first-attack turn:

    handwritten_v26  4.63   55% by turn 4      v22            5.00   37%
    v32              4.70   50%                rule_v24       5.09   43%
    the coalition    4.89   45%                v26_reexport   5.29   48%
                                               v28            5.51   37%

So give the setup turns to handwritten_v26 and leave everything after them
alone. The override sits at the very end of main.py's guard chain, after the
router and all four guards, and only fires while turn <= FAST_SETUP_TURNS -- the
window where the only thing that matters is getting a Stage 2 online. It is
legality-checked like every other guard, and falls through untouched if that
policy throws or proposes something illegal.

Note this is a SPEED hypothesis, not a strength one: handwritten_v26 and the
coalition field-test as a tie (0.6329 vs 0.6376, n=1116 each), so this is not
"the better policy wins", it is "the faster one drives the opening".

  python work/tools/build_fastsetup.py --turns 4
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
SRC = "w8_grimm_tuned"

IMPORT_ANCHOR = "from coalition_expert import agent as coalition_expert\n"
IMPORT_NEW = (
    "from coalition_expert import agent as coalition_expert\n"
    "# fastest setup of the six sub-policies (first attack turn 4.63 vs the\n"
    "# coalition's 4.89); used only for the opening, see the guard below\n"
    "from policies.handwritten_v26 import main as _fast_setup_policy\n"
)

TAIL_ANCHOR = """    if not _legal(chosen, select, option_count):
        chosen = fallback
"""
TAIL_NEW = """    # ---- SETUP-SPEED OVERRIDE -------------------------------------------
    # The mirror is a pure race at 180 damage a side, and on the ladder we lose
    # it 0.333 because we attack on turn 5 while real Grimmsnarl decks attack on
    # 3-4. During the opening the only thing that matters is getting a Stage 2
    # online, and handwritten_v26 does that a third of a turn sooner than the
    # coalition (4.63 vs 4.89, 55% vs 45% by turn 4). After the window closes,
    # the coalition and every guard above are back in charge untouched.
    try:
        _turn = int((obs.get("current") or {}).get("turn", 99))
    except Exception:
        _turn = 99
    if _turn <= FAST_SETUP_TURNS:
        try:
            _fast = list(_fast_setup_policy.agent(obs))
        except Exception:
            _fast = None
        if _fast is not None and _legal(_fast, select, option_count):
            chosen = _fast
    # ---------------------------------------------------------------------
    if not _legal(chosen, select, option_count):
        chosen = fallback
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    name = args.name or f"w9_fastsetup{args.turns}"

    dst = os.path.join(AGENTS, name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(AGENTS, SRC), dst,
                    ignore=shutil.ignore_patterns("__pycache__"))

    mp = os.path.join(dst, "main.py")
    with open(mp, encoding="utf-8") as fh:
        src = fh.read()
    for anchor, label in ((IMPORT_ANCHOR, "import"), (TAIL_ANCHOR, "guard")):
        if src.count(anchor) != 1:
            raise SystemExit(f"{label} anchor found {src.count(anchor)}x, expected 1")
    src = src.replace(IMPORT_ANCHOR, IMPORT_NEW)
    src = src.replace(TAIL_ANCHOR, TAIL_NEW)
    src = src.replace("EXPECTED_DECK = [",
                      f"FAST_SETUP_TURNS = {args.turns}\nEXPECTED_DECK = [", 1)
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write(src)
    compile(src, "main.py", "exec")

    # prove it loads and that the override is actually reachable
    import subprocess
    probe = (
        "import os,sys,tempfile\n"
        f"sys.path.insert(0, r'{dst}')\n"
        f"sys.path.insert(0, r'{os.path.join(WORK, 'lib')}')\n"
        "os.chdir(tempfile.mkdtemp(prefix='w9probe_'))\n"
        f"src=open(r'{mp}',encoding='utf-8').read()\n"
        "env={}\nexec(compile(src,'main.py','exec'),env)\n"
        "print('FAST_SETUP_TURNS', env.get('FAST_SETUP_TURNS'))\n"
        "print('fast policy loaded', env.get('_fast_setup_policy') is not None)\n"
        "fns=[v for v in env.values() if callable(v)]\n"
        "r=fns[-1]({'current':None,'select':None})\n"
        "print('SETUP_OK', isinstance(r,(list,tuple)) and len(r)==60)\n")
    p = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "").strip()
    print(out or (p.stderr or "")[-700:])
    if "SETUP_OK True" not in out or "fast policy loaded True" not in out:
        raise SystemExit("w9 does not load correctly")
    print(f"built work/agents/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
