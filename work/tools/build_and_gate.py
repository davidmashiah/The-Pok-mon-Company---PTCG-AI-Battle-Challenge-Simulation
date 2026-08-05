"""Build a submission tarball and GATE it. Exits non-zero on any failure.

Every check here exists because it is a way this competition can silently eat a
submission slot. Checks are named; add to the list, never remove.

Usage:
  python work/tools/build_and_gate.py --agent v1_greedy [--games 30]
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
LIB = os.path.join(WORK, "lib")

FAILURES = []
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


# --------------------------------------------------------------- deck legality
def gate_deck(deck_path, cards):
    print("\n== deck legality ==")
    with open(deck_path) as f:
        rows = [ln.strip() for ln in f if ln.strip()]
    ids = []
    bad = []
    for r in rows:
        try:
            ids.append(int(r))
        except ValueError:
            bad.append(r)
    check("deck.csv parses as ints", not bad, f"unparseable: {bad[:5]}")
    check("deck has exactly 60 cards", len(ids) == 60, f"got {len(ids)}")

    unknown = [i for i in ids if i not in cards]
    check("all card IDs exist in engine pool", not unknown, f"unknown: {sorted(set(unknown))[:8]}")

    cnt = Counter(ids)
    # >4 copies allowed only for Basic Energy (cardType 5)
    over = []
    for cid, n in cnt.items():
        c = cards.get(cid)
        if c is None:
            continue
        if n > 4 and int(c.cardType) != 5:
            over.append((cid, c.name, n))
    check("no non-basic-energy card exceeds 4 copies", not over, f"{over[:5]}")

    ace = [(cid, cards[cid].name, n) for cid, n in cnt.items()
           if cid in cards and cards[cid].aceSpec]
    n_ace = sum(n for _, _, n in ace)
    check("at most 1 ACE SPEC card", n_ace <= 1, f"{n_ace} -> {ace}")

    basics = sum(n for cid, n in cnt.items()
                 if cid in cards and int(cards[cid].cardType) == 0 and cards[cid].basic)
    check("deck contains >=1 Basic Pokemon", basics >= 1, f"{basics} basics")
    if basics < 8:
        print(f"       (note: only {basics} Basic Pokemon — high mulligan rate)")
    return ids


# ------------------------------------------------- deck we SHIP == deck we PLAY
DECK_IDENTITY_RUNNER = r'''
import json, os, sys, tempfile
AG = sys.argv[1]
sys.path.insert(0, AG)
# The point of this check: run from a cwd that is NOT the agent dir. The rest of
# the gate chdir'd into the bundle, which hid the failure it is looking for.
os.chdir(tempfile.mkdtemp(prefix="deckid_"))
with open(os.path.join(AG, "main.py"), encoding="utf-8") as fh:
    _src = fh.read()
_env = {}
exec(compile(_src, "main.py", "exec"), _env)
played = None
for _n in ("my_deck", "DECK", "MY_DECK", "deck"):
    _v = _env.get(_n)
    if isinstance(_v, list) and len(_v) == 60:
        played = [int(x) for x in _v]
        break
print(json.dumps({"played": played,
                  "wrote_files": sorted(os.listdir(os.getcwd()))}))
'''


def gate_deck_identity(stage_dir, py):
    """Does the agent play the decklist we are shipping?

    v34_stadium shipped a 3x Gravity Mountain deck.csv and played v32's 1x list.
    main.py did Path("deck.csv").write_text(DECK) at import and then read
    "deck.csv" back, so the bundled file lost to a hardcoded constant every
    time. The ladder result of that submission (754.7 against v32's 692.2) was
    therefore a noise measurement between two byte-equivalent agents.

    Nothing in the old gate could see it: gate_deck validated the staged
    deck.csv, and the runtime gate chdir'd INTO the bundle, so the file it
    validated was the one main.py had just overwritten.
    """
    print("\n== deck identity (ship == play) ==")
    with open(os.path.join(stage_dir, "deck.csv"), encoding="utf-8") as f:
        shipped = sorted(int(x) for x in f.read().split() if x.strip())
    r = subprocess.run([py, "-c", DECK_IDENTITY_RUNNER, stage_dir],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        check("agent loads from a foreign cwd", False, r.stderr[-400:])
        return
    try:
        info = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        check("deck-identity probe returned JSON", False, r.stdout[-300:])
        return
    played = info.get("played")
    check("agent exposes a 60-card decklist", played is not None)
    if played is None:
        return
    check("deck PLAYED == deck SHIPPED in deck.csv", sorted(played) == shipped,
          "shipped %s / played %s" % (
              _fmt_diff(shipped, sorted(played)) or "(same)",
              _fmt_diff(sorted(played), shipped) or "(same)"))
    check("agent writes nothing into cwd at import",
          not info.get("wrote_files"),
          "created %s -- a module-level write races other processes sharing "
          "the cwd and can clobber the bundled deck" % info.get("wrote_files"))


def _fmt_diff(a, b):
    """Cards in a that are not accounted for by b, as 'id xN'."""
    from collections import Counter
    d = Counter(a) - Counter(b)
    return ", ".join(f"{k} x{v}" for k, v in sorted(d.items()))


# --------------------------------------------------------------- tar structure
def gate_tar(tar_path):
    print("\n== tarball structure ==")
    with tarfile.open(tar_path, "r:gz") as tf:
        names = tf.getnames()
    top = [n for n in names if "/" not in n.strip("./").rstrip("/") or n.count("/") == 0]
    check("main.py at TOP level (not nested)", "main.py" in names,
          f"top-level entries: {sorted(set(n.split('/')[0] for n in names))}")
    check("deck.csv at TOP level", "deck.csv" in names)
    check("cg/ package bundled", any(n.startswith("cg/") for n in names))
    check("policy.py bundled", "policy.py" in names)
    # Kaggle runs Linux x86_64 — the .so MUST be present or the agent dies on load
    check("LINUX engine libcg.so present", "cg/libcg.so" in names,
          "Kaggle runs Linux; cg.dll alone would fail")
    check("cg/api.py present", "cg/api.py" in names)
    check("cg/__init__.py present", "cg/__init__.py" in names)
    check("no nested duplicate of the agent dir", not any(
        n.startswith("sample_submission/") or n.count("main.py") > 1 for n in names))
    check("no __pycache__ shipped", not any("__pycache__" in n for n in names))
    size = os.path.getsize(tar_path)
    check("tarball < 100 MB", size < 100 * 1024 * 1024, f"{size/1e6:.1f} MB")
    return names


# --------------------------------------------------- run from a foreign cwd
RUNNER = r'''
import json, os, random, sys, time
AG = sys.argv[1]
N  = int(sys.argv[2])
sys.path.insert(0, AG)
os.chdir(AG)   # harness runs with the agent dir as cwd; deck.csv is relative

# ---------------------------------------------------------------------------
# Load main.py EXACTLY as kaggle_environments does:
#   kaggle_environments/agent.py::get_last_callable ->
#       exec(compile(src, path, "exec"), env)   # NOTE: __file__ NOT defined
#   then it takes the LAST callable in that namespace.
# Loading via `import main` instead would define __file__ and hide the very
# bug that killed submission 55194301.
# ---------------------------------------------------------------------------
with open(os.path.join(AG, "main.py")) as fh:
    _src = fh.read()
_env = {}
exec(compile(_src, "main.py", "exec"), _env)
_items = [(k, v) for k, v in _env.items() if callable(v)]
if not _items:
    print("ERR no callable defined in main.py"); sys.exit(5)
_lastname, _last = _items[-1]
# Bind by the DICT key, not __name__: a wrapper assigned to an existing name
# keeps that name's original dict position, so the last callable can be an
# inner/original function whose __name__ still reads "agent".
if _lastname != "agent":
    print("ERR last callable in module dict is %r (fn __name__=%r); "
          "kaggle_environments would call that, not agent"
          % (_lastname, getattr(_last, "__name__", None))); sys.exit(6)

class agent_mod:
    agent = staticmethod(_last)
    # Do NOT assume the deck lives in a module global. Ours uses DECK, one
    # public agent uses my_deck, another has no global at all and reads
    # deck.csv lazily inside agent(). Two working agents were failed by our own
    # assertion before this. deck.csv is the contract; read that.
    DECK = None

for _n in ("DECK", "my_deck", "MY_DECK", "deck"):
    _v = _env.get(_n)
    if isinstance(_v, list) and len(_v) == 60:
        agent_mod.DECK = list(_v)
        break
if agent_mod.DECK is None:
    with open(os.path.join(AG, "deck.csv")) as _fh:
        agent_mod.DECK = [int(x.strip()) for x in _fh if x.strip()][:60]

from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish

deck = agent_mod.DECK
assert deck is not None and len(deck) == 60, "agent DECK not 60"

t0 = time.time(); worst = 0.0; steps = 0; res = []
per_game = []          # agent-only seconds per episode (the 600s budget)
for g in range(N):
    obs, sd = battle_start(list(deck), list(deck))
    if obs is None:
        print("ERR battle_start None"); sys.exit(3)
    gtime = 0.0
    for _ in range(4000):
        o = to_observation_class(obs)
        if o.current is not None and o.current.result != -1:
            res.append(o.current.result); break
        ts = time.time()
        sel = agent_mod.agent(obs)
        dt = time.time() - ts
        gtime += dt
        worst = max(worst, dt); steps += 1
        s = o.select
        assert isinstance(sel, list), "agent returned non-list"
        assert all(isinstance(i, int) for i in sel), "non-int in selection"
        assert len(set(sel)) == len(sel), "duplicate indices"
        assert s.minCount <= len(sel) <= s.maxCount, (
            "count out of range %d not in [%d,%d]" % (len(sel), s.minCount, s.maxCount))
        assert all(0 <= i < len(s.option) for i in sel), "index out of range"
        obs = battle_select(sel)
    else:
        print("ERR game did not terminate"); sys.exit(4)
    per_game.append(gtime)
    battle_finish()
_dz = {}
try:
    # stats live in the exec'd module namespace (_env), NOT on the agent_mod shim.
    # Only report them for agents that actually wire in the net -- otherwise the
    # fire-rate check would fail every agent that legitimately does not use it.
    if "_DZ_STATS" in _env:
        _dz = dict(_env.get("_DZ_STATS") or {})
        _dz["ok"] = bool(_env.get("_DZ_OK"))
except Exception:
    pass
_ft = {}
try:
    if "_FT_STATS" in _env:
        _ft = dict(_env.get("_FT_STATS") or {})
except Exception:
    pass
_vz = {}
try:
    if "_VZ_STATS" in _env:
        _vz = dict(_env.get("_VZ_STATS") or {})
except Exception:
    pass
print(json.dumps({"vz": _vz, "ft": _ft, "dz": _dz, "games": N, "results": res, "worst_move_s": worst,
                  "steps": steps, "total_s": time.time()-t0,
                  "worst_episode_agent_s": max(per_game) if per_game else 0.0,
                  "mean_episode_agent_s": sum(per_game)/len(per_game) if per_game else 0.0}))
'''


def gate_source(stage_dir):
    """Static checks on shipped source. Each is a past catastrophe by name."""
    print("\n== source checks (past catastrophes, re-checked by name) ==")
    mp = os.path.join(stage_dir, "main.py")
    with open(mp, encoding="utf-8-sig") as f:   # tolerate a BOM
        src = f.read()

    # CATASTROPHE 2026-08-02, submission 55194301:
    # NameError: name '__file__' is not defined  (harness exec()s main.py)
    code_lines = [ln for ln in src.splitlines()
                  if not ln.strip().startswith("#")]
    code = "\n".join(code_lines)
    # strip docstrings crudely so prose mentioning __file__ doesn't trip it
    for q in ('"""', "'''"):
        parts = code.split(q)
        code = "".join(parts[::2]) if len(parts) > 2 else code
    # Guarded use is fine and is what a careful agent does:
    #     try: ROOT = __file__
    #     except NameError: ROOT = None
    # Flagging that is a false positive -- it failed a public agent that had
    # handled the trap correctly. Only unguarded use is fatal.
    guarded = ("except NameError" in code) or ("except Exception" in code and
                                               "__file__" in code and
                                               "try:" in code)
    check("main.py never references __file__ UNGUARDED "
          "(harness exec has no __file__)",
          ("__file__" not in code) or guarded,
          "kaggle_environments exec()s main.py; __file__ is undefined there"
          + ("  [guarded use detected - OK]" if guarded else ""))

    # The harness takes the LAST callable in the exec'd namespace.
    # AST source order is NOT what the harness uses. get_last_callable takes the
    # last callable in the exec'd module DICT, and rebinding an existing name
    # (`_inner = agent; def agent(...)`) leaves `agent` at its original dict
    # position -- so a wrapper can look last in the source while the UNWRAPPED
    # original is what actually gets called. That happened here and the wrapper
    # silently never ran. The runtime check below is the authoritative one; the
    # AST check is kept only as a readability signal.
    import ast
    try:
        tree = ast.parse(src)
        defs = [n.name for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        if not (defs and defs[-1] == "agent"):
            print(f"       (note: last *source* definition is {defs[-1] if defs else None!r}; "
                  f"runtime order is what matters)")
    except SyntaxError as e:
        check("main.py parses", False, str(e))

    check("fsearch.py bundled", os.path.exists(os.path.join(stage_dir, "fsearch.py")))

    # A feature guarded by try/except ImportError does not announce its own
    # absence. Assert the opponent-deck library actually loads and is non-empty
    # from the STAGED bundle, so "declared but silently disabled" cannot ship.
    import subprocess as _sp
    probe = (
        "import sys; sys.path.insert(0, r'%s');\n"
        "import fsearch;\n"
        "d = fsearch.meta_decks();\n"
        "print('METADECKS', len(d))\n" % stage_dir
    )
    try:
        r = _sp.run([sys.executable, "-c", probe], capture_output=True,
                    text=True, timeout=180, cwd=stage_dir)
        n = 0
        for ln in (r.stdout or "").splitlines():
            if ln.startswith("METADECKS"):
                n = int(ln.split()[1])
        check("opponent-deck library loads from the bundle (non-empty)", n > 0,
              f"{n} decklists visible to fsearch at runtime")
    except Exception as e:
        check("opponent-deck library loads from the bundle (non-empty)", False,
              f"{type(e).__name__}: {e}")

    # Same rule as meta_decks: a component imported inside try/except does not
    # announce its own absence. If main.py wires in the learned re-ranker, then
    # the weights must load FROM THE STAGED BUNDLE and produce finite scores --
    # otherwise the agent silently degrades to plain v14 and reports nothing.
    with open(mp) as fh:
        _msrc = fh.read()
    if "import dznp" in _msrc:
        pr = os.path.join(HERE, "dz_probe.py")
        try:
            r2 = _sp.run([sys.executable, pr, stage_dir], capture_output=True,
                         text=True, timeout=180, cwd=stage_dir)
            good = any(ln.startswith("DZOK") for ln in (r2.stdout or "").splitlines())
            check("learned re-ranker loads + scores from the bundle", good,
                  (r2.stdout or "").strip().splitlines()[-1] if good
                  else (r2.stderr or "")[-300:])
        except Exception as e:
            check("learned re-ranker loads + scores from the bundle", False,
                  f"{type(e).__name__}: {e}")
        check("dz_weights.npz bundled",
              os.path.exists(os.path.join(stage_dir, "dz_weights.npz")))

    for extra in ("policy.py", "fsearch.py"):
        p = os.path.join(stage_dir, extra)
        if os.path.exists(p):
            with open(p) as f:
                s = f.read()
            check(f"{extra} never references __file__", "__file__" not in s)


def gate_runtime(stage_dir, games, py):
    print("\n== runtime validation (exec-loaded exactly like the harness) ==")
    runner = os.path.join(stage_dir, "_runner.py")
    with open(runner, "w") as f:
        f.write(RUNNER)
    t0 = time.time()
    p = subprocess.run([py, runner, stage_dir, str(games)],
                       capture_output=True, text=True, timeout=1800)
    out = (p.stdout or "").strip().splitlines()
    ok = p.returncode == 0
    if not ok:
        print(textwrap.indent((p.stdout or "")[-1500:], "      "))
        print(textwrap.indent((p.stderr or "")[-2500:], "      "))
    check(f"agent plays {games} self-play games with no crash / illegal move", ok,
          "" if ok else f"exit {p.returncode}")
    stats = {}
    if ok and out:
        import json as _j
        try:
            stats = _j.loads(out[-1])
        except Exception:
            pass
    if stats:
        _v2 = stats.get("vz") or {}
        if _v2:
            check("learned VALUE net actually scored leaves during play",
                  _v2.get("leaf_calls", 0) > 0,
                  f"{_v2.get('leaf_calls',0)} leaf evaluations, "
                  f"{_v2.get('searches',0)} searches, "
                  f"{_v2.get('reranked',0)} re-ranked the policy's top pick")
        _f2 = stats.get("ft") or {}
        if _f2:
            print("       full-turn search:", _f2)
        _d = stats.get("dz") or {}
        if _d:
            _c, _f, _ch = _d.get("calls", 0), _d.get("fired", 0), _d.get("changed", 0)
            check("learned re-ranker actually fired during play", _f > 0,
                  f"{_f}/{_c} decisions above margin, {_ch} changed the pick")
        wm = stats.get("worst_move_s", 0)
        we = stats.get("worst_episode_agent_s", 0)
        me = stats.get("mean_episode_agent_s", 0)
        print(f"       {stats['steps']} moves, worst move {wm*1000:.0f} ms, "
              f"episode agent-time mean {me:.1f}s / worst {we:.1f}s "
              f"({stats['total_s']:.1f}s wall)")
        # THE real constraint: cabt.json has actTimeout=0 and
        # observation.remainingOverageTime=600 -> each agent gets 600 s of
        # thinking for the WHOLE episode. There is no per-move limit; every
        # second draws from that one pool. Gate at 50% of it: local hardware
        # is not Kaggle hardware, and a long game costs more than a short one.
        check("worst episode agent-time < 300 s (50% of the 600 s pool)",
              we < 300.0, f"{we:.1f}s of 600s")
        if we > 120:
            print("       WARNING: >20% of the episode budget used locally; "
                  "scale search against remainingOverageTime, not a fixed "
                  "per-move constant")
        r = Counter(stats.get("results", []))
        print(f"       results (self-play, expect ~even): {dict(r)}")
    os.remove(runner)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    agent_dir = os.path.join(WORK, "agents", args.agent)
    if not os.path.isdir(agent_dir):
        raise SystemExit(f"no such agent dir: {agent_dir}")

    py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    if not os.path.exists(py):
        py = sys.executable

    print(f"== building '{args.agent}' ==")
    stage = tempfile.mkdtemp(prefix="ptcg_build_")
    # 1. agent files
    for fn in os.listdir(agent_dir):
        if fn == "__pycache__":
            continue
        shutil.copy2(os.path.join(agent_dir, fn), os.path.join(stage, fn))
    # 2. shared libs + engine
    # meta_decks.py was missing from every earlier bundle. fsearch imports it
    # inside a bare try/except, so its absence silently disabled opponent-deck
    # matching instead of failing -- the same swallowed-import failure this
    # project found in the public agents. Gate below now asserts it LOADS.
    for shared in ("policy.py", "fsearch.py", "meta_decks.py", "pimc.py",
                   "dznp.py", "dzfeat.py", "dz_weights.npz",
                   "vznp.py", "vz_weights.npz"):
        sp = os.path.join(LIB, shared)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(stage, shared))
    shutil.copytree(os.path.join(LIB, "cg"), os.path.join(stage, "cg"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    print(f"  staged at {stage}")
    print("  contents:", sorted(os.listdir(stage)))

    # deck legality needs the engine card table
    sys.path.insert(0, LIB)
    from cg.api import all_card_data
    cards = {c.cardId: c for c in all_card_data()}
    gate_deck(os.path.join(stage, "deck.csv"), cards)
    gate_deck_identity(stage, py)
    gate_source(stage)

    # 3. runtime validation on the STAGED copy (what actually ships)
    stats = gate_runtime(stage, args.games, py)

    # 4. tar it — main.py must be at top level, so add members by basename
    out = args.out or os.path.join(WORK, "out", f"{args.agent}.tar.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    with tarfile.open(out, "w:gz") as tf:
        for fn in sorted(os.listdir(stage)):
            if fn == "__pycache__":
                continue
            tf.add(os.path.join(stage, fn), arcname=fn,
                   filter=lambda ti: None if "__pycache__" in ti.name else ti)
    print(f"\n  wrote {out} ({os.path.getsize(out)/1e6:.2f} MB)")

    gate_tar(out)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"GATE FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 60)
        return 1
    print(f"GATE PASSED: {len(CHECKS)} checks. Submittable:")
    print(f"  {out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
