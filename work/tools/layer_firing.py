"""How often does each override layer actually CHANGE the live agent's answer?

Before spending hundreds of games ablating eight layers, ask the free question:
which of them ever do anything? w34_koroll runs a chain of guards in series and
each may replace the action chosen so far --

    GBM model -> strategic -> matchup_router -> human -> advisor
              -> residual -> tactical -> development -> search_validator

-- but a layer that agrees with its input every time is a no-op, and ablating it
can only ever measure noise. Game budget spent there is wasted, and worse, eight
underpowered ablations produce one spurious 2-sigma winner by construction.

So instrument instead of guessing. This rewrites main.py so every guard reports
two counts before it hands the action on:

    eligible   the guard returned a legal action and was allowed to override
    changed    that action actually DIFFERED from what it was handed

`changed` is the only one that matters. It is the layer's true firing rate, and
it bounds how much that layer can possibly be worth: a guard that changes 1% of
decisions cannot move a win rate by 0.05, so no ablation of it is worth running.

Prints per-layer counts plus the running order, so the chain can be read as what
it is -- a sequence of vetoes where the LAST one to fire wins.

  python work/tools/layer_firing.py --agent w34_koroll --games 30
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
sys.path.insert(0, os.path.join(WORK, "lib"))

# Each guard is "if <cond>:\n        chosen = <value>". Record before assigning.
GUARDS = [
    ("human", "    if routed is not None and _legal(routed, select, option_count):\n"
               "        chosen = routed",
              "    if routed is not None and _legal(routed, select, option_count):\n"
               "        _fire('human', chosen, routed)\n"
               "        chosen = routed"),
    ("advisor", "    if advised_action is not None and _legal(advised_action, select, option_count):\n"
                "        chosen = advised_action",
                "    if advised_action is not None and _legal(advised_action, select, option_count):\n"
                "        _fire('advisor', chosen, advised_action)\n"
                "        chosen = advised_action"),
    ("residual", "    if residual_action is not None and _legal(residual_action, select, option_count):\n"
                 "        chosen = residual_action",
                 "    if residual_action is not None and _legal(residual_action, select, option_count):\n"
                 "        _fire('residual', chosen, residual_action)\n"
                 "        chosen = residual_action"),
    ("tactical", "    if tactical_action is not None and _legal(tactical_action, select, option_count):\n"
                 "        chosen = tactical_action",
                 "    if tactical_action is not None and _legal(tactical_action, select, option_count):\n"
                 "        _fire('tactical', chosen, tactical_action)\n"
                 "        chosen = tactical_action"),
    ("development", "    if development_action is not None and _legal(development_action, select, option_count):\n"
                    "        chosen = development_action",
                    "    if development_action is not None and _legal(development_action, select, option_count):\n"
                    "        _fire('development', chosen, development_action)\n"
                    "        chosen = development_action"),
    ("validator", "    if validated is not None and _legal(validated, select, option_count):\n"
                  "        chosen = validated",
                  "    if validated is not None and _legal(validated, select, option_count):\n"
                  "        _fire('validator', chosen, validated)\n"
                  "        chosen = validated"),
]

# The model/route decisions are not guards, so they are counted separately.
EXTRA = [
    ("model", "    chosen = fallback if action is None else action",
              "    _fire('model', fallback, fallback if action is None else action)\n"
              "    chosen = fallback if action is None else action"),
    ("route", "    if baseline_route is None or not _legal(baseline_route, select, option_count):\n"
              "        baseline_route = chosen",
              "    if baseline_route is None or not _legal(baseline_route, select, option_count):\n"
              "        baseline_route = chosen\n"
              "    _fire('route', chosen, baseline_route)"),
]

PREAMBLE = '''
_FIRE = {}


def _fire(name, old, new):
    d = _FIRE.setdefault(name, [0, 0])
    d[0] += 1
    if list(old or []) != list(new or []):
        d[1] += 1

'''


def instrument(base, name):
    src = open(os.path.join(AGENTS, base, "main.py"),
               encoding="utf-8-sig").read()
    applied = []
    for label, old, new in GUARDS + EXTRA:
        if src.count(old) != 1:
            print(f"  ! anchor for {label!r} found {src.count(old)}x -- skipped")
            continue
        src = src.replace(old, new)
        applied.append(label)
    marker = "def agent(obs):"
    if src.count(marker) != 1:
        raise SystemExit("cannot locate agent() to insert the recorder")
    src = src.replace(marker, PREAMBLE + marker)
    compile(src, "main.py", "exec")
    dst = os.path.join(AGENTS, name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(AGENTS, base),
                    dst, ignore=shutil.ignore_patterns("__pycache__"))
    with open(os.path.join(dst, "main.py"), "w", encoding="utf-8") as f:
        f.write(src)
    return dst, applied


def load(name):
    full = os.path.join(AGENTS, name)
    if full not in sys.path:
        sys.path.insert(0, full)
    cwd = os.getcwd()
    env = {}
    try:
        os.chdir(full)
        exec(compile(open(os.path.join(full, "main.py"),
                          encoding="utf-8-sig").read(), "main.py", "exec"), env)
        fn = env.get("agent") or [v for v in env.values() if callable(v)][-1]
        try:
            d = fn({"current": None, "select": None, "logs": []})
        except Exception:
            d = None
    finally:
        os.chdir(cwd)
        for nm, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None) or ""
            if f.startswith(full + os.sep) or f.startswith(full + "/"):
                del sys.modules[nm]
        while full in sys.path:
            sys.path.remove(full)
    if not (isinstance(d, (list, tuple)) and len(d) == 60):
        d = [int(x) for x in open(os.path.join(full, "deck.csv"),
                                  encoding="utf-8").read().split() if x.strip()]
    return fn, [int(x) for x in d], env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="w34_koroll")
    ap.add_argument("--opponent", default="w5_grimmsnarl")
    ap.add_argument("-n", "--games", type=int, default=30)
    ap.add_argument("--slot", default="_fireprobe")
    a = ap.parse_args()

    _, applied = instrument(a.agent, a.slot)
    print(f"instrumented {len(applied)} layers: {', '.join(applied)}\n")

    from cg.api import to_observation_class  # noqa: F401
    from cg.game import battle_finish, battle_select, battle_start

    fa, da, env = load(a.slot)
    fb, db, _ = load(a.opponent)

    decisions = 0
    for g in range(a.games):
        a_first = (g % 2 == 0)
        p0, p1 = (fa, fb) if a_first else (fb, fa)
        d0, d1 = (da, db) if a_first else (db, da)
        for f in (fa, fb):
            try:
                f({"current": None, "select": None, "logs": []})
            except Exception:
                pass
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                st = o.current
                if st is None or st.result != -1:
                    break
                who = st.yourIndex
                if (who == 0) == a_first:
                    decisions += 1
                obs = battle_select(list((p0 if who == 0 else p1)(obs)))
        except Exception:
            pass
        finally:
            battle_finish()

    fire = env.get("_FIRE", {})
    print(f"{a.agent} vs {a.opponent}, {a.games} games, "
          f"{decisions} of OUR decisions\n")
    print(f"{'layer':14s} {'eligible':>9} {'CHANGED':>9} {'change rate':>12}")
    print("-" * 48)
    order = [lbl for lbl, _, _ in EXTRA] + [lbl for lbl, _, _ in GUARDS]
    for label in order:
        d = fire.get(label)
        if d is None:
            print(f"{label:14s} {'--':>9} {'--':>9} {'never called':>12}")
            continue
        elig, chg = d
        print(f"{label:14s} {elig:9d} {chg:9d} {chg / max(1, elig):12.3f}")
    print("\nCHANGED is the layer's real firing rate and it BOUNDS the layer's "
          "value:\na guard that alters 1% of decisions cannot move a win rate "
          "by 0.05, so\nablating it would only measure noise. Spend games on "
          "the busy layers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
