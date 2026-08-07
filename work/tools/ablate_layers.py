"""Turn off each override layer of the live agent and see which ones were costing us.

w34_koroll -- the bundle actually on the ladder at 917 -- does not compute one
action. It computes an action and then passes it through a chain of guards, each
of which may replace it:

    GBM model -> strategic fallback -> matchup_router -> human -> advisor
              -> residual -> tactical -> development -> search_validator

Every one of those layers was written and tuned by its author against THEIR
configuration of the rest of the stack. We inherited the whole chain and have
never once asked whether any individual link earns its place in ours. A layer
that fires often and is wrong more than it is right is a pure, free loss, and
removing it costs nothing but a diff.

This measures that directly. For each layer: rebuild the bundle with that link's
verdict ignored (the layer is still CALLED, so its internal state stays warm and
we are measuring the override, not the bookkeeping), then play the matchup.

Why the mirror is the default opponent: it is 0.30 of the top-50 field, the
single largest share, and it is the only column where we sit near 0.55 while
every other column is 0.66 to 0.84. The room is there and nowhere else.

Discipline, unchanged from the rest of this repo:
  * gauntlet.py content-hashes the bundle, so an ablated bundle can never pool
    its results with the intact one
  * this is a SCREEN. Anything that looks good here gets re-measured on the full
    weighted panel with field_test.py before it is believed. A layer that helps
    the mirror and quietly wrecks Alakazam is a loss, and only the panel sees it

  python work/tools/ablate_layers.py --base w34_koroll --games 400
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
AGENTS = os.path.join(WORK, "agents")
OUT = os.path.join(WORK, "out")

# name -> (exact source line, replacement). Each disables ONE link's verdict
# while leaving its call site intact.
LAYERS = {
    "model": (
        "    chosen = fallback if action is None else action",
        "    chosen = fallback  # ABLATED: ignore the GBM ensemble",
    ),
    "route": (
        "        \"baseline_route\": baseline_route,",
        "        \"baseline_route\": chosen,  # ABLATED: ignore matchup_router",
    ),
    "human": (
        "    if routed is not None and _legal(routed, select, option_count):",
        "    if False and routed is not None and _legal(routed, select, option_count):",
    ),
    "advisor": (
        "    if advised_action is not None and _legal(advised_action, select, option_count):",
        "    if False and advised_action is not None and _legal(advised_action, select, option_count):",
    ),
    "residual": (
        "    if residual_action is not None and _legal(residual_action, select, option_count):",
        "    if False and residual_action is not None and _legal(residual_action, select, option_count):",
    ),
    "tactical": (
        "    if tactical_action is not None and _legal(tactical_action, select, option_count):",
        "    if False and tactical_action is not None and _legal(tactical_action, select, option_count):",
    ),
    "development": (
        "    if development_action is not None and _legal(development_action, select, option_count):",
        "    if False and development_action is not None and _legal(development_action, select, option_count):",
    ),
    "validator": (
        "    if validated is not None and _legal(validated, select, option_count):",
        "    if False and validated is not None and _legal(validated, select, option_count):",
    ),
}


def build(base, name, off):
    """Copy `base` to `name` with every layer in `off` neutralised."""
    src_path = os.path.join(AGENTS, base, "main.py")
    src = open(src_path, encoding="utf-8-sig").read()
    for layer in off:
        old, new = LAYERS[layer]
        if src.count(old) != 1:
            raise SystemExit(
                f"layer {layer!r}: anchor line found {src.count(old)} times in "
                f"{base}/main.py -- refusing to guess which one")
        src = src.replace(old, new)
    compile(src, "main.py", "exec")
    dst = os.path.join(AGENTS, name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(AGENTS, base), dst,
                    ignore=shutil.ignore_patterns("__pycache__"))
    with open(os.path.join(dst, "main.py"), "w", encoding="utf-8") as f:
        f.write(src)
    return dst


def evaluate(name, opp, games, workers):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    subprocess.run([sys.executable, "-u", os.path.join(HERE, "gauntlet.py"),
                    "--agents", f"{name},{opp}", "--games", str(games),
                    "--workers", str(workers)], cwd=ROOT, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sys.path.insert(0, HERE)
    import gauntlet
    store = json.load(open(os.path.join(OUT, "gauntlet.json")))
    ka = f"{name}@{gauntlet.bundle_hash(name)}|{opp}@{gauntlet.bundle_hash(opp)}"
    kb = f"{opp}@{gauntlet.bundle_hash(opp)}|{name}@{gauntlet.bundle_hash(name)}"
    for k, c in store.items():
        n = c["wa"] + c["wb"]
        if k == ka:
            return (c["wa"], n) if n else (0, 0)
        if k == kb:
            return (c["wb"], n) if n else (0, 0)
    return (0, 0)


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - hw), min(1.0, c + hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="w34_koroll")
    ap.add_argument("--opponent", default="w5_grimmsnarl")
    ap.add_argument("-n", "--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--slot", default="_abl")
    ap.add_argument("--only", default="", help="comma-separated subset")
    a = ap.parse_args()

    layers = ([x for x in a.only.split(",") if x.strip()]
              if a.only else list(LAYERS))
    for layer in layers:
        if layer not in LAYERS:
            raise SystemExit(f"unknown layer {layer!r}")

    w, n = evaluate(a.base, a.opponent, a.games, a.workers)
    p0, lo0, hi0 = wilson(w, n)
    print(f"intact {a.base} vs {a.opponent}: {w}/{n} = {p0:.4f} "
          f"[{lo0:.4f}, {hi0:.4f}]\n")

    rows = []
    for layer in layers:
        name = f"{a.slot}_{layer}"
        try:
            build(a.base, name, [layer])
        except SystemExit as exc:
            print(f"  {layer:12s} SKIPPED: {exc}")
            continue
        w, n = evaluate(name, a.opponent, a.games, a.workers)
        if not n:
            print(f"  {layer:12s} no games")
            continue
        p, lo, hi = wilson(w, n)
        # sd of the difference of two independent proportions
        sd = math.sqrt(p * (1 - p) / n + p0 * (1 - p0) / max(1, n))
        z = (p - p0) / sd if sd else 0.0
        rows.append((p - p0, z, layer, p, n))
        print(f"  off:{layer:12s} {w:4d}/{n:4d} = {p:.4f} [{lo:.4f}, {hi:.4f}]"
              f"   delta {p - p0:+.4f}  ({z:+.1f} sd)", flush=True)

    rows.sort(reverse=True)
    print("\nranked by gain from REMOVING the layer:")
    for d, z, layer, p, n in rows:
        verdict = ("layer is COSTING us" if z >= 2 else
                   "layer is earning its place" if z <= -2 else "noise")
        print(f"  {layer:12s} {d:+.4f}  ({z:+.1f} sd)  {verdict}")
    print("\nThis is a SCREEN on one column. Re-measure any winner on the full "
          "panel\nwith field_test.py -- a layer can help the mirror and lose "
          "more elsewhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
