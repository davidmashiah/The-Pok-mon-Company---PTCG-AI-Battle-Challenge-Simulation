"""Hill-climb the hand-written policy's card-preference orderings.

Five hand-reasoned fixes to this policy were measured today and all five made it
worse (0.1208 / 0.4770 / 0.4892 / 0.5083 / and the search bolt-on) against a
control of 0.5523. Reasoning my way to a rule is 0 for 5, so this converts the
problem into one compute can attack instead.

The policy's real knobs are discrete: dozens of `ids=[...]` preference lists that
decide which card to fetch, promote or discard in a given situation. A genome
here is a set of permutations of those lists. Mutate one, rebuild the bundle,
measure, keep it only if it beats the incumbent by more than noise.

Discipline, all of it paid for by this repo already:
  * every candidate is measured through gauntlet.py, which content-hashes the
    bundle -- a mutated policy is a different hash, so results can never pool
    with its parent's
  * a searcher CANNOT certify its own result. Anything this accepts must be
    re-measured with field_test.py before it is believed, let alone submitted
  * acceptance needs the lower confidence bound of the candidate to clear the
    incumbent's point estimate, not just a better mean at n=200

  python work/tools/tune_prefs.py --rounds 40 --games 200
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
AGENTS = os.path.join(WORK, "agents")
OUT = os.path.join(WORK, "out")
OPPONENT = "w5_grimmsnarl"

# Which basin to climb. The bundle ships six complete policies and they are NOT
# equivalent -- measured vs the mirror under the clean harness they span 0.39 to
# 0.56, and on the full panel _sub_v28 reaches 0.6687 against
# _sub_handwritten_v26's 0.6495. Climbing one starting point can only find that
# basin's peak, so the base is a parameter.
BASINS = {
    "_sub_v28": os.path.join("policies", "v28", "v26_manual_policy.py"),
    "_sub_handwritten_v26": os.path.join("policies", "handwritten_v26",
                                         "manual_policy.py"),
}
BASE = "_sub_v28"
POLICY = BASINS[BASE]
STATE = os.path.join(OUT, "tune_prefs.json")

IDS_RE = re.compile(r"^(\s*)ids\s*(\+?=)\s*\[([A-Z_0-9,\s]+)\]\s*$", re.M)


def find_lists(text):
    """Every `ids=[A,B,C]` line: (start, end, indent, op, [names])."""
    out = []
    for m in IDS_RE.finditer(text):
        names = [t.strip() for t in m.group(3).split(",") if t.strip()]
        if len(names) >= 2:
            out.append((m.start(), m.end(), m.group(1), m.group(2), names))
    return out


def apply_genome(text, genome):
    """genome: {list_index: permutation}. Rebuild the source with it."""
    lists = find_lists(text)
    for i in sorted(genome, reverse=True):
        if i >= len(lists):
            continue
        s, e, indent, op, names = lists[i]
        perm = [names[j] for j in genome[i]]
        text = text[:s] + f"{indent}ids {op} [{', '.join(perm)}]" + text[e:]
    return text


def build(name, genome, base_src):
    dst = os.path.join(AGENTS, name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(AGENTS, BASE), dst,
                    ignore=shutil.ignore_patterns("__pycache__"))
    src = apply_genome(base_src, genome)
    compile(src, POLICY, "exec")
    with open(os.path.join(dst, POLICY), "w", encoding="utf-8") as f:
        f.write(src)
    return dst


def evaluate(name, games, workers):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    subprocess.run([sys.executable, "-u", os.path.join(HERE, "gauntlet.py"),
                    "--agents", f"{name},{OPPONENT}", "--games", str(games),
                    "--workers", str(workers)], cwd=ROOT, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sys.path.insert(0, HERE)
    import gauntlet
    store = json.load(open(os.path.join(OUT, "gauntlet.json")))
    key_a = f"{name}@{gauntlet.bundle_hash(name)}|{OPPONENT}@{gauntlet.bundle_hash(OPPONENT)}"
    key_b = f"{OPPONENT}@{gauntlet.bundle_hash(OPPONENT)}|{name}@{gauntlet.bundle_hash(name)}"
    for k, c in store.items():
        if k == key_a:
            n = c["wa"] + c["wb"]
            return (c["wa"], n) if n else (0, 0)
        if k == key_b:
            n = c["wa"] + c["wb"]
            return (c["wb"], n) if n else (0, 0)
    return (0, 0)


def wilson_lo(k, n, z=1.645):
    if not n:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--slot", default="_pref_cand")
    ap.add_argument("--base", default="_sub_v28", choices=sorted(BASINS))
    global BASE, POLICY, STATE
    a = ap.parse_args()
    BASE = a.base
    POLICY = BASINS[BASE]
    STATE = os.path.join(OUT, f"tune_prefs_{BASE}.json")

    base_src = open(os.path.join(AGENTS, BASE, POLICY), encoding="utf-8").read()
    lists = find_lists(base_src)
    print(f"{len(lists)} mutable preference lists in {BASE}")
    if not lists:
        raise SystemExit("no ids=[...] lists found -- the regex needs updating")

    state = {"best": {}, "best_rate": None, "history": []}
    if os.path.exists(STATE):
        try:
            state = json.load(open(STATE))
            state["best"] = {int(k): v for k, v in state["best"].items()}
        except Exception:
            pass

    if state["best_rate"] is None:
        w, n = evaluate(BASE, a.games, a.workers)
        state["best_rate"] = w / max(1, n)
        print(f"incumbent {BASE}: {w}/{n} = {state['best_rate']:.4f}")

    rng = random.Random(20260807)
    for r in range(a.rounds):
        genome = dict(state["best"])
        i = rng.randrange(len(lists))
        names = lists[i][4]
        perm = genome.get(i, list(range(len(names))))
        perm = list(perm)
        if len(perm) >= 2:                       # swap two positions
            x, y = rng.sample(range(len(perm)), 2)
            perm[x], perm[y] = perm[y], perm[x]
        genome[i] = perm

        name = a.slot
        try:
            build(name, genome, base_src)
        except Exception as exc:
            print(f"  round {r}: build failed ({type(exc).__name__})")
            continue
        w, n = evaluate(name, a.games, a.workers)
        if not n:
            print(f"  round {r}: no games")
            continue
        rate, lo = w / n, wilson_lo(w, n)
        keep = lo > state["best_rate"]
        tag = "ACCEPT" if keep else "reject"
        print(f"  round {r:3d} list {i:3d} {[names[j] for j in perm]} "
              f"-> {rate:.4f} (lo {lo:.4f}) vs {state['best_rate']:.4f}  {tag}",
              flush=True)
        state["history"].append({"round": r, "list": i, "perm": perm,
                                 "rate": rate, "n": n, "kept": keep})
        if keep:
            state["best"] = genome
            state["best_rate"] = rate
            shutil.rmtree(os.path.join(AGENTS, "_pref_best"), ignore_errors=True)
            shutil.copytree(os.path.join(AGENTS, name),
                            os.path.join(AGENTS, "_pref_best"),
                            ignore=shutil.ignore_patterns("__pycache__"))
        json.dump({"best": {str(k): v for k, v in state["best"].items()},
                   "best_rate": state["best_rate"],
                   "history": state["history"]},
                  open(STATE, "w"), indent=1)

    print(f"\nbest {state['best_rate']:.4f} with {len(state['best'])} mutated "
          f"lists -> work/agents/_pref_best")
    print("A searcher cannot certify itself. Re-measure with field_test.py "
          "before believing this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
