"""Do our agents agree with a STRONG pilot playing OUR deck? Seconds, not minutes.

Why this metric: both local proxies we had are dead. The mirror gauntlet ranked
v23_dz ABOVE v14 (0.5297 over 438 games) and meta_arena could not separate them
(0.7105 vs 0.7239) -- while the ladder put them 300 points apart (field win rate
0.407 vs 0.609). The common flaw is that OUR ~700 policy pilots the opponents in
both, and against weak opposition almost anything wins ~71%.

This metric has no such flaw: the opponent is a real competitor at ladder
strength, and the yardstick is their own choice in the position. It needs no
engine games at all -- just replay their observations through our agent.

Reads the pre-filtered cache from cache_deck_games.py so it does not re-scan
700 MB per question.

Usage: python work/tools/fast_agree.py <agent> [agent2 ...]
"""
import json
import os
import sys
import time
import zipfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
LIB = os.path.join(WORK, "lib")
CACHE = os.path.join(WORK, "out", "games_678.zip")

sys.path.insert(0, LIB)
from cg.api import SelectContext, to_observation_class  # noqa: E402


def load_agent(name):
    """exec main.py exactly as kaggle_environments does (no __file__)."""
    ag = os.path.join(WORK, "agents", name)
    for p in (ag, LIB):
        if p not in sys.path:
            sys.path.insert(0, p)
    # Run from the REPO ROOT, not the agent dir: dznp resolves its weights
    # relative to cwd, and chdir-ing into the agent dir made that lookup fail
    # silently -- v23_dz then ran as plain v14 and "agreed" 997/5391 vs v14's
    # 999/5391, which looked like a real (null) result. Same silent-fallback
    # class of bug this project has hit five times.
    root = os.path.dirname(WORK)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        with open(os.path.join(ag, "main.py"), encoding="utf-8") as fh:
            src = fh.read()
        env = {}
        exec(compile(src, "main.py", "exec"), env)
    finally:
        os.chdir(cwd)
    fns = [(k, v) for k, v in env.items() if callable(v)]

    # If the agent claims to use the learned model, it MUST have loaded.
    if "import dznp" in src:
        if not env.get("_DZ_OK"):
            raise SystemExit(
                f"{name}: imports dznp but the weights did not load -- it would "
                f"run as plain v14 and the measurement would be meaningless")
        print(f"  [{name}: learned model loaded and live]")
    return fns[-1][1], env


def main():
    names = sys.argv[1:] or ["v14_search_noloop2"]
    if not os.path.exists(CACHE):
        print(f"no cache at {CACHE}; run cache_deck_games.py first")
        return 2
    zf = zipfile.ZipFile(CACHE)
    files = [n for n in zf.namelist() if n.endswith(".json")]
    print(f"cache: {len(files)} games played with our deck by strong pilots\n")

    results = {}
    for name in names:
        fn, env = load_agent(name)
        t0 = time.time()
        agree = total = 0
        by_ctx = Counter()
        tot_ctx = Counter()
        for f in files:
            d = json.loads(zf.open(f).read().decode("utf-8"))
            rw = d.get("rewards") or []
            if 1 not in rw:
                continue
            w = rw.index(1)
            for st in d.get("steps", []):
                if w >= len(st):
                    continue
                ag = st[w]
                if ag.get("status") != "ACTIVE":
                    continue
                obs = ag.get("observation") or {}
                act = ag.get("action")
                if not act or not isinstance(act, list) or len(act) == 60:
                    continue
                sel = obs.get("select")
                if not sel or len(sel.get("option") or []) < 2:
                    continue
                try:
                    o = to_observation_class(obs)
                    ours = fn(obs)
                except Exception:
                    continue
                if not ours:
                    continue
                try:
                    ctx = int(o.select.context)
                except Exception:
                    ctx = -1
                total += 1
                tot_ctx[ctx] += 1
                if list(ours)[:len(act)] == list(act):
                    agree += 1
                    by_ctx[ctx] += 1
        st = env.get("_DZ_STATS") or {}
        if st:
            print(f"  [{name}: model fired {st.get('fired',0)}/{st.get('calls',0)}, "
                  f"changed {st.get('changed',0)} picks]")
        rate = agree / max(total, 1)
        results[name] = (rate, agree, total)
        print(f"{name:<24} agreement {rate:.4f}  ({agree}/{total})  "
              f"[{time.time()-t0:.0f}s]")

    if len(results) > 1:
        print("\nranking by agreement with the strong pilot:")
        for n, (r, a, t) in sorted(results.items(), key=lambda x: -x[1][0]):
            print(f"  {r:.4f}  {n}")
        print("\nLadder truth to check against: v14 field winrate 0.609 (734.2),")
        print("v23_dz 0.407 (389.3). If this metric puts them in that order with a")
        print("clear gap, it is the first local measure that tracks reality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
