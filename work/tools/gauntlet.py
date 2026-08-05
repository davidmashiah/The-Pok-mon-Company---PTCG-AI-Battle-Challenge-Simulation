"""Parallel, accumulating gauntlet between agent bundles.

Why this shape:
  * PARALLEL — the search agent costs ~17 s/episode; serial measurement is too
    slow to iterate against.
  * ACCUMULATING — results are appended to a JSON store keyed by
    (agent_a, agent_b, content-hash of each bundle). Re-running adds games to
    the same cell instead of starting over, so the CI tightens over the day.
    Any edit to an agent changes its hash and starts a fresh cell, so we can
    never silently pool results across two different versions of the code.
  * SIDE-SWAPPED — every pair alternates who moves first.

Usage:
  python work/tools/gauntlet.py --agents v2_lucario,v3_lucario_search --games 120
  python work/tools/gauntlet.py --report
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
STORE = os.path.join(WORK, "out", "gauntlet.json")


# ------------------------------------------------------------------ helpers
def bundle_hash(agent):
    """Content hash of everything that affects play: agent files + shared libs."""
    h = hashlib.sha256()
    parts = []
    d = os.path.join(WORK, "agents", agent)
    for fn in sorted(os.listdir(d)):
        if fn == "__pycache__":
            continue
        parts.append(os.path.join(d, fn))
    # The learned model lives outside the agent dir. If it is not hashed, a
    # retrained net silently pools its games with the OLD net's results in the
    # same cell -- the one thing this store exists to prevent. Only agents that
    # actually import it are affected, so hashing it for the others would
    # needlessly invalidate their accumulated games.
    # meta_decks.py MUST be hashed: it is the opponent-deck library fsearch uses
    # to determinize hidden zones, so rebuilding it changes how every
    # search-using agent plays. Leaving it out would pool results from before and
    # after the rebuild in the same cell -- the one thing this store exists to
    # prevent.
    shared_files = ["policy.py", "fsearch.py", "meta_decks.py"]
    try:
        with open(os.path.join(d, "main.py"), encoding="utf-8") as _fh:
            if "import dznp" in _fh.read():
                shared_files += ["dznp.py", "dzfeat.py", "dz_weights.npz"]
    except Exception:
        pass
    for shared in shared_files:
        p = os.path.join(WORK, "lib", shared)
        if os.path.exists(p):
            parts.append(p)
    for p in parts:
        if os.path.isfile(p):
            h.update(os.path.basename(p).encode())
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:12]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - hw), min(1.0, c + hw)


def load_store():
    if os.path.exists(STORE):
        try:
            with open(STORE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_store(s):
    """Atomic-ish save. os.replace can transiently fail on Windows (AV scan or
    an open handle from a killed worker); retry rather than lose the run."""
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    tmp = f"{STORE}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=1)
    for attempt in range(6):
        try:
            os.replace(tmp, STORE)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))
    try:                       # last resort: non-atomic but do not lose data
        with open(STORE, "w") as f:
            json.dump(s, f, indent=1)
        os.remove(tmp)
    except Exception:
        pass


# ------------------------------------------------------------------ worker
def _worker(job):
    """Play `n` games between two bundles. Runs in its own process."""
    a_dir, b_dir, n, seed0 = job
    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    def load(d):
        full = os.path.join(WORK, "agents", d)
        if full not in sys.path:
            sys.path.insert(0, full)
        cwd = os.getcwd()
        try:
            os.chdir(full)
            with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
                src = fh.read()
            env = {}
            exec(compile(src, "main.py", "exec"), env)
        finally:
            os.chdir(cwd)
        fns = [v for v in env.values() if callable(v)]
        # Do NOT assume the decklist is a module global. Some public agents
        # (w2_archaludon) read deck.csv lazily inside agent() and expose no
        # global at all, which made this raise "'NoneType' object is not
        # iterable" and kill the whole run. deck.csv is the contract; fall back
        # to it -- the same rule build_and_gate.py already uses.
        d = env.get("DECK") or env.get("my_deck") or env.get("MY_DECK")
        if not d:
            with open(os.path.join(full, "deck.csv"), encoding="utf-8") as fh:
                d = [int(x) for x in fh.read().split() if x.strip()]
        return fns[-1], list(d)

    fa, da = load(a_dir)
    fb, db = load(b_dir)

    wa = wb = draw = err = 0
    atime = 0.0
    for g in range(n):
        a_first = ((seed0 + g) % 2 == 0)
        p0, p1 = (fa, fb) if a_first else (fb, fa)
        d0, d1 = (da, db) if a_first else (db, da)
        a_idx = 0 if a_first else 1
        obs, sd = battle_start(list(d0), list(d1))
        if obs is None:
            err += 1
            continue
        res = None
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                if o.current is not None and o.current.result != -1:
                    res = o.current.result
                    break
                who = o.current.yourIndex if o.current is not None else 0
                t0 = time.time()
                sel = (p0 if who == 0 else p1)(obs)
                if (who == 0) == a_first:
                    atime += time.time() - t0
                obs = battle_select(list(sel))
        except Exception:
            res = None
        finally:
            battle_finish()
        if res is None:
            err += 1
        elif res == 2:
            draw += 1
        elif res == a_idx:
            wa += 1
        else:
            wb += 1
    return {"wa": wa, "wb": wb, "draw": draw, "err": err, "a_time": atime}


# ------------------------------------------------------------------ report
def report(store, only=None):
    print(f"\n{'matchup':<46} {'games':>6} {'A wins':>7} {'winrate':>9} "
          f"{'Wilson 95%':>18}  verdict")
    print("-" * 104)
    for key in sorted(store):
        c = store[key]
        if only and not any(o in key for o in only):
            continue
        n = c["wa"] + c["wb"]
        if n == 0:
            continue
        p, lo, hi = wilson(c["wa"], n)
        if lo > 0.5:
            v = "A BETTER"
        elif hi < 0.5:
            v = "B better"
        else:
            v = "inconclusive"
        label = key.replace("|", "  vs  ")
        print(f"{label:<46} {n:>6} {c['wa']:>7} {p:>9.4f} "
              f"[{lo:.4f},{hi:.4f}]  {v}"
              + (f"  ERR={c['err']}" if c.get("err") else ""))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    store = load_store()
    if args.report or not args.agents:
        report(store)
        return 0

    names = [a.strip() for a in args.agents.split(",") if a.strip()]
    hashes = {n: bundle_hash(n) for n in names}
    print("bundle hashes:", hashes)

    pairs = [(names[i], names[j])
             for i in range(len(names)) for j in range(i + 1, len(names))]

    for a, b in pairs:
        key = f"{a}@{hashes[a]}|{b}@{hashes[b]}"
        cell = store.setdefault(key, {"wa": 0, "wb": 0, "draw": 0,
                                      "err": 0, "a_time": 0.0})
        per = max(1, args.games // args.workers)
        jobs = [(a, b, per, w * 7919) for w in range(args.workers)]
        print(f"\n== {a} vs {b} : {per*args.workers} games on "
              f"{args.workers} workers (existing: {cell['wa']+cell['wb']}) ==")
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_worker, j) for j in jobs]
            for f in as_completed(futs):
                r = f.result()
                for k in ("wa", "wb", "draw", "err", "a_time"):
                    cell[k] += r[k]
                done += 1
                n = cell["wa"] + cell["wb"]
                p, lo, hi = wilson(cell["wa"], n)
                print(f"   worker {done}/{len(jobs)} -> cumulative "
                      f"{cell['wa']}-{cell['wb']}  {p:.3f} [{lo:.3f},{hi:.3f}]")
                save_store(store)
        dt = time.time() - t0
        n = cell["wa"] + cell["wb"]
        print(f"   {dt:.0f}s  ({(per*args.workers)/max(dt,1e-9):.2f} games/s), "
              f"A agent-time/episode ~{cell['a_time']/max(n,1):.1f}s")

    save_store(store)
    report(store, only=names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
