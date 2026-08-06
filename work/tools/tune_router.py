"""Search which expert handles which archetype -- the layer that actually decides.

Instrumenting the live agent over 933 real decisions against Grimmsnarl:

    matchup_router overrode   351  (38% of all decisions)
    residual / tactical / development guards   1 each
    advisor                     0
    coalition_expert            0     <- NEVER CONSULTED

The coalition is gated behind `profile == "grass_fast" and confidence >= 0.45`
and never fires in the mirror, so every hour spent tuning coalition_weights.json
was tuning a dead knob -- which is exactly why its "gains" kept collapsing when
measured independently (0.6920 -> 0.6114). The router is where the decisions are.

And the shipped router barely routes:

    profile 'mirror' -> mirror expert
    profile 'grass'  -> tempo expert
    wall, arch, psychic, unknown -> baseline, all of them

That fall-through IS the non-Grimmsnarl 70%, the slice that has to go from 0.72
to 0.92 for 1000. Three experts across six profiles is 729 assignments, small
enough to search properly and discrete enough that a change is either real or
obviously not.

Objective and discipline are unchanged from the rest of the repo: the weighted
TOP-BAND field win rate, screened then CONFIRMED on disjoint seeds, with the bar
set at the live agent's own measured rate so nothing can be adopted that does
not beat what is on the ladder.

  python work/tools/tune_router.py --rounds 120 --screen 66 --confirm 180
"""
import argparse
import itertools
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
OUT = os.path.join(WORK, "out")
STATE = os.path.join(OUT, "tune_router.json")
BEST = os.path.join(OUT, "route_table_best.json")

BASE = "w11_router"
RFILE = "route_table.json"
EXPERTS = ("baseline", "mirror", "tempo")
PROFILES = ("mirror", "grass", "wall", "arch", "psychic", "unknown")

FIELD = [
    ("w5_grimmsnarl", 0.30),
    ("w1_alakazam",   0.16),
    ("p3_crustle",    0.08),
    ("s_dragapult",   0.06),
    ("z_roman950",    0.02),
    ("w2_archaludon", 0.02),
]
ANCHOR_FIELD = 0.6376        # w8's measured field rate; live 829.5 -> 849.6


def worker_dir(i):
    d = os.path.join(AGENTS, f"_route_w{i}")
    if not os.path.isdir(d):
        shutil.copytree(os.path.join(AGENTS, BASE), d,
                        ignore=shutil.ignore_patterns("__pycache__"))
    return d


def _play(job):
    widx, table, n, seed0, opp = job
    d = worker_dir(widx)
    with open(os.path.join(d, RFILE), "w", encoding="utf-8") as fh:
        json.dump(table, fh)

    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    def load(full):
        # Put THIS bundle first and evict every module already imported from an
        # agent directory. main.py does `from matchup_router import choose`, and
        # once any copy of that module is in sys.modules the import is a no-op --
        # so a candidate would silently run the previous bundle's routing table
        # and the whole search would measure the base over and over.
        while full in sys.path:
            sys.path.remove(full)
        sys.path.insert(0, full)
        agents_root = os.path.join(WORK, "agents")
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None) or ""
            if f.startswith(agents_root):
                del sys.modules[name]
        cwd = os.getcwd()
        try:
            os.chdir(full)
            with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
                src = fh.read()
            env = {}
            exec(compile(src, "main.py", "exec"), env)
            fn = [v for v in env.values() if callable(v)][-1]
            got = None
            try:
                r = fn({"current": None, "select": None})
                if isinstance(r, (list, tuple)) and len(r) == 60:
                    got = [int(x) for x in r]
            except Exception:
                pass
        finally:
            os.chdir(cwd)
        if got is None:
            with open(os.path.join(full, "deck.csv"), encoding="utf-8") as fh:
                got = [int(x) for x in fh.read().split() if x.strip()]
        return fn, got

    fa, da = load(d)
    # The table is read at IMPORT of matchup_router, so a stale worker directory
    # would silently evaluate the shipped routing under the candidate's name.
    mr = sys.modules.get("matchup_router")
    if mr is None or any(getattr(mr, "_ROUTE", {}).get(k) != v
                         for k, v in table.items()):
        return {"err": 1, "w": 0, "l": 0}
    route_mod = sys.modules.get("matchup_router")
    fb, db = load(os.path.join(AGENTS, opp))
    # restore the candidate's module view: loading the opponent purged it
    if route_mod is not None:
        sys.modules["matchup_router"] = route_mod

    w = l = 0
    for g in range(n):
        a_first = ((seed0 + g) % 2 == 0)
        p0, p1 = (fa, fb) if a_first else (fb, fa)
        d0, d1 = (da, db) if a_first else (db, da)
        a_idx = 0 if a_first else 1
        for f in (p0, p1):
            try:
                f({"current": None, "select": None})
            except Exception:
                pass
        obs, _sd = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        res = None
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                if o.current is not None and o.current.result != -1:
                    res = o.current.result
                    break
                who = o.current.yourIndex if o.current is not None else 0
                obs = battle_select(list((p0 if who == 0 else p1)(obs)))
        except Exception:
            res = None
        finally:
            battle_finish()
        if res == a_idx:
            w += 1
        elif res is not None and res != 2:
            l += 1
    return {"err": 0, "w": w, "l": l}


def evaluate(table, games, workers, seed0):
    live = [(o, sh) for o, sh in FIELD
            if os.path.isdir(os.path.join(AGENTS, o))]
    tot = sum(sh for _, sh in live)
    jobs, plan = [], []
    for wi, (opp, share) in enumerate(live):
        k = max(2, int(round(games * share / tot)))
        jobs.append((wi % max(1, workers), table, k, seed0 + wi * 7919, opp))
        plan.append((opp, share))
    num = den = 0.0
    n_tot = bad = 0
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for (opp, share), r in zip(plan, ex.map(_play, jobs)):
            bad += r["err"]
            n_i = r["w"] + r["l"]
            n_tot += n_i
            if n_i:
                num += share * (r["w"] / n_i)
                den += share
    if bad:
        raise SystemExit("routing table did not bind -- fix before searching")
    return (num / den if den else -1.0), n_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--screen", type=int, default=66)
    ap.add_argument("--confirm", type=int, default=180)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    base = {"mirror": "mirror", "grass": "tempo", "wall": "baseline",
            "arch": "baseline", "psychic": "baseline", "unknown": "baseline"}
    if args.report:
        st = json.load(open(STATE))
        print(json.dumps({k: v for k, v in st.items() if k != "history"}, indent=1))
        return 0

    st = {"round": 0, "accepted": 0, "screened": 0, "rejected_confirm": 0,
          "best": dict(base), "best_field": ANCHOR_FIELD, "history": []}
    if os.path.exists(STATE):
        try:
            st = json.load(open(STATE))
            print(f"resuming at round {st['round']}, {st['accepted']} accepted")
        except Exception:
            pass
    print(f"shipped routing: {base}")
    print(f"objective: top-band field; accept above {st['best_field']:.4f}")

    rng = random.Random(args.seed + st["round"])
    tried = {json.dumps(st["best"], sort_keys=True)}
    t0 = time.time()
    for _ in range(args.rounds):
        st["round"] += 1
        cand = dict(st["best"])
        for k in rng.sample(PROFILES, rng.choice([1, 1, 2])):
            cand[k] = rng.choice(EXPERTS)
        key = json.dumps(cand, sort_keys=True)
        if key in tried:
            continue
        tried.add(key)
        chg = {k: v for k, v in cand.items() if base.get(k) != v}
        bar = st["best_field"]
        scr, n1 = evaluate(cand, args.screen, args.workers, 4000 + st["round"] * 13)
        st["screened"] += 1
        if scr <= bar:
            print(f"r{st['round']:3d} screen {scr:.3f} ({n1}) reject | {chg}")
            json.dump(st, open(STATE, "w"))
            continue
        conf, n2 = evaluate(cand, args.confirm, args.workers,
                            900000 + st["round"] * 977)
        pooled = (scr * n1 + conf * n2) / (n1 + n2)
        ok = conf > bar and pooled > bar + 0.02
        print(f"r{st['round']:3d} screen {scr:.3f} confirm {conf:.3f} "
              f"pooled {pooled:.3f} {'ACCEPT' if ok else 'reject'} | {chg}")
        if ok:
            st["best"] = cand
            st["best_field"] = pooled
            st["accepted"] += 1
            st["history"].append({"round": st["round"], "pooled": pooled,
                                  "table": cand})
            json.dump(cand, open(BEST, "w"), indent=1)
        else:
            st["rejected_confirm"] += 1
        json.dump(st, open(STATE, "w"))
    print(f"\n{st['accepted']} accepted / {st['screened']} screened, "
          f"{(time.time()-t0)/60:.0f} min")
    print("best routing:", st["best"], f"field {st['best_field']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
