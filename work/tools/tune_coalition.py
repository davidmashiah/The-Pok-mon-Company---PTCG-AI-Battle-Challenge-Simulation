"""Re-fit w7's ensemble arbitration for WIN RATE instead of imitation accuracy.

w7_grimm_safe is a coalition of six sub-policies. Which one wins a disagreement
is decided by five smoothing scalars in coalition_weights.json, and those were
fit offline to REPRODUCE a reference agent's choices -- 49 replays, 5,241
decisions, 88.6% accuracy, 88.2% cross-validated.

That is imitation, and this repo has already paid to learn that offline
correctness does not convert. The damage-model audit cut per-attack KO
prediction error from 11.8% to 4.2% with phantom knockouts to zero, and the
win rate did not move at all (0.4770 over 239 games). An arbitration rule tuned
to agree with a reference agent is optimising the wrong loss.

So re-fit the arbitration for the thing that pays: the weighted win rate across
the TOP-BAND field, the same objective and the same panel field_test.py uses,
anchored on our own measured submission (v61 = 726.1).

The search space is deliberately tiny -- five scalars and one integer threshold
-- because that is the whole arbitration surface, and a small space with an
honest confirmation stage is worth more than a large one without. Discipline is
unchanged: screen, then CONFIRM on disjoint seeds, because a screen-only
optimiser here once reported 13 improvements of which all 13 were false.

  python work/tools/tune_coalition.py --rounds 200 --screen 60 --confirm 180
  python work/tools/tune_coalition.py --report
"""
import argparse
import json
import math
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
STATE = os.path.join(OUT, "tune_coalition.json")
BEST = os.path.join(OUT, "coalition_best.json")

BASE = "w8_grimm_tuned"
WFILE = "coalition_weights.json"

# Top-band shares, from top_decks.py over the top 50 teams. Same table as
# field_test.py; renormalised over the archetypes we can actually pilot.
FIELD = [
    ("w5_grimmsnarl", 0.30),      # the mirror, and the largest slice of the field
    ("w1_alakazam",   0.16),
    ("p3_crustle",    0.08),
    ("s_dragapult",   0.06),
    ("z_roman950",    0.02),
    ("w2_archaludon", 0.02),
]
ANCHOR_FIELD = 0.6376            # w8's measured field rate; live score 829.5

# name -> (low, high). The support bonus is the majority-vote weight; the
# others are Dirichlet-style smoothing counts, so they are scale-free and
# searched multiplicatively.
BOUNDS = {
    "agent_global":     (0.5, 200.0),
    "coalition_global": (0.5, 200.0),
    "coalition_family": (0.05, 100.0),
    "tie_agent_family": (0.25, 100.0),
    "support_bonus":    (0.0, 2.0),
}


def load_weights():
    with open(os.path.join(AGENTS, BASE, WFILE), encoding="utf-8") as fh:
        return json.load(fh)


def worker_dir(i):
    d = os.path.join(AGENTS, f"_coal_w{i}")
    if not os.path.isdir(d):
        shutil.copytree(os.path.join(AGENTS, BASE), d,
                        ignore=shutil.ignore_patterns("__pycache__"))
    return d


def _play(job):
    widx, smoothing, n, seed0, opp = job
    d = worker_dir(widx)
    w = load_weights()
    w["smoothing"] = smoothing
    with open(os.path.join(d, WFILE), "w", encoding="utf-8") as fh:
        json.dump(w, fh)

    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    def load(full):
        if full not in sys.path:
            sys.path.insert(0, full)
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
    # Assert the genome bound. coalition_expert reads the file at IMPORT, so a
    # stale worker directory would evaluate the base's arbitration under the
    # candidate's name and the search would spend a night finding nothing.
    try:
        import importlib
        sys.path.insert(0, d)
        ce = importlib.import_module("coalition_expert")
        importlib.reload(ce)
        got_sm = ce._W.get("smoothing", {})
        if any(abs(float(got_sm.get(k, 1e18)) - float(v)) > 1e-9
               for k, v in smoothing.items()):
            return {"err": 1, "w": 0, "l": 0}
    except Exception:
        pass

    fb, db = load(os.path.join(AGENTS, opp))

    w_ = l_ = 0
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
            w_ += 1
        elif res is not None and res != 2:
            l_ += 1
    return {"err": 0, "w": w_, "l": l_}


def evaluate(smoothing, games, workers, seed0):
    live = [(o, sh) for o, sh in FIELD
            if os.path.isdir(os.path.join(AGENTS, o))]
    tot = sum(sh for _, sh in live)
    jobs, plan = [], []
    for wi, (opp, share) in enumerate(live):
        k = max(2, int(round(games * share / tot)))
        jobs.append((wi % max(1, workers), smoothing, k, seed0 + wi * 7919, opp))
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
        raise SystemExit("candidate arbitration did not bind -- coalition_weights"
                         ".json is not being read; fix before searching")
    return (num / den if den else -1.0), n_tot


def mutate(sm, rng):
    out = dict(sm)
    for key in rng.sample(list(BOUNDS), rng.choice([1, 1, 2])):
        lo, hi = BOUNDS[key]
        cur = float(out.get(key, lo))
        if key == "support_bonus":
            cand = cur + rng.uniform(-0.25, 0.25)
        else:
            cand = cur * rng.choice([0.4, 0.6, 0.8, 1.25, 1.7, 2.5])
        out[key] = round(min(max(cand, lo), hi), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--screen", type=int, default=60)
    ap.add_argument("--confirm", type=int, default=180)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    base_sm = load_weights()["smoothing"]
    if args.report:
        st = json.load(open(STATE))
        print(json.dumps({k: v for k, v in st.items() if k != "best"}, indent=1))
        print("base:", base_sm)
        print("best:", st["best"])
        return 0

    print(f"base smoothing: {base_sm}")
    print(f"objective: top-band field; accept above {ANCHOR_FIELD:.4f}")
    # best_field ratchets. The first version compared every candidate against
    # the FIXED anchor, so once the best had climbed to 0.6245 a candidate at
    # 0.6231 still cleared "anchor + 0.02" and the search would have walked
    # downhill while reporting accepts.
    st = {"round": 0, "accepted": 0, "screened": 0, "rejected_confirm": 0,
          "best": dict(base_sm), "best_field": ANCHOR_FIELD, "history": []}
    if os.path.exists(STATE):
        try:
            st = json.load(open(STATE))
            print(f"resuming at round {st['round']}, {st['accepted']} accepted")
        except Exception:
            pass

    rng = random.Random(args.seed + st["round"])
    t0 = time.time()
    for _ in range(args.rounds):
        st["round"] += 1
        cand = mutate(st["best"], rng)
        if cand == st["best"]:
            continue
        scr, n1 = evaluate(cand, args.screen, args.workers,
                           3000 + st["round"] * 13)
        st["screened"] += 1
        chg = {k: v for k, v in cand.items() if base_sm.get(k) != v}
        bar = st.get("best_field", ANCHOR_FIELD)
        if scr <= bar:
            print(f"r{st['round']:3d} screen {scr:.3f} ({n1}) reject | {chg}")
            json.dump(st, open(STATE, "w"))
            continue
        conf, n2 = evaluate(cand, args.confirm, args.workers,
                            800000 + st["round"] * 977)
        pooled = (scr * n1 + conf * n2) / (n1 + n2)
        ok = conf > bar and pooled > bar + 0.02
        print(f"r{st['round']:3d} screen {scr:.3f} confirm {conf:.3f} "
              f"pooled {pooled:.3f} {'ACCEPT' if ok else 'reject'} | {chg}")
        if ok:
            st["best"] = cand
            st["best_field"] = pooled
            st["accepted"] += 1
            st["history"].append({"round": st["round"], "pooled": pooled,
                                  "smoothing": cand})
            json.dump(cand, open(BEST, "w"), indent=1)
        else:
            st["rejected_confirm"] += 1
        json.dump(st, open(STATE, "w"))
    print(f"\n{st['accepted']} accepted / {st['screened']} screened, "
          f"{(time.time()-t0)/60:.0f} min")
    print("best:", st["best"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
