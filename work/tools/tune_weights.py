"""Hill-climb the adopted base's 66 priority weights against the base itself.

Why this is the lever that is left. Four structural attacks on `v61_codex_safe`
all landed inside the noise at n=240 -- four times the determinizations
(0.5375), a live-state correctness fix (0.5208), terminal playouts with no
evaluator (0.4625), real opponent decklists in the belief (0.4542). Together
v65 and v70 say the search is neither variance-limited nor evaluator-limited,
so "search harder" is spent. What has NOT been touched is the policy those
searches are ranking, and its author left an explicit hook for exactly this:

    for _p in ("alak_w.json", ..., "/kaggle_simulations/agent/alak_w.json"):
        if os.path.exists(_p):
            WEIGHTS.update(json.load(open(_p)))

66 numbers, overridable by a JSON file, no code change. Their baked values came
from the author's own memetic search against the author's own opponents, so the
space is genuinely new for us.

Two design choices that this repo has already paid to learn:

  * OPTIMISE THE METRIC WE ACTUALLY CARE ABOUT. `evolve.py` maximised fitness
    against a PANEL and produced a genome that was dead even head-to-head
    (0.5050 over 400 games) -- panel fitness and head-to-head are different
    questions. Here the objective IS the head-to-head against the reigning
    champion, so there is no proxy to be wrong about.
  * NOTHING IS ADOPTED ON A SCREEN. Every candidate that screens better must
    also win a confirmation run on DISJOINT seeds. That stage rejected roughly
    8 of every 10 candidates last time, and a screen-only optimiser here once
    reported 13 improvements of which all 13 were false.

Each worker gets its own agent directory, because the weights are read from a
file in the cwd and workers sharing one would overwrite each other's genome
mid-read -- the same class of race that made a shared deck.csv look like a
flaky engine.

  python work/tools/tune_weights.py --rounds 60 --screen 40 --confirm 80
  python work/tools/tune_weights.py --report
"""
import argparse
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
STATE = os.path.join(OUT, "tune_weights.json")

BASE = "v61_codex_safe"          # champion, and the opponent we optimise against
WEIGHT_FILE = "alak_w.json"

# Weights whose meaning is a hard ordering rather than a preference. The policy
# uses ~20000 to mean "play a Pokemon before anything else" and ~30000 to mean
# "fire abilities first"; perturbing those re-orders whole phases of the turn
# rather than tuning a preference, and ranking abilities below the energy
# attachment has already been measured at 0.4250 in this repo. Leave them.
FROZEN = {
    "play_pokemon_base", "ability_dudun", "ability_fez", "ability_fanrotom",
    "ability_default", "nz_ex", "cage_counter", "cage_snipe", "mine_counter",
    "jamming_tools", "jamming_counter",
}


def base_weights():
    """Read W out of the agent by exec-loading it, not by parsing the source."""
    lib = os.path.join(WORK, "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)          # main.py imports cg.api at module level
    d = os.path.join(AGENTS, BASE)
    cwd = os.getcwd()
    try:
        os.chdir(d)
        env = {}
        with open("main.py", encoding="utf-8") as fh:
            exec(compile(fh.read(), "main.py", "exec"), env)
    finally:
        os.chdir(cwd)
    w = env.get("W") or env.get("WEIGHTS")
    if not isinstance(w, dict):
        raise SystemExit("could not read WEIGHTS out of " + BASE)
    return {k: float(v) for k, v in w.items()}


def worker_dir(i):
    """A private copy of the bundle per worker; genome written into it."""
    d = os.path.join(AGENTS, f"_tune_w{i}")
    os.makedirs(d, exist_ok=True)
    for fn in ("main.py", "deck.csv"):
        src = os.path.join(AGENTS, BASE, fn)
        dst = os.path.join(d, fn)
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            shutil.copy(src, dst)
    return d


def _play(job):
    """Play `n` games: candidate (with genome) vs the unmodified base."""
    widx, genome, n, seed0 = job
    d = worker_dir(widx)
    with open(os.path.join(d, WEIGHT_FILE), "w") as fh:
        json.dump(genome, fh)

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
            deck = None
            try:
                r = fn({"current": None, "select": None})
                if isinstance(r, (list, tuple)) and len(r) == 60:
                    deck = [int(x) for x in r]
            except Exception:
                pass
        finally:
            os.chdir(cwd)
        if deck is None:
            with open(os.path.join(full, "deck.csv"), encoding="utf-8") as fh:
                deck = [int(x) for x in fh.read().split() if x.strip()]
        return fn, deck, env

    fa, da, env_a = load(d)
    # Assert the genome actually took. A weight file that silently failed to
    # load would make every candidate score exactly like the base, and the
    # search would spend a night reporting that nothing helps.
    wa = env_a.get("W") or {}
    applied = sum(1 for k, v in genome.items() if abs(float(wa.get(k, 1e18)) - v) < 1e-6)
    if applied < len(genome):
        return {"err_genome": len(genome) - applied, "w": 0, "l": 0}

    fb, db, _ = load(os.path.join(AGENTS, BASE))

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
    return {"err_genome": 0, "w": w, "l": l}


def evaluate(genome, games, workers, seed0):
    per = max(1, games // workers)
    jobs = [(i, genome, per, seed0 + i * 7919) for i in range(workers)]
    w = l = bad = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_play, jobs):
            w += r["w"]
            l += r["l"]
            bad += r["err_genome"]
    if bad:
        raise SystemExit(f"genome did not load in the agent ({bad} weights differ)"
                         f" -- alak_w.json is not being read; fix before tuning")
    n = w + l
    return (w / n if n else 0.0), n


def mutate(base, genome, rng, k=4):
    out = dict(genome)
    keys = [k_ for k_ in base if k_ not in FROZEN]
    for key in rng.sample(keys, min(k, len(keys))):
        cur = out.get(key, base[key])
        scale = rng.choice([0.7, 0.85, 1.18, 1.4])
        jitter = rng.uniform(-0.06, 0.06) * max(abs(base[key]), 1.0)
        out[key] = round(cur * scale + jitter, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--screen", type=int, default=40)
    ap.add_argument("--confirm", type=int, default=80)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        if os.path.exists(STATE):
            st = json.load(open(STATE))
            print(json.dumps({k: v for k, v in st.items() if k != "best"}, indent=1))
            print("best genome deltas vs base:")
            base = base_weights()
            for k, v in sorted(st.get("best", {}).items()):
                if abs(v - base.get(k, v)) > 1e-6:
                    print(f"   {k:26s} {base.get(k)} -> {v}")
        return 0

    base = base_weights()
    print(f"{len(base)} weights, {len(base)-len(FROZEN & set(base))} tunable")
    st = {"round": 0, "accepted": 0, "screened": 0, "rejected_confirm": 0,
          "best": dict(base), "history": []}
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
        cand = mutate(base, st["best"], rng)
        scr, n1 = evaluate(cand, args.screen, args.workers, 1000 + st["round"] * 13)
        st["screened"] += 1
        if scr <= 0.50:
            print(f"r{st['round']:3d} screen {scr:.3f} ({n1})  reject")
            json.dump(st, open(STATE, "w"), indent=1)
            continue
        # DISJOINT seeds -- the whole point of the confirmation stage
        conf, n2 = evaluate(cand, args.confirm, args.workers,
                            500000 + st["round"] * 977)
        pooled = (scr * n1 + conf * n2) / (n1 + n2)
        # Deliberately stricter than "beat 0.500". Over ~80 rounds a 0.50 bar
        # accepts noise several times per run and the climb walks sideways into
        # whichever genome got lucky -- the exact mechanism behind 13 false
        # improvements here. The final genome is re-validated at n=240 through
        # the gauntlet regardless; this bar only decides what the climb carries.
        ok = conf > 0.50 and pooled > 0.54
        print(f"r{st['round']:3d} screen {scr:.3f} ({n1})  confirm {conf:.3f} "
              f"({n2})  pooled {pooled:.3f}  {'ACCEPT' if ok else 'reject'}")
        if ok:
            st["best"] = cand
            st["accepted"] += 1
            st["history"].append({"round": st["round"], "screen": scr,
                                  "confirm": conf, "pooled": pooled})
            with open(os.path.join(OUT, "alak_w_best.json"), "w") as fh:
                json.dump(cand, fh, indent=1)
        else:
            st["rejected_confirm"] += 1
        json.dump(st, open(STATE, "w"), indent=1)
    print(f"\n{st['accepted']} accepted / {st['screened']} screened, "
          f"{st['rejected_confirm']} died at confirmation, "
          f"{(time.time()-t0)/60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
