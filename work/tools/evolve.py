"""Hill-climb the policy weights by actually playing the strongest agents we have.

The policy is a pile of hand-chosen magic numbers (ABILITY 30000, PLAY_MON
20000, EVOLVE 9000, ENERGY_BASE 8000, ...). Their RELATIVE order is the whole
strategy: it decides that an Ability always resolves before the turn's energy
attachment, that Boss's Orders outranks Carmine, and so on. Nobody ever tuned
them against a real opponent.

Every earlier tuning attempt in this repo (deck_opt.py, best_deck.py) scored
candidates with OUR OWN policy piloting the other side. That is the measurement
flaw behind every failed local metric here: against weak opposition almost
anything wins, so real differences are invisible or inverted. FINDINGS S11
records the consequence -- a screen-only optimiser reported 13 improvements and
all 13 were false.

So this differs in exactly two ways:

  * the opponents are REAL published agents (z_roman950, the LB-950 baseline;
    w1_alakazam, the 5th-place agent; w2_archaludon), weighted toward the
    strongest one and toward the archetypes our own replays say we actually meet;

  * nothing is adopted on a screen alone. A candidate must beat the champion on
    the cheap screen AND again on a confirmation run with disjoint seeds, which
    is the stage that killed all 13 false positives last time.

Usage:
  python work/tools/evolve.py --rounds 20 --screen 30 --confirm 90
  python work/tools/evolve.py --report
"""
import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
OUT = os.path.join(WORK, "out")
STORE = os.path.join(OUT, "evolve.json")
AGENT = "tuned"

# Fixed opponents, plus SELF -- the reigning champion.
#
# Without SELF this loop has a hard ceiling: the strongest sparring partner
# available is z_roman950, an LB-950 agent, so tuning to beat it can only ever
# produce ~950-level play. Making every candidate also beat the CURRENT champion
# turns the panel into a moving target and is what lets the search climb past
# the strength of anything we can download.
PANEL = [("z_roman950", 0.25), ("w1_alakazam", 0.20),
         ("w2_archaludon", 0.15), ("SELF", 0.40)]

# knob -> (low, high). Ranges are wide enough to REORDER the policy, because the
# ordering is the strategy; a +-5% jitter could only ever polish tie-breaks.
BOUNDS = {
    "ABILITY": (500, 32000), "PLAY_MON": (500, 32000),
    "ITEM_DEFAULT": (500, 32000), "EVOLVE": (500, 32000),
    "ENERGY_BASE": (500, 32000), "HERO_CAPE": (500, 32000),
    "SWITCH": (500, 32000), "PPP": (500, 32000),
    "GRAVITY": (-1, 32000), "BOSS": (500, 32000),
    "LILLIE": (-1, 32000), "CARMINE": (-1, 32000),
    "RETREAT": (-1, 32000), "ATTACK_MATCH": (500, 32000),
    "LOW_DECK": (0, 25), "PRIZE_W": (200, 12000),
}


# Cards the policy has an explicit rule for, plus Basic {F} Energy. Restricting
# the deck search to these keeps behaviour interpretable: an unrecognised card
# falls through _score_play to the default 10000, which would get it played
# ahead of the energy attachment every turn. The search re-allocates COPIES
# among known cards rather than inventing a new archetype.
DECK_POOL = {
    6: (8, 20),      # Basic {F} Energy
    673: (0, 4), 674: (0, 4),        # Makuhita / Hariyama
    675: (1, 4), 676: (1, 4),        # Lunatone / Solrock (Cosmic Beam needs both)
    677: (2, 4), 678: (2, 4),        # Riolu / Mega Lucario ex
    1102: (0, 4),                    # Dusk Ball
    1123: (0, 4),                    # Switch
    1141: (0, 4),                    # Premium Power Pro
    1142: (0, 4),                    # Fighting Gong
    1152: (0, 4),                    # Poke Pad
    1159: (0, 1),                    # Hero's Cape (ACE SPEC, max 1 in the deck)
    1182: (0, 4),                    # Boss's Orders
    1192: (0, 4),                    # Carmine
    1213: (0, 4),                    # Judge
    1227: (0, 4),                    # Lillie's Determination
    1252: (0, 4),                    # Gravity Mountain
}


def mutate_deck(counts, rng):
    """Move one copy from one card to another, keeping the deck legal at 60."""
    out = dict(counts)
    for _ in range(12):
        src = rng.choice([k for k in out if out.get(k, 0) > DECK_POOL[k][0]])
        dst = rng.choice([k for k in DECK_POOL if out.get(k, 0) < DECK_POOL[k][1]])
        if src == dst:
            continue
        out[src] -= 1
        out[dst] = out.get(dst, 0) + 1
        if sum(out.values()) == 60 and all(
                DECK_POOL[k][0] <= v <= DECK_POOL[k][1] for k, v in out.items() if v):
            return {k: v for k, v in out.items() if v > 0}
        out = dict(counts)
    return dict(counts)


def counts_to_deck(counts):
    c = {int(k): int(v) for k, v in counts.items()}
    return [k for k in sorted(c) for _ in range(c[k])]


def _load_opp(agent_dir):
    full = os.path.join(WORK, "agents", agent_dir)
    if full not in sys.path:
        sys.path.insert(0, full)
    with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
        src = fh.read()
    env = {}
    cwd = os.getcwd()
    try:
        os.chdir(full)
        exec(compile(src, "main.py", "exec"), env)
    finally:
        os.chdir(cwd)
    d = env.get("my_deck") or env.get("DECK")
    if not d:
        with open(os.path.join(full, "deck.csv"), encoding="utf-8") as fh:
            d = [int(x) for x in fh.read().split() if x.strip()]
    return [v for v in env.values() if callable(v)][-1], list(d)


def _worker(job):
    params, deck_counts, opp_dir, n, seed0, champ_params, champ_deck = job
    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    fa, da = _load_opp(AGENT)
    # P is read inside the scoring functions, so overriding it after import
    # changes behaviour without rewriting the file. LOW_DECK_COUNT is the one
    # knob consumed at import time, so it has to be re-derived by hand.
    import importlib
    full = os.path.join(WORK, "agents", AGENT)
    env = {}
    cwd = os.getcwd()
    try:
        os.chdir(full)
        with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
            exec(compile(fh.read(), "main.py", "exec"), env)
    finally:
        os.chdir(cwd)
    env["P"].update(params)
    env["LOW_DECK_COUNT"] = int(env["P"]["LOW_DECK"])
    fa = [v for v in env.values() if callable(v)][-1]
    da = counts_to_deck(deck_counts) if deck_counts else list(env["my_deck"])
    env["my_deck"][:] = list(da)
    if opp_dir == "SELF":
        # the reigning champion, same file, its own weights and decklist
        env2 = {}
        cwd2 = os.getcwd()
        try:
            os.chdir(full)
            with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
                exec(compile(fh.read(), "main.py", "exec"), env2)
        finally:
            os.chdir(cwd2)
        env2["P"].update(champ_params)
        env2["LOW_DECK_COUNT"] = int(env2["P"]["LOW_DECK"])
        db = counts_to_deck(champ_deck) if champ_deck else list(env2["my_deck"])
        env2["my_deck"][:] = list(db)
        fb = [v for v in env2.values() if callable(v)][-1]
    else:
        fb, db = _load_opp(opp_dir)

    w = 0
    played = 0
    for g in range(n):
        first = ((seed0 + g) % 2 == 0)
        d0, d1 = (da, db) if first else (db, da)
        p0, p1 = (fa, fb) if first else (fb, fa)
        me = 0 if first else 1
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                c = o.current
                if c is not None and c.result != -1:
                    if c.result == me:
                        w += 1
                    played += 1
                    break
                who = c.yourIndex if c is not None else 0
                obs = battle_select(list((p0 if who == 0 else p1)(obs)))
        except Exception:
            pass
        finally:
            battle_finish()
    return w, played


def evaluate(params, deck, games, seed0, workers=5, champ=None, champ_deck=None):
    """Weighted win rate against the panel. Returns (fitness, detail)."""
    fit = 0.0
    detail = {}
    for opp, wt in PANEL:
        per = max(1, games // workers)
        jobs = [(params, deck, opp, per, seed0 + i * 1013, champ or params, champ_deck or deck) for i in range(workers)]
        w = n = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for f in as_completed([ex.submit(_worker, j) for j in jobs]):
                a, b = f.result()
                w += a
                n += b
        p = w / max(1, n)
        detail[opp] = (w, n, round(p, 4))
        fit += wt * p
    return fit, detail


def mutate(params, rng):
    out = dict(params)
    for k in rng.sample(list(BOUNDS), rng.randint(1, 3)):
        lo, hi = BOUNDS[k]
        cur = out[k]
        if rng.random() < 0.35:
            out[k] = rng.randint(int(lo), int(hi))          # jump
        else:
            span = max(1, abs(cur) * 0.5)                    # local step
            out[k] = int(max(lo, min(hi, cur + rng.gauss(0, span))))
    return out


def load_store():
    if os.path.exists(STORE):
        try:
            return json.load(open(STORE, encoding="utf-8"))
        except Exception:
            pass
    return {"champion": None, "history": []}


def save_store(s):
    json.dump(s, open(STORE, "w", encoding="utf-8"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--screen", type=int, default=30)
    ap.add_argument("--confirm", type=int, default=90)
    ap.add_argument("--mutants", type=int, default=5)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    store = load_store()
    if a.report:
        print(json.dumps(store.get("champion"), indent=1))
        for h in store["history"][-25:]:
            print(h)
        return

    # read the default weights straight out of the agent
    env = {}
    full = os.path.join(WORK, "agents", AGENT)
    cwd = os.getcwd()
    try:
        os.chdir(full)
        sys.path.insert(0, os.path.join(WORK, "lib"))
        sys.path.insert(0, full)
        with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
            exec(compile(fh.read(), "main.py", "exec"), env)
    finally:
        os.chdir(cwd)
    champ = dict(store.get("champion") or env["P"])

    rng = random.Random(20260805)
    t0 = time.time()
    print(f"panel: {PANEL}\nchampion: {champ}\n")
    # JSON round-trips dict keys to strings; DECK_POOL and the engine both
    # need ints, and a str key silently produces a deck of strings.
    cdeck = {int(k): int(v) for k, v in (store.get('deck') or {}).items()}
    if not cdeck:
        from collections import Counter as _C
        cdeck = dict(_C(env['my_deck']))
    cf, cd = evaluate(champ, cdeck, a.confirm, seed0=7, workers=a.workers,
                      champ=champ, champ_deck=cdeck)
    print(f"champion fitness {cf:.4f}  {cd}\n")

    for r in range(a.rounds):
        best = None
        for m in range(a.mutants):
            # half the proposals move a card, half move a weight -- the deck was
            # inherited from romanrozen v10 and has never been validated against
            # a real opponent either
            if rng.random() < 0.5:
                cand, cdk = dict(champ), mutate_deck(cdeck, rng)
            else:
                cand, cdk = mutate(champ, rng), dict(cdeck)
            sf, sd = evaluate(cand, cdk, a.screen, seed0=1000 + r * 97 + m,
                              workers=a.workers, champ=champ, champ_deck=cdeck)
            flag = ""
            if sf > cf:
                flag = "  <- screens better"
                if best is None or sf > best[0]:
                    best = (sf, cand, sd, cdk)
            print(f"r{r} m{m} screen {sf:.4f}{flag}")
        if best is None:
            print(f"r{r}: nothing screened better ({time.time()-t0:.0f}s)")
            continue
        # confirmation on DISJOINT seeds -- the stage that killed 13 of 13 false
        # positives the last time this repo ran an optimiser
        vf, vd = evaluate(best[1], best[3], a.confirm, seed0=50000 + r,
                          workers=a.workers, champ=champ, champ_deck=cdeck)
        print(f"r{r}: screen {best[0]:.4f} -> confirm {vf:.4f} vs champ {cf:.4f}")
        if vf > cf:
            champ, cf, cd, cdeck = best[1], vf, vd, best[3]
            store["champion"] = champ
            store["deck"] = cdeck
            store["history"].append(
                {"round": r, "fitness": round(vf, 4), "detail": vd,
                 "params": champ, "deck": cdeck})
            save_store(store)
            print(f"   ADOPTED. new champion fitness {cf:.4f}  {vd}")
        else:
            print("   rejected on confirmation")
    print(f"\nfinal champion fitness {cf:.4f}\n{json.dumps(champ)}")
    store["champion"] = champ
    store["deck"] = cdeck
    save_store(store)


if __name__ == "__main__":
    main()
