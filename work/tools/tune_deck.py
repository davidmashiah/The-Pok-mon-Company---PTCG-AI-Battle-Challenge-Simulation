"""Search the DECK against the champion, inside the card pool the policy already knows.

Why the deck and not the policy. A paired playout search -- common random
numbers across candidates, graded by prize differential, ~150 paired
determinizations per decision -- measures the edge between the heuristic's top
choice and the best alternative at a MEAN of 0.005, and 0.0018 when even the
options the heuristic rejects are allowed in. Its move selection is saturated.
No search inside that candidate set can pay, which is why eight structural
attacks all landed inside the noise.

What is not saturated is the DECK, and there is a specific reason to think so.
The policy's own weight table is written for cards the shipped decklist does not
contain:

    ability_fez        38066   <- the single highest weight in the table
    ability_fanrotom   29500
    play_genesect      20100   play_psyduck 20300   play_shaymin 19807
    cape_alak 9800   balloon 7300   helmet 7000   mist_retreat 9400 ...

There is no Fezandipiti ex, no Fan Rotom, no Genesect, Psyduck, Shaymin, no
capes, balloons, helmets, no Mist or Enriching Energy in the 60. A large part of
the policy is dead code with the list it ships with. The author published a
stripped-down deck; the weights are tuned for a richer one.

So the pool here is exactly the set of cards the policy NAMES -- every scoring
branch that exists but never fires today is reachable, and nothing outside the
policy's vocabulary is proposed, because a card the policy cannot score is a
card it will play at random.

Discipline is the same as tune_weights.py and for the same reason: screen, then
CONFIRM on disjoint seeds, because a screen-only optimiser here once reported 13
improvements of which all 13 were false.

  python work/tools/tune_deck.py --rounds 200 --screen 42 --confirm 120 --workers 6
  python work/tools/tune_deck.py --report
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
OUT = os.path.join(WORK, "out")
STATE = os.path.join(OUT, "tune_deck.json")
BEST = os.path.join(OUT, "deck_best.csv")

BASE = "v61_codex_safe"

# THE OBJECTIVE IS THE FIELD, NOT THE CHAMPION.
# Optimising a head-to-head against v61 optimises the mirror, which we already
# win 0.934 -- it cannot move the rating. What maps to rating is the weighted
# win rate across the archetypes we are actually matched into, and the anchor
# from field_test.py is unambiguous about where the rating is being lost:
#
#   Alakazam mirror  0.23  0.934      Crustle control 0.13  0.550
#   Mega Lucario     0.22  0.818      Archaludon      0.12  0.778
#   Grimmsnarl       0.20  0.207      Dragapult       0.10  0.449
#
#   field 0.6461  ==  726.1 on the ladder
#
# Grimmsnarl is a fifth of the field and we win one game in five. Games are
# allocated by share, so a candidate is rewarded for exactly what the ladder
# pays for and a mirror-only gain scores nothing.
FIELD = [
    ("w1_alakazam",   0.23),
    ("z_roman950",    0.22),
    ("w5_grimmsnarl", 0.20),
    ("p3_crustle",    0.13),
    ("w2_archaludon", 0.12),
    ("s_dragapult",   0.10),
]

# Cards the policy must keep or the deck stops functioning: the Abra evolution
# line it wins with, and its Psychic energy. Everything else is fair game.
CORE_MIN = {741: 4, 742: 3, 743: 3, 19: 4}
ACE_SPEC = {1247}          # at most one ACE SPEC in a legal deck
ANCHOR_FIELD = 0.6461      # v61_codex_safe's measured field rate == 726.1


def load_pool():
    """Card ids the policy names, i.e. every id it has a scoring branch for."""
    import re
    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import all_card_data
    cards = {c.cardId: c for c in all_card_data()}
    src = open(os.path.join(AGENTS, BASE, "main.py"), encoding="utf-8").read()
    pool = {}
    for m in re.finditer(r"^([A-Z][A-Za-z0-9_]*)\s*=\s*(\d{1,4})\s*$", src, re.M):
        v = int(m.group(2))
        if v in cards:
            pool[v] = cards[v]
    # SEARCH_MAX_OPTS = 24 is a search parameter that happens to collide with a
    # card id. Including it would propose a card the policy has never heard of.
    pool.pop(24, None)
    return pool, cards


def legal(deck, cards):
    if len(deck) != 60:
        return False
    c = Counter(deck)
    for cid, n in c.items():
        card = cards.get(cid)
        if card is None:
            return False
        # basic energy is exempt from the 4-copy rule
        is_basic_energy = int(getattr(card, "cardType", -1)) == 5
        if not is_basic_energy and n > 4:
            return False
    if sum(n for cid, n in c.items() if cid in ACE_SPEC) > 1:
        return False
    for cid, n in CORE_MIN.items():
        if c.get(cid, 0) < n:
            return False
    # One Basic is legal but unplayable: round 3 of the first field run
    # proposed a deck that produced ZERO completed games, because a hand with
    # no Basic is a mulligan and the engine never got a battle started. The
    # base ships 7 Basics and the gate already flags that as a high mulligan
    # rate, so treat 6 as the floor rather than 1.
    basics = sum(n for cid, n in c.items()
                 if int(getattr(cards[cid], "cardType", -1)) == 0
                 and getattr(cards[cid], "basic", False))
    if basics < 6:
        return False
    return True


def mutate(deck, pool, cards, rng):
    """Move 1-3 copies between cards, staying legal and staying in the pool."""
    for _ in range(60):
        d = list(deck)
        c = Counter(d)
        k = rng.choice([1, 1, 2, 3])
        ok = True
        for _ in range(k):
            # drop a copy of something we are allowed to reduce
            droppable = [cid for cid, n in c.items()
                         if n > 0 and c[cid] > CORE_MIN.get(cid, 0)]
            if not droppable:
                ok = False
                break
            drop = rng.choice(droppable)
            c[drop] -= 1
            if c[drop] == 0:
                del c[drop]
            # Bias toward cards the policy SUPPORTS BUT THE DECK OMITS. Those
            # are the scoring branches that never fire today -- ability_fez is
            # the highest weight in the table and there is no Fezandipiti ex in
            # the 60 -- so that is where an untouched gain would be. Uniform
            # sampling over the whole pool would spend most rounds nudging
            # counts of cards that are already there.
            absent = [c_ for c_ in pool if c_ not in c]
            if absent and rng.random() < 0.6:
                add = rng.choice(absent)
            else:
                add = rng.choice(list(pool))
            c[add] += 1
        if not ok:
            continue
        out = []
        for cid, n in c.items():
            out.extend([cid] * n)
        if legal(out, cards):
            return sorted(out)
    return list(deck)


def worker_dir(i):
    d = os.path.join(AGENTS, f"_deck_w{i}")
    os.makedirs(d, exist_ok=True)
    src = os.path.join(AGENTS, BASE, "main.py")
    dst = os.path.join(d, "main.py")
    if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
        shutil.copy(src, dst)
    return d


def _play(job):
    widx, deck, n, seed0, opp_dir = job
    d = worker_dir(widx)
    with open(os.path.join(d, "deck.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(map(str, deck)) + "\n")

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
    # The candidate must actually be PLAYING the mutated list. The agent resolves
    # deck.csv from cwd, and a resolution order that quietly preferred the base's
    # copy would make every candidate score exactly like the base -- a night
    # spent concluding that no deck helps.
    if sorted(da) != sorted(deck):
        return {"err_deck": 1, "w": 0, "l": 0}
    fb, db = load(os.path.join(AGENTS, opp_dir))

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
    return {"err_deck": 0, "w": w, "l": l}


def evaluate(deck, games, workers, seed0):
    """Weighted field win rate. Games are allocated to each archetype by share,
    so the objective is the quantity that converts to a rating."""
    live = [(o, sh) for o, sh in FIELD
            if os.path.isdir(os.path.join(AGENTS, o))]
    tot_share = sum(sh for _, sh in live)
    jobs = []
    plan = []
    for wi, (opp, share) in enumerate(live):
        k = max(1, int(round(games * share / tot_share)))
        plan.append((opp, share, k))
        jobs.append((wi % max(1, workers), deck, k,
                     seed0 + wi * 7919, opp))
    w = l = bad = 0
    num = den = 0.0
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for (opp, share, k), r in zip(plan, ex.map(_play, jobs)):
            bad += r["err_deck"]
            n_i = r["w"] + r["l"]
            w += r["w"]
            l += r["l"]
            if n_i:
                num += share * (r["w"] / n_i)
                den += share
    if bad:
        raise SystemExit("candidate did not play the mutated deck -- deck.csv "
                         "resolution is wrong; fix before searching")
    return (num / den if den else -1.0), w + l


def describe(deck, base, cards):
    a, b = Counter(deck), Counter(base)
    out = []
    for cid in sorted(set(a) | set(b)):
        if a.get(cid, 0) != b.get(cid, 0):
            nm = getattr(cards.get(cid), "name", f"id{cid}")
            out.append(f"{b.get(cid,0)}->{a.get(cid,0)} {nm}")
    return "; ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--screen", type=int, default=42)
    ap.add_argument("--confirm", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    pool, cards = load_pool()
    base = sorted(int(x) for x in
                  open(os.path.join(AGENTS, BASE, "deck.csv"),
                       encoding="utf-8").read().split() if x.strip())

    if args.report:
        st = json.load(open(STATE))
        print(f"round {st['round']} accepted {st['accepted']} "
              f"died_at_confirm {st['rejected_confirm']}")
        print("best deck vs base:", describe(st["best"], base, cards) or "(same)")
        return 0

    print(f"pool: {len(pool)} policy-known cards; base deck has "
          f"{len(set(base))} distinct")
    st = {"round": 0, "accepted": 0, "screened": 0, "rejected_confirm": 0,
          "best": base, "history": []}
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
        cand = mutate(st["best"], pool, cards, rng)
        if sorted(cand) == sorted(st["best"]):
            continue
        scr, n1 = evaluate(cand, args.screen, args.workers,
                           2000 + st["round"] * 13)
        st["screened"] += 1
        if scr < 0.0:
            print(f"r{st['round']:3d} UNPLAYABLE (0 games) | "
                  f"{describe(cand, st['best'], cards)[:70]}")
            json.dump(st, open(STATE, "w"))
            continue
        if scr <= ANCHOR_FIELD:
            print(f"r{st['round']:3d} screen {scr:.3f} ({n1})  reject   "
                  f"| {describe(cand, st['best'], cards)[:80]}")
            json.dump(st, open(STATE, "w"))
            continue
        conf, n2 = evaluate(cand, args.confirm, args.workers,
                            700000 + st["round"] * 977)
        pooled = (scr * n1 + conf * n2) / (n1 + n2)
        ok = conf > ANCHOR_FIELD and pooled > ANCHOR_FIELD + 0.02
        print(f"r{st['round']:3d} screen {scr:.3f} confirm {conf:.3f} "
              f"pooled {pooled:.3f}  {'ACCEPT' if ok else 'reject'} "
              f"| {describe(cand, st['best'], cards)[:80]}")
        if ok:
            st["best"] = cand
            st["accepted"] += 1
            st["history"].append({"round": st["round"], "pooled": pooled,
                                  "change": describe(cand, base, cards)})
            with open(BEST, "w", encoding="utf-8") as fh:
                fh.write("\n".join(map(str, cand)) + "\n")
        else:
            st["rejected_confirm"] += 1
        json.dump(st, open(STATE, "w"))
    print(f"\n{st['accepted']} accepted / {st['screened']} screened, "
          f"{(time.time()-t0)/60:.0f} min")
    print("best deck vs base:", describe(st["best"], base, cards) or "(same)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
