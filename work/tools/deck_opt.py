"""Hill-climb our decklist against the REAL metagame, using paired comparisons.

Three deliberate design choices, each answering a way this can lie to you:

1. OBJECTIVE IS THE FIELD, NOT THE MIRROR.
   Every earlier measurement here was our deck vs our deck. A mirror match
   optimises for beating yourself, which is exactly the validation that
   mispredicts a diverse ladder. Candidates are scored against scraped
   leaderboard decklists instead.

2. PAIRED COMPARISON.
   The champion and the candidate face the SAME opponents with the SAME seeds
   and the same first-player pattern. Deck matchups and coin flips then affect
   both sides identically and cancel, so a real difference shows up in far
   fewer games than an unpaired test would need.

3. TWO STAGES, AND THE SECOND ONE DECIDES.
   Picking the best of N candidates by a noisy score is a max-over-N, not an
   unbiased estimate. A cheap screen proposes; a much larger independent
   confirmation run disposes, and only its interval can trigger adoption.

Usage:
  python work/tools/deck_opt.py --agent v2_lucario --rounds 40
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
OUT = os.path.join(WORK, "out")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def _field(job):
    """Play `deck` against a slice of the field. Returns (wins, decided)."""
    agent_dir, deck, opps, games_each, seed0 = job
    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    full = os.path.join(WORK, "agents", agent_dir)
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
    fn = [v for v in env.values() if callable(v)][-1]

    wins = decided = 0
    for oi, opp in enumerate(opps):
        for g in range(games_each):
            me_first = ((seed0 + oi * 31 + g) % 2 == 0)
            d0, d1 = (deck, opp) if me_first else (opp, deck)
            me_idx = 0 if me_first else 1
            obs, _ = battle_start(list(d0), list(d1))
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
                    env["my_deck"] = list(d0 if who == 0 else d1)
                    env["DECK"] = env["my_deck"]
                    obs = battle_select(list(fn(obs)))
            except Exception:
                res = None
            finally:
                battle_finish()
            if res is None or res == 2:
                continue
            decided += 1
            if res == me_idx:
                wins += 1
    return wins, decided


def field_score(agent, deck, opps, games_each, workers, seed):
    chunks = [opps[i::workers] for i in range(workers)]
    jobs = [(agent, deck, c, games_each, seed) for c in chunks if c]
    w = d = 0
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        for f in as_completed([ex.submit(_field, j) for j in jobs]):
            a, b = f.result()
            w += a
            d += b
    return w, d


def legal(deck, cards):
    if len(deck) != 60:
        return False
    cnt = Counter(deck)
    ace = basics = 0
    for cid, n in cnt.items():
        c = cards.get(cid)
        if c is None:
            return False
        if n > 4 and int(c.cardType) != 5:
            return False
        if c.aceSpec:
            ace += n
        if int(c.cardType) == 0 and c.basic:
            basics += n
    return ace <= 1 and basics >= 1


def mutate(deck, pool, cards, rng):
    # legal() below is what stops a swap from building an illegal list (e.g.
    # adding a 5th copy of a card already at the 4-copy maximum). A hand-rolled
    # variant without this check produced a deck where every battle_start
    # returned None and the whole run scored n=0.
    for _ in range(80):
        d = list(deck)
        i = rng.randrange(60)
        out = d[i]
        new = rng.choice(pool)
        if new == out:
            continue
        d[i] = new
        if legal(d, cards):
            return d, out, new
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="v2_lucario")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--screen-games", type=int, default=3)
    ap.add_argument("--confirm-games", type=int, default=10)
    ap.add_argument("--decks", type=int, default=20)
    ap.add_argument("--hard", action="store_true",
                    help="optimise against the discriminating subset only")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import all_card_data
    cards = {c.cardId: c for c in all_card_data()}

    base = [int(x.strip()) for x in
            open(os.path.join(WORK, "agents", args.agent, "deck.csv"))
            if x.strip()][:60]
    assert legal(base, cards), "base deck illegal"

    with open(os.path.join(OUT, "meta_decks.json"), encoding="utf-8") as f:
        store = json.load(f)
    # Against the full field every agent wins ~89% and differences vanish into
    # the noise. These are the opponents the champion scores 0.59-0.83 against,
    # where a deck change can actually register.
    HARD_SCORES = {1010.8, 1109.6, 1060.3, 1034.9, 1275.3, 1063.4, 1104.6}
    seen, opps = set(), []
    for t in sorted(store["teams"].values(), key=lambda t: -t.get("score", 0)):
        d = t.get("deck")
        if not d or len(d) != 60:
            continue
        k = tuple(sorted(d))
        if k in seen:
            continue
        if args.hard and round(t.get("score", 0), 1) not in HARD_SCORES:
            continue
        seen.add(k)
        opps.append(d)
        if len(opps) >= args.decks:
            break

    pool = set(base)
    for t in store["teams"].values():
        for cid in (t.get("deck") or []):
            pool.add(cid)
    pool = sorted(pool)

    print(f"field: {len(opps)} decks | candidate pool: {len(pool)} cards")
    rng = random.Random(args.seed)
    champ = list(base)
    log = []
    t0 = time.time()

    cw, cd = field_score(args.agent, champ, opps, args.screen_games,
                         args.workers, args.seed)
    p, lo, hi = wilson(cw, cd)
    print(f"champion baseline vs field: {p:.4f} [{lo:.4f},{hi:.4f}] n={cd}\n")

    for rnd in range(1, args.rounds + 1):
        cand, out, new = mutate(champ, pool, cards, rng)
        if cand is None:
            continue
        tag = f"-{cards[out].name[:20]} +{cards[new].name[:20]}"
        seed = args.seed + rnd * 977

        # paired screen: identical opponents, identical seed
        aw, ad = field_score(args.agent, cand, opps, args.screen_games,
                             args.workers, seed)
        bw, bd = field_score(args.agent, champ, opps, args.screen_games,
                             args.workers, seed)
        if ad == 0 or bd == 0:
            continue
        if aw / ad <= bw / bd:
            print(f"[{rnd:>3}] screen {aw/ad:.3f} vs champ {bw/bd:.3f}  "
                  f"REJECT  {tag}")
            log.append({"round": rnd, "swap": tag, "screen": aw / ad,
                        "champ": bw / bd, "adopted": False})
            continue

        # confirmation on fresh seeds; only this decides
        s2 = seed + 500000
        aw2, ad2 = field_score(args.agent, cand, opps, args.confirm_games,
                               args.workers, s2)
        bw2, bd2 = field_score(args.agent, champ, opps, args.confirm_games,
                               args.workers, s2)
        pa, loa, hia = wilson(aw2, ad2)
        pb, lob, hib = wilson(bw2, bd2)
        ok = loa > pb          # candidate's lower bound clears champ's point
        print(f"[{rnd:>3}] screen {aw/ad:.3f}>{bw/bd:.3f} -> CONFIRM "
              f"cand {pa:.4f}[{loa:.4f},{hia:.4f}] vs champ {pb:.4f} "
              f"n={ad2}  {'ADOPT' if ok else 'reject (screen was noise)'}  {tag}")
        log.append({"round": rnd, "swap": tag, "cand": pa, "cand_lo": loa,
                    "champ": pb, "n": ad2, "adopted": ok})
        if ok:
            champ = cand
            with open(os.path.join(OUT, "deck_opt_champion.csv"), "w") as f:
                f.write("\n".join(map(str, champ)) + "\n")
        with open(os.path.join(OUT, "deck_opt_log.json"), "w") as f:
            json.dump(log, f, indent=1)

    adopted = sum(1 for r in log if r.get("adopted"))
    print(f"\n{time.time()-t0:.0f}s | {adopted} adopted of {len(log)} tried")
    add = Counter(champ) - Counter(base)
    rem = Counter(base) - Counter(champ)
    print("CHANGES vs base:")
    for cid, n in add.items():
        print(f"   +{n} {cards[cid].name}")
    for cid, n in rem.items():
        print(f"   -{n} {cards[cid].name}")


if __name__ == "__main__":
    main()
