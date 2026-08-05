"""Evaluate an agent against the REAL leaderboard metagame, not its own mirror.

Every earlier measurement here was our deck vs our deck with our policy on both
sides. That is the classic anti-correlated-validation trap: the ladder is a
diverse field (about a third of the top 34 teams play Marnie's Grimmsnarl ex,
and exactly one plays our Mega Lucario ex), so a mirror match measures almost
nothing about how we do against it.

This plays our agent, on our deck, against the same policy piloting each
scraped opponent decklist, and reports per-archetype win rates.

Known limitation, stated rather than hidden: our rule-based policy pilots
someone else's deck badly, so these opponents are weaker than the real ones.
The RANKING of which archetypes trouble us is the signal; the absolute win rate
is optimistic. Do not quote it as an expected ladder score.

Usage:
  python work/tools/meta_arena.py --agent v2_lucario --games 120
"""
import argparse
import json
import math
import os
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


def _play(job):
    agent_dir, my_deck, opp_deck, n, seed0 = job
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

    wins = losses = errs = 0
    for g in range(n):
        me_first = ((seed0 + g) % 2 == 0)
        d0, d1 = (my_deck, opp_deck) if me_first else (opp_deck, my_deck)
        me_idx = 0 if me_first else 1
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            errs += 1
            continue
        res = None
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                if o.current is not None and o.current.result != -1:
                    res = o.current.result
                    break
                who = o.current.yourIndex if o.current is not None else 0
                # the agent resolves its own list at import; point it at
                # whichever deck the acting side is holding
                env["my_deck"] = list(d0 if who == 0 else d1)
                env["DECK"] = env["my_deck"]
                obs = battle_select(list(fn(obs)))
        except Exception:
            res = None
        finally:
            battle_finish()
        if res is None:
            errs += 1
        elif res == 2:
            pass
        elif res == me_idx:
            wins += 1
        else:
            losses += 1
    return wins, losses, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="v2_lucario")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--max-decks", type=int, default=25)
    ap.add_argument("--contains", type=int, default=0,
                    help="only opponents whose deck contains this card id "
                         "(e.g. 648 = Marnie's Grimmsnarl ex, 53%% of the field)")
    ap.add_argument("--hard", action="store_true",
                    help="evaluate ONLY on the known-hard decks")
    args = ap.parse_args()

    # Measuring against the whole field saturates: every variant wins ~89%, so
    # real differences compress into a band narrower than the noise. The hard
    # subset sits near 60-78%, where a change can actually register.
    #
    # Select by MEASURED difficulty, not by marker card. A first attempt keyed
    # on card ids and pulled in 0.96+ matchups, because Fezandipiti ex is a
    # tech card that appears in many otherwise-easy decks. These are the exact
    # opponents the champion scored below 0.83 against, identified by the
    # owning team's leaderboard score.
    HARD_SCORES = {1010.8, 1109.6, 1060.3, 1034.9, 1275.3, 1063.4, 1104.6}

    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import all_card_data
    cards = {c.cardId: c for c in all_card_data()}

    my_deck = [int(x.strip()) for x in
               open(os.path.join(WORK, "agents", args.agent, "deck.csv"))
               if x.strip()][:60]

    with open(os.path.join(OUT, "meta_decks.json"), encoding="utf-8") as f:
        store = json.load(f)
    seen, opps = set(), []
    for t in sorted(store["teams"].values(), key=lambda t: -t.get("score", 0)):
        d = t.get("deck")
        if not d or len(d) != 60:
            continue
        key = tuple(sorted(d))
        if key in seen:
            continue
        if args.hard and round(t.get("score", 0), 1) not in HARD_SCORES:
            continue
        # Half of all ladder opponents play one archetype, so its matchup win
        # rate dominates our rating far more than any other single number.
        if args.contains and args.contains not in set(d):
            continue
        seen.add(key)
        opps.append(t)
        if len(opps) >= args.max_decks:
            break

    def label(deck):
        cnt = Counter(deck)
        ex = [cards[c].name for c, n in cnt.items()
              if c in cards and int(cards[c].cardType) == 0
              and (cards[c].megaEx or cards[c].ex)]
        return " / ".join(sorted(set(ex))[:2]) or "(no ex)"

    print(f"agent={args.agent}  vs {len(opps)} distinct meta decks, "
          f"{args.games} games each\n")
    rows = []
    tot_w = tot_l = 0
    t0 = time.time()
    for t in opps:
        per = max(1, args.games // args.workers)
        jobs = [(args.agent, my_deck, t["deck"], per,
                 args.seed + w * 7919) for w in range(args.workers)]
        w = l = e = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(_play, j) for j in jobs]):
                a, b, c = f.result()
                w += a
                l += b
                e += c
        n = w + l
        p, lo, hi = wilson(w, n)
        tot_w += w
        tot_l += l
        rows.append((p, lo, hi, n, t, e))
        print(f"  {p:6.3f} [{lo:.3f},{hi:.3f}] n={n:<4} "
              f"vs {t['score']:>7.1f} {label(t['deck'])[:44]}"
              + (f"  ERR={e}" if e else ""))

    rows.sort(key=lambda r: r[0])
    print(f"\n--- WORST MATCHUPS (fix these first) ---")
    for p, lo, hi, n, t, e in rows[:6]:
        print(f"  {p:6.3f} [{lo:.3f},{hi:.3f}]  {t['score']:>7.1f}  "
              f"{label(t['deck'])[:50]}")
    n = tot_w + tot_l
    p, lo, hi = wilson(tot_w, n)
    print(f"\nOVERALL vs field: {p:.4f} [{lo:.4f},{hi:.4f}]  n={n}  "
          f"({time.time()-t0:.0f}s)")
    print("NOTE: opponents are piloted by OUR policy, so they underplay their "
          "decks. Treat the ranking as signal, the absolute number as "
          "optimistic.")


if __name__ == "__main__":
    main()
