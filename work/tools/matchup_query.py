"""Query the matchup index: how does OUR deck really fare, and against whom?

Reads the index built by index_matchups.py, so every question is instant.
All numbers come from games between real players -- our own code never touches
the play, which is the whole point: locally we beat Grimmsnarl 94% because our
policy pilots it, while the ladder says our overall win rate is 61%.

Usage: python work/tools/matchup_query.py [our_marker_card_id]
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))
from cg.api import all_card_data  # noqa: E402

CARDS = {c.cardId: c.name for c in all_card_data()}
IDX = os.path.join(WORK, "out", "matchups.json")
OURS = int(sys.argv[1]) if len(sys.argv) > 1 else 678       # Mega Lucario ex


def wilson(w, n):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def name_of(ids):
    if not ids:
        return "(no ex line)"
    return " / ".join(sorted(CARDS.get(i, str(i)) for i in ids)[:2])


def main():
    if not os.path.exists(IDX):
        print("no index; run index_matchups.py first")
        return 2
    rows = json.load(open(IDX, encoding="utf-8"))
    print(f"indexed episodes: {len(rows)}\n")

    # overall archetype popularity and win rate, counting each SIDE once
    pop = Counter()
    wins = Counter()
    for r in rows:
        for side, key in ((0, "a"), (1, "b")):
            k = tuple(r[key])
            pop[k] += 1
            if r["w"] == side:
                wins[k] += 1
    print("FIELD (each deck appearance counted once):")
    print(f"{'archetype':<44}{'games':>7}{'winrate':>9}")
    for k, n in pop.most_common(10):
        p, lo, hi = wilson(wins[k], n)
        print(f"{name_of(k)[:43]:<44}{n:>7}{p:>9.3f}")

    # our deck's real matchup spread
    ours_rows = [r for r in rows if OURS in r["a"] or OURS in r["b"]]
    print(f"\nOUR DECK ({CARDS.get(OURS)}): {len(ours_rows)} games by real players")
    tot_w = 0
    vs = defaultdict(lambda: [0, 0])
    for r in ours_rows:
        me = 0 if OURS in r["a"] else 1
        opp = tuple(r["b"] if me == 0 else r["a"])
        won = int(r["w"] == me)
        tot_w += won
        vs[opp][0] += won
        vs[opp][1] += 1
    p, lo, hi = wilson(tot_w, len(ours_rows))
    print(f"  overall {p:.3f} [{lo:.3f},{hi:.3f}]  ({tot_w}/{len(ours_rows)})\n")
    print(f"  {'vs archetype':<44}{'n':>5}{'winrate':>9}{'  95% CI':>16}")
    for opp, (w, n) in sorted(vs.items(), key=lambda x: -x[1][1]):
        if n < 8:
            continue
        p, lo, hi = wilson(w, n)
        print(f"  {name_of(opp)[:43]:<44}{n:>5}{p:>9.3f}   [{lo:.3f},{hi:.3f}]")

    # THE question: how do we do against the deck that is half the field?
    GRIMM = 648
    g = [r for r in ours_rows
         if GRIMM in (r["b"] if OURS in r["a"] else r["a"])]
    gw = sum(1 for r in g if r["w"] == (0 if OURS in r["a"] else 1))
    if g:
        p, lo, hi = wilson(gw, len(g))
        print(f"\n>>> OUR DECK vs Marnie's Grimmsnarl ex (the 53%% matchup):")
        print(f"    {p:.3f} [{lo:.3f},{hi:.3f}]   ({gw}/{len(g)} real games)")
        print("    Locally we 'win' this 94.1% -- that number was our own weak")
        print("    policy piloting their deck. This one is real players.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
