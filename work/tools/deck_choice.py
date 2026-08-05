"""Which archetype should we actually play, against the field we actually face?

Two inputs, deliberately from different sources:

  win rates   `work/out/matchups.json` -- 13,444 REAL episodes between real
              players. Our code never touches the play, so these are not
              contaminated by our own policy piloting someone else's deck.

  weights     the archetype shares measured from OUR OWN ladder replays
              (`loss_autopsy.py`). This is the correction that matters: the
              whole-ladder distribution is ~53% Marnie's Grimmsnarl, but our
              matchmaking band is 23.5% mirror / 18.8% Alakazam / 15.4% Crustle
              / 9.4% Archaludon / 7.4% Grimmsnarl. Every deck decision before
              this was weighted by the wrong population.

Reports, for each candidate archetype, its win rate against each archetype we
actually meet, and the share-weighted expectation. Also reports the same figure
weighted by the WHOLE ladder, because as our rating climbs the band we are
matched into drifts toward that distribution.

Usage: python work/tools/deck_choice.py
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

# signature card -> archetype. Order matters: first hit wins.
ARCH = [
    (648, "Marnie's Grimmsnarl ex"),
    (96, "Teal Mask Ogerpon ex"),
    (756, "Mega Kangaskhan ex"),
    (849, "Mega Lopunny ex"),
    (1071, "Meowth ex"),
    (272, "Lillie's Clefairy ex"),
    (117, "Cornerstone Ogerpon ex"),
    (184, "Latias ex"),
    (121, "Dragapult ex"),
    (381, "Cynthia's Garchomp ex"),
    (431, "TR Mewtwo ex"),
    (861, "Mega Froslass ex"),
    (678, "Mega Lucario ex"),
    (140, "Fezandipiti ex"),
    (306, "Dudunsparce ex"),
    (63, "Raging Bolt ex"),
    (108, "Wellspring Ogerpon ex"),
    (326, "Blaziken ex"),
    (652, "Mega Venusaur ex"),
]

# what WE actually face, measured from our own replays (149 games)
# The index names a deck by its ex / Mega-ex Pokemon, so decks whose engine is a
# plain Stage 2 (Alakazam, Crustle, Cinderace) are invisible here -- they show up
# under whatever ex they splash, usually Fezandipiti ex or Dudunsparce ex. That
# means this file answers "which deck beats the LADDER", and our own replay
# autopsy answers "what our current band plays"; the two must be read together.
OUR_BAND = {
    "Marnie's Grimmsnarl ex": 0.30,
    "Fezandipiti ex": 0.20,
    "Mega Lopunny ex": 0.15,
    "Teal Mask Ogerpon ex": 0.12,
    "Mega Kangaskhan ex": 0.10,
    "Dragapult ex": 0.08,
    "Mega Lucario ex": 0.05,
}


def classify(cards):
    s = set(cards)
    for cid, name in ARCH:
        if cid in s:
            return name
    return None


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    eps = json.load(open(IDX, encoding="utf-8"))
    # wins[A][B] = [wins by A, games]
    rec = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    seen = Counter()
    for e in eps:
        a, b = classify(e.get("a") or []), classify(e.get("b") or [])
        if a is None or b is None or a == b:
            continue
        w = e.get("w")
        if w not in (0, 1):
            continue
        seen[a] += 1
        seen[b] += 1
        rec[a][b][1] += 1
        rec[b][a][1] += 1
        if w == 0:
            rec[a][b][0] += 1
        else:
            rec[b][a][0] += 1

    ladder_total = sum(seen.values())
    ladder_share = {k: v / ladder_total for k, v in seen.items()}

    cands = [k for k, _ in sorted(seen.items(), key=lambda kv: -kv[1])]
    print(f"archetypes seen in {len(eps)} real episodes: "
          + ", ".join(f"{k} {seen[k]}" for k in cands))

    print(f"\n{'candidate deck':<20} " + " ".join(f"{k[:9]:>10}" for k in OUR_BAND)
          + f" {'OUR band':>10} {'whole LB':>10}")
    print("-" * 128)
    rows = []
    for a in cands:
        cells = []
        num = den = 0.0
        for b, share in OUR_BAND.items():
            w, n = rec[a][b]
            if n < 12:
                cells.append(f"{'-':>10}")
                continue
            p = w / n
            cells.append(f"{p:>7.3f}/{n:<3}"[:10].rjust(10))
            num += share * p
            den += share
        ours = num / den if den > 0.5 else None
        lnum = lden = 0.0
        for b, share in ladder_share.items():
            w, n = rec[a][b]
            if n >= 12:
                lnum += share * (w / n)
                lden += share
        lb = lnum / lden if lden > 0.4 else None
        rows.append((ours or -1, a, cells, ours, lb))
    for _, a, cells, ours, lb in sorted(rows, reverse=True):
        print(f"{a:<20} " + " ".join(cells)
              + f" {(f'{ours:.3f}' if ours else 'n/a'):>10}"
              + f" {(f'{lb:.3f}' if lb else 'n/a'):>10}")
    print("\ncells are winrate/games for the ROW deck against the COLUMN deck.")
    print("'-' means fewer than 12 real games, which decides nothing.")
    print("'OUR band' weights by the archetypes we actually meet; 'whole LB' by")
    print("the full ladder, which is where we drift as the rating climbs.")


if __name__ == "__main__":
    main()
