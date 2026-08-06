"""Deck variants derived from Alakazam's card text, not from random search.

The mechanism, read off the printed card rather than inferred:

    Alakazam (743)  "Place 2 damage counters on your opponent's Active Pokemon
                     for each card in YOUR hand."

Our damage is 20 x our own hand size. A Marnie's Grimmsnarl ex has 320 HP, so
one-shotting it needs SIXTEEN cards in hand, and 10 cards is 200 damage. That
deck's whole plan is hand disruption. It is not a bad matchup by accident -- it
attacks the exact quantity our damage is computed from, which is why we win it
0.207 while winning the mirror 0.934.

So the lever is hand SIZE, and the shipped list barely uses it:

    Lillie's Determination (1227)  1 copy   shuffle hand into deck, draw 6
                                            (8 if you still hold 6 prizes)
    Enriching Energy (13)          0 copies attach it, draw 4 -- and it is
                                            {C}, so it still powers an attack

Both are already scored by the policy (`lillie` 3400, `enriching_1st` 2000,
`enriching_2nd` 6249), so nothing here asks it to play a card it cannot reason
about. Enriching Energy is one of the ~28 policy-known cards the published deck
omits entirely.

What gets cut is the anti-energy tech, which does nothing against a deck whose
threat is a Stage 2 that is already powered: Enhanced Hammer discards a Special
Energy, and Grimmsnarl runs Darkness.

  python work/tools/build_hand_decks.py
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
BASE = "v61_codex_safe"

LILLIE, ENRICH, HAMMER, XEROSIC = 1227, 13, 1081, 1197
SACRED_ASH, LANA, NIGHT_STRETCHER = 1129, 1184, 1097

# (name, {card_id: delta}) -- deltas must net to zero
VARIANTS = [
    ("vG1_lillie", {
        # the minimal, most direct read of the card: more refills after Marnie
        LILLIE: +3, HAMMER: -3,
    }),
    ("vG2_lillie_enrich", {
        # add the draw-4 energy the policy already scores and the deck omits
        LILLIE: +3, ENRICH: +2, HAMMER: -3, SACRED_ASH: -1, XEROSIC: -1,
    }),
    ("vG3_handmax", {
        # push it: every cheap way to be holding more cards when we attack
        LILLIE: +3, ENRICH: +3, LANA: +2, NIGHT_STRETCHER: +1,
        HAMMER: -3, SACRED_ASH: -1, XEROSIC: -3, 1152: -2,
    }),
]


def load_base():
    p = os.path.join(AGENTS, BASE, "deck.csv")
    with open(p, encoding="utf-8") as fh:
        return [int(x) for x in fh.read().split() if x.strip()]


def build(name, deltas, base, cards):
    c = Counter(base)
    for cid, d in deltas.items():
        c[cid] = c.get(cid, 0) + d
        if c[cid] <= 0:
            c.pop(cid, None)
    deck = []
    for cid, n in c.items():
        deck.extend([cid] * n)
    deck.sort()

    problems = []
    if len(deck) != 60:
        problems.append(f"{len(deck)} cards, not 60")
    for cid, n in Counter(deck).items():
        card = cards.get(cid)
        if card is None:
            problems.append(f"unknown card {cid}")
            continue
        if int(getattr(card, "cardType", -1)) != 5 and n > 4:
            problems.append(f"{n}x {getattr(card,'name',cid)} exceeds 4")
    basics = sum(n for cid, n in Counter(deck).items()
                 if int(getattr(cards[cid], "cardType", -1)) == 0
                 and getattr(cards[cid], "basic", False))
    if basics < 6:
        problems.append(f"only {basics} Basic Pokemon")
    if problems:
        raise SystemExit(f"{name}: " + "; ".join(problems))

    out = os.path.join(AGENTS, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(AGENTS, BASE, "main.py"), encoding="utf-8") as fh:
        src = fh.read()
    # the base resolves its decklist bundle -> cwd -> inlined constant, so the
    # constant has to move with the file or the variant would quietly play the
    # base's 60 whenever deck.csv is not found from the cwd in use
    import re
    src = re.sub(r"^_CODEX_DECK = \[.*?\]$", "_CODEX_DECK = " + repr(deck),
                 src, count=1, flags=re.M | re.S)
    with open(os.path.join(out, "main.py"), "w", encoding="utf-8") as fh:
        fh.write(src)
    with open(os.path.join(out, "deck.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(map(str, deck)) + "\n")
    return deck


def main():
    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import all_card_data
    cards = {c.cardId: c for c in all_card_data()}
    base = load_base()
    bc = Counter(base)
    for name, deltas in VARIANTS:
        deck = build(name, deltas, base, cards)
        dc = Counter(deck)
        diff = "; ".join(
            f"{bc.get(cid,0)}->{dc.get(cid,0)} {getattr(cards.get(cid),'name',cid)}"
            for cid in sorted(set(bc) | set(dc)) if bc.get(cid, 0) != dc.get(cid, 0))
        print(f"{name}: {diff}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
