"""Which card IDs identify each panel archetype on sight?

A router has to name its opponent from the board before it can pick a pilot, and
it gets one look: whatever Pokemon they have revealed. So the useful set is not
"every card in their deck" -- it is the Pokemon that appear in EXACTLY ONE of
the panel's decks. A card two archetypes share tells the router nothing.

Prints, per opponent, its Pokemon and which of them are unique to it.

  python work/tools/archetype_ids.py
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import all_card_data  # noqa: E402

CARDS = {c.cardId: c for c in all_card_data()}
PANEL = ["w5_grimmsnarl", "w1_alakazam", "p3_crustle", "s_dragapult",
         "z_roman950", "w2_archaludon", "x_lopunny"]


def deck(name):
    p = os.path.join(AGENTS, name, "deck.csv")
    if not os.path.exists(p):
        return []
    return [int(x) for x in open(p, encoding="utf-8-sig").read().split()
            if x.strip()]


def is_mon(cid):
    c = CARDS.get(cid)
    return c is not None and getattr(c, "hp", 0)


def main():
    mons = {}
    for name in PANEL:
        d = deck(name)
        mons[name] = sorted({c for c in d if is_mon(c)})

    owner = defaultdict(set)
    for name, ms in mons.items():
        for m in ms:
            owner[m].add(name)

    print("Pokemon per panel deck; * = unique to that deck (a usable tell)\n")
    for name in PANEL:
        if not mons.get(name):
            print(f"{name}: (no deck)")
            continue
        print(name)
        uniq = []
        for m in mons[name]:
            c = CARDS[m]
            mark = "*" if len(owner[m]) == 1 else " "
            if mark == "*":
                uniq.append(m)
            print(f"   {mark} {m:5d}  {str(c.name)[:28]:28s} hp={c.hp}")
        print(f"   unique ids: {uniq}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
