"""Resolve a deck.csv against OUR engine and say whether the engine accepts it.

yu0307 (ladder rank 11) published 16 real City League top-cut lists already in
this competition's card-id numbering. Two things must be checked before any of
them is worth building a pilot for, and neither is safe to assume:
  1. the ids resolve to the cards the filename claims in the engine WE run
  2. battle_start actually accepts the 60 (Enriching Energy was rejected
     outright once, after a build had already been written around it)

Also reports the anti-Grimmsnarl question directly: how much damage the deck's
best attacker does to a 320 HP Grass-weak Stage 2, and what it costs.

  python work/tools/deck_report.py work/mined/yu0307_decks/*.csv
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import all_attack, all_card_data  # noqa: E402
from cg.game import battle_finish, battle_start  # noqa: E402

ET = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D", 8: "M",
      9: "N", 10: "*", 11: "TR"}
cards = {c.cardId: c for c in all_card_data()}
atk = {a.attackId: a for a in all_attack()}

GRIMM_HP = 320          # Marnie's Grimmsnarl ex
GRIMM_WEAK = 1          # Grass


def report(path):
    ids = [int(x) for x in open(path, encoding="utf-8-sig").read().split()
           if x.strip()]
    name = os.path.basename(path)
    print("=" * 96)
    print(f"{name}   {len(ids)} cards, {len(set(ids))} unique")
    if len(ids) != 60:
        print("  NOT 60 CARDS -- unusable as-is")
        return

    unknown = [i for i in ids if i not in cards]
    if unknown:
        print(f"  unknown card ids: {sorted(set(unknown))}")

    obs, _ = battle_start(list(ids), list(ids))
    ok = obs is not None
    battle_finish()
    print(f"  engine accepts: {'YES' if ok else 'NO -- REJECTED'}")

    best = None
    for cid, n in Counter(ids).items():
        c = cards.get(cid)
        if c is None or int(c.cardType) != 0:
            continue
        for aid in c.attacks:
            a = atk.get(aid)
            if not a or not a.damage:
                continue
            eff = a.damage * (2 if int(c.energyType) == GRIMM_WEAK else 1)
            if best is None or (eff, -len(a.energies)) > (best[0], -best[1]):
                best = (eff, len(a.energies), c, a, n)
    if best:
        eff, ne, c, a, n = best
        stage = ("Basic" if c.basic else "St1" if c.stage1
                 else "St2" if c.stage2 else "?")
        cost = "".join(ET.get(int(e), "?") for e in a.energies)
        print(f"  best vs Grimmsnarl: {eff} dmg  ({a.damage} base"
              f"{' x2 GRASS' if int(c.energyType) == GRIMM_WEAK else ''})"
              f"  [{cost}] {ne}E  x{n} {c.name} {stage} HP{c.hp}"
              f"  -> {'ONE-SHOTS 320' if eff >= GRIMM_HP else 'no OHKO'}")

    pk = [(n, cards[i].name) for i, n in Counter(ids).items()
          if i in cards and int(cards[i].cardType) == 0]
    print("  pokemon: " + ", ".join(f"{n}x {nm}" for n, nm in
                                    sorted(pk, key=lambda t: -t[0])))


for p in sys.argv[1:]:
    report(p)
