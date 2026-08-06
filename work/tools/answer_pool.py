"""What in the WHOLE pool answers a 320 HP Stage-2 ex for cheap?

The Grass survey answered "what Grass card one-shots Grimmsnarl". This asks the
question one level up, because the constraint that actually binds is energy and
evolution cost, not type: an answer we can splash into a deck that already runs
Darkness energy is worth far more than one that needs its own energy line.

Sections:
  1. cheap OHKO -- >=160 damage for <=2 Energy (x2 weakness clears 320)
  2. colorless-only costs -- playable in ANY deck's energy line
  3. raw >=300 -- one-shots without needing weakness at all
  4. damage boosters -- turn a 140 into a 160
  5. conditional bonuses vs ex -- the opponent's whole field is 2-prize ex mons
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import all_attack, all_card_data  # noqa: E402

ET = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D", 8: "M",
      9: "N", 10: "*", 11: "TR"}
cards = {c.cardId: c for c in all_card_data()}
atk = {a.attackId: a for a in all_attack()}
mons = [c for c in cards.values() if int(c.cardType) == 0]


def stage(c):
    return "Basic" if c.basic else ("St1" if c.stage1 else
                                    ("St2" if c.stage2 else "?"))


def prizes(c):
    return 3 if c.megaEx else (2 if c.ex else 1)


def row(c, a):
    cost = "".join(ET.get(int(e), "?") for e in a.energies)
    fl = "".join(k for k, v in [("ex", c.ex), ("M", c.megaEx),
                                ("T", c.tera)] if v)
    return (f"  {a.damage or 0:>4} [{cost or '-':<4}] {c.name:<28} {stage(c):<5} "
            f"HP{c.hp:<4} {ET.get(int(c.energyType),'?'):<2} {fl:<3} "
            f"{prizes(c)}pz  {a.name[:20]:<20} {(a.text or '')[:66]}")


def pairs():
    for c in mons:
        for aid in c.attacks:
            a = atk.get(aid)
            if a:
                yield c, a


print("=" * 118)
print("1. CHEAP OHKO: >=160 damage for <=2 Energy  (x2 weakness clears 320 HP)")
print("=" * 118)
for c, a in sorted(pairs(), key=lambda p: (len(p[1].energies), -(p[1].damage or 0))):
    if (a.damage or 0) >= 160 and len(a.energies) <= 2:
        print(row(c, a))

print()
print("=" * 118)
print("2. COLORLESS-ONLY COST, >=90 base  (splashable into any energy line)")
print("=" * 118)
for c, a in sorted(pairs(), key=lambda p: (len(p[1].energies), -(p[1].damage or 0))):
    if not a.energies or (a.damage or 0) < 90:
        continue
    if all(int(e) == 0 for e in a.energies) and len(a.energies) <= 3:
        print(row(c, a))

print()
print("=" * 118)
print("3. RAW >=300 for <=4 Energy  (no weakness needed)")
print("=" * 118)
for c, a in sorted(pairs(), key=lambda p: -(p[1].damage or 0)):
    if (a.damage or 0) >= 300 and len(a.energies) <= 4:
        print(row(c, a))

print()
print("=" * 118)
print("4. DAMAGE BOOSTERS (abilities that add damage before weakness)")
print("=" * 118)
for c in sorted(mons, key=lambda c: c.cardId):
    for sk in c.skills:
        t = (sk.text or "").replace("\n", " ")
        if re.search(r"do(es)? \d+ more damage", t):
            print(f"  [{c.cardId}] {c.name:<26} {stage(c):<5} HP{c.hp:<4} "
                  f"{ET.get(int(c.energyType),'?'):<2} {prizes(c)}pz * {sk.name}: {t[:90]}")

print()
print("=" * 118)
print("5. ATTACKS WITH A BONUS VS POKEMON ex  (the whole top field is 2-prize ex)")
print("=" * 118)
for c, a in sorted(pairs(), key=lambda p: len(p[1].energies)):
    t = (a.text or "").replace("\n", " ")
    if "{ex}" in t or "Pokémon ex" in t or "Pokemon ex" in t:
        print(row(c, a))

print()
print("=" * 118)
print("6. TOOLS / STADIUMS that change the damage math")
print("=" * 118)
for c in sorted(cards.values(), key=lambda c: c.cardId):
    if int(c.cardType) not in (2, 4):
        continue
    t = " | ".join((s.text or "").replace("\n", " ") for s in c.skills)
    if re.search(r"more damage|less damage|HP|Knocked Out", t):
        print(f"  [{c.cardId}] {'TOOL ' if int(c.cardType)==2 else 'STAD '}"
              f"{c.name:<26} {t[:96]}")
