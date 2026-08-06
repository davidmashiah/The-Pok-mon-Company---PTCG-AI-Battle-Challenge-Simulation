"""Full engine record for named cards -- attacks, costs, abilities, evolution.

Used to check assembly cost before committing to a build. The Decidueye lead was
picked off a damage-per-energy sort and its real problem turned out to be the
evolution chain, which that sort does not show.

  python work/tools/card_detail.py 75 96 652 481 322 924 920 1022 150
  python work/tools/card_detail.py --name Ogerpon Venusaur
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import all_attack, all_card_data  # noqa: E402

ET = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D", 8: "M",
      9: "N", 10: "*", 11: "TR"}
CT = {0: "POKEMON", 1: "ITEM", 2: "TOOL", 3: "SUPPORTER", 4: "STADIUM",
      5: "BASIC_ENERGY", 6: "SPECIAL_ENERGY"}

cards = {c.cardId: c for c in all_card_data()}
atk = {a.attackId: a for a in all_attack()}


def dump(c):
    print("=" * 96)
    line = f"[{c.cardId}] {c.name}  ({CT[int(c.cardType)]})"
    print(line)
    if int(c.cardType) == 0:
        stage = ("Basic" if c.basic else "Stage1" if c.stage1
                 else "Stage2" if c.stage2 else "?")
        fl = "".join(k for k, v in [("ex", c.ex), ("MEGA", c.megaEx),
                                    ("tera", c.tera)] if v)
        print(f"    {stage}  type={ET.get(int(c.energyType), '?')}  HP{c.hp}  "
              f"retreat{c.retreatCost}  weakness={ET.get(int(c.weakness), c.weakness)}"
              f"  flags={fl or '-'}")
        # every field the engine exposes, so evolution wiring is read not guessed
        for f in ("evolveFrom", "evolvesFrom", "preEvolution", "parentId",
                  "megaFrom", "baseName", "aceSpec", "ruleBox"):
            v = getattr(c, f, None)
            if v not in (None, "", 0, False):
                print(f"    {f} = {v!r}")
        for aid in c.attacks:
            a = atk.get(aid)
            if not a:
                continue
            cost = "".join(ET.get(int(e), "?") for e in a.energies)
            print(f"    ATTACK {a.name}  [{cost or '-'}] ({len(a.energies)}E)"
                  f"  dmg={a.damage}")
            if a.text:
                print(f"        {a.text}")
    for sk in c.skills:
        print(f"    ABILITY {sk.name}")
        if sk.text:
            print(f"        {sk.text}")


ap = argparse.ArgumentParser()
ap.add_argument("ids", nargs="*")
ap.add_argument("--name", nargs="*", default=[])
a = ap.parse_args()

for i in a.ids:
    c = cards.get(int(i))
    if c is None:
        print(f"[{i}] <unknown>")
    else:
        dump(c)
for pat in a.name:
    for c in sorted(cards.values(), key=lambda c: c.cardId):
        if pat.lower() in c.name.lower():
            dump(c)
