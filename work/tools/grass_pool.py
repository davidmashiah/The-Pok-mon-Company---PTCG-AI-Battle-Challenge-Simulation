"""Every Grass attacker in the pool, ranked by how well it answers Grimmsnarl.

The Decidueye lead in the handoff was found by a damage-per-energy sort and then
committed to. Before spending days on its Stage-2 consistency problem, enumerate
the whole Grass slice: the requirement is not "Decidueye", it is "one-shots a
320 HP Grass-weak attacker cheaply and survives the return hit". Anything Basic
or Stage 1 that clears 160 damage is strictly easier to assemble than a Stage 2.

Also prints the Marnie's line itself so the damage targets are read from the
engine rather than assumed.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import all_attack, all_card_data  # noqa: E402

ET = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D", 8: "M",
      9: "N", 10: "*", 11: "TR"}

cards = {c.cardId: c for c in all_card_data()}
atk = {a.attackId: a for a in all_attack()}


def stage_of(c):
    return "Basic" if c.basic else ("Stage1" if c.stage1 else
                                    ("Stage2" if c.stage2 else "?"))


def flags(c):
    return "".join(k for k, v in [("ex", c.ex), ("MEGA", c.megaEx),
                                  ("tera", c.tera)] if v)


print("=" * 100)
print("THE TARGETS: the Marnie's / Grimmsnarl line as the engine defines it")
print("=" * 100)
for c in sorted(cards.values(), key=lambda c: c.cardId):
    if int(c.cardType) != 0:
        continue
    if "Grimmsnarl" in c.name or "Impidimp" in c.name or "Morgrem" in c.name:
        wk = getattr(c, "weakness", None)
        print(f"  [{c.cardId}] {c.name:<30} {stage_of(c):<6} HP{c.hp:<4} "
              f"weakness={ET.get(int(wk), wk) if wk is not None else '?'} "
              f"retreat{c.retreatCost} <{flags(c)}> prizes~{2 if c.ex else 1}")
        for aid in c.attacks:
            a = atk.get(aid)
            if a:
                cost = "".join(ET.get(int(e), "?") for e in a.energies)
                print(f"        - {a.name:<22} [{cost or '-':<4}] dmg={a.damage}"
                      f"  {(a.text or '')[:80]}")

print()
print("=" * 100)
print("EVERY GRASS ATTACKER, sorted by (damage >= 160 first, then cost, then stage)")
print("  160 x2 weakness = 320 = exactly lethal on Marnie's Grimmsnarl ex")
print("=" * 100)
rows = []
for c in cards.values():
    if int(c.cardType) != 0 or int(c.energyType) != 1:
        continue
    for aid in c.attacks:
        a = atk.get(aid)
        if not a:
            continue
        rows.append((a.damage or 0, len(a.energies), c, a))

# stage rank: Basic easiest to assemble
srank = {"Basic": 0, "Stage1": 1, "Stage2": 2, "?": 3}
rows.sort(key=lambda r: (-(r[0] >= 160), r[1], srank[stage_of(r[2])], -r[0]))
for dmg, ne, c, a in rows:
    if dmg < 60:
        continue
    cost = "".join(ET.get(int(e), "?") for e in a.energies)
    ohko = "OHKO-320" if dmg * 2 >= 320 else ("OHKO-290" if dmg * 2 >= 290 else "")
    print(f"  {dmg:>4} [{cost or '-':<4}] {c.name:<30} {stage_of(c):<6} "
          f"HP{c.hp:<4} {flags(c):<5} {a.name[:22]:<22} {ohko:<9} "
          f"{(a.text or '')[:70]}")

print()
print("=" * 100)
print("GRASS SUPPORT: abilities on Grass Pokemon (accel / draw / search)")
print("=" * 100)
for c in sorted(cards.values(), key=lambda c: c.cardId):
    if int(c.cardType) != 0 or int(c.energyType) != 1:
        continue
    for sk in c.skills:
        print(f"  [{c.cardId}] {c.name:<26} {stage_of(c):<6} HP{c.hp:<4} "
              f"* {sk.name}: {(sk.text or '').replace(chr(10), ' ')[:110]}")
