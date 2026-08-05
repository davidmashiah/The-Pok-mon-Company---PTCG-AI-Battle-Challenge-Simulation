"""Inspect the card pool and the official sample deck."""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import all_attack, all_card_data  # noqa: E402

CT = {0: "POKEMON", 1: "ITEM", 2: "TOOL", 3: "SUPPORTER", 4: "STADIUM",
      5: "BASIC_ENERGY", 6: "SPECIAL_ENERGY"}
ET = {0: "Colorless", 1: "Grass", 2: "Fire", 3: "Water", 4: "Lightning",
      5: "Psychic", 6: "Fighting", 7: "Darkness", 8: "Metal", 9: "Dragon",
      10: "Rainbow", 11: "TeamRocket"}

cards = {c.cardId: c for c in all_card_data()}
atk = {a.attackId: a for a in all_attack()}


def show(cid, n=None):
    c = cards.get(cid)
    if c is None:
        return f"  {cid}: <unknown>"
    pre = f"  x{n} " if n else "  "
    s = f"{pre}[{cid}] {c.name} ({CT[int(c.cardType)]}"
    if int(c.cardType) == 0:
        stage = "Basic" if c.basic else ("Stage1" if c.stage1 else ("Stage2" if c.stage2 else "?"))
        flags = "".join(k for k, v in
                        [("ex", c.ex), ("MEGA", c.megaEx), ("tera", c.tera)] if v)
        s += f" {stage} {ET.get(int(c.energyType),'?')} HP{c.hp} retreat{c.retreatCost}"
        if flags:
            s += f" <{flags}>"
    if c.aceSpec:
        s += " <ACESPEC>"
    s += ")"
    if int(c.cardType) == 0:
        for aid in c.attacks:
            a = atk.get(aid)
            if a:
                cost = "".join(ET.get(int(e), "?")[0] for e in a.energies)
                s += f"\n        - {a.name} [{cost or '-'}] dmg={a.damage}"
                if a.text:
                    s += f" :: {a.text[:150]}"
    for sk in c.skills:
        s += f"\n        * {sk.name}: {(sk.text or '')[:150]}"
    return s


print("=" * 70)
print("SAMPLE DECK (official)")
print("=" * 70)
with open(os.path.join(WORK, "lib", "sample_deck.csv")) as f:
    deck = [int(x.strip()) for x in f if x.strip()][:60]
for cid, n in Counter(deck).most_common():
    print(show(cid, n))

print()
print("=" * 70)
print("POOL: highest-damage single attacks (dmg, cost<=3)")
print("=" * 70)
rows = []
for c in cards.values():
    if int(c.cardType) != 0:
        continue
    for aid in c.attacks:
        a = atk.get(aid)
        if a and a.damage:
            rows.append((a.damage, len(a.energies), c, a))
rows.sort(key=lambda r: (-r[0], r[1]))
seen = 0
for dmg, ne, c, a in rows:
    if ne > 3:
        continue
    stage = "Basic" if c.basic else ("S1" if c.stage1 else ("S2" if c.stage2 else "?"))
    cost = "".join(ET.get(int(e), "?")[0] for e in a.energies)
    print(f"  {dmg:>4} [{cost or '-':<3}] {c.name:<28} {stage:<5} HP{c.hp:<4} "
          f"{'ex' if c.ex else '':<4} {a.name[:26]:<26} {(a.text or '')[:60]}")
    seen += 1
    if seen >= 35:
        break

print()
print("=" * 70)
print("POOL: Basic Pokemon with best raw damage per energy (no evolution needed)")
print("=" * 70)
best = []
for c in cards.values():
    if int(c.cardType) != 0 or not c.basic:
        continue
    for aid in c.attacks:
        a = atk.get(aid)
        if a and a.damage and len(a.energies) <= 3:
            best.append((a.damage / max(1, len(a.energies)), a.damage, len(a.energies), c, a))
best.sort(key=lambda r: -r[0])
for eff, dmg, ne, c, a in best[:25]:
    cost = "".join(ET.get(int(e), "?")[0] for e in a.energies)
    print(f"  {eff:6.1f}/E  {dmg:>4} [{cost or '-':<3}] {c.name:<26} HP{c.hp:<4} "
          f"{'ex' if c.ex else '':<3} {a.name[:24]:<24} {(a.text or '')[:55]}")

print()
print("=" * 70)
print("POOL: Supporters (draw/search engine candidates)")
print("=" * 70)
sup = [c for c in cards.values() if int(c.cardType) == 3]
print(f"  {len(sup)} supporters")
for c in sorted(sup, key=lambda c: c.cardId)[:60]:
    t = " | ".join((s.text or "").replace("\n", " ") for s in c.skills)
    print(f"  [{c.cardId}] {c.name:<26} {t[:110]}")
