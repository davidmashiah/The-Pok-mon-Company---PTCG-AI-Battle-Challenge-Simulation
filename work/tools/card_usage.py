"""Which cards in our deck actually get used, and which are dead weight?

Deck slots are the scarcest resource in the list. A card that is drawn but
almost never played is a slot that could hold something that is. This measures
per-card: how often it was drawn, and how often it was actually played /
attached / evolved / used.

Deliberately measured against the hard subset of opponents, because that is
where the games are decided.

Usage: python work/tools/card_usage.py <agent> <games>
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import all_card_data, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

CARDS = {c.cardId: c for c in all_card_data()}
CT = {0: "Pokemon", 1: "Item", 2: "Tool", 3: "Supporter", 4: "Stadium",
      5: "B.Energy", 6: "S.Energy"}

AG = sys.argv[1] if len(sys.argv) > 1 else "v13_noloop2"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60

full = os.path.join(WORK, "agents", AG)
sys.path.insert(0, full)
cwd = os.getcwd()
os.chdir(full)
env = {}
exec(compile(open("main.py", encoding="utf-8-sig").read(), "main.py", "exec"), env)
os.chdir(cwd)
fn = [v for v in env.values() if callable(v)][-1]
mine = list(env.get("DECK") or env.get("my_deck"))

store = json.load(open(os.path.join(WORK, "out", "meta_decks.json"),
                      encoding="utf-8"))
HARD = {1010.8, 1109.6, 1060.3, 1034.9, 1275.3, 1063.4, 1104.6}
opps, seen = [], set()
for t in sorted(store["teams"].values(), key=lambda t: -t.get("score", 0)):
    d = t.get("deck")
    if not d or len(d) != 60:
        continue
    k = tuple(sorted(d))
    if k in seen or round(t.get("score", 0), 1) not in HARD:
        continue
    seen.add(k)
    opps.append(d)

drawn = Counter()
used = Counter()
games = 0
# LogType: 4 DRAW, 10 PLAY, 11 ATTACH, 12 EVOLVE, 15 ATTACK
for g in range(N):
    opp = opps[g % len(opps)]
    first = (g % 2 == 0)
    d0, d1 = (mine, opp) if first else (opp, mine)
    me_idx = 0 if first else 1
    obs, _ = battle_start(list(d0), list(d1))
    if obs is None:
        continue
    games += 1
    try:
        for _ in range(4000):
            o = to_observation_class(obs)
            if o.current is not None and o.current.result != -1:
                break
            for lg in (o.logs or []):
                if lg.playerIndex != me_idx or lg.cardId is None:
                    continue
                t = int(lg.type)
                if t == 4:
                    drawn[lg.cardId] += 1
                elif t in (10, 11, 12):
                    used[lg.cardId] += 1
            who = o.current.yourIndex if o.current is not None else 0
            env["my_deck"] = list(d0 if who == 0 else d1)
            env["DECK"] = env["my_deck"]
            obs = battle_select(list(fn(obs)))
    except Exception:
        pass
    finally:
        battle_finish()

print(f"agent={AG}  {games} games vs {len(opps)} hard decks\n")
print(f"{'card':<30} {'n':>3} {'type':<9} {'drawn':>6} {'used':>6} {'used/drawn':>10}")
print("-" * 72)
counts = Counter(mine)
rows = []
for cid, n in counts.items():
    c = CARDS.get(cid)
    dr, us = drawn[cid], used[cid]
    rows.append((us / max(dr, 1), dr, us, cid, n, c))
for ratio, dr, us, cid, n, c in sorted(rows, key=lambda r: r[0]):
    print(f"{(c.name if c else cid)[:29]:<30} x{n:<2} "
          f"{CT.get(int(c.cardType), '?') if c else '?':<9} "
          f"{dr:>6} {us:>6} {ratio:>10.2f}")
print("\nlow ratio = drawn but rarely played -> candidate slots to reallocate")
