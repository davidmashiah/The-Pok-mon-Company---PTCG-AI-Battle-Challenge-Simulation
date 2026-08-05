"""Compare the #1 agent's tempo with ours, from replay logs.

We and the rank-1 team play the SAME archetype (Mega Lucario ex), so the ~325
point gap is policy quality, not deck choice. Replays expose their whole action
sequence, so we can measure concretely where they are faster.

Metrics per player:
  * turn Mega Lucario ex first reaches the field
  * turn of first attack, and first Mega Brave
  * energy attached by turn 3
  * how often Hero's Cape is attached, and to what
"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))
from cg.api import all_card_data  # noqa: E402

CARDS = {c.cardId: c for c in all_card_data()}
RIOLU, MEGA_LUC, HERO_CAPE, F_ENERGY = 677, 678, 1159, 6


def analyse(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    names = d["info"]["TeamNames"]
    rewards = d.get("rewards") or [0, 0]
    out = {}
    for pi in (0, 1):
        st = {"mega_turn": None, "first_attack_turn": None,
              "mega_brave_turn": None, "energy_by_t3": 0,
              "cape_on": Counter(), "attacks": 0, "evolves": 0}
        out[pi] = st

    turn = 0
    for step in d["steps"]:
        for ai, ag in enumerate(step):
            obs = ag.get("observation") or {}
            cur = obs.get("current") or {}
            if cur.get("turn"):
                turn = max(turn, cur["turn"])
            for lg in (obs.get("logs") or []):
                t = lg.get("type")
                pl = lg.get("playerIndex")
                if pl is None or pl not in out:
                    continue
                s = out[pl]
                # 12 = EVOLVE, 11 = ATTACH, 15 = ATTACK
                if t == 12:
                    s["evolves"] += 1
                    if lg.get("cardId") == MEGA_LUC and s["mega_turn"] is None:
                        s["mega_turn"] = turn
                elif t == 11:
                    cid = lg.get("cardId")
                    if cid == F_ENERGY and turn <= 6:
                        s["energy_by_t3"] += 1
                    if cid == HERO_CAPE:
                        tgt = lg.get("cardIdTarget")
                        s["cape_on"][CARDS[tgt].name if tgt in CARDS
                                     else str(tgt)] += 1
                elif t == 15:
                    s["attacks"] += 1
                    if s["first_attack_turn"] is None:
                        s["first_attack_turn"] = turn
                    if lg.get("cardId") == MEGA_LUC and s["mega_brave_turn"] is None:
                        s["mega_brave_turn"] = turn
    return names, rewards, out, turn


def main():
    rd = os.path.join(WORK, "out", "replays")
    files = [os.path.join(rd, f) for f in os.listdir(rd)
             if f.endswith("-replay.json")]
    if not files:
        raise SystemExit("no replays in work/out/replays")
    agg = defaultdict(list)
    for p in files:
        names, rewards, out, turns = analyse(p)
        print(f"\n=== {os.path.basename(p)}  ({turns} turns) ===")
        for pi in (0, 1):
            s = out[pi]
            res = "WON " if rewards[pi] == 1 else ("lost" if rewards[pi] == -1
                                                   else "draw")
            print(f"  [{res}] {names[pi][:26]:<26} "
                  f"MegaLucario@turn={s['mega_turn']} "
                  f"1st_attack@turn={s['first_attack_turn']} "
                  f"MegaBrave@turn={s['mega_brave_turn']} "
                  f"energy_by_t6={s['energy_by_t3']} "
                  f"attacks={s['attacks']} evolves={s['evolves']}")
            if s["cape_on"]:
                print(f"       Hero's Cape -> {dict(s['cape_on'])}")
            if s["mega_turn"]:
                agg["mega_turn"].append(s["mega_turn"])
            if s["first_attack_turn"]:
                agg["first_attack"].append(s["first_attack_turn"])
    print("\n=== aggregate over replays ===")
    for k, v in agg.items():
        v.sort()
        print(f"  {k}: n={len(v)} min={v[0]} median={v[len(v)//2]} max={v[-1]}")


if __name__ == "__main__":
    main()
