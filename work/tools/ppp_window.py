"""How often did the missing +30 actually cost us a KO? Count it, don't guess.

v14's attack planner never added Premium Power Pro's +30, so it underestimated
our damage whenever a copy was in hand. The question that decides whether the
fix is worth a submission slot is simply: how often does a position occur where
that 30 is the difference between "no KO" and "KO"?

The window is: opponent's ACTIVE has HP in (base_damage, base_damage + 30], we
hold at least one Premium Power Pro, and our attacker is the Fighting Pokemon
whose attack we would use. In that window the old planner scored the attack as
a fraction of a KO -- and, through the same `prize` variable, could score a
GAME-WINNING attack as non-lethal.

Measured on real games from the cached archive, so the positions are real.

Usage: python work/tools/ppp_window.py
"""
import json
import os
import sys
import zipfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))
from cg.api import EnergyType, all_attack, all_card_data  # noqa: E402

CARDS = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}
CACHE = os.path.join(WORK, "out", "games_678.zip")

MEGA_LUCARIO, HARIYAMA, PPP = 678, 674, 1141


def best_attack_dmg(cid):
    c = CARDS.get(cid)
    if not c:
        return 0
    best = 0
    for aid in (c.attacks or []):
        a = ATK.get(aid)
        if a and a.damage:
            best = max(best, a.damage)
    return best


def main():
    zf = zipfile.ZipFile(CACHE)
    files = [n for n in zf.namelist() if n.endswith(".json")]
    stat = Counter()
    windows = Counter()
    for f in files:
        d = json.loads(zf.open(f).read().decode("utf-8"))
        rw = d.get("rewards") or []
        if 1 not in rw:
            continue
        w = rw.index(1)
        for st in d.get("steps", []):
            if w >= len(st):
                continue
            ag = st[w]
            if ag.get("status") != "ACTIVE":
                continue
            obs = ag.get("observation") or {}
            cur = obs.get("current") or {}
            pls = cur.get("players") or []
            if len(pls) < 2:
                continue
            me, opp = pls[w if w < 2 else 0], pls[1 - (w if w < 2 else 0)]
            mine = (me.get("active") or [None])[0]
            theirs = (opp.get("active") or [None])[0]
            if not isinstance(mine, dict) or not isinstance(theirs, dict):
                continue
            hand = me.get("hand") or []
            n_ppp = sum(1 for c in hand if isinstance(c, dict) and c.get("id") == PPP)
            stat["decisions"] += 1
            if n_ppp == 0:
                continue
            stat["with_ppp_in_hand"] += 1
            mc = CARDS.get(mine.get("id"))
            if mc is None or int(mc.energyType) != int(EnergyType.FIGHTING):
                continue
            base = best_attack_dmg(mine.get("id"))
            if not base:
                continue
            hp = theirs.get("hp") or 0
            tc = CARDS.get(theirs.get("id"))
            # weakness doubling applies AFTER the +30
            wk = tc is not None and tc.weakness is not None and \
                int(tc.weakness) == int(EnergyType.FIGHTING)
            dmg_old = base * 2 if wk else base
            dmg_new = (base + 30) * 2 if wk else base + 30
            if dmg_old < hp <= dmg_new:
                stat["KO_ONLY_VISIBLE_WITH_PPP"] += 1
                windows[(mine.get("id"), theirs.get("id"), hp)] += 1

    print(f"real decisions scanned                 : {stat['decisions']}")
    print(f"  with a Premium Power Pro in hand     : {stat['with_ppp_in_hand']}")
    print(f"  where +30 flips NO-KO -> KO          : "
          f"{stat['KO_ONLY_VISIBLE_WITH_PPP']}")
    if stat["with_ppp_in_hand"]:
        print(f"  => {100*stat['KO_ONLY_VISIBLE_WITH_PPP']/stat['with_ppp_in_hand']:.2f}% "
              f"of PPP-in-hand decisions were mis-scored as 'no KO'")
    print("\nmost common such positions (attacker, target, target HP):")
    for (a, t, hp), n in windows.most_common(10):
        print(f"  {n:>4}x  {str(CARDS.get(a).name)[:20]:<20} -> "
              f"{str(CARDS.get(t).name)[:26]:<26} at {hp} HP")


if __name__ == "__main__":
    main()
