"""Find positions where a KO was available and our agent does not take it.

This is the audit that found the +81 Premium Power Pro bug, generalised. It is
not a win-rate proxy -- those are all disproven here -- it is an objective
correctness check: compute, with the real card rules, whether any attack we
could legally make this turn knocks out the opponent's Active, then ask what our
agent actually chose.

Damage model used (matching the printed cards):
  base attack damage
  + 30  per Premium Power Pro in hand, if the attacker is {F} and target is Active
        ("before applying Weakness and Resistance")
  x 2   if the target is weak to the attacker's type
  - 30  if it resists
against the target's CURRENT hp (the engine has already applied stadium effects
such as Gravity Mountain's -30 to Stage 2).

Every hit is a concrete, reproducible mistake with a prize attached, and each one
is worth more than another round of self-play measurement.

Usage: python work/tools/missed_ko.py <agent> [limit_games]
"""
import json
import os
import sys
import zipfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
LIB = os.path.join(WORK, "lib")
sys.path.insert(0, LIB)
from cg.api import (  # noqa: E402
    EnergyType, OptionType, all_attack, all_card_data, to_observation_class,
)

CARDS = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}
CACHE = os.path.join(WORK, "out", "games_678.zip")
PPP = 1141

AGENT = sys.argv[1] if len(sys.argv) > 1 else "v32_ppp"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 108


def load_agent(name):
    ag = os.path.join(WORK, "agents", name)
    for p in (ag, LIB):
        if p not in sys.path:
            sys.path.insert(0, p)
    root = os.path.dirname(WORK)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        with open(os.path.join(ag, "main.py"), encoding="utf-8") as fh:
            src = fh.read()
        env = {}
        exec(compile(src, "main.py", "exec"), env)
    finally:
        os.chdir(cwd)
    return [v for k, v in env.items() if callable(v)][-1]


def opts_of(obs):
    return ((obs.get("select") or {}).get("option") or [])


def killing_attacks(obs, me):
    """-> set of option indices whose attack KOs the opponent's Active."""
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    cur = obs.get("current") or {}
    pls = cur.get("players") or []
    if len(pls) < 2:
        return set(), None
    mine, opp = pls[me], pls[1 - me]
    act = (mine.get("active") or [None])[0]
    tgt = (opp.get("active") or [None])[0]
    if not isinstance(act, dict) or not isinstance(tgt, dict):
        return set(), None
    ac = CARDS.get(act.get("id"))
    tc = CARDS.get(tgt.get("id"))
    if ac is None or tc is None:
        return set(), None
    hand = mine.get("hand") or []
    n_ppp = sum(1 for c in hand if isinstance(c, dict) and c.get("id") == PPP)
    hp = tgt.get("hp") or 0
    out = set()
    for i, o in enumerate(opts):
        if o.get("type") != int(OptionType.ATTACK):
            continue
        a = ATK.get(o.get("attackId"))
        if a is None or not a.damage:
            continue
        # not enough energy attached -> the engine would not offer it, but be safe
        if len(act.get("energies") or []) < len(a.energies or []):
            continue
        dmg = a.damage
        if n_ppp > 0 and int(ac.energyType) == int(EnergyType.FIGHTING):
            dmg += 30
        if tc.weakness is not None and int(tc.weakness) == int(ac.energyType):
            dmg *= 2
        elif tc.resistance is not None and int(tc.resistance) == int(ac.energyType):
            dmg -= 30
        if dmg >= hp:
            out.add(i)
    return out, (str(ac.name), str(tc.name), hp)


def main():
    fn = load_agent(AGENT)
    zf = zipfile.ZipFile(CACHE)
    files = [n for n in zf.namelist() if n.endswith(".json")][:LIMIT]
    n_ko_avail = n_taken = n_ended_with_ko = 0
    misses = Counter()
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
            if not (obs.get("select") or {}).get("option"):
                continue
            kills, info = killing_attacks(obs, w)
            if not kills:
                continue
            n_ko_avail += 1
            try:
                ours = fn(obs)
            except Exception:
                continue
            if ours and ours[0] in kills:
                n_taken += 1
            elif ours and info:
                # Attacking ENDS THE TURN, so declining a KO to attach energy or
                # play a supporter first is correct -- the agent may take the same
                # KO later that turn, which this replay cannot follow. The only
                # unambiguous mistake is ENDING THE TURN with the KO still there.
                o = opts_of(obs)
                pick = o[ours[0]] if 0 <= ours[0] < len(o) else None
                if pick is not None and pick.get("type") == int(OptionType.END):
                    misses[info] += 1
                    n_ended_with_ko += 1

    print(f"agent: {AGENT}")
    print(f"positions with a KO on the opponent's Active available : {n_ko_avail}")
    print(f"  our agent takes it                                   : {n_taken}")
    print(f"  MISSED                                               : {n_ko_avail-n_taken}")
    if n_ko_avail:
        print(f"  -> takes {100*n_taken/n_ko_avail:.1f}% of available knockouts")
    print("\nmost common misses (attacker -> target @ HP):")
    for (a, t, hp), c in misses.most_common(12):
        print(f"  {c:>4}x  {a[:22]:<22} -> {t[:26]:<26} at {hp} HP")


if __name__ == "__main__":
    main()
