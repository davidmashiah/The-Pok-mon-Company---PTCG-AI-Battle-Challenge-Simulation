"""w40_cape = the hand-written Grimmsnarl policy + 3 Hero's Cape.

The one deck edge available to us, and the arithmetic behind it:

    Marnie's Grimmsnarl ex   320 HP        Shadow Bullet 180
      320 HP -> dies to 2 Shadow Bullets
      420 HP -> dies to 3

Hero's Cape is +100 HP, verified in this engine over 253 observations
(work/tools/cape_check.py), not taken from card text. In the mirror -- 16 of the
top 50 teams play this exact deck -- an extra hit on our attacker is an extra
turn of attacking, on both sides of the prize race.

Why this base and not the one we ship. w8_grimm_tuned CANNOT take a new card:
its main.py asserts a fixed 60, and its learned policy scores options through a
closed 180-entry intent vocabulary keyed to card ids, so a card outside the deck
has no representation to be scored with. The hand-written v26 sub-policy is the
same deck at the same strength -- 0.5269 vs w5_grimmsnarl against w8's 0.5297,
and 0.6329 field against 0.6376 -- but it is plain Python with explicit card
ids, so it can hold a card the model has never seen.

The guard runs AFTER the policy and never overrides an attack: attacking ends
the turn, so pre-empting a lethal swing to put on a coat would be strictly
worse. It only ever fires on a Grimmsnarl ex with no tool yet.

  python work/tools/build_cape.py --name w40_cape --capes 3
"""
import argparse
import os
import shutil
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
BASE = "_sub_handwritten_v26"

CAPE = 1159
# Hero's Cape is an ACE SPEC and this format allows exactly ONE ACE SPEC per
# deck. The engine rejects the 60 outright otherwise -- it refused even a single
# added copy, which is how the rule was found. So this is not a 3-card package,
# it is a 1-for-1 trade against the ACE SPEC we already run:
#
#   Unfair Stamp  -- after one of our Pokemon is KO'd, both players shuffle
#                    their hand away; we draw 5, they draw 2.
#   Hero's Cape   -- +100 HP, turning our attacker from a 2-hit body into a
#                    3-hit one in the 32%-of-the-field mirror.
#
# 4x Team Rocket's Petrel searches ANY Trainer, so a 1-of is genuinely findable
# rather than a card we draw half the time.
ACE_SPEC_IN_DECK = 1080            # Unfair Stamp
CUTS = [ACE_SPEC_IN_DECK]

GUARD = r'''"""Put the coat on the 320 HP body, and never instead of attacking.

Hero's Cape is offered as OptionType.ATTACH (8) from the hand, with
inPlayArea/inPlayIndex naming the target -- one option per legal target, so a
single choice both plays the card and picks who wears it. Confirmed against a
real game rather than assumed (work/tools/cape_check.py).
"""
ATTACH = 8
ATTACK = 13
CAPE = 1159
GRIMMSNARL = 648
ACTIVE, BENCH = 4, 5


def _opt_type(o):
    try:
        return int(o.get("type", -1) if o.get("type") is not None else -1)
    except Exception:
        return -1


def choose(obs, base):
    """Return a replacement selection, or None to keep the policy's own."""
    sel = obs.get("select") or {}
    try:
        if int(sel.get("context", -1) if sel.get("context") is not None
               else -1) != 0:
            return None                      # MAIN phase only
    except Exception:
        return None
    opts = sel.get("option") or []
    if not opts:
        return None
    try:
        lo = int(sel.get("minCount", 0) or 0)
        hi = int(sel.get("maxCount", 0) or 0)
    except Exception:
        return None
    if lo > 1 or hi < 1:
        return None                          # this guard only makes 1-picks

    # Never pre-empt an attack. Attacking ends the turn, so the policy putting
    # a Cape on ahead of a knockout would cost the prize it was buying time for.
    for i in (base or []):
        if 0 <= i < len(opts) and _opt_type(opts[i]) == ATTACK:
            return None

    cur = obs.get("current") or {}
    try:
        me = (cur.get("players") or [])[int(cur.get("yourIndex", 0) or 0)]
    except Exception:
        return None
    hand = me.get("hand") or []

    best, best_rank = None, 0
    for i, o in enumerate(opts):
        if _opt_type(o) != ATTACH:
            continue
        try:
            ix = int(o.get("index", -1))
            if not (0 <= ix < len(hand)):
                continue
            if int((hand[ix] or {}).get("id", 0) or 0) != CAPE:
                continue
            area = int(o.get("inPlayArea", 0) or 0)
            idx = int(o.get("inPlayIndex", -1))
            zone = me.get("active") if area == ACTIVE else me.get("bench")
            if not zone or not (0 <= idx < len(zone)):
                continue
            mon = zone[idx]
            if not mon:
                continue
            if mon.get("tools"):
                continue                     # one coat per body
            if int(mon.get("id", 0) or 0) != GRIMMSNARL:
                continue                     # only the 320 -> 420 breakpoint
        except Exception:
            continue
        rank = 2 if area == ACTIVE else 1
        if rank > best_rank:
            best, best_rank = i, rank
    return [best] if best is not None else None
'''

MAIN = r'''import policy_features as pf
from policies.handwritten_v26 import main as _sub
import cape_guard
DECK=list(pf.DECK)
def _legal(a,s,n):
    try:
        lo=int(s.get('minCount',0) or 0); hi=int(s.get('maxCount',0) or 0)
        return (lo<=len(a)<=max(hi,lo) and len(a)==len(set(a))
                and all(isinstance(i,int) and 0<=i<n for i in a))
    except Exception:
        return False
def agent(obs):
    if not obs or obs.get('select') is None:
        try: _sub.agent({})
        except Exception: pass
        return list(DECK)
    s=obs.get('select') or {}
    n=len(s.get('option') or [])
    try:
        base=list(_sub.agent(obs))
    except Exception:
        lo=int(s.get('minCount',0) or 0); hi=min(n,int(s.get('maxCount',0) or 0))
        base=list(range(max(lo,hi)))
    try:
        ov=cape_guard.choose(obs,base)
    except Exception:
        ov=None
    if ov is not None and _legal(ov,s,n):
        return ov
    return base
def w40_cape_entry(obs):
    return agent(obs)
'''


def rewrite_deck(path, deck, varname="DECK"):
    """Replace the DECK literal in a file, asserting it was actually there."""
    text = open(path, encoding="utf-8").read()
    start = text.find(f"{varname}=[")
    style = f"{varname}=["
    if start < 0:
        start = text.find(f"{varname} = [")
        style = f"{varname} = ["
    if start < 0:
        raise SystemExit(f"no {varname} literal in {path}")
    end = text.find("]", start)
    if end < 0:
        raise SystemExit(f"unterminated {varname} literal in {path}")
    new = style + ", ".join(str(x) for x in deck) + "]"
    text = text[:start] + new + text[end + 1:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="w40_cape")
    ap.add_argument("--capes", type=int, default=3)
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.game import battle_finish, battle_start

    src = os.path.join(AGENTS, BASE)
    dst = os.path.join(AGENTS, a.name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))

    base_deck = [int(x) for x in
                 open(os.path.join(src, "deck.csv"), encoding="utf-8")
                 .read().split() if x.strip()]
    if len(base_deck) != 60:
        raise SystemExit(f"base deck is {len(base_deck)} cards")

    deck = list(base_deck)
    for k in range(a.capes):
        victim = CUTS[k % len(CUTS)]
        if victim not in deck:
            raise SystemExit(f"cannot cut {victim}: not in the deck")
        deck.remove(victim)
        deck.append(CAPE)
    deck.sort()
    if len(deck) != 60:
        raise SystemExit(f"deck ended at {len(deck)} cards")
    if deck.count(CAPE) != a.capes:
        raise SystemExit("cape count wrong")
    for cid, n in Counter(deck).items():
        if cid != 7 and n > 4:
            raise SystemExit(f"{n}x card {cid} exceeds the 4-copy limit")

    obs, _ = battle_start(list(deck), list(deck))
    ok = obs is not None
    battle_finish()
    if not ok:
        raise SystemExit("engine REJECTED the deck")

    with open(os.path.join(dst, "cape_guard.py"), "w", encoding="utf-8") as f:
        f.write(GUARD)
    compile(GUARD, "cape_guard.py", "exec")
    with open(os.path.join(dst, "main.py"), "w", encoding="utf-8") as f:
        f.write(MAIN)
    compile(MAIN, "main.py", "exec")
    with open(os.path.join(dst, "deck.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, deck)) + "\n")

    # Both copies of the 60 must agree or the agent returns one list and plays
    # another -- a deck/constant mismatch already voided one submission here.
    rewrite_deck(os.path.join(dst, "policy_features.py"), deck)
    rewrite_deck(os.path.join(dst, "policies", "handwritten_v26",
                              "manual_policy.py"), deck)

    changed = Counter(deck) - Counter(base_deck)
    gone = Counter(base_deck) - Counter(deck)
    print(f"built work/agents/{a.name} from {BASE}")
    print(f"  + {dict(changed)}   - {dict(gone)}")
    print("  engine accepts the 60; deck.csv, policy_features and "
          "manual_policy all agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
