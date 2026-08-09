"""How do we actually lose on the LADDER, per archetype?

We cannot fix Alakazam by tuning against `w1_alakazam`: we beat it 0.758 while
real Alakazam players beat us (we win 0.407), and the supposedly stronger
`v61_codex_safe` is weaker still -- we beat it 0.841. No strong Alakazam agent
is published anywhere, so there is no local opponent to tune against and no
local measurement that would transfer.

What we do have is 155 real games against real players. This reads the losses
out of them and describes HOW they ended, per archetype, so a fix can be aimed
at a failure that actually happened rather than at a proxy that flatters us.

Per game it records the things that separate the ways this deck dies:

  prizes each side took     a 6-0 is a different disease from a 5-6
  turn the game ended       fast losses are setup failures, slow ones attrition
  our Grimmsnarl ex arrival the deck is a Stage 2; if it never lands, nothing
                            else matters
  our attacks               zero attacks is a different bug from losing a race
  cards left in our deck    deck-out shows up here and nowhere else

and prints WINS beside LOSSES, because a number is only diagnostic if it differs
between them.

  python work/tools/real_losses.py --archetype Alakazam
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from our_field import TELLS, classify, deck_of, OURS  # noqa: E402

GRIMM, MORGREM, IMPIDIMP = 648, 647, 646


def side_cards(player):
    out = []
    for zone in ("active", "bench"):
        for c in (player.get(zone) or []):
            if isinstance(c, dict):
                out.append(c)
    return out


def describe(steps, me):
    """Walk one replay from our point of view."""
    r = {"turns": 0, "grimm_turn": None, "attacks": 0, "prizes_me": 0,
         "prizes_opp": 0, "deck_left": 0}
    last = None
    for si, st in enumerate(steps):
        if me >= len(st):
            continue
        obs = (st[me].get("observation") or {})
        cur = obs.get("current") or {}
        players = cur.get("players") or []
        if len(players) < 2:
            continue
        last = (cur, players)
        turn = int(cur.get("turn") or 0)
        r["turns"] = max(r["turns"], turn)
        mine = players[me]
        if r["grimm_turn"] is None:
            for c in side_cards(mine):
                if int(c.get("id", 0) or 0) == GRIMM:
                    r["grimm_turn"] = turn
                    break
        sel = obs.get("select") or {}
        act = st[me].get("action")
        # an ATTACK option chosen: OptionType.ATTACK == 13
        if sel and isinstance(act, list) and len(act) != 60:
            opts = sel.get("option") or []
            for i in act:
                if isinstance(i, int) and 0 <= i < len(opts):
                    if int((opts[i] or {}).get("type", -1) or -1) == 13:
                        r["attacks"] += 1
    if last:
        cur, players = last
        mine, opp = players[me], players[1 - me]
        r["prizes_me"] = 6 - len(mine.get("prize") or [])
        r["prizes_opp"] = 6 - len(opp.get("prize") or [])
        r["deck_left"] = int(mine.get("deckCount") or 0)
    return r


def summarise(rows, label):
    if not rows:
        print(f"  ({label}: none)")
        return
    n = len(rows)

    def avg(k):
        return sum(x[k] or 0 for x in rows) / n
    got = [x for x in rows if x["grimm_turn"] is not None]
    print(f"  {label:7s} n={n:3d}  prizes {avg('prizes_me'):.2f}-"
          f"{avg('prizes_opp'):.2f}  turns {avg('turns'):.1f}  "
          f"attacks {avg('attacks'):.1f}  "
          f"attacks/turn {avg('attacks')/max(1e-9,avg('turns')):.3f}  "
          f"deck_left {avg('deck_left'):.1f}")
    print(f"          Grimmsnarl ex landed: {len(got)}/{n} "
          f"({100*len(got)/n:.0f}%)"
          + (f", median turn {sorted(x['grimm_turn'] for x in got)[len(got)//2]}"
             if got else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetype", default="")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(WORK, "out", "our_replays", "*.json")))
    paths += sorted(glob.glob(os.path.join(WORK, "out", "replays", "*.json")))
    per = collections.defaultdict(lambda: {"win": [], "loss": []})

    for p in paths:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        rewards = d.get("rewards") or []
        steps = d.get("steps") or []
        info = d.get("info") or {}
        teams = info.get("TeamNames") or []
        if not steps or 1 not in rewards:
            continue
        me = None
        for i, t in enumerate(teams):
            if (t or "").strip() in OURS:
                me = i
                break
        if me is None:
            continue
        arch = classify(deck_of(steps, 1 - me))
        if a.archetype and arch != a.archetype:
            continue
        rec = describe(steps, me)
        per[arch]["win" if rewards.index(1) == me else "loss"].append(rec)

    for arch in sorted(per, key=lambda k: -(len(per[k]["win"]) +
                                            len(per[k]["loss"]))):
        w, l = per[arch]["win"], per[arch]["loss"]
        tot = len(w) + len(l)
        print(f"\n=== {arch}  ({tot} games, win rate "
              f"{len(w)/max(1,tot):.3f}) ===")
        summarise(w, "WINS")
        summarise(l, "LOSSES")
    print("\nA number matters only where WINS and LOSSES differ. Read "
          "attacks/turn and\nthe Grimmsnarl landing rate first -- they "
          "separate 'never set up' from\n'set up and lost the race'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
