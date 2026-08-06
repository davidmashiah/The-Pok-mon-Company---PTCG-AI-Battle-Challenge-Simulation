"""What do the top-50 teams playing OUR deck run that we do not?

16 of the top 50 pilot Marnie's Grimmsnarl ex, at 1044-1100, while our copy of
that archetype sits at ~853. Their real 60s are already in work/out/top_decks.json
-- read verbatim out of the setup frame of their own replays, not inferred -- so
the comparison costs nothing and is the most direct evidence available about
what a good version of our deck looks like.

Read the output with the coupling caveat firmly in mind: a decklist is NOT
transferable without the policy it was tuned with (p4_crustle_live scored 0.1167
running a top team's real list under a foreign policy). This is for spotting
single cards worth testing, not for wholesale adoption.

  python work/tools/mirror_diff.py
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import all_card_data  # noqa: E402

cards = {c.cardId: c for c in all_card_data()}
GRIMM = 648
STORE = os.path.join(WORK, "out", "top_decks.json")

ours = [int(x) for x in open(os.path.join(WORK, "agents", "w8_grimm_tuned",
                                          "deck.csv"), encoding="utf-8")
        .read().split() if x.strip()]
ours_c = collections.Counter(ours)

store = json.load(open(STORE, encoding="utf-8"))
rows = []
for team in store.values():
    for deck in (team.get("decks") or []):
        if deck and deck.count(GRIMM) > 0:
            rows.append((team.get("rank"), team.get("name"),
                         float(team.get("score") or 0), collections.Counter(deck)))
            break

rows.sort(key=lambda r: -r[2])
print(f"{len(rows)} top-50 teams piloting Marnie's Grimmsnarl ex\n")

# how often each card appears across their lists, vs our count
freq = collections.Counter()
totals = collections.Counter()
for _, _, _, c in rows:
    for cid, n in c.items():
        freq[cid] += 1
        totals[cid] += n

print(f"{'card':34s} {'ours':>4} {'theirs avg':>10} {'teams':>6}")
print("-" * 60)
allids = set(freq) | set(ours_c)
def sort_key(cid):
    theirs = totals[cid] / len(rows) if rows else 0
    return -(theirs - ours_c.get(cid, 0))
for cid in sorted(allids, key=sort_key):
    theirs = totals[cid] / len(rows) if rows else 0
    mine = ours_c.get(cid, 0)
    if abs(theirs - mine) < 0.25:
        continue
    nm = getattr(cards.get(cid), "name", str(cid))
    ace = " <ACE>" if getattr(cards.get(cid), "aceSpec", False) else ""
    print(f"[{cid}] {nm[:26]:26s}{ace:6s} {mine:>4} {theirs:>10.2f} "
          f"{freq[cid]:>4}/{len(rows)}")

print("\nper-team lists (score, then what they run that we do not):")
for rank, name, score, c in rows[:8]:
    extra = sorted((c - ours_c).items(), key=lambda kv: -kv[1])
    miss = sorted((ours_c - c).items(), key=lambda kv: -kv[1])
    e = ", ".join(f"{n}x {getattr(cards.get(i),'name',i)}" for i, n in extra)
    m = ", ".join(f"{n}x {getattr(cards.get(i),'name',i)}" for i, n in miss)
    print(f"\n  rank {rank} {name[:22]:22s} {score}")
    print(f"     + {e[:150]}")
    print(f"     - {m[:150]}")
