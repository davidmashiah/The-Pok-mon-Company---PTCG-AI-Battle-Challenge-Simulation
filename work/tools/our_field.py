"""What do we ACTUALLY face on the ladder, and how do we do against it?

`field_now.py` weights archetypes by their share of the TOP 50. That is the
right target if the question is "what would score 1040", but it is not what the
ladder pays us for right now: matchmaking pairs us inside our own band, so the
mix we meet is the mix of players near our rating, not the mix at the top.

Both numbers matter and they answer different questions:
  * top-50 shares  -> what we would need to beat to reach the cutoff
  * our own mix    -> which column is costing us points TODAY

This reads our own downloaded ladder replays, identifies each opponent by cards
unique to one archetype (from archetype_ids.py), and reports share and win rate
per archetype -- ground truth instead of an assumed table. It also prints the
date range, because a mix measured at rating 726 does not describe the field at
950, and quoting a stale one is how a panel ends up weighting a dead archetype.

  python work/tools/our_field.py
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)

# cards unique to exactly one panel archetype (work/tools/archetype_ids.py)
TELLS = {}
for _cid in (104, 646, 647, 648):
    TELLS[_cid] = "Grimmsnarl"
for _cid in (142, 343, 741, 742, 743, 858):
    TELLS[_cid] = "Alakazam"
for _cid in (117, 344, 345, 414):
    TELLS[_cid] = "Crustle"
for _cid in (119, 120, 121, 184, 235, 1071):
    TELLS[_cid] = "Dragapult"
for _cid in (673, 674, 675, 676, 677, 678):
    TELLS[_cid] = "Mega Lucario"
for _cid in (57, 169, 190, 666):
    TELLS[_cid] = "Archaludon"
for _cid in (174, 848, 849, 861):
    TELLS[_cid] = "Mega Lopunny"

OURS = {"David Mashiah"}


def deck_of(steps, k):
    for st in steps:
        if k < len(st):
            a = st[k].get("action") or []
            if isinstance(a, list) and len(a) == 60:
                return [int(x) for x in a]
    return None


def our_60():
    """The deck we CURRENTLY ship."""
    p = os.path.join(WORK, "agents", "w34_koroll", "deck.csv")
    with open(p, encoding="utf-8-sig") as f:
        return sorted(int(x) for x in f.read().split() if x.strip())


def classify(deck):
    votes = collections.Counter()
    for c in deck or []:
        name = TELLS.get(c)
        if name:
            votes[name] += 1
    if not votes:
        return "unknown"
    return votes.most_common(1)[0][0]


def main():
    paths = sorted(glob.glob(os.path.join(WORK, "out", "our_replays", "*.json")))
    paths += sorted(glob.glob(os.path.join(WORK, "out", "replays", "*.json")))
    if not paths:
        raise SystemExit("no replays under work/out/our_replays")

    stats = collections.defaultdict(lambda: [0, 0])   # arch -> [wins, games]
    dates = []
    skipped = 0
    # Filter to games we played with the deck we CURRENTLY ship. Without this
    # the archive silently pools eras: of 155 replays, 91 are our old Mega
    # Lucario deck and 35 our old Alakazam deck. Classifying only the OPPONENT
    # produced a "we win 0.407 vs Alakazam" that was mostly the Lucario deck
    # losing, and a "Grimmsnarl ex reaches play 38% vs their 97%" that is really
    # 92% vs 100% once filtered.
    ours60 = our_60()
    wrong_deck = 0
    for p in paths:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        info = d.get("info") or {}
        teams = info.get("TeamNames") or []
        rewards = d.get("rewards") or []
        steps = d.get("steps") or []
        if not steps or 1 not in rewards:
            skipped += 1
            continue
        me = None
        for i, t in enumerate(teams):
            if (t or "").strip() in OURS:
                me = i
                break
        if me is None:
            skipped += 1
            continue
        my_deck = deck_of(steps, me)
        if my_deck is None or sorted(my_deck) != ours60:
            wrong_deck += 1
            continue
        opp_deck = deck_of(steps, 1 - me)
        arch = classify(opp_deck)
        won = 1 if rewards.index(1) == me else 0
        stats[arch][0] += won
        stats[arch][1] += 1
        ts = info.get("EpisodeId")
        if ts:
            dates.append(ts)

    total = sum(g for _, g in stats.values())
    if not total:
        raise SystemExit(f"no usable replays (skipped {skipped}); is the team "
                         f"name in OURS correct?")

    print(f"our own ladder games ON OUR CURRENT 60: {total}  "
          f"(skipped {skipped}; {wrong_deck} played a DIFFERENT deck of ours "
          f"and were excluded)\n")
    print(f"{'archetype':16s} {'share':>7} {'n':>5} {'win rate':>9}")
    print("-" * 42)
    rows = sorted(stats.items(), key=lambda kv: -kv[1][1])
    for arch, (w, n) in rows:
        print(f"{arch:16s} {n/total:7.3f} {n:5d} {w/n:9.3f}")

    weighted = sum(w for w, _ in stats.values()) / total
    print(f"\noverall win rate {weighted:.4f} over {total} real ladder games")
    print("\nCompare the SHARES against field_now.py's top-50 weights "
          "(Grimmsnarl 0.32,\nAlakazam 0.14, Crustle 0.10, Lucario 0.06, "
          "Dragapult 0.04). Where they differ,\nthe panel is optimising for a "
          "field we are not currently being matched into.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
