"""Index every episode once: who played what, and who won.

Local simulation cannot answer matchup questions -- our own policy pilots the
opponent, so v14 beats Grimmsnarl 94% locally while winning 61% on the real
ladder. The entire gap is pilot skill, so any "which deck beats which" answer
has to come from games between real players.

This builds a compact index (one row per episode: both decks' ex/Mega-ex lines,
the winner, the players) so archetype-vs-archetype questions become instant
instead of costing a 13,444-episode / 2 GB rescan each time.

Usage: python work/tools/index_matchups.py [out.json]
"""
import glob
import json
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
sys.path.insert(0, os.path.join(WORK, "lib"))
from cg.api import all_card_data  # noqa: E402

CARDS = {c.cardId: c for c in all_card_data()}
# the ex / Mega-ex Pokemon that name an archetype
NAMERS = {cid for cid, c in CARDS.items()
          if int(c.cardType) == 0 and (getattr(c, "ex", False)
                                       or getattr(c, "megaEx", False))}

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    WORK, "out", "matchups.json")
ZIPS = sorted(glob.glob(os.path.join(ROOT, "data", "episodes", "*", "*.zip")))


def decks_of(d):
    """-> {player_index: [60 card ids]} from each side's deck-submission action."""
    out = {}
    for st in d.get("steps", []):
        for i, ag in enumerate(st):
            if i in out:
                continue
            a = ag.get("action")
            if isinstance(a, list) and len(a) == 60:
                out[i] = a
        if len(out) >= 2:
            break
    return out


def main():
    rows = []
    t0 = time.time()
    n = 0
    for zp in ZIPS:
        try:
            zf = zipfile.ZipFile(zp)
        except Exception:
            continue
        day = os.path.basename(os.path.dirname(zp))
        for name in [x for x in zf.namelist() if x.endswith(".json")]:
            n += 1
            try:
                d = json.loads(zf.open(name).read().decode("utf-8"))
            except Exception:
                continue
            rw = d.get("rewards") or []
            if len(rw) < 2:
                continue
            dk = decks_of(d)
            if len(dk) < 2:
                continue
            info = d.get("info") or {}
            tn = info.get("TeamNames") or ["?", "?"]
            rows.append({
                "f": f"{day}/{name}",
                "w": rw.index(1) if 1 in rw else -1,
                "t": [str(tn[0])[:24], str(tn[1])[:24]] if len(tn) > 1 else ["?", "?"],
                # only the archetype-naming cards, so the index stays small
                "a": sorted({c for c in dk[0] if c in NAMERS}),
                "b": sorted({c for c in dk[1] if c in NAMERS}),
            })
            if n % 2000 == 0:
                print(f"  {n} episodes, {len(rows)} indexed "
                      f"({time.time()-t0:.0f}s)", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    print(f"\nindexed {len(rows)} episodes of {n} in {time.time()-t0:.0f}s")
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
