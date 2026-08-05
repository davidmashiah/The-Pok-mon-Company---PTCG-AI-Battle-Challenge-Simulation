 """Cache the episodes played with OUR deck, once, so analysis stops re-scanning.

Every deck-filtered analysis so far re-read a 700 MB archive and json.loads()'d
all ~4,600 episodes to find the ~100 that matter. That is minutes of wall clock
per question asked, repeated all night.

Two fixes:
  * cheap byte prefilter -- the card id must literally appear in the raw JSON
    text before it is worth parsing, which skips ~95% of files
  * write the survivors to one small archive, so every later run is instant

Usage: python work/tools/cache_deck_games.py [marker_card_id] [out.zip]
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

MARKER = int(sys.argv[1]) if len(sys.argv) > 1 else 678
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    WORK, "out", f"games_{MARKER}.zip")
ZIPS = sorted(glob.glob(os.path.join(ROOT, "data", "episodes", "*", "*.zip")))
needle = str(MARKER).encode()

t0 = time.time()
kept = scanned = parsed = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as out:
    for zp in ZIPS:
        try:
            zf = zipfile.ZipFile(zp)
        except Exception:
            continue
        day = os.path.basename(os.path.dirname(zp))
        for name in [n for n in zf.namelist() if n.endswith(".json")]:
            scanned += 1
            raw = zf.open(name).read()
            if needle not in raw:          # cheap prefilter, no JSON parse
                continue
            parsed += 1
            try:
                d = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            rw = d.get("rewards") or []
            if 1 not in rw:
                continue
            w = rw.index(1)
            deck = None
            for st in d.get("steps", []):
                if w < len(st):
                    a0 = st[w].get("action") or []
                    if isinstance(a0, list) and len(a0) == 60:
                        deck = set(a0)
                        break
            if not deck or MARKER not in deck:
                continue
            out.writestr(f"{day}/{name}", raw)
            kept += 1
        if scanned % 1000 < 5:
            print(f"  scanned {scanned}, kept {kept} ({time.time()-t0:.0f}s)",
                  flush=True)

print(f"\nscanned {scanned} episodes, byte-prefilter passed {parsed} "
      f"({100*parsed/max(scanned,1):.1f}%), kept {kept} won with card {MARKER}")
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB) in {time.time()-t0:.0f}s")
