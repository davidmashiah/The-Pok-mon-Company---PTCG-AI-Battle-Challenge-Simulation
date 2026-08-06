"""What are our two ACTIVE submissions doing on the ladder right now?

Only the latest two submissions score, and the board takes their MAX, so this
prints exactly the number that decides our rank -- plus how far each draw has
converged, because a fresh submission starts at 600 and climbs. Reading a score
before ~25 episodes is reading noise, and this project has twice acted on one.

  python work/tools/live_watch.py
  python work/tools/live_watch.py --loop 600     # re-read every 10 minutes
"""
import argparse
import csv
import io
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
COMP = "pokemon-tcg-ai-battle"
TOKEN = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
EXE = os.path.join(ROOT, ".venv", "Scripts", "kaggle.exe")


def rows():
    env = dict(os.environ, KAGGLE_API_TOKEN=TOKEN, PYTHONIOENCODING="utf-8")
    out = subprocess.run([EXE, "competitions", "submissions", "-c", COMP,
                          "--csv"], capture_output=True, text=True, env=env,
                         encoding="utf-8", errors="replace").stdout
    i = out.find("ref,")
    return list(csv.DictReader(io.StringIO(out[i:]))) if i >= 0 else []


def show():
    r = rows()
    if not r:
        print("  (no submissions returned)")
        return
    active = r[:2]
    scores = []
    print(f"  {'ref':>9}  {'agent':22s} {'submitted':19s} {'score':>8}  status")
    for row in active:
        s = row.get("publicScore") or ""
        try:
            scores.append(float(s))
        except ValueError:
            pass
        print(f"  {row['ref']:>9}  {row['fileName'].replace('.tar.gz',''):22s} "
              f"{row['date'][:19]:19s} {s or 'pending':>8}  "
              f"{row['status'].replace('SubmissionStatus.','')}")
    if scores:
        print(f"\n  BOARD SCORE = max of the two actives = {max(scores):.1f}")
    best = max((float(x["publicScore"]) for x in r
                if (x.get("publicScore") or "").replace(".", "").isdigit()),
               default=None)
    if best is not None:
        print(f"  (best we have ever held, active or not: {best:.1f})")
    print("  a fresh submission starts at 600 and climbs; under ~25 episodes "
          "it means nothing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0,
                    help="seconds between reads; 0 = once")
    a = ap.parse_args()
    while True:
        print(time.strftime("\n=== %Y-%m-%d %H:%M:%S UTC ===", time.gmtime()))
        try:
            show()
        except Exception as exc:
            print("  read failed:", type(exc).__name__, exc)
        if not a.loop:
            return 0
        time.sleep(a.loop)


if __name__ == "__main__":
    sys.exit(main())
