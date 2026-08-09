"""The only measurement that has ever predicted this ladder: real episode results.

Our local field test ranked `_sub_v28` +15.9 rating above `w34_koroll`. Across
144 real episodes the two are indistinguishable -- 0.528 and 0.529. The local
panel does not transfer, and this is the direct evidence, independent of any
theory about why.

`competitions.EpisodeService/ListEpisodes` returns, per episode, our agent's
reward and both sides' before/after scores. That gives, for any submission:

  * the REAL win rate against real opponents at our own rating
  * the strength of the opponents we are actually matched into
  * the score trajectory, so an unconverged submission is obvious rather than
    quoted as a result (a fresh one starts at 600 and climbs; ours read 1005.3
    at ~10 episodes and settled near 900)

Use it to convert a target rating into a required win rate. Beating a field of
median R at rate p is worth about R + 400*log10(p/(1-p)); at p=0.528 against
R=899 that is ~918, and reaching 1000 needs p=0.64 -- i.e. +0.11, not +0.02.

  python work/tools/live_winrate.py --submission 55387402 --submission 55310358
"""
import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request

TOKEN_PATH = os.path.join(os.path.expanduser("~"), ".kaggle", "access_token")
BASE = "https://www.kaggle.com/api/i/competitions.EpisodeService"


def token():
    t = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not t and os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH) as fh:
            t = fh.read().strip()
    if not t:
        raise SystemExit("no access token at ~/.kaggle/access_token")
    return t


def list_episodes(sub):
    data = json.dumps({"submissionId": sub}).encode()
    req = urllib.request.Request(
        f"{BASE}/ListEpisodes", data=data, method="POST",
        headers={"Authorization": f"Bearer {token()}",
                 "Content-Type": "application/json",
                 "User-Agent": "ptcg-analysis"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read()).get("episodes") or []
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} for submission {sub}")
        return []


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=int, action="append", required=True)
    ap.add_argument("--target", type=float, default=1000.0)
    a = ap.parse_args()

    for sub in a.submission:
        eps = list_episodes(sub)
        w = l = 0
        first = last = None
        opp = []
        for e in sorted(eps, key=lambda x: x.get("createTime") or ""):
            mine = [x for x in (e.get("agents") or [])
                    if x.get("submissionId") == sub]
            other = [x for x in (e.get("agents") or [])
                     if x.get("submissionId") != sub]
            if not mine:
                continue
            m = mine[0]
            r = m.get("reward", 0) or 0
            if r > 0:
                w += 1
            elif r < 0:
                l += 1
            if first is None:
                first = m.get("initialScore")
            last = m.get("updatedScore")
            if other and other[0].get("initialScore"):
                opp.append(other[0]["initialScore"])
        n = w + l
        p, lo, hi = wilson(w, n)
        print(f"\nsubmission {sub}: {len(eps)} episodes, {n} decided")
        print(f"  real win rate {p:.4f}  [{lo:.4f}, {hi:.4f}]   (W-L {w}-{l})")
        if first is not None and last is not None:
            print(f"  score {first:.1f} -> {last:.1f}"
                  + ("   <- UNCONVERGED, under ~25 episodes means little"
                     if n < 25 else ""))
        if opp:
            opp.sort()
            med = opp[len(opp) // 2]
            print(f"  opponents: median {med:.0f}, "
                  f"range {opp[0]:.0f}-{opp[-1]:.0f}")
            implied = med + 400 * math.log10(max(p, 1e-6) / max(1 - p, 1e-6))
            need = 1.0 / (1.0 + 10 ** ((med - a.target) / 400.0))
            print(f"  implied rating ~{implied:.0f}")
            print(f"  to reach {a.target:.0f} against this field you must win "
                  f"{need:.3f}  (you win {p:.3f}, gap {need - p:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
