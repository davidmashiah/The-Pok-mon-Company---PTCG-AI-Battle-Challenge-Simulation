"""Predict a candidate's LADDER RATING, by beating a field instead of one agent.

The mistake this replaces. Every candidate in this project has been judged by a
head-to-head against the reigning champion, and that number does not convert:

    v61_codex_safe beat v51_roman_safe 0.8333 over 240 games
    v61_codex_safe then measured 726.1 on the ladder over 58 episodes
    v51_roman_safe measured ~700-780

0.83 head-to-head bought roughly +30. An Elo reading of 0.83 would predict
+300. The head-to-head is enormous because Alakazam beats Mega Lucario -- it is
a MATCHUP, not strength -- and the ladder does not pay for matchups, it pays for
the win rate against the mix of archetypes you are actually matched into.

So measure that mix. The panel below is every distinct archetype we have a real
policy for, weighted by how often we actually meet it, taken from 126 of our own
downloaded ladder replays and cross-checked against a survey of what the top 20
teams play. Then calibrate: v61 scores a known 726.1, so its field win rate is
the anchor, and a candidate's field win rate converts to a rating through it.

The conversion is Elo on the FIELD result, which is the quantity that behaves
like a rating, unlike a single head-to-head:

    rating(candidate) = 726.1 + 400 * log10(p / (1 - p)) - 400 * log10(pv / (1 - pv))

where p and pv are the field win rates of the candidate and of v61. Both are
measured against the identical panel with the identical weights, so everything
common to the panel cancels.

This is a PREDICTION and it is worth stating its limits plainly: the panel's
pilots are weaker than the ladder's, unranked opponents inflate every local
number, and 400 is a convention rather than this ladder's measured slope. Treat
it as "does this clear the bar by a wide margin", never as a promise of a score.

  python work/tools/field_test.py --agent v61_codex_safe --games 200
  python work/tools/field_test.py --report
"""
import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
STORE = os.path.join(WORK, "out", "gauntlet.json")
CAL = os.path.join(WORK, "out", "field_calibration.json")

# Archetype -> (agent we have a real policy for, share of the field).
#
# THESE ARE TOP-BAND SHARES, from work/tools/top_decks.py over the top 50 teams'
# own replays. The first version of this table used shares from OUR replays at
# ~726 rating and it was badly wrong for the question being asked. Compare:
#
#                      our band (726)   top 50      v61 win rate
#   Grimmsnarl              0.20         0.30          0.207
#   Mega Lopunny            0.00         0.20          (no opponent)
#   Alakazam                0.23         0.16          0.934
#   Mega Lucario            0.22         0.02          0.818
#
# Weighting Mega Lucario at 0.22 -- where v61 wins 0.818 -- when it is 2% of the
# top field, and Grimmsnarl at 0.20 when it is 30%, flattered us by ~0.16 of
# field win rate. You cannot measure your way to 1000 against the field of the
# band you are already in.
#
# KNOWN GAP, stated rather than hidden: Mega Lopunny ex is 20% of the top 50 and
# no Lopunny agent is published anywhere, so it is absent here. "unknown"
# archetypes are a further 18%. The shares below are renormalised over what we
# can actually pilot, so this measures 62% of the top field, not all of it.
PANEL = [
    ("Grimmsnarl",       "w5_grimmsnarl",   0.30),
    ("Alakazam",         "w1_alakazam",     0.16),
    ("Crustle control",  "p3_crustle",      0.08),
    ("Dragapult",        "s_dragapult",     0.06),
    ("Mega Lucario",     "z_roman950",      0.02),
    ("Archaludon",       "w2_archaludon",   0.02),
]

ANCHOR_AGENT = "v61_codex_safe"
ANCHOR_RATING = 726.1        # measured: submission 55274352, 58 episodes


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - hw), min(1.0, c + hw)


def bundle_hash(agent):
    sys.path.insert(0, HERE)
    import gauntlet
    return gauntlet.bundle_hash(agent)


def cell(a, b):
    try:
        store = json.load(open(STORE))
    except Exception:
        return None
    for key, c in store.items():
        if key == f"{a}@{bundle_hash(a)}|{b}@{bundle_hash(b)}":
            n = c["wa"] + c["wb"]
            return (c["wa"], n) if n else None
        if key == f"{b}@{bundle_hash(b)}|{a}@{bundle_hash(a)}":
            n = c["wa"] + c["wb"]
            return (c["wb"], n) if n else None
    return None


def run(a, b, games, workers):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    subprocess.run([sys.executable, "-u", os.path.join(HERE, "gauntlet.py"),
                    "--agents", f"{a},{b}", "--games", str(games),
                    "--workers", str(workers)], cwd=ROOT, env=env)


def field(agent, games, workers, quiet=False):
    """Weighted win rate across the panel. Returns (p, rows, total_n)."""
    rows = []
    num = den = 0.0
    tot = 0
    for name, opp, share in PANEL:
        if not os.path.isdir(os.path.join(WORK, "agents", opp)):
            continue
        c = cell(agent, opp)
        if c is None or c[1] < games:
            run(agent, opp, games, workers)
            c = cell(agent, opp)
        if c is None:
            continue
        w, n = c
        p, lo, hi = wilson(w, n)
        rows.append((name, opp, share, p, lo, hi, n))
        num += share * p
        den += share
        tot += n
    if den == 0:
        return None, rows, 0
    return num / den, rows, tot


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log10(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    print(f"\n=== field test: {args.agent} ===")
    p, rows, tot = field(args.agent, args.games, args.workers)
    if p is None:
        raise SystemExit("no panel results")
    print(f"\n{'archetype':20s} {'opponent':16s} {'share':>6} {'winrate':>8} "
          f"{'Wilson 95%':>18} {'n':>6}")
    print("-" * 82)
    for name, opp, share, wp, lo, hi, n in rows:
        print(f"{name:20s} {opp:16s} {share:6.2f} {wp:8.3f} "
              f"[{lo:.3f},{hi:.3f}] {n:6d}")
    print(f"\nFIELD WIN RATE: {p:.4f}   (total {tot} games)")

    anchor = None
    if os.path.exists(CAL):
        try:
            anchor = json.load(open(CAL)).get(ANCHOR_AGENT)
        except Exception:
            anchor = None
    if args.agent == ANCHOR_AGENT:
        cal = {}
        if os.path.exists(CAL):
            try:
                cal = json.load(open(CAL))
            except Exception:
                cal = {}
        cal[ANCHOR_AGENT] = {"field": p, "rating": ANCHOR_RATING}
        json.dump(cal, open(CAL, "w"), indent=1)
        print(f"\nanchor stored: field {p:.4f} == {ANCHOR_RATING} on the ladder")
        print("run this on a candidate to get a projected rating")
        return 0

    if not anchor:
        print(f"\nNo anchor yet. Run:  python work/tools/field_test.py "
              f"--agent {ANCHOR_AGENT}")
        return 1

    pv = anchor["field"]
    est = anchor["rating"] + 400.0 * (logit(p) - logit(pv))
    print(f"\nanchor: {ANCHOR_AGENT} field {pv:.4f} == {anchor['rating']}")
    print(f"PROJECTED LADDER RATING: {est:.0f}")
    print(f"  ({'CLEARS' if est >= 1000 else 'does NOT clear'} 1000)")
    print("\nThis is a prediction, not a score. The panel's pilots are weaker "
          "than the ladder's,\nand 400 is a convention rather than this "
          "ladder's measured slope. Use it to reject\ncandidates cheaply and to "
          "size a gap -- never as a promise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
