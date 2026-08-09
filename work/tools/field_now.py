"""Field win rate under the CURRENT top-50 shares, with a like-for-like anchor.

`field_test.py`'s PANEL has two errors that the 2026-08-06 top-50 re-survey
found and that nothing has acted on:

    Archaludon   weighted 0.02, actually **0 of 50 teams** -- stale
    Mega Lucario weighted 0.02, actually **3 of 50 teams (6%)** -- and it is our
                 WORST matchup, so understating it flatters every candidate

Both errors push the same way: they hide our weakness. This recomputes the
weighted rate from the same gauntlet store under the corrected shares.

It reads cells only -- it never plays games. So it cannot invent a number that
was not measured, and a matchup with no cell is reported as MISSING rather than
silently dropped, because dropping it renormalises the rest and quietly inflates
the total.

The rating conversion is deliberately RELATIVE. Absolute predictions here have a
poor record (predicted 830 vs 853 actual, and the live twins of one identical
bundle read 55.6 points apart). Comparing two candidates through the same anchor
cancels everything common to the panel, which is the only part that has held up.

  python work/tools/field_now.py --agents w34_koroll,w70_route,w80_val
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
STORE = os.path.join(WORK, "out", "gauntlet.json")
sys.path.insert(0, HERE)

# archetype -> (opponent agent, corrected share of the top 50)
PANEL = [
    ("Grimmsnarl", "w5_grimmsnarl", 0.32),
    ("Alakazam", "w1_alakazam", 0.14),
    ("Crustle", "p3_crustle", 0.10),
    ("Mega Lucario", "z_roman950", 0.06),
    ("Dragapult", "s_dragapult", 0.04),
]
# Mega Lopunny (18%) and unknown (16%) have no published pilot; this therefore
# measures 66% of the top field and says nothing about the other 34%.


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def cells(store, agent, opp):
    """Every cell for this pair, POOLED across bundle-hash eras.

    Hashes changed when bundle_hash was fixed to walk subdirectories (commit
    c778ca4). That renamed the cells; it did not change the agents. Pooling is
    therefore correct here and is what makes the older, larger samples usable.
    """
    w = n = 0
    for key, c in store.items():
        left, _, right = key.partition("|")
        la = left.split("@")[0]
        ra = right.split("@")[0]
        if la == agent and ra == opp:
            w += c["wa"]
            n += c["wa"] + c["wb"]
        elif la == opp and ra == agent:
            w += c["wb"]
            n += c["wa"] + c["wb"]
    return w, n


def field(store, agent):
    rows = []
    num = den = 0.0
    for name, opp, share in PANEL:
        w, n = cells(store, agent, opp)
        if not n:
            rows.append((name, share, None, None, None, 0))
            continue
        p, lo, hi = wilson(w, n)
        rows.append((name, share, p, lo, hi, n))
        num += share * p
        den += share
    return (num / den if den else None), rows, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", required=True)
    ap.add_argument("--anchor", default="w34_koroll")
    ap.add_argument("--anchor-rating", type=float, default=886.0)
    a = ap.parse_args()

    store = json.load(open(STORE))
    names = [x.strip() for x in a.agents.split(",") if x.strip()]

    base, _, _ = field(store, a.anchor)
    if base is None:
        raise SystemExit(f"anchor {a.anchor} has no measured cells")

    def logit(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log10(p / (1 - p))

    for name in names:
        f, rows, den = field(store, name)
        print(f"\n=== {name} ===")
        print(f"{'archetype':14s} {'share':>6} {'n':>6} {'rate':>7} "
              f"{'95% CI':>18}")
        for arch, share, p, lo, hi, n in rows:
            if p is None:
                print(f"{arch:14s} {share:6.2f} {'--':>6} {'MISSING':>7}"
                      f"{'':>19}  <- not measured")
            else:
                print(f"{arch:14s} {share:6.2f} {n:6d} {p:7.4f} "
                      f"[{lo:.4f},{hi:.4f}]")
        if f is None:
            print("  no cells")
            continue
        covered = den / sum(s for _, _, s in PANEL)
        delta = 400.0 * (logit(f) - logit(base))
        print(f"  field {f:.4f} over {covered*100:.0f}% of the panel"
              f"   vs {a.anchor} {base:.4f}"
              f"   delta {delta:+.1f} rating "
              f"(~{a.anchor_rating + delta:.0f})")

    print("\nRelative deltas only. The live noise floor on this ladder is "
          "+/-55-85\npoints -- one identical bundle scored 882.8 and 519.1 -- "
          "so anything under\n~+60 cannot be confirmed by a single submission.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
