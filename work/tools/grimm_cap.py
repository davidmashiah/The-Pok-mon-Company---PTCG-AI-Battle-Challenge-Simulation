"""Reject a candidate from ONE panel cell instead of six.

Grimmsnarl is 0.30 of the top-50 field and the panel renormalises over the 0.64
it can actually pilot, so that one archetype carries ~0.47 of the field score.
That makes its cell a hard CEILING: whatever a candidate does elsewhere,

    field <= w_g * p_grimmsnarl + (1 - w_g) * 1.0

Set p to the measured Grimmsnarl rate, give the candidate a free 1.000 against
every other archetype, and if the resulting rating still does not beat what we
already ship, the candidate is dead and the remaining five pairs are wasted
compute. This is not a projection of how good a candidate is -- it is only ever
a proof that it cannot be good enough.

Paid for by w24_tientrum: its author sits at ladder rank 88 and the notebook
reports 1034.6 live on 2026-07-05, but it wins 0.2525 (n=198) against
Grimmsnarl, which caps it at exactly the 0.638 we already have. Grimmsnarl grew
into 30% of the top field after that build was live.

  python work/tools/grimm_cap.py --agent w26_arist_prob --games 200
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import field_test as ft  # noqa: E402

GRIMM = "w5_grimmsnarl"
SHIPPED_FIELD = 0.6376        # w8_grimm_tuned, measured; live 848.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    share = dict((opp, s) for _, opp, s in ft.PANEL)[GRIMM]
    den = sum(s for _, opp, s in ft.PANEL
              if os.path.isdir(os.path.join(ft.WORK, "agents", opp)))
    w_g = share / den

    c = ft.cell(a.agent, GRIMM)
    if c is None or c[1] < a.games:
        ft.run(a.agent, GRIMM, a.games, a.workers)
        c = ft.cell(a.agent, GRIMM)
    if c is None:
        raise SystemExit("no result")
    w, n = c
    p, lo, hi = ft.wilson(w, n)

    # Be generous to the candidate at every step: use the UPPER confidence bound
    # on its Grimmsnarl rate, and hand it a perfect 1.000 everywhere else. A
    # rejection under those terms is a real rejection.
    cap = w_g * hi + (1 - w_g) * 1.0

    anchor = None
    try:
        import json
        anchor = json.load(open(ft.CAL)).get(ft.ANCHOR_AGENT)
    except Exception:
        pass

    print(f"\n=== Grimmsnarl cap: {a.agent} ===")
    print(f"vs {GRIMM}: {w}/{n} = {p:.4f}  Wilson95 [{lo:.4f},{hi:.4f}]")
    print(f"Grimmsnarl carries {w_g:.3f} of the renormalised panel weight")
    print(f"\nCEILING on field win rate (perfect 1.000 vs all others,")
    print(f"                            upper CI bound on Grimmsnarl): {cap:.4f}")
    print(f"already shipped (w8_grimm_tuned):                          "
          f"{SHIPPED_FIELD:.4f}")

    if anchor:
        est = anchor["rating"] + 400.0 * (ft.logit(cap) - ft.logit(anchor["field"]))
        print(f"\nBEST POSSIBLE projected rating: {est:.0f}")

    if cap <= SHIPPED_FIELD:
        print("\nVERDICT: REJECTED. Cannot beat what we already ship even if it "
              "wins every\n         other matchup outright. Do not spend the "
              "other five pairs on it.")
    else:
        print("\nVERDICT: survives the cap -- run the full field_test.py to see "
              "what it\n         ACTUALLY scores. Surviving this is not "
              "evidence of being good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
