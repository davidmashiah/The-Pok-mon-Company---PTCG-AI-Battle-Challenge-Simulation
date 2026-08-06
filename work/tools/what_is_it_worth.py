"""How many RATING POINTS is a given matchup improvement actually worth?

This is the sizing check that should precede any build, and did not precede
several in this repo. Rating moves with the LOGIT of the field win rate, so a
matchup that is 47% of the panel still buys very little once the field rate is
already near 0.64 -- the curve is flat there in the units we care about.

Run it before deciding a lever is worth days:

  python work/tools/what_is_it_worth.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import field_test as ft  # noqa: E402

# w8_grimm_tuned, as measured. Live 848.8.
RATES = {
    "w5_grimmsnarl": 0.5297,
    "w1_alakazam": 0.747,
    "p3_crustle": 0.811,
    "s_dragapult": 0.720,
    "z_roman950": 0.446,
    "w2_archaludon": 0.629,
}
TARGET = 1040.0


def field(rates):
    num = den = 0.0
    for _, opp, share in ft.PANEL:
        if opp in rates:
            num += share * rates[opp]
            den += share
    return num / den


def rating(p, anchor):
    return anchor["rating"] + 400.0 * (ft.logit(p) - ft.logit(anchor["field"]))


def main():
    import json
    anchor = json.load(open(ft.CAL))[ft.ANCHOR_AGENT]
    base_f = field(RATES)
    base_r = rating(base_f, anchor)
    print(f"base: field {base_f:.4f} -> {base_r:.0f}  (w8 live 848.8)\n")

    print("what one matchup is worth, holding everything else fixed:")
    print(f"{'matchup':18s} {'share':>6} {'now':>6} {'->':>3} {'new':>6} "
          f"{'field':>7} {'rating':>7} {'gain':>6}")
    print("-" * 68)
    for _, opp, share in ft.PANEL:
        if opp not in RATES:
            continue
        for new in (RATES[opp] + 0.05, RATES[opp] + 0.20, 0.95):
            if new <= RATES[opp] or new > 1.0:
                continue
            r2 = dict(RATES)
            r2[opp] = new
            f2 = field(r2)
            g = rating(f2, anchor) - base_r
            print(f"{opp:18s} {share:6.2f} {RATES[opp]:6.3f} {'->':>3} "
                  f"{new:6.3f} {f2:7.4f} {rating(f2, anchor):7.0f} "
                  f"{g:+6.0f}")
        print()

    print("=" * 68)
    need = ft.logit(anchor["field"]) + (TARGET - anchor["rating"]) / 400.0
    need_p = 1.0 / (1.0 + 10.0 ** (-need))
    print(f"to reach {TARGET:.0f} the field win rate must be {need_p:.4f} "
          f"(now {base_f:.4f})")
    print("\nEVERY matchup at 0.95 except the mirror, mirror left at 0.530:")
    r3 = dict((k, 0.95) for k in RATES)
    r3["w5_grimmsnarl"] = RATES["w5_grimmsnarl"]
    f3 = field(r3)
    print(f"   field {f3:.4f} -> {rating(f3, anchor):.0f}")
    print("\nMIRROR at 0.95, everything else left exactly as it is:")
    r4 = dict(RATES)
    r4["w5_grimmsnarl"] = 0.95
    f4 = field(r4)
    print(f"   field {f4:.4f} -> {rating(f4, anchor):.0f}")
    print("\nBOTH -- mirror 0.90 and every other matchup 0.90:")
    r5 = dict((k, 0.90) for k in RATES)
    f5 = field(r5)
    print(f"   field {f5:.4f} -> {rating(f5, anchor):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
