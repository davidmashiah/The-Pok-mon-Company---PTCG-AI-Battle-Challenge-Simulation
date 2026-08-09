"""Field score against the opponents we ACTUALLY meet, at the shares we meet them.

Both halves of the old panel were wrong, and both errors flattered us.

SHARES. `field_now.py` weights by top-50 composition. `our_field.py` reads our
own 155 ladder replays and finds a different field entirely:

    archetype      real share   old weight
    Grimmsnarl        0.187        0.32
    Mega Lucario      0.181        0.06
    Alakazam          0.174        0.14
    Crustle           0.129        0.10
    Archaludon        0.123        0.02
    Dragapult         0.110        0.04
    Mega Lopunny      0.026        0.18 (top-50)

OPPONENTS. The bigger error. Against our local benchmarks we score far above
what the same archetypes do to us in real games:

    Alakazam     local 0.758   real 0.407
    Lucario      local 0.672   real 0.464
    Archaludon   local 0.710   real 0.526
    Grimmsnarl   local 0.518   real 0.483   <- honest
    Crustle      local 0.784   real 0.750   <- honest

A 0.35 gap on Alakazam is not noise, it is the wrong opponent. So the panel
below prefers the STRONGEST pilot we have per archetype, by author leaderboard
rank rather than by notebook title:

    v61_codex_safe   jazivxt's Alakazam, author LB rank 121   (vs w1_alakazam)
    v51_roman_safe   romanrozen's public LB-950 Lucario       (vs z_roman950)

Set --strict to fail loudly on a missing cell instead of renormalising over what
happens to be measured -- dropping a column silently inflates the total, and the
columns most likely to be missing are the new hard ones.

  python work/tools/field_honest.py --agents _sub_v28,w96_mirrorroute
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
STORE = os.path.join(WORK, "out", "gauntlet.json")

# archetype -> (preferred opponent, fallback opponent, real share)
PANEL = [
    ("Grimmsnarl",   "w5_grimmsnarl",  None,            0.187),
    ("Mega Lucario", "v51_roman_safe", "z_roman950",    0.181),
    ("Alakazam",     "v61_codex_safe", "w1_alakazam",   0.174),
    ("Crustle",      "p3_crustle",     None,            0.129),
    ("Archaludon",   "w2_archaludon",  None,            0.123),
    ("Dragapult",    "s_dragapult",    None,            0.110),
]
# Mega Lopunny is 0.026 of the real field and has no published pilot; omitted
# rather than faked. "unknown" was 0.071 and we won 0.909 of it.

# what the same archetypes actually do to us on the ladder (our_field.py)
REAL = {"Grimmsnarl": 0.483, "Mega Lucario": 0.464, "Alakazam": 0.407,
        "Crustle": 0.750, "Archaludon": 0.526, "Dragapult": 0.706}


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def cells(store, agent, opp):
    w = n = 0
    for key, c in store.items():
        left, _, right = key.partition("|")
        la, ra = left.split("@")[0], right.split("@")[0]
        if la == agent and ra == opp:
            w += c["wa"]
            n += c["wa"] + c["wb"]
        elif la == opp and ra == agent:
            w += c["wb"]
            n += c["wa"] + c["wb"]
    return w, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    store = json.load(open(STORE))

    for name in [x.strip() for x in a.agents.split(",") if x.strip()]:
        print(f"\n=== {name} ===")
        print(f"{'archetype':14s} {'share':>6} {'opponent':16s} {'n':>6} "
              f"{'rate':>7} {'real':>7} {'gap':>7}")
        num = den = 0.0
        missing = []
        for arch, pref, fallback, share in PANEL:
            w, n = cells(store, name, pref)
            used = pref
            if not n and fallback:
                w, n = cells(store, name, fallback)
                used = fallback + " (weak)"
            if not n:
                missing.append(arch)
                print(f"{arch:14s} {share:6.3f} {pref:16s} {'--':>6} "
                      f"{'MISSING':>7}")
                continue
            p, lo, hi = wilson(w, n)
            real = REAL.get(arch)
            gap = (p - real) if real is not None else None
            print(f"{arch:14s} {share:6.3f} {used:16s} {n:6d} {p:7.4f} "
                  + (f"{real:7.3f} {gap:+7.3f}" if real is not None else ""))
            num += share * p
            den += share
        if missing and a.strict:
            raise SystemExit(f"missing cells: {missing}")
        if den:
            covered = den / sum(s for _, _, _, s in PANEL)
            print(f"  honest field {num/den:.4f} over {covered*100:.0f}% "
                  f"of the real field"
                  + (f"   MISSING {missing}" if missing else ""))
    print("\n'gap' is local minus real. A large POSITIVE gap means that "
          "opponent is a\nstrawman and any tuning done against it is not "
          "transferring to the ladder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
