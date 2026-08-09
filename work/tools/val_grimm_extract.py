"""Value-net data for the deck we ACTUALLY play, from top-50 pilots.

The search in our live agent is not broken and it is not slow -- it runs at 2225
decisions/s and `search_liveness.py` confirms it executes every playout it is
asked for. What is wrong is the leaf: it evaluates a position by GREEDY ROLLOUT,
which is biased, so buying more playouts converges faster onto a wrong number.
Measured: un-muzzling it (72 playouts, margin 250) took the mirror from 0.4915 to
0.3025, about -4.5 sigma.

Replacing that leaf is the one untried change with the size to matter, and this
is the data for it.

Why this is a different bet from the behavioural cloning that just failed. BC
asked "which option did the pilot take", and on the 465 held-out frames where the
answer was not option 0 it scored 0.2452 against 0.2154 random -- no skill, with
the author's own 262-feature pipeline. That target is hostage to the engine's
option ordering. "Is this position winning" is not: the replay labels it exactly,
no ordering prior can fake it, and every position in every game is a label rather
than one per decision. 167 replays is ~25k labelled positions against BC's 4332.

`val_extract.py` already does this correctly -- both points of view so the net
sees losing boards, labels discounted toward 0.5 by distance from the end, one
row per decision. Its only flaw is that it reads a 2026-08-02 episode ZIP from
the era when we played Mega Lucario. Every value net in the ledger, like every
BC run in it, was trained on a deck we no longer play.

This reuses that featurizer verbatim and swaps the source for the top-50
Grimmsnarl replays, with the same rank join `bc_grimm_extract.py` uses: pilots
above a score floor, not merely whoever won.

  python work/tools/val_grimm_extract.py --out work/out/val_grimm.npz
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
sys.path.insert(0, HERE)

from val_extract import FEATNAMES, featurize  # noqa: E402

GRIMMSNARL_EX = 648
GAMMA = 0.97


def leaderboard():
    pats = [os.path.join(WORK, "out", "lb_now", "*.csv"),
            os.path.join(ROOT, "data", "*publicleaderboard*.csv")]
    files = sorted(sum((glob.glob(p) for p in pats), []))
    if not files:
        return {}
    best = {}
    with open(files[-1], encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("TeamName") or "").strip()
            try:
                score = float(row.get("Score") or 0)
            except ValueError:
                continue
            if name and score > best.get(name, -1e9):
                best[name] = score
    return best


def deck_of(steps, k):
    for st in steps:
        if k < len(st):
            a = st[k].get("action") or []
            if isinstance(a, list) and len(a) == 60:
                return sorted(int(x) for x in a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=os.path.join(WORK, "out",
                                                      "top_replays"))
    ap.add_argument("--out", default=os.path.join(WORK, "out",
                                                  "val_grimm.npz"))
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--grimm-only", action="store_true",
                    help="only positions where the POV player runs Grimmsnarl")
    a = ap.parse_args()

    lb = leaderboard()
    X, Y, W = [], [], []      # features, discounted label, which game (for split)
    games = 0
    scores_used = []

    for path in sorted(glob.glob(os.path.join(a.replays, "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        rewards = d.get("rewards") or []
        if 1 not in rewards:
            continue
        steps = d.get("steps") or []
        if not steps:
            continue
        winner = rewards.index(1)
        teams = ((d.get("info") or {}).get("TeamNames") or [])

        # BOTH points of view -- a net trained only on winners learns
        # "everything is a win".
        used = False
        for me in (0, 1):
            deck = deck_of(steps, me)
            if deck is None:
                continue
            if a.grimm_only and GRIMMSNARL_EX not in deck:
                continue
            team = teams[me] if me < len(teams) else None
            score = lb.get(team)
            if a.min_score > 0 and (score is None or score < a.min_score):
                continue
            won = 1.0 if me == winner else 0.0
            rows = []
            for si, st in enumerate(steps):
                if me >= len(st):
                    continue
                obs = st[me].get("observation") or {}
                if not obs.get("select"):
                    continue
                try:
                    ft = featurize(obs, me)
                except Exception:
                    continue
                if ft is None:
                    continue
                rows.append((si, ft))
            if not rows:
                continue
            last = rows[-1][0]
            for si, ft in rows:
                dist = last - si
                X.append(ft)
                Y.append(0.5 + (won - 0.5) * (GAMMA ** (dist / 4.0)))
                W.append(games)
            used = True
            if score is not None:
                scores_used.append(score)
        if used:
            games += 1

    if not X:
        raise SystemExit("no rows extracted -- check --replays and the filters")
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    W = np.asarray(W, dtype=np.int32)
    np.savez_compressed(a.out, X=X, Y=Y, G=W,
                        names=np.array(FEATNAMES, dtype=object))
    print(f"games {games}   positions {len(X)}   features {X.shape[1]}")
    if scores_used:
        scores_used.sort()
        print(f"pilots: median ladder score "
              f"{scores_used[len(scores_used)//2]:.1f}, "
              f"range {scores_used[0]:.1f}-{scores_used[-1]:.1f}")
    print(f"label mean {Y.mean():.4f} (0.5 = balanced; far from 0.5 means the "
          f"POV split is broken)")
    print(f"-> {a.out}")
    print("\nJudge it on AUC and on beating the prize-difference baseline, not "
          "on loss:\na net that predicts 0.5 everywhere has fine loss and is "
          "useless as a leaf.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
