"""Value-net data through valfeat.features -- the SAME code the search leaf runs.

`val_grimm_extract.py` produced a usable net (AUC 0.743 against a 0.631
prize-difference baseline) but it featurised replay DICTS with
`val_extract.featurize`, while the search leaf holds an observation CLASS from
`search_step`. Two featurisers over two APIs is a silent train/inference
mismatch waiting to happen: no exception, no warning, just a net scoring noise
at the leaf and an experiment that reads as "value nets do not help".

So this converts each replay observation with `to_observation_class` and calls
`valfeat.features` -- the identical function the agent will call. Whatever these
features mean, they mean the same thing in both places.

Keeps the parts of val_extract.py that were already right:
  * BOTH points of view, so the net sees losing boards. Training on winners
    alone teaches "everything is a win".
  * labels discounted toward 0.5 by distance from the end, because a turn-2
    board barely predicts the outcome and a hard 1/0 there is mostly noise.
  * one row per decision -- this evaluates a STATE, not an option.

  python work/tools/valfeat_extract.py --out work/out/val_vf.npz
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
sys.path.insert(0, os.path.join(WORK, "lib"))

import valfeat  # noqa: E402
from cg.api import to_observation_class  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=os.path.join(WORK, "out",
                                                      "top_replays"))
    ap.add_argument("--out", default=os.path.join(WORK, "out", "val_vf.npz"))
    ap.add_argument("--min-score", type=float, default=0.0)
    a = ap.parse_args()

    lb = leaderboard()
    X, Y, G = [], [], []
    games = 0
    failed = 0
    scores_used = []

    for path in sorted(glob.glob(os.path.join(a.replays, "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        rewards = d.get("rewards") or []
        if 1 not in rewards:
            continue
        winner = rewards.index(1)
        steps = d.get("steps") or []
        if not steps:
            continue
        teams = ((d.get("info") or {}).get("TeamNames") or [])

        used = False
        for me in (0, 1):
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
                    o = to_observation_class(obs)
                    ft = valfeat.features(o, me)
                except Exception:
                    failed += 1
                    continue
                if ft is None or len(ft) != valfeat.NF:
                    failed += 1
                    continue
                rows.append((si, ft))
            if not rows:
                continue
            last = rows[-1][0]
            for si, ft in rows:
                X.append(ft)
                Y.append(0.5 + (won - 0.5) * (GAMMA ** ((last - si) / 4.0)))
                G.append(games)
            used = True
            if score is not None:
                scores_used.append(score)
        if used:
            games += 1

    if not X:
        raise SystemExit("no rows extracted")
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    G = np.asarray(G, dtype=np.int32)
    np.savez_compressed(a.out, X=X, Y=Y, G=G,
                        names=np.array(valfeat.NAMES, dtype=object))
    print(f"games {games}   positions {len(X)}   features {X.shape[1]}   "
          f"featurise failures {failed}")
    if scores_used:
        scores_used.sort()
        print(f"pilots: median ladder score "
              f"{scores_used[len(scores_used)//2]:.1f}")
    print(f"label mean {Y.mean():.4f} (0.5 = the POV split is balanced)")
    nz = int((X.std(axis=0) > 0).sum())
    print(f"features with any variance: {nz}/{X.shape[1]}"
          + ("" if nz == X.shape[1] else "  <- constant columns learn nothing"))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
