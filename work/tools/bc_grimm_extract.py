"""Behavioural cloning on the deck we ACTUALLY play, from the pilots we need to beat.

The ledger files "behavioural cloning 389.3" as refuted. That number is real but
it does not say what it has been read as saying: `bc_train.py` defaults to
`bc_lucario.npz`, and every BC dataset in work/out is from 2026-08-03 and is
Lucario or Lopunny -- the deck we played in a previous era. **No Grimmsnarl BC
dataset has ever existed.** Cloning the deck we ship has not been tried.

It is also the one mechanism aimed at the thing that is actually wrong. The deck
is solved and shared -- 13 of the top-50 Grimmsnarl teams run our byte-identical
60 -- so the entire spread from our ~886 to the leader's 1188 is piloting. And
`divergence_audit.py` measured our agreement with a 1100+ pilot's choices at
**0.267 over 2731 decisions**. We are not playing the same game they are.

Two disciplines this applies that the old extractor could not:

  * **Rank join, not "whoever won".** `info.TeamNames` in each replay is joined
    to the live leaderboard CSV, so rows come only from pilots above a score
    floor. A winner rated 600 beat someone rated 550; imitating them teaches us
    to be rated 600. Titles and win/loss are marketing, rank is not -- the same
    lesson that made mine_notebook.py work.
  * **Deck match, not archetype match.** The winner's 60 comes from
    `steps[0][k]['action']`, so we can require the byte-identical list rather
    than merely "has a Grimmsnarl in it". A near-mirror pilot's choices are
    keyed to cards we do not hold.

Emits the same (X, Y, G) contract `bc_train.py` consumes: one row per OPTION,
labelled 1 if the pilot chose it, grouped by decision so the split never
straddles one frame.

  python work/tools/bc_grimm_extract.py --out work/out/bc_grimm.npz --min-score 1000
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

from bc_extract import FEATNAMES, featurize  # noqa: E402

GRIMMSNARL_EX = 648
OUR_DECK_PATH = os.path.join(WORK, "agents", "w34_koroll", "deck.csv")


def our_deck():
    with open(OUR_DECK_PATH, encoding="utf-8-sig") as f:
        return sorted(int(x) for x in f.read().split() if x.strip())


def leaderboard():
    """TeamName -> best score, from the newest leaderboard CSV we have."""
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
    print(f"leaderboard: {len(best)} teams from {os.path.basename(files[-1])}")
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
    ap.add_argument("--out", default=os.path.join(WORK, "out", "bc_grimm.npz"))
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="only clone pilots at or above this ladder score")
    ap.add_argument("--exact-deck", action="store_true",
                    help="require the byte-identical 60, not just a Grimmsnarl")
    a = ap.parse_args()

    ours = our_deck()
    lb = leaderboard()

    X, Y, G = [], [], []
    gid = 0
    kept = skipped_deck = skipped_rank = 0
    scores_used = []

    for path in sorted(glob.glob(os.path.join(a.replays, "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        rewards = d.get("rewards") or []
        if 1 not in rewards:
            continue
        w = rewards.index(1)
        steps = d.get("steps") or []
        if not steps:
            continue

        deck = deck_of(steps, w)
        if deck is None:
            continue
        if a.exact_deck:
            if deck != ours:
                skipped_deck += 1
                continue
        elif GRIMMSNARL_EX not in deck:
            skipped_deck += 1
            continue

        names = ((d.get("info") or {}).get("TeamNames") or [])
        team = names[w] if w < len(names) else None
        score = lb.get(team)
        if a.min_score > 0:
            if score is None or score < a.min_score:
                skipped_rank += 1
                continue
        if score is not None:
            scores_used.append(score)
        kept += 1

        for st in steps:
            if w >= len(st):
                continue
            ag = st[w]
            obs = ag.get("observation") or {}
            act = ag.get("action")
            if not act or not isinstance(act, list) or len(act) == 60:
                continue
            sel = obs.get("select")
            if not sel:
                continue
            opts = sel.get("option") or []
            if len(opts) < 2:
                continue
            me = (obs.get("current") or {}).get("yourIndex", 0)
            chosen = set(act)
            rows = []
            for i, o in enumerate(opts):
                try:
                    rows.append((featurize(obs, sel, o, me, len(opts)),
                                 1 if i in chosen else 0))
                except Exception:
                    rows = []
                    break
            if not rows or not any(y for _, y in rows):
                continue
            for feat, lab in rows:
                X.append(feat)
                Y.append(lab)
                G.append(gid)
            gid += 1

    if not X:
        raise SystemExit("no rows extracted -- check --replays and the filters")

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)
    G = np.asarray(G, dtype=np.int32)
    np.savez_compressed(a.out, X=X, Y=Y, G=G,
                        names=np.array(FEATNAMES, dtype=object))
    print(f"\nepisodes kept {kept}   skipped: deck {skipped_deck}, "
          f"rank {skipped_rank}")
    if scores_used:
        scores_used.sort()
        mid = scores_used[len(scores_used) // 2]
        print(f"cloned pilots: median ladder score {mid:.1f}, "
              f"range {scores_used[0]:.1f}-{scores_used[-1]:.1f}")
    print(f"rows {len(X)}  decisions {gid}  features {X.shape[1]}  "
          f"positive rate {Y.mean():.4f}")
    print(f"-> {a.out}")
    print("\nNext: python work/tools/bc_train.py " + a.out)
    print("The bar is top-1 accuracy 0.240 (best rule-based agent on "
          "same-archetype\nwinners); we currently agree with a 1100+ pilot "
          "on 0.267 of decisions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
