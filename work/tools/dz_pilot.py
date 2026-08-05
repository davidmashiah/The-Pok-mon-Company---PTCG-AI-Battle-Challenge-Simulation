"""Clone a specific strong PILOT of our own deck, instead of the whole field.

Why: 102 of the 103 Mega Lucario ex winner-games in the published episodes
belong to one competitor. That is not a field sample, it is a single agent --
and it plays the deck we play. Imitating the field taught our net other decks'
strategies and scored 0.3555 on our deck against a 0.4088 always-first
baseline, i.e. worse than trivial. Imitating THIS pilot is the aligned target.

Cloning a policy does not require the game to have been won, so this takes the
pilot's decisions from every game they played, roughly doubling the data over
winner-only extraction. It also prints their win rate and exact decklist, since
if they beat us with a different 60 we should know that too.

Usage: python work/tools/dz_pilot.py <TeamName> <out.npz> [--wins-only]
"""
import glob
import json
import os
import sys
import zipfile
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
sys.path.insert(0, os.path.join(WORK, "lib"))
from dzfeat import (  # noqa: E402
    History, MAX_CAND, NF as STATE_NF, encode_options, featurize,
)

PLAYER = sys.argv[1] if len(sys.argv) > 1 else "Majkel1337"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(WORK, "out", "dz_pilot.npz")
WINS_ONLY = "--wins-only" in sys.argv
ZIPS = sorted(glob.glob(os.path.join(ROOT, "data", "episodes", "*", "*.zip")))


def main():
    S, A, AC, M, HT, HC, Ylab, G = [], [], [], [], [], [], [], []
    games = wins = losses = 0
    decks = Counter()
    opponents = Counter()
    for zp in ZIPS:
        try:
            zf = zipfile.ZipFile(zp)
        except Exception:
            continue
        day = os.path.basename(os.path.dirname(zp))
        found = 0
        for name in [n for n in zf.namelist() if n.endswith(".json")]:
            try:
                d = json.loads(zf.open(name).read().decode("utf-8"))
            except Exception:
                continue
            tn = ((d.get("info") or {}).get("TeamNames")) or []
            if PLAYER not in tn:
                continue
            p = tn.index(PLAYER)
            rw = d.get("rewards") or []
            if len(rw) < 2:
                continue
            won = rw[p] == 1
            found += 1
            games += 1
            wins += int(won)
            losses += int(not won)
            opponents[tn[1 - p] if len(tn) > 1 else "?"] += 1
            if WINS_ONLY and not won:
                continue

            hist = History()
            for st in d.get("steps", []):
                if p >= len(st):
                    continue
                ag = st[p]
                if ag.get("status") != "ACTIVE":
                    continue
                obs = ag.get("observation") or {}
                act = ag.get("action")
                hist.push(obs)
                if isinstance(act, list) and len(act) == 60:
                    decks[tuple(sorted(act))] += 1
                    continue
                if not act or not isinstance(act, list):
                    continue
                sel = obs.get("select")
                if not sel:
                    continue
                opts = sel.get("option") or []
                if len(opts) < 2 or len(opts) > MAX_CAND:
                    continue
                me = (obs.get("current") or {}).get("yourIndex", 0)
                sf = featurize(obs, me)
                if sf is None:
                    continue
                chosen = act[0]
                if not isinstance(chosen, int) or chosen >= len(opts):
                    continue
                af, ac, mk = encode_options(obs, opts, me)
                ht, hc = hist.arrays()
                S.append(sf); A.append(af); AC.append(ac); M.append(mk)
                HT.append(ht); HC.append(hc); Ylab.append(chosen); G.append(games)
        print(f"  {day}: {found} games by {PLAYER}", flush=True)

    print(f"\n{PLAYER}: {games} games, {wins} W / {losses} L "
          f"= {wins/max(games,1):.3f} win rate")
    print(f"decisions extracted: {len(S)}  (wins_only={WINS_ONLY})")
    if opponents:
        print(f"distinct opponents faced: {len(opponents)}")
    if decks:
        dk, dn = decks.most_common(1)[0]
        print(f"\ntheir decklist (seen {dn}x):\n{list(dk)}")
        try:
            from cg.api import all_card_data
            C = {c.cardId: c.name for c in all_card_data()}
            cnt = Counter(dk)
            print("\ndecklist by name:")
            for cid, n in sorted(cnt.items(), key=lambda x: -x[1]):
                print(f"  {n}x {cid:>5}  {C.get(cid)}")
            ours = [int(x) for x in open(
                os.path.join(WORK, "agents", "v14_search_noloop2", "deck.csv")
            ).read().split()]
            oc = Counter(ours)
            diff_add = [(c, n - oc.get(c, 0)) for c, n in cnt.items() if n > oc.get(c, 0)]
            diff_rem = [(c, oc[c] - cnt.get(c, 0)) for c in oc if oc[c] > cnt.get(c, 0)]
            print(f"\nvs OUR deck -- they run MORE of: "
                  f"{[(C.get(c), n) for c, n in diff_add]}")
            print(f"vs OUR deck -- they run LESS of: "
                  f"{[(C.get(c), n) for c, n in diff_rem]}")
        except Exception as e:
            print("deck compare failed:", e)

    if not S:
        print("no decisions extracted")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT,
        S=np.asarray(S, np.float32), A=np.asarray(A, np.float32),
        AC=np.asarray(AC, np.int32), M=np.asarray(M, np.float32),
        HT=np.asarray(HT, np.int32), HC=np.asarray(HC, np.int32),
        Y=np.asarray(Ylab, np.int64), G=np.asarray(G, np.int32))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
