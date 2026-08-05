"""Value-net data with the DouZero encoder: state + card identities + history.

The value net we have is a 46-scalar linear model and it scores AUC 0.7121 --
indistinguishable from the single `prize_diff` feature (0.7122). It cannot
improve a search that already weights prizes x1000, which is exactly why the
hand-written evaluate() failed as a leaf function (v30 0.3500, v31 0.0530).

The fix is representation, not the model class: give the evaluator the same
inputs the action-scorer got -- learned card embeddings and a GRU over the move
log -- but train it to predict WIN PROBABILITY instead of which option a human
clicked. That matters because value prediction is not imitation: it never
copies a move out of its plan, which is the failure that sank every cloning
attempt (v23_dz 389.3 on the ladder).

Labels: final outcome, discounted toward 0.5 by distance from the end, from
BOTH points of view so the net sees losing positions.

Usage: python work/tools/vz_extract.py <n_episodes> <out.npz>
"""
import glob
import json
import os
import sys
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
sys.path.insert(0, os.path.join(WORK, "lib"))
from dzfeat import HIST, History, NF, featurize  # noqa: E402

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(WORK, "out", "vz_data.npz")
GAMMA = 0.97
ZIPS = sorted(glob.glob(os.path.join(ROOT, "data", "episodes", "*", "*.zip")))


def main():
    S, HT, HC, Y, G = [], [], [], [], []
    eps = 0
    t0 = time.time()
    for zp in ZIPS:
        if eps >= LIMIT:
            break
        try:
            zf = zipfile.ZipFile(zp)
        except Exception:
            continue
        for name in [n for n in zf.namelist() if n.endswith(".json")]:
            if eps >= LIMIT:
                break
            try:
                d = json.loads(zf.open(name).read().decode("utf-8"))
            except Exception:
                continue
            rw = d.get("rewards") or []
            if len(rw) != 2 or 1 not in rw:
                continue
            eps += 1
            steps = d.get("steps") or []
            n = len(steps)
            for me in (0, 1):                      # BOTH POVs: see losses too
                won = 1.0 if rw[me] == 1 else 0.0
                hist = History()
                for si, st in enumerate(steps):
                    if me >= len(st):
                        continue
                    ag = st[me]
                    if ag.get("status") != "ACTIVE":
                        continue                   # INACTIVE frames repeat logs
                    obs = ag.get("observation") or {}
                    hist.push(obs)
                    if not obs.get("current") or not ag.get("action"):
                        continue
                    ft = featurize(obs, me)
                    if ft is None:
                        continue
                    dist = max(0, n - si)
                    ht, hc = hist.arrays()
                    S.append(ft)
                    HT.append(ht)
                    HC.append(hc)
                    # a turn-2 board barely predicts the winner; a hard 1/0 there
                    # is mostly noise, so pull distant labels toward 0.5
                    Y.append(0.5 + (won - 0.5) * (GAMMA ** (dist / 4.0)))
                    G.append(eps)
            if eps % 250 == 0:
                print(f"  {eps} eps, {len(S)} rows ({time.time()-t0:.0f}s)",
                      flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT, S=np.asarray(S, np.float32), HT=np.asarray(HT, np.int32),
        HC=np.asarray(HC, np.int32), Y=np.asarray(Y, np.float32),
        G=np.asarray(G, np.int32))
    print(f"\nepisodes={eps} rows={len(S)} state_nf={NF} hist={HIST}")
    print(f"label mean={np.mean(Y):.4f} (0.5 == balanced, both POVs kept)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
