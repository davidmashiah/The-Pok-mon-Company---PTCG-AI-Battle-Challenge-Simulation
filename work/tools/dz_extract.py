"""DouZero-style data: (state, history, per-candidate action encodings) -> chosen.

The previous BC run scored static per-option features and landed at 0.3719
against a 0.3665 always-first baseline: it learned the engine's option ordering
and nothing else. Three things were missing, and all three are here:

  1. ACTION ENCODING  -- each candidate carries its own card id (for a learned
     embedding) plus type/target/damage, so the net reasons about what the
     action IS, not where it sits in the list.
  2. HISTORY          -- the last H log events (type + card id) feed a GRU.
     Card-game decisions depend on what has already been discarded, attached
     and attacked with.
  3. LEARNED EMBEDDINGS over the raw card-id vocabulary, not a 64-way hash.
     Hashing destroys the structure that lets similar cards share meaning.

Output is padded/masked per decision so the model can score a variable number
of candidates and pick an argmax over legal options only.

Usage: python work/tools/dz_extract.py <n_episodes> <out.npz> [markers]
"""
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
import glob as _glob
# every day we have downloaded; deck-matched games are ~2% of each day, so
# more days is the only way to get a usable amount of OUR deck's play
ZIPS = sorted(_glob.glob(os.path.join(ROOT, "data", "episodes", "*", "*.zip")))
assert ZIPS, "no episode archives found under data/episodes/*/"

sys.path.insert(0, os.path.join(WORK, "lib"))
from dzfeat import (  # noqa: E402
    History, MAX_CAND, HIST, CARD_VOCAB, ACT_NF, NF as STATE_NF,
    encode_options, featurize as state_featurize,
)




def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(WORK, "out", "dz_data.npz")
    markers = set()
    if len(sys.argv) > 3 and sys.argv[3] != "-":
        markers = {int(x) for x in sys.argv[3].split(",")}

    S, A, AC, M, HT, HC, Ylab, G = [], [], [], [], [], [], [], []
    eps = 0
    names = []
    for zp in ZIPS:
        try:
            with zipfile.ZipFile(zp) as zf:
                names += [(zp, n) for n in zf.namelist() if n.endswith(".json")]
        except Exception:
            continue
    print(f"scanning {len(names)} episodes across {len(ZIPS)} day(s)", flush=True)
    _open = {}
    if True:
        for zp, name in names:
            if eps >= limit:
                break
            zf = _open.get(zp) or _open.setdefault(zp, zipfile.ZipFile(zp))
            try:
                d = json.loads(zf.open(name).read().decode("utf-8"))
            except Exception:
                continue
            rw = d.get("rewards") or []
            if 1 not in rw:
                continue
            w = rw.index(1)
            if markers:
                wd = None
                for st in d.get("steps", []):
                    if w < len(st):
                        a0 = st[w].get("action") or []
                        if len(a0) == 60:
                            wd = set(a0)
                            break
                if wd is None or not (wd & markers):
                    continue
            eps += 1
            hist = History()
            for st in d.get("steps", []):
                if w >= len(st):
                    continue
                ag = st[w]
                # INACTIVE frames repeat the previous observation verbatim
                # (measured 9969/9969), so their logs would be counted twice --
                # and the live agent never sees them. Skip them entirely.
                if ag.get("status") != "ACTIVE":
                    continue
                obs = ag.get("observation") or {}
                act = ag.get("action")
                hist.push(obs)
                if not act or not isinstance(act, list) or len(act) == 60:
                    continue
                sel = obs.get("select")
                if not sel:
                    continue
                opts = sel.get("option") or []
                if len(opts) < 2 or len(opts) > MAX_CAND:
                    continue
                me = (obs.get("current") or {}).get("yourIndex", 0)
                sf = state_featurize(obs, me)
                if sf is None:
                    continue
                chosen = act[0]
                if not isinstance(chosen, int) or chosen >= len(opts):
                    continue

                af, ac, mk = encode_options(obs, opts, me)
                ht, hc = hist.arrays()

                S.append(sf)
                A.append(af)
                AC.append(ac)
                M.append(mk)
                HT.append(ht)
                HC.append(hc)
                Ylab.append(chosen)
                G.append(eps)
            if eps % 20 == 0:
                print(f"  {eps} eps, {len(S)} decisions", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        S=np.asarray(S, dtype=np.float32), A=np.asarray(A, dtype=np.float32),
        AC=np.asarray(AC, dtype=np.int32), M=np.asarray(M, dtype=np.float32),
        HT=np.asarray(HT, dtype=np.int32), HC=np.asarray(HC, dtype=np.int32),
        Y=np.asarray(Ylab, dtype=np.int64), G=np.asarray(G, dtype=np.int32))
    print(f"\nepisodes={eps} decisions={len(S)} state_nf={STATE_NF} "
          f"max_cand={MAX_CAND} hist={HIST}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
