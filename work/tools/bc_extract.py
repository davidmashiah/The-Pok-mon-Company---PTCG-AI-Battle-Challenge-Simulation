"""Behavioural cloning, step 1: turn winner replays into (features, label) rows.

The action space is variable -- each frame offers a different number and mix of
options -- so this is framed as pointwise ranking rather than classification:
every OPTION becomes a row, labelled 1 if the winner chose it and 0 otherwise.
At inference we score each legal option and take the argmax.

Features are deliberately cheap and dependency-free so the shipped model can be
a dot product in pure Python.

Usage:
  python work/tools/bc_extract.py <n_episodes> <out.npz> [markers]
"""
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
ZIP = os.path.join(ROOT, "data", "episodes", "d0802",
                   "pokemon-tcg-ai-battle-episodes-2026-08-02.zip")

sys.path.insert(0, os.path.join(WORK, "lib"))
from cg.api import all_card_data, all_attack  # noqa: E402

CARDS = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}

N_OPT_TYPE = 17
N_CTX = 50
N_CARD_HASH = 64          # hashed card identity

FEATNAMES = (
    [f"opt_type_{i}" for i in range(N_OPT_TYPE)]
    + [f"ctx_{i}" for i in range(N_CTX)]
    + [f"card_h{i}" for i in range(N_CARD_HASH)]
    + ["is_mine", "area", "index_norm",
       "card_is_pokemon", "card_is_item", "card_is_tool", "card_is_supporter",
       "card_is_stadium", "card_is_basic_energy", "card_is_special_energy",
       "card_hp", "card_retreat", "card_is_basic", "card_is_stage1",
       "card_is_stage2", "card_is_ex", "card_is_megaex", "card_is_acespec",
       "card_best_dmg", "card_prize_value",
       "atk_damage", "atk_cost",
       "my_prizes", "op_prizes", "prize_diff", "my_hand", "op_hand",
       "my_deck", "op_deck", "turn", "turn_actions",
       "my_active_hp", "my_active_maxhp", "op_active_hp", "op_active_maxhp",
       "my_bench", "op_bench", "my_active_energy", "op_active_energy",
       "supporter_played", "energy_attached", "retreated", "n_options"]
)
NF = len(FEATNAMES)


def prize_value(c):
    if c is None:
        return 1.0
    if getattr(c, "megaEx", False):
        return 3.0
    if getattr(c, "ex", False):
        return 2.0
    return 1.0


def card_at(obs, opt, me):
    a = opt.get("area")
    i = opt.get("index")
    pi = opt.get("playerIndex")
    if pi is None:
        pi = me
    cur = obs.get("current") or {}
    pls = cur.get("players") or []
    if not isinstance(pi, int) or pi >= len(pls) or not isinstance(i, int):
        return None
    p = pls[pi]
    try:
        if a == 1:
            return ((obs.get("select") or {}).get("deck") or [])[i]
        if a == 2:
            return (p.get("hand") or [])[i]
        if a == 3:
            return (p.get("discard") or [])[i]
        if a == 4:
            return (p.get("active") or [])[i]
        if a == 5:
            return (p.get("bench") or [])[i]
        if a == 6:
            return (p.get("prize") or [])[i]
    except Exception:
        return None
    return None


def featurize(obs, sel, opt, me, n_opts):
    f = np.zeros(NF, dtype=np.float32)
    k = 0
    t = opt.get("type")
    if isinstance(t, int) and 0 <= t < N_OPT_TYPE:
        f[k + t] = 1.0
    k += N_OPT_TYPE
    ctx = sel.get("context")
    if isinstance(ctx, int) and 0 <= ctx < N_CTX:
        f[k + ctx] = 1.0
    k += N_CTX

    cur = obs.get("current") or {}
    pls = cur.get("players") or [{}, {}]
    mine = pls[me] if me < len(pls) else {}
    opp = pls[1 - me] if (1 - me) < len(pls) else {}

    pi = opt.get("playerIndex")
    f[k] = 1.0 if (pi is None or pi == me) else 0.0
    k += 1
    f[k] = float(opt.get("area") or 0)
    k += 1
    idx = opt.get("index")
    f[k] = float(idx) / 10.0 if isinstance(idx, int) else 0.0
    k += 1

    cd = card_at(obs, opt, me)
    cid = cd.get("id") if isinstance(cd, dict) else None
    if isinstance(cid, int):
        f[k + (cid % N_CARD_HASH)] = 1.0
    k += N_CARD_HASH
    c = CARDS.get(cid)
    ct = int(c.cardType) if c is not None else -1
    for j, want in enumerate((0, 1, 2, 3, 4, 5, 6)):
        f[k + j] = 1.0 if ct == want else 0.0
    k += 7
    f[k] = (c.hp or 0) / 100.0 if c is not None else 0.0
    k += 1
    f[k] = float(c.retreatCost or 0) if c is not None else 0.0
    k += 1
    for attr in ("basic", "stage1", "stage2", "ex", "megaEx", "aceSpec"):
        f[k] = 1.0 if (c is not None and getattr(c, attr, False)) else 0.0
        k += 1
    best = 0
    if c is not None:
        for aid in (c.attacks or []):
            a = ATK.get(aid)
            if a and a.damage:
                best = max(best, a.damage)
    f[k] = best / 100.0
    k += 1
    f[k] = prize_value(c)
    k += 1

    aid = opt.get("attackId")
    a = ATK.get(aid) if aid is not None else None
    f[k] = (a.damage or 0) / 100.0 if a else 0.0
    k += 1
    f[k] = float(len(a.energies)) if a else 0.0
    k += 1

    def act0(p, key):
        arr = p.get("active") or []
        if arr and isinstance(arr[0], dict):
            return arr[0].get(key) or 0
        return 0

    vals = [
        len(mine.get("prize") or []), len(opp.get("prize") or []),
        len(opp.get("prize") or []) - len(mine.get("prize") or []),
        mine.get("handCount") or 0, opp.get("handCount") or 0,
        (mine.get("deckCount") or 0) / 10.0, (opp.get("deckCount") or 0) / 10.0,
        (cur.get("turn") or 0) / 10.0, (cur.get("turnActionCount") or 0) / 10.0,
        act0(mine, "hp") / 100.0, act0(mine, "maxHp") / 100.0,
        act0(opp, "hp") / 100.0, act0(opp, "maxHp") / 100.0,
        len(mine.get("bench") or []), len(opp.get("bench") or []),
        len((mine.get("active") or [{}])[0].get("energies") or []) if (mine.get("active") or [{}])[0] else 0,
        len((opp.get("active") or [{}])[0].get("energies") or []) if (opp.get("active") or [{}])[0] else 0,
        1.0 if cur.get("supporterPlayed") else 0.0,
        1.0 if cur.get("energyAttached") else 0.0,
        1.0 if cur.get("retreated") else 0.0,
        n_opts / 10.0,
    ]
    for v in vals:
        f[k] = float(v)
        k += 1
    return f


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(WORK, "out", "bc_data.npz")
    markers = set()
    if len(sys.argv) > 3 and sys.argv[3] != "-":
        markers = {int(x) for x in sys.argv[3].split(",")}

    X, Y, G = [], [], []
    gid = 0
    eps = 0
    with zipfile.ZipFile(ZIP) as zf:
        for name in [n for n in zf.namelist() if n.endswith(".json")]:
            if eps >= limit:
                break
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
                        a = st[w].get("action") or []
                        if len(a) == 60:
                            wd = set(a)
                            break
                if wd is None or not (wd & markers):
                    continue
            eps += 1
            if eps % 10 == 0:
                print(f"  {eps} eps, {len(X)} rows", flush=True)
            for st in d.get("steps", []):
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
                for i, o in enumerate(opts):
                    try:
                        X.append(featurize(obs, sel, o, me, len(opts)))
                    except Exception:
                        continue
                    Y.append(1 if i in chosen else 0)
                    G.append(gid)
                gid += 1

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)
    G = np.asarray(G, dtype=np.int32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, X=X, Y=Y, G=G, names=np.array(FEATNAMES))
    print(f"\nepisodes={eps} decisions={gid} rows={len(X)} features={NF}")
    print(f"positive rate={Y.mean():.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
