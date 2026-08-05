"""Value-net data: (state -> did THIS player win), from both points of view.

Different target from the failed BC run. That one asked "was this option
chosen", which collapsed to the engine's option ordering (0.3719 vs a 0.3665
always-first baseline). This asks "is this position winning", which the replay
labels exactly and which no ordering prior can fake.

Three things that matter for label quality:
  * BOTH POVs of every game, so the net sees losing states. The dataset is
    winner-biased; training only on winners teaches "everything is a win".
  * Labels DISCOUNTED toward 0.5 by distance from the end. A turn-2 board
    barely predicts the outcome; a hard 1/0 there is mostly noise.
  * One row per DECISION, not per option -- this is a state evaluator.

Usage: python work/tools/val_extract.py <n_episodes> <out.npz>
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

FEATNAMES = [
    "my_prizes", "op_prizes", "prize_diff",
    "my_hand", "op_hand", "hand_diff",
    "my_deck", "op_deck",
    "turn", "is_first",
    "my_act_hp", "my_act_maxhp", "my_act_hpfrac", "my_act_energy",
    "op_act_hp", "op_act_maxhp", "op_act_hpfrac", "op_act_energy",
    "my_bench", "op_bench",
    "my_board_hp", "op_board_hp", "board_hp_diff",
    "my_board_energy", "op_board_energy",
    "my_prize_liability", "op_prize_liability",
    "my_discard", "op_discard",
    "my_act_bestdmg", "op_act_bestdmg", "dmg_diff",
    "my_poisoned", "my_burned", "my_asleep", "my_paralyzed", "my_confused",
    "op_poisoned", "op_burned", "op_asleep", "op_paralyzed", "op_confused",
    "my_act_is_ex", "op_act_is_ex", "supporter_played", "energy_attached",
]
NF = len(FEATNAMES)


def _mon(p):
    a = p.get("active") or []
    return a[0] if (a and isinstance(a[0], dict)) else None


def _bestdmg(mon):
    if not mon:
        return 0.0
    c = CARDS.get(mon.get("id"))
    if c is None:
        return 0.0
    best = 0
    for aid in (c.attacks or []):
        a = ATK.get(aid)
        if a and a.damage:
            best = max(best, a.damage)
    return best / 100.0


def _prize_liability(p):
    """How many prizes the opponent collects for knocking our board out."""
    tot = 0.0
    for mon in ([_mon(p)] + list(p.get("bench") or [])):
        if not isinstance(mon, dict):
            continue
        c = CARDS.get(mon.get("id"))
        if c is None:
            tot += 1
        elif getattr(c, "megaEx", False):
            tot += 3
        elif getattr(c, "ex", False):
            tot += 2
        else:
            tot += 1
    return tot


def _boardsum(p, key):
    tot = 0.0
    for mon in ([_mon(p)] + list(p.get("bench") or [])):
        if isinstance(mon, dict):
            if key == "hp":
                tot += mon.get("hp") or 0
            else:
                tot += len(mon.get("energies") or [])
    return tot


def featurize(obs, me):
    cur = obs.get("current") or {}
    pls = cur.get("players") or []
    if len(pls) < 2:
        return None
    mine, opp = pls[me], pls[1 - me]
    ma, oa = _mon(mine), _mon(opp)
    f = np.zeros(NF, dtype=np.float32)
    v = [
        len(mine.get("prize") or []), len(opp.get("prize") or []),
        len(opp.get("prize") or []) - len(mine.get("prize") or []),
        mine.get("handCount") or 0, opp.get("handCount") or 0,
        (mine.get("handCount") or 0) - (opp.get("handCount") or 0),
        (mine.get("deckCount") or 0) / 10.0, (opp.get("deckCount") or 0) / 10.0,
        (cur.get("turn") or 0) / 10.0,
        1.0 if cur.get("firstPlayer") == me else 0.0,
        (ma or {}).get("hp", 0) / 100.0, (ma or {}).get("maxHp", 0) / 100.0,
        ((ma or {}).get("hp", 0) / max((ma or {}).get("maxHp", 1), 1)) if ma else 0.0,
        len((ma or {}).get("energies") or []),
        (oa or {}).get("hp", 0) / 100.0, (oa or {}).get("maxHp", 0) / 100.0,
        ((oa or {}).get("hp", 0) / max((oa or {}).get("maxHp", 1), 1)) if oa else 0.0,
        len((oa or {}).get("energies") or []),
        len(mine.get("bench") or []), len(opp.get("bench") or []),
        _boardsum(mine, "hp") / 100.0, _boardsum(opp, "hp") / 100.0,
        (_boardsum(mine, "hp") - _boardsum(opp, "hp")) / 100.0,
        _boardsum(mine, "e"), _boardsum(opp, "e"),
        _prize_liability(mine), _prize_liability(opp),
        len(mine.get("discard") or []) / 10.0, len(opp.get("discard") or []) / 10.0,
        _bestdmg(ma), _bestdmg(oa), _bestdmg(ma) - _bestdmg(oa),
    ]
    for p in (mine, opp):
        for k in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
            v.append(1.0 if p.get(k) else 0.0)
    for mon in (ma, oa):
        c = CARDS.get((mon or {}).get("id"))
        v.append(1.0 if (c is not None and (getattr(c, "ex", False)
                                            or getattr(c, "megaEx", False))) else 0.0)
    v.append(1.0 if cur.get("supporterPlayed") else 0.0)
    v.append(1.0 if cur.get("energyAttached") else 0.0)
    for i, x in enumerate(v[:NF]):
        f[i] = float(x)
    return f


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(WORK, "out", "val_data.npz")
    GAMMA = 0.97

    X, Y, W, G = [], [], [], []
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
            if len(rw) != 2 or 1 not in rw:
                continue
            eps += 1
            steps = d.get("steps") or []
            n = len(steps)
            # BOTH points of view: one win row set, one loss row set
            for me in (0, 1):
                won = 1.0 if rw[me] == 1 else 0.0
                rows = []
                for si, st in enumerate(steps):
                    if me >= len(st):
                        continue
                    obs = st[me].get("observation") or {}
                    if not obs.get("current"):
                        continue
                    if not (st[me].get("action")):
                        continue
                    ft = featurize(obs, me)
                    if ft is None:
                        continue
                    rows.append((si, ft))
                for si, ft in rows:
                    dist = max(0, n - si)
                    # discount a hard win/loss label toward 0.5 far from the end
                    lab = 0.5 + (won - 0.5) * (GAMMA ** (dist / 4.0))
                    X.append(ft)
                    Y.append(lab)
                    W.append(1.0)
                    G.append(eps)          # group by GAME for an honest split
            if eps % 25 == 0:
                print(f"  {eps} eps, {len(X)} rows", flush=True)

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    G = np.asarray(G, dtype=np.int32)
    np.savez_compressed(out, X=X, Y=Y, G=G, names=np.array(FEATNAMES))
    print(f"\nepisodes={eps} rows={len(X)} features={NF}")
    print(f"label mean={Y.mean():.4f} (0.5 == balanced; both POVs kept)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
