"""Single source of truth for DouZero features -- used by BOTH the offline
extractor and the live agent.

This module exists because of a measured bug. The replay stores per-frame log
DELTAS, and INACTIVE frames repeat the previous observation verbatim (measured:
9969/9969 INACTIVE frames duplicate the prior logs, vs 41/9969 ACTIVE frames).
Accumulating logs from every frame therefore double-counts history, and worse,
produces a stream the live agent -- which is only invoked on ACTIVE frames --
can never reproduce. Training on one distribution and serving on another is the
silent-failure mode that has already cost this project five components.

So: history is accumulated ONLY from frames the agent would actually see, and
both sides of the pipeline call the same functions in this file.
"""
import numpy as np

from cg.api import all_card_data, all_attack

CARDS = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}

MAX_CAND = 24          # candidates per decision (pad/mask beyond)
HIST = 24              # log events of history
CARD_VOCAB = 1400      # card ids run to ~1267; 0 reserved for "none"
ACT_NF = 10            # numeric features per candidate

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


# ------------------------------------------------------------ card descr ----
# Measured problem this solves: our deck's Pokemon appear ~0 times in the
# field's games (Makuhita/Hariyama/Dusk Ball: 0 of 264,093 option observations;
# Mega Lucario ex: 12). A per-id learned embedding is therefore untrained noise
# for exactly the cards we play. These descriptors are computed from the engine
# card table, so an unseen card still arrives with hp / stage / typing / attack
# costs / damage attached and the net can reason about it by ATTRIBUTE.
N_CARDTYPE = 8
N_ENERGY = 12
DESC_D = 16 + N_CARDTYPE + N_ENERGY          # 36
_DESC_TABLE = None


def _desc(c):
    d = np.zeros(DESC_D, dtype=np.float32)
    d[0] = (c.hp or 0) / 100.0
    d[1] = (c.retreatCost or 0) / 4.0
    atks = [ATK.get(a) for a in (c.attacks or [])]
    atks = [a for a in atks if a is not None]
    d[2] = len(atks) / 2.0
    if atks:
        dmgs = [(a.damage or 0) for a in atks]
        costs = [len(a.energies or []) for a in atks]
        d[3] = max(dmgs) / 100.0
        d[4] = min(dmgs) / 100.0
        d[5] = min(costs) / 4.0
        d[6] = max(costs) / 4.0
        # damage per energy: the single most useful attack-efficiency scalar
        d[7] = max((dm / max(co, 1)) for dm, co in zip(dmgs, costs)) / 100.0
        d[8] = 1.0 if any(co == 0 for co in costs) else 0.0
    d[9] = 1.0 if c.basic else 0.0
    d[10] = 1.0 if c.stage1 else 0.0
    d[11] = 1.0 if c.stage2 else 0.0
    d[12] = 1.0 if getattr(c, "ex", False) else 0.0
    d[13] = 1.0 if getattr(c, "megaEx", False) else 0.0
    d[14] = 1.0 if getattr(c, "aceSpec", False) else 0.0
    d[15] = 1.0 if (c.skills or []) else 0.0          # has an ability
    ct = int(c.cardType) if c.cardType is not None else -1
    if 0 <= ct < N_CARDTYPE:
        d[16 + ct] = 1.0
    et = int(c.energyType) if c.energyType is not None else -1
    if 0 <= et < N_ENERGY:
        d[16 + N_CARDTYPE + et] = 1.0
    return d


def card_desc_table():
    """(CARD_VOCAB, DESC_D) fixed table. Row i == card id i-1; row 0 == 'none'.

    Recomputed from the bundled engine on both sides rather than shipped as
    weights, so the training table and the runtime table cannot disagree.
    """
    global _DESC_TABLE
    if _DESC_TABLE is None:
        T = np.zeros((CARD_VOCAB, DESC_D), dtype=np.float32)
        for cid, c in CARDS.items():
            if isinstance(cid, int) and 0 <= cid < CARD_VOCAB - 1:
                try:
                    T[cid + 1] = _desc(c)
                except Exception:
                    pass
        _DESC_TABLE = T
    return _DESC_TABLE


# ---------------------------------------------------------------- state -----
def g(o, k, default=None):
    """Read a field from either a replay dict or an engine Observation object.

    Training data arrives as raw JSON dicts; the search evaluates engine objects
    with the same field names as attributes. Routing both through one accessor
    keeps a SINGLE featurize implementation -- duplicating it is exactly how
    train/serve skew gets introduced, and this project has already paid for that
    once with the history bug.
    """
    if o is None:
        return default
    if isinstance(o, dict):
        v = o.get(k, default)
    else:
        v = getattr(o, k, default)
    return default if v is None else v


def _mon(p):
    a = g(p, "active") or []
    try:
        return a[0] if len(a) and a[0] is not None else None
    except Exception:
        return None


def _bestdmg(mon):
    if mon is None:
        return 0.0
    c = CARDS.get(g(mon, "id"))
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
    for mon in ([_mon(p)] + list(g(p, "bench") or [])):
        if mon is None:
            continue
        c = CARDS.get(g(mon, "id"))
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
    for mon in ([_mon(p)] + list(g(p, "bench") or [])):
        if mon is not None:
            if key == "hp":
                tot += g(mon, "hp", 0) or 0
            else:
                tot += len(g(mon, "energies") or [])
    return tot


def featurize(obs, me):
    """obs may be a replay dict OR an engine Observation; see g()."""
    cur = g(obs, "current")
    pls = g(cur, "players") or []
    if len(pls) < 2:
        return None
    mine, opp = pls[me], pls[1 - me]
    ma, oa = _mon(mine), _mon(opp)
    f = np.zeros(NF, dtype=np.float32)
    v = [
        len(g(mine, "prize") or []), len(g(opp, "prize") or []),
        len(g(opp, "prize") or []) - len(g(mine, "prize") or []),
        g(mine, "handCount", 0), g(opp, "handCount", 0),
        g(mine, "handCount", 0) - g(opp, "handCount", 0),
        g(mine, "deckCount", 0) / 10.0, g(opp, "deckCount", 0) / 10.0,
        g(cur, "turn", 0) / 10.0,
        1.0 if g(cur, "firstPlayer", -1) == me else 0.0,
        g(ma, "hp", 0) / 100.0, g(ma, "maxHp", 0) / 100.0,
        (g(ma, "hp", 0) / max(g(ma, "maxHp", 1), 1)) if ma is not None else 0.0,
        len(g(ma, "energies") or []),
        g(oa, "hp", 0) / 100.0, g(oa, "maxHp", 0) / 100.0,
        (g(oa, "hp", 0) / max(g(oa, "maxHp", 1), 1)) if oa is not None else 0.0,
        len(g(oa, "energies") or []),
        len(g(mine, "bench") or []), len(g(opp, "bench") or []),
        _boardsum(mine, "hp") / 100.0, _boardsum(opp, "hp") / 100.0,
        (_boardsum(mine, "hp") - _boardsum(opp, "hp")) / 100.0,
        _boardsum(mine, "e"), _boardsum(opp, "e"),
        _prize_liability(mine), _prize_liability(opp),
        len(g(mine, "discard") or []) / 10.0, len(g(opp, "discard") or []) / 10.0,
        _bestdmg(ma), _bestdmg(oa), _bestdmg(ma) - _bestdmg(oa),
    ]
    for p in (mine, opp):
        for k in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
            v.append(1.0 if g(p, k) else 0.0)
    for mon in (ma, oa):
        c = CARDS.get(g(mon, "id"))
        v.append(1.0 if (c is not None and (getattr(c, "ex", False)
                                            or getattr(c, "megaEx", False))) else 0.0)
    v.append(1.0 if g(cur, "supporterPlayed") else 0.0)
    v.append(1.0 if g(cur, "energyAttached") else 0.0)
    for i, x in enumerate(v[:NF]):
        f[i] = float(x)
    return f


# --------------------------------------------------------------- actions ----
def card_of(obs, opt, me):
    a, i = opt.get("area"), opt.get("index")
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


def act_feats(obs, opt, me):
    """Numeric part of the action encoding (the card id is embedded separately)."""
    f = np.zeros(ACT_NF, dtype=np.float32)
    t = opt.get("type")
    f[0] = float(t if isinstance(t, int) else -1)
    f[1] = float(opt.get("area") or 0)
    idx = opt.get("index")
    f[2] = float(idx) / 10.0 if isinstance(idx, int) else 0.0
    pi = opt.get("playerIndex")
    f[3] = 1.0 if (pi is None or pi == me) else 0.0
    f[4] = float(opt.get("inPlayArea") or 0)
    aid = opt.get("attackId")
    a = ATK.get(aid) if aid is not None else None
    f[5] = (a.damage or 0) / 100.0 if a else 0.0
    f[6] = float(len(a.energies)) if a else 0.0
    cd = card_of(obs, opt, me)
    c = CARDS.get(cd.get("id")) if isinstance(cd, dict) else None
    f[7] = (c.hp or 0) / 100.0 if c is not None else 0.0
    f[8] = 1.0 if (c is not None and (getattr(c, "ex", False)
                                      or getattr(c, "megaEx", False))) else 0.0
    f[9] = float(int(c.cardType)) if c is not None else -1.0
    return f


def act_card_id(obs, opt, me):
    cd = card_of(obs, opt, me)
    cid = cd.get("id") if isinstance(cd, dict) else None
    return int(cid) + 1 if isinstance(cid, int) and 0 <= cid < CARD_VOCAB - 1 else 0


def encode_options(obs, opts, me):
    """-> (act_feats MAX_CANDxACT_NF, act_cards MAX_CAND, mask MAX_CAND)."""
    af = np.zeros((MAX_CAND, ACT_NF), dtype=np.float32)
    ac = np.zeros(MAX_CAND, dtype=np.int32)
    mk = np.zeros(MAX_CAND, dtype=np.float32)
    for i, o in enumerate(opts[:MAX_CAND]):
        af[i] = act_feats(obs, o, me)
        ac[i] = act_card_id(obs, o, me)
        mk[i] = 1.0
    return af, ac, mk


# --------------------------------------------------------------- history ----
class History:
    """Rolling window of log events, fed ONLY from frames the agent sees.

    Offline, callers must pass frames with status == "ACTIVE"; INACTIVE frames
    repeat the same logs and would double-count. Live, every call is an ACTIVE
    frame by construction, so the two streams match.
    """

    def __init__(self):
        self.t = []
        self.c = []

    def push(self, obs):
        for lg in (obs.get("logs") or []):
            self.t.append(int(lg.get("type") or 0))
            cid = lg.get("cardId")
            self.c.append(int(cid) + 1 if isinstance(cid, int)
                          and 0 <= cid < CARD_VOCAB - 1 else 0)
        if len(self.t) > 4 * HIST:                    # bound memory over a game
            self.t = self.t[-HIST:]
            self.c = self.c[-HIST:]

    def arrays(self):
        ht = np.zeros(HIST, dtype=np.int32)
        hc = np.zeros(HIST, dtype=np.int32)
        tt, tc = self.t[-HIST:], self.c[-HIST:]
        if tt:
            ht[-len(tt):] = tt
            hc[-len(tc):] = tc
        return ht, hc
