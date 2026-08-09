"""Position features, computed ONCE, for both training and search inference.

The value net has to score two kinds of object: replay observations, which are
plain dicts, and search leaves, which `search_step` hands back as observation
CLASSES. Writing one featurizer per API is how a train/inference mismatch gets
in -- the net would then score noise at the leaf and the experiment would read
as "value nets do not help" when what failed was feature alignment. That failure
is invisible: no exception, no warning, just a worse agent.

So this module defines the features once against the class API, and the training
extractor converts replay dicts with `to_observation_class` before calling it.
Same code path, same numbers, by construction.

Order matters and is fixed by NAMES -- valnet.W is a plain weight vector, so a
reordering here silently rescores every feature. Do not reorder without
retraining.
"""

NAMES = [
    'my_prizes', 'op_prizes', 'prize_diff', 'my_hand', 'op_hand', 'hand_diff',
    'my_deck', 'op_deck', 'turn', 'is_first', 'my_act_hp', 'my_act_maxhp',
    'my_act_hpfrac', 'my_act_energy', 'op_act_hp', 'op_act_maxhp',
    'op_act_hpfrac', 'op_act_energy', 'my_bench', 'op_bench', 'my_board_hp',
    'op_board_hp', 'board_hp_diff', 'my_board_energy', 'op_board_energy',
    'my_prize_liability', 'op_prize_liability', 'my_discard', 'op_discard',
    'my_act_bestdmg', 'op_act_bestdmg', 'dmg_diff', 'my_poisoned', 'my_burned',
    'my_asleep', 'my_paralyzed', 'my_confused', 'op_poisoned', 'op_burned',
    'op_asleep', 'op_paralyzed', 'op_confused', 'my_act_is_ex',
    'op_act_is_ex', 'supporter_played', 'energy_attached',
]
NF = len(NAMES)

_ATTACKS = None
_CARDS = None


def _tables():
    global _ATTACKS, _CARDS
    if _ATTACKS is None:
        try:
            from cg.api import all_attack, all_card_data
            _ATTACKS = {a.attackId: a for a in all_attack()}
            _CARDS = {c.cardId: c for c in all_card_data()}
        except Exception:
            _ATTACKS, _CARDS = {}, {}
    return _ATTACKS, _CARDS


def _n(x, d=0.0):
    try:
        if x is None:
            return d
        return float(x)
    except Exception:
        return d


def _side(p):
    """Active plus bench, skipping empty slots."""
    out = []
    try:
        act = p.active or []
        if act and act[0] is not None:
            out.append(act[0])
    except Exception:
        pass
    try:
        for b in (p.bench or []):
            if b is not None:
                out.append(b)
    except Exception:
        pass
    return out


def _active(p):
    try:
        act = p.active or []
        return act[0] if act and act[0] is not None else None
    except Exception:
        return None


def _energy(mon):
    try:
        return float(len(mon.energyCards or []))
    except Exception:
        return 0.0


def _prize_value(mon, cards):
    c = cards.get(getattr(mon, "id", None))
    if c is None:
        return 1.0
    if getattr(c, "megaEx", False):
        return 3.0
    if getattr(c, "ex", False):
        return 2.0
    return 1.0


def _bestdmg(mon, attacks):
    best = 0.0
    try:
        for a in (mon.attacks or []):
            aid = getattr(a, "attackId", None)
            atk = attacks.get(aid) if aid is not None else None
            dmg = _n(getattr(atk, "damage", None) if atk is not None
                     else getattr(a, "damage", None))
            if dmg > best:
                best = dmg
    except Exception:
        pass
    return best


def _status(mon, key):
    try:
        v = getattr(mon, key, None)
        if v is None:
            return 0.0
        return 1.0 if v else 0.0
    except Exception:
        return 0.0


def features(o, me_idx):
    """Class observation -> list[float] in NAMES order, or None."""
    attacks, cards = _tables()
    try:
        st = o.current
        if st is None:
            return None
        players = st.players or []
        if len(players) < 2:
            return None
        me = players[me_idx]
        op = players[1 - me_idx]
    except Exception:
        return None

    # prize LISTS shrink as their owner takes prizes
    my_prizes = 6.0 - len(getattr(me, "prize", None) or [])
    op_prizes = 6.0 - len(getattr(op, "prize", None) or [])
    my_hand = _n(getattr(me, "handCount", None))
    op_hand = _n(getattr(op, "handCount", None))
    my_deck = _n(getattr(me, "deckCount", None))
    op_deck = _n(getattr(op, "deckCount", None))

    ma, oa = _active(me), _active(op)
    my_hp = _n(getattr(ma, "hp", None)) if ma is not None else 0.0
    my_max = _n(getattr(ma, "maxHp", None)) if ma is not None else 0.0
    op_hp = _n(getattr(oa, "hp", None)) if oa is not None else 0.0
    op_max = _n(getattr(oa, "maxHp", None)) if oa is not None else 0.0

    my_side, op_side = _side(me), _side(op)
    my_board_hp = sum(_n(getattr(p, "hp", None)) for p in my_side)
    op_board_hp = sum(_n(getattr(p, "hp", None)) for p in op_side)
    my_board_en = sum(_energy(p) for p in my_side)
    op_board_en = sum(_energy(p) for p in op_side)
    my_liab = sum(_prize_value(p, cards) for p in my_side)
    op_liab = sum(_prize_value(p, cards) for p in op_side)
    my_disc = _n(len(getattr(me, "discard", None) or []))
    op_disc = _n(len(getattr(op, "discard", None) or []))
    my_dmg = _bestdmg(ma, attacks) if ma is not None else 0.0
    op_dmg = _bestdmg(oa, attacks) if oa is not None else 0.0

    try:
        is_first = 1.0 if int(getattr(st, "firstPlayer", -1)) == me_idx else 0.0
    except Exception:
        is_first = 0.0

    return [
        my_prizes, op_prizes, my_prizes - op_prizes,
        my_hand, op_hand, my_hand - op_hand,
        my_deck, op_deck, _n(getattr(st, "turn", None)), is_first,
        my_hp, my_max, (my_hp / my_max) if my_max else 0.0,
        _energy(ma) if ma is not None else 0.0,
        op_hp, op_max, (op_hp / op_max) if op_max else 0.0,
        _energy(oa) if oa is not None else 0.0,
        float(max(0, len(my_side) - 1)), float(max(0, len(op_side) - 1)),
        my_board_hp, op_board_hp, my_board_hp - op_board_hp,
        my_board_en, op_board_en,
        my_liab, op_liab,
        my_disc, op_disc,
        my_dmg, op_dmg, my_dmg - op_dmg,
        _status(ma, "poisoned"), _status(ma, "burned"), _status(ma, "asleep"),
        _status(ma, "paralyzed"), _status(ma, "confused"),
        _status(oa, "poisoned"), _status(oa, "burned"), _status(oa, "asleep"),
        _status(oa, "paralyzed"), _status(oa, "confused"),
        1.0 if (ma is not None and getattr(cards.get(getattr(ma, "id", None)),
                                           "ex", False)) else 0.0,
        1.0 if (oa is not None and getattr(cards.get(getattr(oa, "id", None)),
                                           "ex", False)) else 0.0,
        1.0 if getattr(st, "supporterPlayed", False) else 0.0,
        1.0 if getattr(st, "energyAttached", False) else 0.0,
    ]
