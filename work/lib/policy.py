"""Heuristic policy for the PTCG cabt engine.

Design rules (these exist because the ladder punishes crashes far harder than
bad play):
  * NEVER raise. Every path ends in a legal selection.
  * Every SelectContext has an explicit or defaulted handler.
  * No wall-clock assumptions: this is pure table lookup, microseconds per move.
"""
from cg.api import (
    AreaType, CardType, EnergyType, Observation, OptionType, SelectContext,
    SelectType, all_attack, all_card_data, to_observation_class,
)

# ---------------------------------------------------------------- card tables
_CARDS = None
_ATTACKS = None


def tables():
    global _CARDS, _ATTACKS
    if _CARDS is None:
        _CARDS = {c.cardId: c for c in all_card_data()}
        _ATTACKS = {a.attackId: a for a in all_attack()}
    return _CARDS, _ATTACKS


def _i(x, default=-1):
    """Coerce enum/int/None to int safely."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- evaluation
def attack_damage(cards, attacks, attacker, defender):
    """Estimate damage of each attack the attacker could use, incl. weakness."""
    out = {}
    acard = cards.get(attacker.id) if attacker else None
    if acard is None:
        return out
    dcard = cards.get(defender.id) if defender else None
    for aid in acard.attacks:
        a = attacks.get(aid)
        if a is None:
            continue
        dmg = a.damage or 0
        if dmg and dcard is not None and dcard.weakness is not None:
            if _i(dcard.weakness) == _i(acard.energyType):
                dmg *= 2
        out[aid] = dmg
    return out


def can_pay(attacker, atk):
    """Does the attacker have enough energy for this attack?

    Energy list on a Pokemon is the resolved per-unit types. We check the
    coloured requirements can be met and the total count is sufficient.
    """
    if atk is None:
        return False
    need = [_i(e) for e in atk.energies]
    have = [_i(e) for e in (attacker.energies or [])]
    if len(have) < len(need):
        return False
    pool = list(have)
    # satisfy coloured costs first, colourless (0) last
    for want in sorted(need, key=lambda w: w == 0):
        if want == 0:
            pool.pop()
            continue
        hit = None
        for j, h in enumerate(pool):
            # RAINBOW(10) pays anything; TEAM_ROCKET(11) pays psychic/dark
            if h == want or h == 10 or (h == 11 and want in (5, 7)):
                hit = j
                break
        if hit is None:
            return False
        pool.pop(hit)
    return True


# ---------------------------------------------------------------- main policy
def choose(obs: Observation, deck_ids):
    """Return a list of option indices. Always legal."""
    sd = obs.select
    n = len(sd.option)
    lo, hi = sd.minCount, min(sd.maxCount, n)
    if n == 0 or hi == 0:
        return []

    cards, attacks = tables()
    st = obs.current
    me = st.players[st.yourIndex] if st else None
    opp = st.players[1 - st.yourIndex] if st else None
    my_act = me.active[0] if (me and me.active and me.active[0]) else None
    op_act = opp.active[0] if (opp and opp.active and opp.active[0]) else None

    stype = _i(sd.type)
    ctx = _i(sd.context)

    # ---- MAIN: the real decision -------------------------------------
    if stype == _i(SelectType.MAIN):
        idx = _main(sd, cards, attacks, st, me, opp, my_act, op_act)
        return [idx]

    # ---- ATTACK selection --------------------------------------------
    if stype == _i(SelectType.ATTACK):
        best, best_s = 0, -1e9
        for i, o in enumerate(sd.option):
            a = attacks.get(o.attackId)
            s = _score_attack(a, my_act, op_act, cards, attacks)
            if ctx == _i(SelectContext.DISABLE_ATTACK):
                s = -s  # we're disabling the opponent's: kill their best
            if s > best_s:
                best, best_s = i, s
        return [best]

    # ---- YES / NO ----------------------------------------------------
    if stype == _i(SelectType.YES_NO):
        want = _yes_no(ctx, st, me, opp)
        for i, o in enumerate(sd.option):
            if _i(o.type) == _i(OptionType.YES if want else OptionType.NO):
                return [i]
        return list(range(lo)) if lo else [0]

    # ---- COUNT: usually take the max (draw more, more counters) -------
    if stype == _i(SelectType.COUNT):
        pick, best_v = 0, None
        for i, o in enumerate(sd.option):
            v = o.number if o.number is not None else 0
            if best_v is None or v > best_v:
                pick, best_v = i, v
        return [pick]

    # ---- CARD-ish selections: score by context ------------------------
    scored = sorted(
        range(n),
        key=lambda i: -_score_card(sd.option[i], ctx, cards, attacks, st,
                                   me, opp, my_act, op_act, deck_ids),
    )
    k = hi if _greedy_max(ctx) else max(lo, 1)
    k = max(lo, min(k, hi))
    return scored[:k]


def _greedy_max(ctx):
    """Contexts where taking the maximum allowed number is right."""
    return ctx in (
        _i(SelectContext.TO_HAND), _i(SelectContext.SETUP_BENCH_POKEMON),
        _i(SelectContext.TO_FIELD), _i(SelectContext.TO_BENCH),
        _i(SelectContext.HEAL), _i(SelectContext.REMOVE_DAMAGE_COUNTER),
        _i(SelectContext.DAMAGE_COUNTER), _i(SelectContext.DAMAGE_COUNTER_ANY),
        _i(SelectContext.DAMAGE), _i(SelectContext.LOOK),
    )


def _score_attack(a, my_act, op_act, cards, attacks):
    if a is None:
        return -1e6
    dmg = a.damage or 0
    if dmg and op_act is not None:
        dcard = cards.get(op_act.id)
        acard = cards.get(my_act.id) if my_act else None
        if dcard is not None and acard is not None and dcard.weakness is not None:
            if _i(dcard.weakness) == _i(acard.energyType):
                dmg *= 2
    s = float(dmg)
    if op_act is not None and dmg >= op_act.hp:
        s += 10000.0  # lethal now
        ocard = cards.get(op_act.id)
        if ocard is not None:
            if ocard.megaEx:
                s += 3000
            elif ocard.ex:
                s += 2000
    txt = (a.text or "").lower()
    if "discard all energy from this" in txt:
        s -= 60
    elif "discard 2 energy from this" in txt:
        s -= 40
    elif "discard an energy from this" in txt or "discard 1 energy" in txt:
        s -= 20
    if "can’t use" in txt or "can't use" in txt:
        s -= 25
    if "damage to itself" in txt:
        s -= 15
    return s


def _yes_no(ctx, st, me, opp):
    if ctx == _i(SelectContext.IS_FIRST):
        # Going first: no attack on turn 1 but you develop the board first and
        # cannot be attacked on turn 1 either. In this engine setup-speed wins.
        return True
    if ctx == _i(SelectContext.MULLIGAN):
        return True          # redraw a bad opener
    if ctx == _i(SelectContext.COIN_HEAD):
        return True          # arbitrary but consistent
    if ctx == _i(SelectContext.ACTIVATE):
        return True          # abilities/effects are opt-in value
    if ctx == _i(SelectContext.FIRST_EFFECT):
        return True
    if ctx == _i(SelectContext.MORE_DEVOLVE):
        return True
    return True


def _card_of(cards, opt):
    return cards.get(opt.cardId) if opt.cardId is not None else None


def _score_card(o, ctx, cards, attacks, st, me, opp, my_act, op_act, deck_ids):
    """Score a CARD/ENERGY/etc option for a non-MAIN selection."""
    otype = _i(o.type)
    c = _card_of(cards, o)
    mine = (o.playerIndex is None or st is None or o.playerIndex == st.yourIndex)

    # value of a pokemon card as a body
    def body(cc):
        if cc is None:
            return 0.0
        v = (cc.hp or 0) / 10.0
        best = 0
        for aid in cc.attacks:
            a = attacks.get(aid)
            if a and a.damage:
                best = max(best, a.damage)
        v += best / 10.0
        if cc.megaEx:
            v += 6
        elif cc.ex:
            v += 4
        v -= (cc.retreatCost or 0) * 0.5
        return v

    # --- put a pokemon into play / active / bench -------------------
    if ctx in (_i(SelectContext.SETUP_ACTIVE_POKEMON), _i(SelectContext.TO_ACTIVE),
               _i(SelectContext.SWITCH)):
        return body(c)
    if ctx in (_i(SelectContext.SETUP_BENCH_POKEMON), _i(SelectContext.TO_BENCH),
               _i(SelectContext.TO_FIELD)):
        return body(c)

    # --- fetch cards to hand: prefer pokemon > supporter > energy ----
    if ctx in (_i(SelectContext.TO_HAND), _i(SelectContext.LOOK)):
        if c is None:
            return 0.0
        ct = _i(c.cardType)
        base = {0: 8.0, 3: 6.0, 1: 4.0, 2: 2.0, 4: 1.0, 6: 3.0, 5: 2.5}.get(ct, 1.0)
        return base + body(c) * 0.3

    # --- discard / to-deck: dump the least useful --------------------
    if ctx in (_i(SelectContext.DISCARD), _i(SelectContext.TO_DECK),
               _i(SelectContext.TO_DECK_BOTTOM),
               _i(SelectContext.DISCARD_CARD_OR_ATTACHED_CARD)):
        if c is None:
            return 0.0
        ct = _i(c.cardType)
        keep = {0: 8.0, 3: 6.0, 1: 4.0, 2: 2.0, 4: 1.0, 6: 3.0, 5: 2.5}.get(ct, 1.0)
        return -(keep + body(c) * 0.3)      # least valuable first

    # --- damage / KO targets: opponent's most valuable ---------------
    if ctx in (_i(SelectContext.DAMAGE), _i(SelectContext.DAMAGE_COUNTER),
               _i(SelectContext.DAMAGE_COUNTER_ANY)):
        return (body(c) + 20) * (-1 if mine else 1)

    # --- heal / remove counters: our most valuable --------------------
    if ctx in (_i(SelectContext.HEAL), _i(SelectContext.REMOVE_DAMAGE_COUNTER)):
        return (body(c) + 20) * (1 if mine else -1)

    # --- attach energy: onto our active, else biggest attacker --------
    if ctx in (_i(SelectContext.ATTACH_FROM), _i(SelectContext.ATTACH_TO),
               _i(SelectContext.EFFECT_TARGET)):
        s = body(c)
        if _i(o.area) == _i(AreaType.ACTIVE) and mine:
            s += 15
        return s

    # --- detach / discard energy: take it off THEIR stuff -------------
    if ctx in (_i(SelectContext.DISCARD_ENERGY), _i(SelectContext.DISCARD_ENERGY_CARD),
               _i(SelectContext.DISCARD_TOOL_CARD), _i(SelectContext.DETACH_FROM),
               _i(SelectContext.SWITCH_ENERGY), _i(SelectContext.SWITCH_ENERGY_CARD),
               _i(SelectContext.TO_HAND_ENERGY), _i(SelectContext.TO_DECK_ENERGY)):
        base = 10.0
        if _i(o.area) == _i(AreaType.ACTIVE):
            base += 5
        return base * (-1 if mine else 1)

    # --- evolution --------------------------------------------------
    if ctx in (_i(SelectContext.EVOLVES_FROM), _i(SelectContext.EVOLVES_TO),
               _i(SelectContext.EVOLVE)):
        return body(c) + (15 if _i(o.area) == _i(AreaType.ACTIVE) else 0)
    if ctx == _i(SelectContext.DEVOLVE):
        return body(c) * (-1 if mine else 1)

    # --- prizes / leave-in-place / misc ------------------------------
    if ctx == _i(SelectContext.NOT_MOVE):
        return body(c)
    if ctx == _i(SelectContext.TO_PRIZE):
        return -body(c)

    if otype == _i(OptionType.ENERGY) or otype == _i(OptionType.ENERGY_CARD):
        return 1.0 * (-1 if mine else 1)
    return body(c)


# ---------------------------------------------------------------- MAIN turn
def _main(sd, cards, attacks, st, me, opp, my_act, op_act):
    """Pick one MAIN option. Returns an index into sd.option."""
    best_i, best_s = 0, -1e18

    # what's the best attack we could use right now?
    lethal_idx, lethal_s = None, -1e18
    for i, o in enumerate(sd.option):
        if _i(o.type) != _i(OptionType.ATTACK):
            continue
        a = attacks.get(o.attackId)
        s = _score_attack(a, my_act, op_act, cards, attacks)
        if s > lethal_s:
            lethal_idx, lethal_s = i, s
    has_lethal = lethal_s >= 10000.0

    for i, o in enumerate(sd.option):
        t = _i(o.type)
        s = 0.0

        if t == _i(OptionType.ATTACK):
            a = attacks.get(o.attackId)
            s = 1000.0 + _score_attack(a, my_act, op_act, cards, attacks)

        elif t == _i(OptionType.ABILITY):
            s = 2600.0                     # free value, use before attacking

        elif t == _i(OptionType.EVOLVE):
            c = _card_of(cards, o)
            s = 2500.0 + ((c.hp or 0) / 10.0 if c else 0)
            if _i(o.inPlayArea) == _i(AreaType.ACTIVE):
                s += 40

        elif t == _i(OptionType.ATTACH):
            c = _card_of(cards, o)
            s = 2300.0
            if _i(o.inPlayArea) == _i(AreaType.ACTIVE):
                s += 60                     # power the attacker first
            if c is not None and _i(c.cardType) == _i(CardType.TOOL):
                s += 10

        elif t == _i(OptionType.PLAY):
            c = _card_of(cards, o)
            s = 2000.0
            if c is not None:
                ct = _i(c.cardType)
                if ct == _i(CardType.POKEMON):
                    # benching more basics early is how you avoid losing to a
                    # single KO; late, extra bodies are dead cards
                    nbench = len(me.bench) if me else 0
                    s += 400 - nbench * 60
                elif ct == _i(CardType.SUPPORTER):
                    s += 250
                elif ct == _i(CardType.ITEM):
                    s += 300
                elif ct == _i(CardType.STADIUM):
                    s += 50
                elif ct in (_i(CardType.BASIC_ENERGY), _i(CardType.SPECIAL_ENERGY)):
                    s += 100

        elif t == _i(OptionType.RETREAT):
            # only worth it if the active is nearly dead and we have a bench
            s = 100.0
            if my_act is not None and my_act.maxHp:
                frac = my_act.hp / float(my_act.maxHp)
                if frac < 0.35 and me and len(me.bench) > 0:
                    s = 2450.0

        elif t == _i(OptionType.DISCARD):
            s = 50.0

        elif t == _i(OptionType.END):
            s = 10.0

        if s > best_s:
            best_i, best_s = i, s

    # if we can KO right now, do it — nothing else this turn matters more
    if has_lethal and lethal_idx is not None:
        return lethal_idx
    return best_i


# ---------------------------------------------------------------- entry point
def act(obs_dict: dict, deck_ids) -> list:
    """Top-level: never raises."""
    try:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return list(deck_ids)
        return choose(obs, deck_ids)
    except Exception:
        # guaranteed-legal fallback
        try:
            sel = obs_dict.get("select") or {}
            n = len(sel.get("option") or [])
            lo = int(sel.get("minCount") or 0)
            hi = int(sel.get("maxCount") or 0)
            k = max(lo, min(1 if hi >= 1 else 0, hi))
            k = min(k, n)
            return list(range(k))
        except Exception:
            return [0]
