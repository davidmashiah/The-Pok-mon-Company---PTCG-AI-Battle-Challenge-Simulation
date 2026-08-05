"""Mega Lopunny ex -- TWO-COPY ROTATION pilot.

Built from turn-by-turn traces of winning games in the host's top-episode
dataset, not from reading card text. The previous attempt (v19) was written from
card text and lost 386 of 396 games.

THE ENGINE, as winners actually run it
    Mega Lopunny ex   S1, Colourless, HP330, retreat 1
      Gale Thrust [C]  60, +170 if this Pokemon moved from the Bench to the
                       Active Spot THIS TURN  -> 230 off ONE energy

    Air Balloon      retreat cost -CC  => Lopunny retreats for FREE
    every energy in the list provides {C}, so any of them powers Gale Thrust

    Run TWO Lopunny, both energised, both holding Air Balloon:
        Lopunny A attacks.
        Next turn retreat A for free -> that PROMOTES Lopunny B from the bench
        -> B satisfies "moved from Bench to Active this turn" -> 230 again.
        Retreat B next turn to promote A. Repeat.
    230 damage every turn, indefinitely.

    The bench engine comes FIRST: winners open with 5x Dunsparce, evolve into
    Dudunsparce (draw 3, then shuffles ITSELF back into the deck, so it recycles),
    and use Hilda to fetch an Evolution Pokemon + an Energy.

v19's mistakes, all visible in the traces:
    one Lopunny parked on the bench      -> winners rotate two
    Air Balloon on the cheap Active      -> winners put it on LOPUNNY
    attack immediately                   -> winners build the bench first
"""
import os
import sys

_KAGGLE = "/kaggle_simulations/agent"
if os.path.isdir(_KAGGLE) and _KAGGLE not in sys.path:
    sys.path.insert(0, _KAGGLE)

import policy  # noqa: E402
from cg.api import (  # noqa: E402
    AreaType, CardType, Observation, OptionType, SelectContext,
    to_observation_class,
)

LOPUNNY, BUNEARY = 849, 848
FROSLASS, SNORUNT = 861, 860
DUNSPARCE, DUDUNSPARCE = 305, 66
AIR_BALLOON = 1174
BATTLE_CAGE = 1264
GALE_THRUST, SPIKY_HOPPER = 1225, 1226
ENERGIES = {11, 13, 3}          # Mist / Enriching / Basic W -- all give {C}


def _i(x, d=-1):
    try:
        return int(x)
    except (TypeError, ValueError):
        return d


def _read_deck():
    for cand in ("deck.csv", os.path.join(_KAGGLE, "deck.csv")):
        try:
            if os.path.exists(cand):
                with open(cand) as fh:
                    rows = [ln.strip() for ln in fh if ln.strip()]
                d = [int(r) for r in rows[:60]]
                if len(d) == 60:
                    return d
        except Exception:
            continue
    raise RuntimeError("deck.csv not found")


DECK = _read_deck()
_last_turn = -10 ** 9
_promoted_turn = -10 ** 9


def _armed(p):
    """A Lopunny that can attack the moment it is promoted."""
    return (p is not None and p.id == LOPUNNY and len(p.energies or []) >= 1)


def _bench_armed_lopunny(me):
    for i, p in enumerate(me.bench or []):
        if _armed(p):
            return i
    return -1


def _score(obs, opts):
    st = obs.current
    me = st.players[st.yourIndex]
    act = me.active[0] if (me.active and me.active[0]) else None
    bench = me.bench or []
    n_bench = len(bench)
    lop_active = act is not None and act.id == LOPUNNY
    bench_lop = _bench_armed_lopunny(me)
    n_lop = sum(1 for p in [act] + list(bench) if p is not None and p.id == LOPUNNY)
    bonus_live = (_promoted_turn == st.turn)
    cards = policy.tables()[0]

    out = []
    for o in opts:
        t = _i(o.type)
        cid = o.cardId
        c = cards.get(cid)
        s = 0.0

        if t == _i(OptionType.ATTACK):
            if o.attackId == GALE_THRUST:
                s = 40000.0 if bonus_live else 5000.0
            elif o.attackId == SPIKY_HOPPER:
                s = 20000.0
            else:
                s = 9000.0

        elif t == _i(OptionType.RETREAT):
            # The rotation. Retreating promotes a benched Pokemon of our choice,
            # so retreating INTO an armed Lopunny is how 230 happens.
            s = 35000.0 if bench_lop >= 0 else 100.0

        elif t == _i(OptionType.EVOLVE):
            if cid == LOPUNNY:
                s = 33000.0 if n_lop < 2 else 26000.0   # want TWO
            elif cid == DUDUNSPARCE:
                s = 30000.0                              # draw engine online
            else:
                s = 24000.0

        elif t == _i(OptionType.ATTACH):
            tgt_bench = _i(o.inPlayArea) == _i(AreaType.BENCH)
            if cid == AIR_BALLOON:
                # on LOPUNNY, so LOPUNNY retreats free -- this is the rotation
                s = 31000.0
            elif cid in ENERGIES or (c is not None and _i(c.cardType) in (
                    _i(CardType.BASIC_ENERGY), _i(CardType.SPECIAL_ENERGY))):
                # one energy per Lopunny is all Gale Thrust needs; spread it
                s = 29000.0 if tgt_bench else 27000.0
            else:
                s = 12000.0

        elif t == _i(OptionType.ABILITY):
            # Dudunsparce draws 3 and recycles itself; that is the engine.
            s = 28000.0

        elif t == _i(OptionType.PLAY):
            if c is None:
                s = 15000.0
            elif _i(c.cardType) == _i(CardType.POKEMON):
                # bench development comes FIRST -- winners open 5x Dunsparce
                s = 32000.0 if n_bench < 4 else 18000.0
                if cid in (BUNEARY, DUNSPARCE, SNORUNT):
                    s += 500
            elif _i(c.cardType) == _i(CardType.SUPPORTER):
                s = 25000.0
            elif _i(c.cardType) == _i(CardType.ITEM):
                s = 23000.0
            elif _i(c.cardType) == _i(CardType.STADIUM):
                s = 16000.0
            else:
                s = 15000.0

        elif t == _i(OptionType.END):
            s = 1.0
        else:
            s = 500.0
        out.append(s)
    return out


def _promote_pick(obs):
    """Whenever we choose who becomes Active, take an armed Lopunny."""
    try:
        sel = obs.select
        st = obs.current
        if sel is None or st is None:
            return None
        if _i(sel.context) not in (_i(SelectContext.TO_ACTIVE),
                                   _i(SelectContext.SWITCH),
                                   _i(SelectContext.SETUP_ACTIVE_POKEMON)):
            return None
        me = st.players[st.yourIndex]
        bench = me.bench or []
        best = None
        for i, o in enumerate(sel.option or []):
            idx = o.index
            if _i(o.area) == _i(AreaType.BENCH) and isinstance(idx, int) \
                    and idx < len(bench) and _armed(bench[idx]):
                return i
            if o.cardId == LOPUNNY and best is None:
                best = i
        return best
    except Exception:
        return None


def agent(obs_dict: dict) -> list:
    global _last_turn, _promoted_turn
    try:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return list(DECK)
        st = obs.current
        if st is not None:
            if st.turn < _last_turn:
                _promoted_turn = -10 ** 9
            _last_turn = st.turn
            me = st.players[st.yourIndex]
            act = me.active[0] if (me.active and me.active[0]) else None
            if act is not None and act.id == LOPUNNY:
                for lg in (obs.logs or []):
                    # SWITCH(8) or MOVE_CARD(6) that brought Lopunny in
                    if _i(lg.type) in (6, 8) and lg.playerIndex == st.yourIndex:
                        _promoted_turn = st.turn
                        break

        pick = _promote_pick(obs)
        if pick is None and _i(obs.select.context) == _i(SelectContext.MAIN) \
                and len(obs.select.option) >= 2:
            sc = _score(obs, obs.select.option)
            pick = max(range(len(sc)), key=lambda i: sc[i])
        if pick is not None:
            sd = obs.select
            if sd.minCount <= 1 <= sd.maxCount and 0 <= pick < len(sd.option):
                return [pick]
        return policy.act(obs_dict, DECK)
    except Exception:
        return policy.act(obs_dict, DECK)
