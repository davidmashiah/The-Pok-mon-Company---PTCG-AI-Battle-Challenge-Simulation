"""Mega Lopunny ex pilot -- executes the Gale Thrust loop.

WHY THIS EXISTS
Measured from the host's top-episode dataset (real games between ~1085-rated
agents, our code touching nothing):

    Mega Lopunny ex variants   0.60 - 0.74 win rate over 169 games
    Marnie's Grimmsnarl ex     0.480 over 735 games  (most played, losing)
    Mega Lucario ex (ours)     under 25 games in 1400 -- extinct at that level

But handing our existing policy this deck won nothing: 201-195 over 396 games,
a tie. The deck's win rate lives in the PILOT, not the cards.

THE ENGINE
    Mega Lopunny ex  Stage 1, Colorless, HP330, retreat 1
      Gale Thrust [C]   60 damage
                        +170 MORE if this Pokemon moved from your Bench to the
                        Active Spot THIS TURN   -> 230 for one Colorless energy
      Spiky Hopper [CC] 160, damage unaffected by the opponent's effects

This list runs no Switch. It runs Air Balloon x3. Retreating sends your Active
to the bench and lets you PROMOTE ANY BENCHED POKEMON -- so retreat is how
Lopunny arrives from the bench, and Air Balloon makes it free.

    keep a cheap Pokemon Active (Air Balloon attached)
    keep Mega Lopunny ex BENCHED with >=1 energy
    retreat  ->  promote Lopunny  ->  Gale Thrust 230

The upstream policy retreats only when the Active is nearly dead. It treats the
deck's win condition as an emergency button.
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

GALE_THRUST = 1225     # [C] 60, +170 if promoted from the bench this turn
SPIKY_HOPPER = 1226    # [CC] 160, damage unaffected by opponent effects
LOPUNNY = 849          # Mega Lopunny ex
BUNEARY = 848          # its Basic
FROSLASS = 861         # Mega Froslass ex, secondary attacker
SNORUNT = 860
AIR_BALLOON = 1174     # free retreat
GALE_THRUST_MAX = 230


# Turn on which we promoted Lopunny from the bench. Gale Thrust only reaches
# 230 on that same turn; otherwise it is a 60 and Spiky Hopper's 160 is
# strictly better. Measured before this fix: 59 Gale Thrusts, 0 Spiky Hoppers,
# only 21% at the 230 tier -- i.e. ~71 attacks throwing away 100 damage each.
_promoted_turn = -10 ** 9
_last_turn = -10 ** 9


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


def _bench_lopunny(me):
    """Benched Mega Lopunny ex with energy, i.e. ready to be promoted."""
    for i, p in enumerate(me.bench or []):
        if p is not None and p.id == LOPUNNY and len(p.energies or []) >= 1:
            return i
    return -1


def _score_main(obs, opts):
    """Score MAIN options around the Gale Thrust loop. Higher is better."""
    st = obs.current
    me = st.players[st.yourIndex]
    act = me.active[0] if (me.active and me.active[0]) else None
    bench_lop = _bench_lopunny(me)
    lop_active = act is not None and act.id == LOPUNNY

    scores = []
    for o in opts:
        t = _i(o.type)
        s = 0.0

        if t == _i(OptionType.ATTACK):
            bonus_live = (_promoted_turn == st.turn)
            if o.attackId == GALE_THRUST:
                s = 9600.0 if bonus_live else 8800.0      # 230 vs 60
            elif o.attackId == SPIKY_HOPPER:
                s = 9200.0 if not bonus_live else 8900.0  # 160
            else:
                s = 9000.0

        elif t == _i(OptionType.RETREAT):
            # THE combo enabler: retreat promotes a benched Pokemon of our
            # choice. With Lopunny benched and powered, this is the whole deck.
            if bench_lop >= 0 and not lop_active:
                s = 30000.0
            elif lop_active and _promoted_turn != st.turn:
                # Lopunny is stranded Active with a dead bonus: cycling it out
                # now re-arms the loop for next turn.
                s = 6000.0
            else:
                s = 200.0

        elif t == _i(OptionType.EVOLVE):
            s = 24000.0                      # Buneary -> Lopunny, always

        elif t == _i(OptionType.ATTACH):
            c = policy.tables()[0].get(o.cardId)
            tgt_bench = _i(o.inPlayArea) == _i(AreaType.BENCH)
            if c is not None and c.cardId == AIR_BALLOON:
                # free retreat matters on whoever is Active
                s = 23000.0 if not tgt_bench else 15000.0
            elif c is not None and _i(c.cardType) in (
                    _i(CardType.BASIC_ENERGY), _i(CardType.SPECIAL_ENERGY)):
                # Gale Thrust costs ONE colorless: energy on the benched
                # Lopunny is what arms the loop.
                s = 22000.0 if tgt_bench else 12000.0
            else:
                s = 11000.0

        elif t == _i(OptionType.PLAY):
            c = policy.tables()[0].get(o.cardId)
            s = 20000.0
            if c is not None and _i(c.cardType) == _i(CardType.POKEMON):
                s = 21000.0 if c.cardId in (BUNEARY, SNORUNT) else 20500.0

        elif t == _i(OptionType.ABILITY):
            s = 19000.0                      # below development, above filler

        elif t == _i(OptionType.END):
            s = 1.0
        else:
            s = 500.0
        scores.append(s)
    return scores


def _lopunny_choice(obs):
    """Return an option index for MAIN, or None to defer to the generic policy."""
    try:
        sel = obs.select
        if sel is None or obs.current is None:
            return None
        if _i(sel.context) != _i(SelectContext.MAIN):
            return None
        opts = sel.option or []
        if len(opts) < 2:
            return None
        sc = _score_main(obs, opts)
        best = max(range(len(opts)), key=lambda i: sc[i])
        return best
    except Exception:
        return None


def _promote_choice(obs):
    """When choosing who to promote (retreat / KO replacement), take Lopunny."""
    try:
        sel = obs.select
        if sel is None or obs.current is None:
            return None
        ctx = _i(sel.context)
        if ctx not in (_i(SelectContext.TO_ACTIVE), _i(SelectContext.SWITCH)):
            return None
        st = obs.current
        for i, o in enumerate(sel.option or []):
            if o.cardId == LOPUNNY:
                return i
            if _i(o.area) == _i(AreaType.BENCH) and o.index is not None:
                me = st.players[st.yourIndex]
                bl = me.bench or []
                if o.index < len(bl) and bl[o.index] is not None \
                        and bl[o.index].id == LOPUNNY:
                    return i
        return None
    except Exception:
        return None


def agent(obs_dict: dict) -> list:
    global _promoted_turn, _last_turn
    try:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return list(DECK)
        st = obs.current
        if st is not None:
            if st.turn < _last_turn:          # new game
                _promoted_turn = -10 ** 9
            _last_turn = st.turn
            me = st.players[st.yourIndex]
            act = me.active[0] if (me.active and me.active[0]) else None
            # record the turn Lopunny becomes Active: that is when Gale Thrust
            # is worth 230
            if act is not None and act.id == LOPUNNY and _promoted_turn != st.turn:
                for lg in (obs.logs or []):
                    if _i(lg.type) in (8, 6) and lg.playerIndex == st.yourIndex:
                        _promoted_turn = st.turn
                        break
        pick = _promote_choice(obs)
        if pick is None:
            pick = _lopunny_choice(obs)
        if pick is not None:
            sd = obs.select
            if sd.minCount <= 1 <= sd.maxCount and 0 <= pick < len(sd.option):
                return [pick]
        return policy.act(obs_dict, DECK)
    except Exception:
        return policy.act(obs_dict, DECK)
