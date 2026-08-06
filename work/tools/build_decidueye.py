"""v200_decidueye: a base built to beat the 30% of the top field we are capped against.

Why this deck and not another. w8_grimm_tuned is live at 829.5 and is the best
anti-Grimmsnarl agent that exists -- ours or anyone's -- at 0.530. Grimmsnarl is
30% of the top 50, so that 0.530 is a hard ceiling on the whole field rate:

    field = 0.30(0.530) + 0.70(rest);  rest 0.72 -> 830,  rest 0.92 -> 1000

No amount of tuning the other 70% gets to 1000 while the largest slice is a coin
flip. The only way through is a deck that BEATS Grimmsnarl, and the card pool
has one, hiding behind a damage-per-energy sort:

    Marnie's Grimmsnarl ex   320 HP, 2 prizes, Shadow Bullet 180 for 2 energy
    Decidueye (id 129)       150 HP, 1 PRIZE,  Power Shot   170 for 1 energy

The whole Marnie's line is weak to Grass, so 170 x 2 = 340 >= 320: we one-shot
their two-prize attacker with a single energy, using a one-prize Pokemon. They
need SIX knockouts to win the game; we need THREE. That is the structural
advantage, and it is the reason to build rather than tune.

Decidueye's second attack matters as much as the first. Power Shot costs "discard
a Basic {G} Energy from your hand", so the deck can stall out holding none --
and Stock Up on Feathers is free and draws until you hold seven. The attacker
refuels itself, which is why this line can run a low energy count and still
attack every turn.

Two published Grass agents already exist and both lose to Grimmsnarl (0.100 and
0.160 over 270 games), so the type advantage alone is worth nothing. Those decks
attack for 240 at FOUR energy; this one attacks for 170 at one. The pilot below
is written for that difference.

  python work/tools/build_decidueye.py
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
NAME = "v200_decidueye"

ROWLET, DARTRIX, DECIDUEYE = 127, 128, 129
GRASS_ENERGY = 1
RARE_CANDY, POFFIN, ULTRA_BALL, POKEGEAR = 1079, 1086, 1121, 1122
BUG_CATCHING_SET, SWITCH = 1094, 1123
LILLIE, BOSS = 1227, 1182
ZARUDE, TAPU_BULU = 178, 920

DECK = (
    [ROWLET] * 4 + [DARTRIX] * 4 + [DECIDUEYE] * 4      # the attacker line
    + [ZARUDE] * 3 + [TAPU_BULU] * 2                    # extra Basics: 4 Rowlet
                                                        # alone mulligans far too
                                                        # often for a Stage 2 deck
    + [RARE_CANDY] * 4                                  # Rowlet -> Decidueye
    + [POFFIN] * 4 + [ULTRA_BALL] * 4 + [POKEGEAR] * 4
    + [BUG_CATCHING_SET] * 4
    + [LILLIE] * 4 + [BOSS] * 2 + [SWITCH] * 2
    + [GRASS_ENERGY] * 15                               # Power Shot spends one
                                                        # from HAND per attack as
                                                        # well as the attached one
)

MAIN = r'''"""Decidueye Power Shot pilot.

One decision rule dominates everything else in this deck: a Decidueye holding a
Basic {G} Energy in hand one-shots any 320 HP Grass-weak attacker for a single
energy, and dies to one hit as a ONE-prize Pokemon. So the plan is to always
have a Decidueye active with an energy attached and an energy in hand, and to
spend the free attack (draw until you hold seven) whenever the hand is empty
rather than attacking for nothing.

Everything is scored, never hardcoded to an option index, because the option
list order is not stable across frames.
"""
import os

from cg.api import (
    AreaType, CardType, OptionType, SelectContext, all_card_data,
    to_observation_class,
)

ROWLET, DARTRIX, DECIDUEYE = 127, 128, 129
GRASS = 1
RARE_CANDY, POFFIN, ULTRA_BALL, POKEGEAR = 1079, 1086, 1121, 1122
BUG_SET, SWITCH, LILLIE, BOSS = 1094, 1123, 1227, 1182
POWER_SHOT_DMG = 170

_CARDS = {c.cardId: c for c in all_card_data()}
_DECK_CONST = __DECK__


def _load_deck():
    """Bundle first, cwd second, constant last -- all three agree.

    The harness exec()s main.py so __file__ is undefined, and a cwd-relative
    read has silently shipped the wrong decklist in this repo before.
    """
    for p in ("/kaggle_simulations/agent/deck.csv", "deck.csv"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = [int(x) for x in f.read().split() if x.strip()]
            if len(d) == 60:
                return d
        except Exception:
            pass
    return list(_DECK_CONST)


my_deck = _load_deck()


def _card(obs, area, index, player_index):
    try:
        st = obs.current
        p = st.players[player_index]
        if area == AreaType.HAND:
            return p.hand[index]
        if area == AreaType.ACTIVE:
            return p.active[index]
        if area == AreaType.BENCH:
            return p.bench[index]
        if area == AreaType.DISCARD:
            return p.discard[index]
        if area == AreaType.DECK:
            return (obs.select.deck or [])[index]
    except Exception:
        return None
    return None


def _score(obs, opt):
    st, sel = obs.current, obs.select
    me = st.players[st.yourIndex]
    opp = st.players[1 - st.yourIndex]
    hand = list(me.hand or [])
    hand_ids = [c.id for c in hand if c is not None]
    active = me.active[0] if (me.active and me.active[0]) else None
    bench = [b for b in (me.bench or []) if b is not None]
    oact = opp.active[0] if (opp.active and opp.active[0]) else None

    grass_in_hand = hand_ids.count(GRASS)
    active_is_deci = active is not None and active.id == DECIDUEYE
    active_energy = len(active.energies or []) if active is not None else 0
    t = opt.type

    if t == OptionType.ATTACK:
        # ATTACKING ENDS THE TURN, so an attack must be either the play that
        # takes a prize or the last thing done in a turn -- never the first.
        # v1 scored every attack at 90000, above EVOLVE, and spent turns 4-8
        # swinging Rowlet's 0-damage "Add On" (0.020 over 150 games). v2 then
        # scored them all at 2 and simply stopped attacking (30% of games).
        # Power Shot outranks development because a prize beats a board; every
        # other attack sits just above END so it closes a turn we have already
        # spent.
        if active_is_deci and active_energy >= 1 and grass_in_hand >= 1:
            return 100000       # 170 x2 weakness = 340, one-shots a 320 HP ex
        if active_is_deci:
            return 10           # Stock Up on Feathers: free, draws to seven
        return 5

    if t == OptionType.EVOLVE:
        c = _card(obs, opt.area, opt.index, st.yourIndex)
        cid = getattr(c, "id", None)
        if cid == DECIDUEYE:
            return 95000        # the only thing that wins games
        if cid == DARTRIX:
            return 70000
        return 60000

    if t == OptionType.PLAY:
        c = _card(obs, opt.area, opt.index, st.yourIndex)
        cid = getattr(c, "id", None)
        if cid == RARE_CANDY:
            # only worth it if it actually reaches Decidueye this turn
            if DECIDUEYE in hand_ids and any(
                    p is not None and p.id == ROWLET
                    for p in ([active] + bench)):
                return 94000
            return -1
        if cid == BOSS:
            # drag up something we can one-shot rather than the wall in front
            if active_is_deci and active_energy >= 1 and grass_in_hand >= 1:
                return 50000
            return 3000
        if cid == POFFIN:
            return 40000 if len(bench) < 4 else 8000
        if cid == ULTRA_BALL:
            return 39000 if DECIDUEYE not in hand_ids else 12000
        if cid == BUG_SET:
            return 38000
        if cid == POKEGEAR:
            return 30000
        if cid == LILLIE:
            # a refill is only free when we are not throwing away a live hand
            return 26000 if len(hand) <= 3 else 5000
        if cid == SWITCH:
            if active is not None and not active_is_deci and any(
                    b.id == DECIDUEYE for b in bench):
                return 45000
            return 2000
        c2 = _CARDS.get(cid)
        if c2 is not None and int(c2.cardType) == 0:
            return 35000 if len(bench) < 4 else 1500
        return 4000

    if t == OptionType.ATTACH:
        # Resolve through the option's OWN area. Hardcoding HAND here made every
        # ATTACH resolve to the wrong card, score as junk, and never fire: the
        # active sat on zero energy for whole games while Grass sat in hand, so
        # Power Shot -- the entire point of the deck -- could never be taken.
        c = _card(obs, opt.area, opt.index, st.yourIndex)
        if getattr(c, "id", None) == GRASS:
            # one energy is all Power Shot needs; a second is wasted tempo, and
            # the hand copy is the ammunition for the discard cost
            if active_is_deci and active_energy == 0:
                return 96000    # nothing matters more than arming Power Shot
            if active_energy == 0:
                return 46000    # arm whoever is up; we may evolve into it
            return 900
        return 800

    if t == OptionType.ABILITY:
        return 25000
    if t == OptionType.RETREAT:
        if active is not None and not active_is_deci and any(
                b.id == DECIDUEYE for b in bench):
            return 44000
        return 500
    if t == OptionType.END:
        return 1
    return 1000


def _pick(obs):
    sel = obs.select
    n = len(sel.option)
    if n == 0:
        return []
    if sel.context != SelectContext.MAIN:
        # Non-MAIN prompts are forced choices (promote, discard, search). Prefer
        # a Decidueye-line card when one is offered, otherwise take the first
        # legal set -- guessing cleverly here has no upside and can go illegal.
        order = sorted(range(n), key=lambda i: -_line_pref(obs, sel.option[i]))
        k = min(max(1, sel.minCount), n)
        return order[:k]
    scores = [_score(obs, o) for o in sel.option]
    order = sorted(range(n), key=lambda i: -scores[i])
    order = [i for i in order if scores[i] > -1] or order
    k = min(max(1, sel.minCount), n)
    k = min(k, max(1, sel.maxCount)) if sel.maxCount else k
    return order[:k]


def _line_pref(obs, opt):
    try:
        c = _card(obs, opt.area, opt.index, obs.current.yourIndex)
        cid = getattr(c, "id", None)
        if cid == DECIDUEYE:
            return 5
        if cid == DARTRIX:
            return 4
        if cid == ROWLET:
            return 3
        if cid == GRASS:
            return 2
    except Exception:
        pass
    return 0


def decidueye_agent(obs_dict):
    # The setup frame: the competition asks for the decklist here and nowhere
    # else, and it is also where per-episode state would be reset.
    if (isinstance(obs_dict, dict) and obs_dict.get("current") is None
            and obs_dict.get("select") is None):
        return list(my_deck)
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return list(my_deck)
        out = _pick(obs)
        s = obs.select
        n = len(s.option)
        if (not isinstance(out, list) or not out
                or len(set(out)) != len(out)
                or not all(isinstance(i, int) and 0 <= i < n for i in out)
                or not (s.minCount <= len(out) <= max(s.maxCount, s.minCount))):
            raise ValueError("illegal selection")
        return out
    except Exception:
        try:
            s = to_observation_class(obs_dict).select
            n = len(s.option)
            return list(range(min(max(1, s.minCount), n)))
        except Exception:
            return [0]
'''


def main():
    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import all_card_data
    from cg.game import battle_finish, battle_start
    cards = {c.cardId: c for c in all_card_data()}

    deck = sorted(DECK)
    if len(deck) != 60:
        raise SystemExit(f"deck has {len(deck)} cards, not 60")
    for cid, n in Counter(deck).items():
        c = cards.get(cid)
        if c is None:
            raise SystemExit(f"unknown card {cid}")
        if int(getattr(c, "cardType", -1)) != 5 and n > 4:
            raise SystemExit(f"{n}x {getattr(c,'name',cid)} exceeds 4")
    basics = sum(n for cid, n in Counter(deck).items()
                 if int(getattr(cards[cid], "cardType", -1)) == 0
                 and getattr(cards[cid], "basic", False))
    print(f"deck: 60 cards, {basics} Basic Pokemon, "
          f"{Counter(deck)[GRASS_ENERGY]} Grass energy")

    obs, _ = battle_start(list(deck), list(deck))
    ok = obs is not None
    battle_finish()
    if not ok:
        raise SystemExit("engine REJECTED the deck (battle_start returned None)")
    print("engine accepts the deck")

    out = os.path.join(AGENTS, NAME)
    os.makedirs(out, exist_ok=True)
    src = MAIN.replace("__DECK__", repr(deck))
    compile(src, "main.py", "exec")
    with open(os.path.join(out, "main.py"), "w", encoding="utf-8") as fh:
        fh.write(src)
    with open(os.path.join(out, "deck.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(map(str, deck)) + "\n")
    print(f"built work/agents/{NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
