"""Generic (archetype-agnostic) policy piloting the best measured deck.

Deck: LB#2's Mega Lopunny ex / Mega Froslass ex list -- 0.741 win rate over 27
games between ~1085-rated agents in the host's top-episode dataset, the highest
of any archetype measured. Our own Mega Lucario ex archetype did not reach 25
games in 1,400 deck instances: it is extinct at that level.

This isolates DECK from POLICY. policy.py has no card-specific rules at all, so
whatever it scores is close to a floor for this deck rather than a tuned result.
"""
import os
import sys

_KAGGLE = "/kaggle_simulations/agent"
if os.path.isdir(_KAGGLE) and _KAGGLE not in sys.path:
    sys.path.insert(0, _KAGGLE)

import policy  # noqa: E402


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


def agent(obs_dict: dict) -> list:
    return policy.act(obs_dict, DECK)
