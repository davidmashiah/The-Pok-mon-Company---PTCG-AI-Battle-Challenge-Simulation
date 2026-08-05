"""PTCG AI Battle agent — v1 greedy heuristic.

!! HARD-WON CONSTRAINTS — do not "improve" these without re-reading why !!

1. NEVER reference __file__ at module level (or anywhere).
   kaggle-environments loads this file with `exec(code_object, env)` in
   kaggle_environments/agent.py::get_last_callable. In that context __file__
   is NOT defined and the agent dies at import with:
       NameError: name '__file__' is not defined
   This cost submission 55194301 (2026-08-02). Resolve paths from cwd and the
   documented /kaggle_simulations/agent/ location instead.

2. `agent` MUST be the LAST callable defined in this module.
   The harness picks up the *last* callable in the exec'd namespace
   (get_last_callable). Define every helper above it.
"""
import os
import sys

_KAGGLE = "/kaggle_simulations/agent"
if os.path.isdir(_KAGGLE) and _KAGGLE not in sys.path:
    sys.path.insert(0, _KAGGLE)

import policy  # noqa: E402


def _read_deck():
    """Read deck.csv from cwd or the Kaggle agent dir. No __file__."""
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
    raise RuntimeError("deck.csv not found or not 60 cards")


DECK = _read_deck()


# --- `agent` must remain the LAST callable defined in this file -------------
def agent(obs_dict: dict) -> list:
    return policy.act(obs_dict, DECK)
