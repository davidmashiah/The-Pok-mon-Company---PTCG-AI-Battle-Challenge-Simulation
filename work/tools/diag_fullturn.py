"""Why is the full-turn search not spending its budget?

v30 is allowed 300 s per episode and uses ~7 s. Either best_action is being
called and returning fast, or it is not really being called. Guessing wasted a
lot of time on this project already -- five components turned out to be silently
dead -- so this counts every exit path instead.

Usage: python work/tools/diag_fullturn.py [agent] [games]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))

AGENT = sys.argv[1] if len(sys.argv) > 1 else "v30_realsearch"
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 4
AG = os.path.join(WORK, "agents", AGENT)
sys.path.insert(0, AG)

from cg.api import SelectContext, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
import fsearch as _fs  # noqa: E402

STATS = {
    "agent_calls": 0, "main_ctx": 0, "ft_called": 0, "ft_ret_none": 0,
    "ft_ret_order": 0, "cand_lt2": 0, "det_build_none": 0,
    "search_begin_none": 0, "scored_lt2": 0, "all_equal": 0, "ft_time": 0.0,
}

_orig_best = _fs.best_action
_orig_build = None


def wrapped_best(obs, det, rollout, candidates, time_budget=1.0, max_candidates=8):
    STATS["ft_called"] += 1
    if len([c for c in candidates][:max_candidates]) < 2:
        STATS["cand_lt2"] += 1
    if det.build(obs) is None:
        STATS["det_build_none"] += 1
    t0 = time.time()
    out = _orig_best(obs, det, rollout, candidates, time_budget=time_budget,
                     max_candidates=max_candidates)
    STATS["ft_time"] += time.time() - t0
    if out is None:
        STATS["ft_ret_none"] += 1
    else:
        STATS["ft_ret_order"] += 1
    return out


_fs.best_action = wrapped_best

with open(os.path.join(AG, "main.py")) as fh:
    src = fh.read()
env = {}
os.chdir(AG)
exec(compile(src, "main.py", "exec"), env)
items = [(k, v) for k, v in env.items() if callable(v)]
agent_fn = items[-1][1]
deck = env.get("DECK") or env.get("my_deck")

t0 = time.time()
for g in range(GAMES):
    obs, _sd = battle_start(list(deck), list(deck))
    for _ in range(4000):
        o = to_observation_class(obs)
        if o.select is None:
            break
        STATS["agent_calls"] += 1
        try:
            if int(o.select.context) == int(SelectContext.MAIN):
                STATS["main_ctx"] += 1
        except Exception:
            pass
        sel = agent_fn(obs)
        obs = battle_select(sel)
    battle_finish()

print(f"{AGENT}: {GAMES} games in {time.time()-t0:.0f}s")
for k, v in STATS.items():
    print(f"  {k:<20} {v:.2f}" if isinstance(v, float) else f"  {k:<20} {v}")
if STATS["ft_called"]:
    print(f"\n  best_action returned a re-ranking on "
          f"{STATS['ft_ret_order']}/{STATS['ft_called']} calls "
          f"({100*STATS['ft_ret_order']/STATS['ft_called']:.0f}%), "
          f"total {STATS['ft_time']:.1f}s")
else:
    print("\n  best_action was NEVER CALLED -- the search is dead code again")
