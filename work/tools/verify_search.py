"""Is the public LB-950 agent's forward search actually running?

Claim under test: SEARCH_ALGO always throws and silently falls back to the
heuristic, so the shipped agent never searches.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))
AG = os.path.join(WORK, "agents", "v2_lucario")
sys.path.insert(0, AG)
os.chdir(AG)

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

src = open(os.path.join(AG, "main.py"), encoding="utf-8").read()
env = {}
exec(compile(src, "main.py", "exec"), env)
agent = env["agent"]

print(f"_SEARCH_OK           = {env.get('_SEARCH_OK')}")
print(f"USE_SEARCH           = {env.get('USE_SEARCH')}")
print(f"SEARCH_TIME_BUDGET   = {env.get('SEARCH_TIME_BUDGET')}")

# --- 1. call search_begin the way the agent does, uncaught -------------
from cg.api import search_begin  # noqa: E402
import inspect  # noqa: E402
print(f"\nreal signature: search_begin{inspect.signature(search_begin)}")

deck = env["my_deck"]
obs, sd = battle_start(list(deck), list(deck))
o = to_observation_class(obs)
# advance to a MAIN selection so SEARCH_ALGO's guard would pass
steps = 0
from cg.api import SelectContext  # noqa: E402
while steps < 400:
    o = to_observation_class(obs)
    if o.current is not None and o.current.result != -1:
        break
    if o.select is not None and o.select.context == SelectContext.MAIN:
        break
    sel = agent(obs)
    obs = battle_select(list(sel))
    steps += 1
o = to_observation_class(obs)
print(f"reached MAIN selection after {steps} steps: "
      f"context={o.select.context}, {len(o.select.option)} options")

print("\n--- calling search_begin exactly as the agent does ---")
yd = [c.id for c in (o.current.players[o.current.yourIndex].hand or [])]
try:
    r = search_begin(o, your_deck=yd)
    print(f"  returned {type(r).__name__}")
except Exception as e:
    print(f"  RAISES {type(e).__name__}: {e}")

# --- 2. does SEARCH_ALGO ever return non-None in a real game? ---------
print("\n--- instrumenting SEARCH_ALGO over a full game ---")
SEARCH_ALGO = env["SEARCH_ALGO"]
calls = {"n": 0, "non_none": 0, "time": 0.0}
orig = SEARCH_ALGO


def wrapped(obs_dict, obs_):
    calls["n"] += 1
    t = time.time()
    r = orig(obs_dict, obs_)
    calls["time"] += time.time() - t
    if r is not None:
        calls["non_none"] += 1
    return r


env["SEARCH_ALGO"] = wrapped
battle_finish()

obs, sd = battle_start(list(deck), list(deck))
n = 0
while n < 2000:
    o = to_observation_class(obs)
    if o.current is not None and o.current.result != -1:
        break
    obs = battle_select(list(agent(obs)))
    n += 1
battle_finish()

print(f"  SEARCH_ALGO called      : {calls['n']}")
print(f"  returned a search result: {calls['non_none']}")
print(f"  total time inside it    : {calls['time']*1000:.1f} ms")
print()
if calls["n"] and calls["non_none"] == 0:
    print("  ==> CONFIRMED: search NEVER succeeds. The agent is pure heuristic.")
elif calls["non_none"]:
    print("  ==> search does run; claim refuted.")
else:
    print("  ==> SEARCH_ALGO never even called (guard rejects every frame).")
