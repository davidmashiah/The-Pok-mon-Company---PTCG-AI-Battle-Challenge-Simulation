"""Precise test: does simulate_action() EVER return a real value?"""
import os, sys, time
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
AG=os.path.join(WORK,"agents","v2_lucario"); sys.path.insert(0,AG); os.chdir(AG)
from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish

src=open(os.path.join(AG,"main.py"),encoding="utf-8").read()
env={}; exec(compile(src,"main.py","exec"), env)

stat={"calls":0,"ok":0,"exc":{}}
orig=env["simulate_action"]
def wrapped(obs, action):
    stat["calls"]+=1
    try:
        v=orig(obs,action)
    except Exception as e:
        stat["exc"][type(e).__name__]=stat["exc"].get(type(e).__name__,0)+1
        raise
    if v!=-float("inf"): stat["ok"]+=1
    return v
env["simulate_action"]=wrapped
agent=env["agent"]; deck=env["my_deck"]

games=0
for g in range(5):
    obs,_=battle_start(list(deck),list(deck)); games+=1
    for _ in range(2000):
        o=to_observation_class(obs)
        if o.current is not None and o.current.result!=-1: break
        obs=battle_select(list(agent(obs)))
    battle_finish()

print(f"games played              : {games}")
print(f"simulate_action calls     : {stat['calls']}")
print(f"  returned a real value   : {stat['ok']}")
print(f"  exceptions by type      : {stat['exc']}")
print()
if stat["calls"]==0:
    print("==> simulate_action NEVER CALLED.")
elif stat["ok"]==0:
    print("==> CONFIRMED: every simulate_action call fails. Zero forward search happens.")
else:
    print("==> search genuinely runs.")
