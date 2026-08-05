"""Does v15's full-turn search actually change decisions, or silently no-op?

best_action() returns None on: not MAIN, <2 candidates, determinization
failure, <2 scored rollouts, or all-equal evaluations. All look identical to
the caller. Same audit pattern that caught the dead meta_decks import.
"""
import json, os, sys, time
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish
import fsearch

AG="v15_deepsearch"; N=int(sys.argv[1]) if len(sys.argv)>1 else 14
full=os.path.join(WORK,"agents",AG); sys.path.insert(0,full)
cwd=os.getcwd(); os.chdir(full)
env={}; exec(compile(open("main.py",encoding="utf-8-sig").read(),"main.py","exec"),env)
os.chdir(cwd)
fn=[v for v in env.values() if callable(v)][-1]
mine=list(env.get("DECK") or env.get("my_deck"))

stat=Counter()
orig=fsearch.best_action
def wrapped(obs, det, rollout, candidates, **kw):
    t=time.time()
    r=orig(obs, det, rollout, candidates, **kw)
    stat["time_ms"]+=int((time.time()-t)*1000)
    stat["calls"]+=1
    if r is None:
        stat["returned_None"]+=1
    else:
        stat["returned_ranking"]+=1
        # did it actually override the heuristic's top pick?
        if list(candidates) and r and r[0]!=list(candidates)[0]:
            stat["CHANGED_top_choice"]+=1
        else:
            stat["agreed_with_heuristic"]+=1
    return r
fsearch.best_action=wrapped

store=json.load(open(os.path.join(WORK,"out","meta_decks.json"),encoding="utf-8"))
HARD={1010.8,1109.6,1060.3,1034.9,1275.3,1063.4,1104.6}
opps=[];seen=set()
for t in sorted(store["teams"].values(), key=lambda t:-t.get("score",0)):
    d=t.get("deck")
    if not d or len(d)!=60: continue
    k=tuple(sorted(d))
    if k in seen or round(t.get("score",0),1) not in HARD: continue
    seen.add(k); opps.append(d)

for g in range(N):
    opp=opps[g%len(opps)]; first=(g%2==0)
    d0,d1=(mine,opp) if first else (opp,mine)
    obs,_=battle_start(list(d0),list(d1))
    if obs is None: continue
    try:
        for _ in range(4000):
            o=to_observation_class(obs)
            if o.current is not None and o.current.result!=-1: break
            who=o.current.yourIndex if o.current is not None else 0
            env["my_deck"]=list(d0 if who==0 else d1); env["DECK"]=env["my_deck"]
            obs=battle_select(list(fn(obs)))
    except Exception: pass
    finally: battle_finish()

c=stat["calls"]
print(f"agent={AG}  {N} games vs hard subset\n")
print(f"best_action called        : {c}")
print(f"  returned a ranking      : {stat['returned_ranking']}")
print(f"  returned None (no-op)   : {stat['returned_None']}")
print(f"  CHANGED the top choice  : {stat['CHANGED_top_choice']}")
print(f"  agreed with heuristic   : {stat['agreed_with_heuristic']}")
print(f"time in full-turn search  : {stat['time_ms']/1000:.1f}s over {N} games "
      f"({stat['time_ms']/1000/max(N,1):.2f}s/game)")
if c: print(f"\ndecision-change rate: {stat['CHANGED_top_choice']/c:.2%} of calls")
