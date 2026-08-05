"""Is the search we pay for actually running in real games, or silently falling back?

fsearch.find_lethal returns None on: gate rejection, determinization failure,
search_begin failure, or any exception. All four look identical to the caller.
This separates them, because "silently disabled" has already bitten this
project twice.
"""
import json, os, sys, time
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class, SelectContext
from cg.game import battle_start, battle_select, battle_finish
import fsearch

AG=sys.argv[1] if len(sys.argv)>1 else "v14_search_noloop2"
N=int(sys.argv[2]) if len(sys.argv)>2 else 20
full=os.path.join(WORK,"agents",AG); sys.path.insert(0,full)
cwd=os.getcwd(); os.chdir(full)
env={}; exec(compile(open("main.py",encoding="utf-8-sig").read(),"main.py","exec"),env)
os.chdir(cwd)
fn=[v for v in env.values() if callable(v)][-1]
mine=list(env.get("DECK") or env.get("my_deck"))

stat=Counter()
orig_gate=fsearch.lethal_plausible
orig_begin=fsearch.search_begin
orig_find=fsearch.find_lethal

def gate(o):
    r=orig_gate(o); stat["gate_pass" if r else "gate_reject"]+=1; return r
def begin(*a,**k):
    try:
        r=orig_begin(*a,**k); stat["search_begin_ok"]+=1; return r
    except Exception as e:
        stat["search_begin_FAIL:"+type(e).__name__]+=1; raise
def find(*a,**k):
    t=time.time(); r=orig_find(*a,**k); stat["find_time_ms"]+=int((time.time()-t)*1000)
    stat["lethal_found" if r is not None else "lethal_none"]+=1; return r
fsearch.lethal_plausible=gate; fsearch.search_begin=begin; fsearch.find_lethal=find

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

print(f"agent={AG}  {N} games vs hard subset\n")
gp,gr=stat["gate_pass"],stat["gate_reject"]
print(f"lethal_plausible gate : pass {gp}  reject {gr}  ({gp/max(gp+gr,1):.1%} pass)")
print(f"search_begin ok       : {stat['search_begin_ok']}")
for k,v in stat.items():
    if k.startswith("search_begin_FAIL"): print(f"  !! {k} : {v}")
print(f"find_lethal called    : {stat['lethal_found']+stat['lethal_none']}")
print(f"   proved a lethal    : {stat['lethal_found']}")
print(f"   no lethal found    : {stat['lethal_none']}")
print(f"total time in search  : {stat['find_time_ms']/1000:.1f}s over {N} games "
      f"({stat['find_time_ms']/1000/max(N,1):.2f}s/game)")
