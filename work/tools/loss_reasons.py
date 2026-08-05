"""WHY do we lose? LogType.RESULT carries a reason code:
   1 = opponent took their last Prize   (a normal loss)
   2 = we started a turn with 0 deck cards (deck-out: self-inflicted)
   3 = we had no Pokemon in the Active Spot (also largely self-inflicted)
   4 = a card effect
Every win found so far came from eliminating self-inflicted losses, so it is
worth knowing the split rather than only the rate.
"""
import json, os, sys
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish

AG=sys.argv[1] if len(sys.argv)>1 else "v13_noloop2"
N=int(sys.argv[2]) if len(sys.argv)>2 else 140
full=os.path.join(WORK,"agents",AG); sys.path.insert(0,full)
cwd=os.getcwd(); os.chdir(full)
env={}; exec(compile(open("main.py",encoding="utf-8-sig").read(),"main.py","exec"),env)
os.chdir(cwd)
fn=[v for v in env.values() if callable(v)][-1]
mine=list(env.get("DECK") or env.get("my_deck"))
store=json.load(open(os.path.join(WORK,"out","meta_decks.json"),encoding="utf-8"))
HARD={1010.8,1109.6,1060.3,1034.9,1275.3,1063.4,1104.6}
opps=[];seen=set()
for t in sorted(store["teams"].values(), key=lambda t:-t.get("score",0)):
    d=t.get("deck")
    if not d or len(d)!=60: continue
    k=tuple(sorted(d))
    if k in seen or round(t.get("score",0),1) not in HARD: continue
    seen.add(k); opps.append(d)

REASON={1:"prizes taken",2:"DECK-OUT (self-inflicted)",3:"no Active Pokemon",4:"card effect"}
wins=Counter(); losses=Counter(); n=0
for g in range(N):
    opp=opps[g%len(opps)]; first=(g%2==0)
    d0,d1=(mine,opp) if first else (opp,mine)
    me=0 if first else 1
    obs,_=battle_start(list(d0),list(d1))
    if obs is None: continue
    reason=None; res=None
    try:
        for _ in range(4000):
            o=to_observation_class(obs)
            for lg in (o.logs or []):
                if int(lg.type)==23:
                    res=lg.result; reason=lg.reason
            if o.current is not None and o.current.result!=-1:
                res=o.current.result; break
            who=o.current.yourIndex if o.current is not None else 0
            env["my_deck"]=list(d0 if who==0 else d1); env["DECK"]=env["my_deck"]
            obs=battle_select(list(fn(obs)))
    except Exception: pass
    finally: battle_finish()
    if res is None: continue
    n+=1
    key=REASON.get(reason,f"reason={reason}")
    if res==me: wins[key]+=1
    elif res!=2: losses[key]+=1

print(f"agent={AG}  {n} decided games vs {len(opps)} hard decks\n")
print("WINS by reason:")
for k,v in wins.most_common(): print(f"   {v:>4}  {k}")
print("\nLOSSES by reason:")
for k,v in losses.most_common(): print(f"   {v:>4}  {k}")
tot=sum(losses.values())
si=sum(v for k,v in losses.items() if "DECK-OUT" in k or "no Active" in k)
if tot: print(f"\nself-inflicted losses: {si}/{tot} = {si/tot:.1%} of all losses")
