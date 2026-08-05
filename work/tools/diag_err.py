"""Why do games against certain decks fail? Errors on the ladder = lost games."""
import json, os, sys, traceback
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish

full=os.path.join(WORK,"agents","v11b_noloop6"); sys.path.insert(0,full)
cwd=os.getcwd(); os.chdir(full)
env={}; exec(compile(open("main.py",encoding="utf-8-sig").read(),"main.py","exec"),env)
os.chdir(cwd)
fn=[v for v in env.values() if callable(v)][-1]
mine=list(env["my_deck"])

store=json.load(open(os.path.join(WORK,"out","meta_decks.json"),encoding="utf-8"))
target=None
for t in store["teams"].values():
    if abs(t.get("score",0)-1055.6)<0.05: target=t; break
if target is None: raise SystemExit("deck not found")
opp=target["deck"]
print("opponent:", target["score"], target["name"][:30])

reasons=Counter(); tb_seen={}
for g in range(40):
    me_first = (g%2==0)
    d0,d1 = (mine,opp) if me_first else (opp,mine)
    obs,sd = battle_start(list(d0),list(d1))
    if obs is None:
        reasons["battle_start None (errorPlayer=%s errorType=%s)"%(sd.errorPlayer,sd.errorType)]+=1
        continue
    steps=0; done=False
    try:
        for steps in range(4000):
            o=to_observation_class(obs)
            if o.current is not None and o.current.result!=-1:
                reasons["ok"]+=1; done=True; break
            who=o.current.yourIndex if o.current is not None else 0
            env["my_deck"]=list(d0 if who==0 else d1); env["DECK"]=env["my_deck"]
            obs=battle_select(list(fn(obs)))
        if not done: reasons["hit 4000-step cap"]+=1
    except Exception as e:
        k="%s: %s"%(type(e).__name__,str(e)[:70])
        reasons[k]+=1
        if k not in tb_seen: tb_seen[k]=traceback.format_exc()
    finally:
        battle_finish()

print("\noutcomes over 40 games:")
for k,v in reasons.most_common(): print("  %3d  %s"%(v,k))
for k,tb in tb_seen.items():
    print("\n--- traceback for %s ---"%k); print(tb[-1200:])
