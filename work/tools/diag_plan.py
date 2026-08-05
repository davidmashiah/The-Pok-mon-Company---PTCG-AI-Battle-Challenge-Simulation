"""How often is `plan` actually populated? Switch/Boss's Orders are gated on it."""
import os, sys, json
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class, SelectContext
from cg.game import battle_start, battle_select, battle_finish
AG="v14_search_noloop2"
full=os.path.join(WORK,"agents",AG); sys.path.insert(0,full)
cwd=os.getcwd(); os.chdir(full)
env={}; exec(compile(open("main.py",encoding="utf-8-sig").read(),"main.py","exec"),env)
os.chdir(cwd)
FN=[v for k,v in env.items() if callable(v)][-1]
deck=list(env["my_deck"])
store=json.load(open(os.path.join(WORK,"out","meta_decks.json"),encoding="utf-8"))
opp=next(t["deck"] for t in store["teams"].values() if abs(t.get("score",0)-1275.3)<0.05)

stat=Counter()
for g in range(14):
    d0,d1=(deck,opp) if g%2==0 else (opp,deck)
    obs,_=battle_start(list(d0),list(d1))
    if obs is None: continue
    try:
        for _ in range(3000):
            o=to_observation_class(obs)
            if o.current is not None and o.current.result!=-1: break
            who=o.current.yourIndex if o.current is not None else 0
            env["my_deck"]=list(d0 if who==0 else d1); env["DECK"]=env["my_deck"]
            sel=FN(obs)
            if o.select is not None and int(o.select.context)==int(SelectContext.MAIN):
                p=env.get("plan")
                stat["main_frames"]+=1
                if p is not None:
                    if getattr(p,"attacker",-1)>0: stat["plan.attacker>0"]+=1
                    if getattr(p,"target",-1)>=1: stat["plan.target>=1"]+=1
                    if getattr(p,"remain_hp",-1)>0: stat["plan.remain_hp>0"]+=1
            obs=battle_select(list(sel))
    except Exception: pass
    finally: battle_finish()
n=stat["main_frames"]
print(f"MAIN frames                      : {n}")
for k in ("plan.attacker>0","plan.target>=1","plan.remain_hp>0"):
    print(f"  {k:<22}: {stat[k]:>6}  ({stat[k]/max(n,1):.1%})")
print()
print("Switch is playable only when plan.attacker>0; Boss's Orders only when plan.target>=1.")
