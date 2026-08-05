"""Does the pilot actually execute the Gale Thrust loop?

Counts attacks where Mega Lopunny ex hit for the 230 tier (promoted from the
bench that turn) versus the 60 tier (already Active). A pilot that does not know
the combo attacks almost entirely at the 60 tier.
"""
import os, sys, json
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish
AG=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 30
LOPUNNY=849
def load(d):
    full=os.path.join(WORK,'agents',d); sys.path.insert(0,full)
    cwd=os.getcwd(); os.chdir(full)
    env={}; exec(compile(open('main.py',encoding='utf-8-sig').read(),'main.py','exec'),env)
    os.chdir(cwd)
    return [v for v in env.values() if callable(v)][-1], list(env.get('DECK') or env.get('my_deck'))
fa,da=load(AG)
fb,db=load("v14_search_noloop2")
big=small=0; retreats=0; att=Counter()
for g in range(N):
    first=(g%2==0)
    d0,d1=(da,db) if first else (db,da)
    me=0 if first else 1
    obs,_=battle_start(list(d0),list(d1))
    if obs is None: continue
    try:
        for _ in range(3000):
            o=to_observation_class(obs)
            if o.current is not None and o.current.result!=-1: break
            for lg in (o.logs or []):
                t=int(lg.type)
                if t==16 and lg.playerIndex is not None and lg.playerIndex!=me and lg.value:
                    v=abs(lg.value)
                    if v>=200: big+=1
                    elif 50<=v<=80: small+=1
                if t==15 and lg.playerIndex==me and lg.cardId==LOPUNNY: att[lg.attackId]+=1
            who=o.current.yourIndex if o.current is not None else 0
            obs=battle_select(list((fa if who==me else fb)(obs)))
    except Exception: pass
    finally: battle_finish()
print(f"agent={AG}  {N} games")
print(f"  damage events >=200 dealt to opponent : {big}   (Gale Thrust 230 tier)")
print(f"  damage events 50-80                   : {small}  (Gale Thrust 60 tier)")
print(f"  Lopunny attacks by id                 : {dict(att)}")
if big+small: print(f"  fraction at the 230 tier: {big/(big+small):.1%}")
