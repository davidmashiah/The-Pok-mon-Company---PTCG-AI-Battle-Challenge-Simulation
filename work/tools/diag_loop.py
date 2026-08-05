"""What is the agent doing in the games that never terminate?"""
import json, os, sys
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class, all_card_data
from cg.game import battle_start, battle_select, battle_finish
cards={c.cardId:c for c in all_card_data()}
OPT={0:"NUMBER",1:"YES",2:"NO",3:"CARD",4:"TOOL_CARD",5:"ENERGY_CARD",6:"ENERGY",
     7:"PLAY",8:"ATTACH",9:"EVOLVE",10:"ABILITY",11:"DISCARD",12:"RETREAT",
     13:"ATTACK",14:"END",15:"SKILL",16:"SPECIAL_CONDITION"}

full=os.path.join(WORK,"agents","v2_lucario"); sys.path.insert(0,full)
cwd=os.getcwd(); os.chdir(full)
env={}; exec(compile(open("main.py",encoding="utf-8-sig").read(),"main.py","exec"),env)
os.chdir(cwd)
fn=[v for v in env.values() if callable(v)][-1]
mine=list(env["my_deck"])
store=json.load(open(os.path.join(WORK,"out","meta_decks.json"),encoding="utf-8"))
opp=next(t["deck"] for t in store["teams"].values() if abs(t.get("score",0)-1055.6)<0.05)

for g in range(12):
    me_first=(g%2==0)
    d0,d1=(mine,opp) if me_first else (opp,mine)
    obs,_=battle_start(list(d0),list(d1))
    if obs is None: continue
    late=Counter(); turns=[]
    for step in range(4000):
        o=to_observation_class(obs)
        if o.current is not None and o.current.result!=-1: break
        who=o.current.yourIndex if o.current is not None else 0
        env["my_deck"]=list(d0 if who==0 else d1); env["DECK"]=env["my_deck"]
        sel=fn(obs)
        if step>3000:      # deep into the stall
            sd=o.select
            opt=sd.option[sel[0]] if sel and sel[0]<len(sd.option) else None
            cid=getattr(opt,"cardId",None)
            nm=cards[cid].name if cid in cards else ""
            late[(who, int(sd.context), OPT.get(int(opt.type) if opt else -1,"?"), nm)]+=1
            turns.append(o.current.turn)
        obs=battle_select(list(sel))
    else:
        print(f"\n=== game {g} STALLED (me_first={me_first}) turn={turns[-1] if turns else '?'} ===")
        print(f"    deck counts: p0={o.current.players[0].deckCount} p1={o.current.players[1].deckCount}")
        print(f"    prizes: p0={len(o.current.players[0].prize)} p1={len(o.current.players[1].prize)}")
        for (w,ctx,t,nm),n in late.most_common(6):
            print(f"    {n:>4}x player{w} ctx={ctx} {t} {nm}")
        battle_finish(); break
    battle_finish()
