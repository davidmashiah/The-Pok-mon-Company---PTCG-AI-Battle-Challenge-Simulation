"""What SPECIFICALLY do we discard / attach differently from the 1275 agent?

ctx 8 (DISCARD) and ctx 22 (ATTACH_TO) are order-independent, so disagreement
there is a real decision gap rather than a within-turn permutation.
"""
import json, os, sys
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
REPLAYS=os.path.join(WORK,"out","replays")
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import all_card_data
cards={c.cardId:c for c in all_card_data()}
def nm(cid): return cards[cid].name if cid in cards else str(cid)

full=os.path.join(WORK,"agents","vDeck_only"); sys.path.insert(0,full)
cwd=os.getcwd(); os.chdir(full)
env={}; exec(compile(open("main.py",encoding="utf-8-sig").read(),"main.py","exec"),env)
os.chdir(cwd)
fn=[v for v in env.values() if callable(v)][-1]

TEAM="majkel"
FOCUS={8:"DISCARD", 22:"ATTACH_TO", 30:"DISCARD_ENERGY", 7:"TO_HAND", 3:"SWITCH"}
pairs={k:Counter() for k in FOCUS}

def card_at(obs, opt, me):
    a=opt.get("area"); i=opt.get("index"); pi=opt.get("playerIndex", me)
    cur=obs.get("current") or {}; pls=cur.get("players") or []
    if pi is None or pi>=len(pls): return None
    p=pls[pi]
    try:
        if a==2: return (p.get("hand") or [])[i]
        if a==1: return ((obs.get("select") or {}).get("deck") or [])[i]
        if a==3: return (p.get("discard") or [])[i]
        if a==4: return (p.get("active") or [])[i]
        if a==5: return (p.get("bench") or [])[i]
    except Exception: return None
    return None

for f in sorted(os.listdir(REPLAYS)):
    if not f.endswith("-replay.json"): continue
    d=json.load(open(os.path.join(REPLAYS,f),encoding="utf-8"))
    names=d.get("info",{}).get("TeamNames") or []
    tgt=next((i for i,n in enumerate(names) if TEAM in (n or "").lower()), None)
    if tgt is None: continue
    for step in d["steps"]:
        a=(step[tgt].get("action") or [])
        if len(a)==60: env["my_deck"]=list(a); env["DECK"]=list(a); break
    for step in d["steps"]:
        ag=step[tgt]; obs=ag.get("observation") or {}; act=ag.get("action")
        if not act or len(act)==60: continue
        sel=obs.get("select") or {}
        ctx=sel.get("context"); opts=sel.get("option") or []
        if ctx not in FOCUS or len(opts)<2: continue
        try: ours=fn(obs)
        except Exception: continue
        if not ours or list(ours)[:len(act)]==list(act): continue
        me=(obs.get("current") or {}).get("yourIndex",0)
        ct=card_at(obs,opts[act[0]],me) if act[0]<len(opts) else None
        co=card_at(obs,opts[ours[0]],me) if ours[0]<len(opts) else None
        tn=nm(ct.get("id")) if isinstance(ct,dict) and ct.get("id") is not None else "?"
        on=nm(co.get("id")) if isinstance(co,dict) and co.get("id") is not None else "?"
        pairs[ctx][(tn,on)]+=1

for ctx,label in FOCUS.items():
    if not pairs[ctx]: continue
    print(f"\n=== ctx {ctx} {label}: they chose -> we chose ===")
    for (t,o),n in pairs[ctx].most_common(8):
        print(f"   {n:>3}x  {t[:34]:<34} -> {o[:34]}")
