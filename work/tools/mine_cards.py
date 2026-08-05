"""What CARDS do same-archetype winners choose where we choose differently?

Percentages are not rules. This resolves the top order-independent contexts
down to card names, so the disagreements become editable policy.
"""
import json, os, sys, zipfile
from collections import Counter, defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE); ROOT=os.path.dirname(WORK)
ZIP=os.path.join(ROOT,"data","episodes","d0802","pokemon-tcg-ai-battle-episodes-2026-08-02.zip")
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import all_card_data
C={c.cardId:c for c in all_card_data()}
def nm(i): return C[i].name if i in C else str(i)

AGENT="v14_search_noloop2"; LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 60
MARK={677,678}
FOCUS={43:"ACTIVATE(yes/no)",22:"ATTACH_TO",8:"DISCARD",7:"TO_HAND",21:"ATTACH_FROM",3:"SWITCH"}
full=os.path.join(WORK,"agents",AGENT); sys.path.insert(0,full)
cwd=os.getcwd(); os.chdir(full)
env={}; exec(compile(open("main.py",encoding="utf-8-sig").read(),"main.py","exec"),env)
os.chdir(cwd)
FN=[v for k,v in env.items() if callable(v)][-1]

def card_of(obs,opt,me):
    a=opt.get("area"); i=opt.get("index"); pi=opt.get("playerIndex")
    if pi is None: pi=me
    cur=obs.get("current") or {}; pls=cur.get("players") or []
    if pi>=len(pls): return None
    p=pls[pi]
    try:
        if a==2: return (p.get("hand") or [])[i]
        if a==1: return ((obs.get("select") or {}).get("deck") or [])[i]
        if a==3: return (p.get("discard") or [])[i]
        if a==4: return (p.get("active") or [])[i]
        if a==5: return (p.get("bench") or [])[i]
        if a==6: return (p.get("prize") or [])[i]
    except Exception: return None
    return None

pairs=defaultdict(Counter); yesno=Counter(); eps=0
with zipfile.ZipFile(ZIP) as zf:
    for name in [n for n in zf.namelist() if n.endswith(".json")]:
        if eps>=LIMIT: break
        try:
            d=json.loads(zf.open(name).read().decode("utf-8"))
        except Exception: continue
        rw=d.get("rewards") or []
        if 1 not in rw: continue
        w=rw.index(1)
        wd=None
        for st in d.get("steps",[]):
            if w<len(st):
                a=st[w].get("action") or []
                if len(a)==60: wd=set(a); break
        if wd is None or not (wd & MARK): continue
        eps+=1
        env["my_deck"]=list(wd); env["DECK"]=list(wd)
        for st in d.get("steps",[]):
            if w>=len(st): continue
            ag=st[w]; obs=ag.get("observation") or {}; act=ag.get("action")
            if not act or len(act)==60: continue
            sel=obs.get("select") or {}
            ctx=sel.get("context"); opts=sel.get("option") or []
            if ctx not in FOCUS or len(opts)<2: continue
            try: ours=FN(obs)
            except Exception: continue
            if not ours or list(ours)[:len(act)]==list(act): continue
            me=(obs.get("current") or {}).get("yourIndex",0)
            if ctx==43:
                if act[0]>=len(opts) or ours[0]>=len(opts): continue
                t=opts[act[0]].get("type"); o=opts[ours[0]].get("type")
                yesno[("them=YES" if t==1 else "them=NO", "us=YES" if o==1 else "us=NO")]+=1
                continue
            ct=card_of(obs,opts[act[0]],me) if act[0]<len(opts) else None
            co=card_of(obs,opts[ours[0]],me) if ours[0]<len(opts) else None
            tn=nm(ct["id"]) if isinstance(ct,dict) and ct.get("id") is not None else "?"
            on=nm(co["id"]) if isinstance(co,dict) and co.get("id") is not None else "?"
            pairs[ctx][(tn,on)]+=1

print(f"same-archetype winners: {eps} episodes\n")
print("=== ctx 43 ACTIVATE (yes/no) ===")
for k,v in yesno.most_common(): print(f"   {v:>4}x  {k[0]} / {k[1]}")
for ctx,label in FOCUS.items():
    if ctx==43 or not pairs[ctx]: continue
    print(f"\n=== ctx {ctx} {label}: winner chose -> we chose ===")
    for (t,o),n in pairs[ctx].most_common(8):
        print(f"   {n:>4}x  {t[:30]:<30} -> {o[:30]}")
