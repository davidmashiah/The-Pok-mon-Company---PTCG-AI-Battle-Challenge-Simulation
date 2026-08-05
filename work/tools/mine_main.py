"""ctx 0 MAIN is 54% of all decisions at 19.7% agreement. Which CARDS do
same-archetype winners play where we play something else?

Caveat kept in view: actions within a turn partly commute, so some of this is
ordering rather than error. A consistent directional bias (we always play X,
they always play Y) is still signal.
"""
import json, os, sys, zipfile
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE); ROOT=os.path.dirname(WORK)
ZIP=os.path.join(ROOT,"data","episodes","d0802","pokemon-tcg-ai-battle-episodes-2026-08-02.zip")
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import all_card_data
C={c.cardId:c for c in all_card_data()}
def nm(i): return C[i].name if i in C else str(i)
OPT={7:"PLAY",8:"ATTACH",9:"EVOLVE",10:"ABILITY",12:"RETREAT",13:"ATTACK",14:"END",11:"DISCARD"}
AGENT="v14_search_noloop2"; LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 45
MARK={677,678}
full=os.path.join(WORK,"agents",AGENT); sys.path.insert(0,full)
cwd=os.getcwd(); os.chdir(full)
env={}; exec(compile(open("main.py",encoding="utf-8-sig").read(),"main.py","exec"),env)
os.chdir(cwd)
FN=[v for k,v in env.items() if callable(v)][-1]

def hand_card(obs,opt,me):
    if opt.get("type")!=7: return None
    i=opt.get("index")
    cur=obs.get("current") or {}; pls=cur.get("players") or []
    if me>=len(pls): return None
    try: return (pls[me].get("hand") or [])[i]
    except Exception: return None

play_pairs=Counter(); type_pairs=Counter(); we_play=Counter(); they_play=Counter(); eps=0
with zipfile.ZipFile(ZIP) as zf:
    for name in [n for n in zf.namelist() if n.endswith(".json")]:
        if eps>=LIMIT: break
        try: d=json.loads(zf.open(name).read().decode("utf-8"))
        except Exception: continue
        rw=d.get("rewards") or []
        if 1 not in rw: continue
        w=rw.index(1); wd=None
        for st in d.get("steps",[]):
            if w<len(st):
                a=st[w].get("action") or []
                if len(a)==60: wd=set(a); break
        if wd is None or not (wd & MARK): continue
        eps+=1; env["my_deck"]=list(wd); env["DECK"]=list(wd)
        for st in d.get("steps",[]):
            if w>=len(st): continue
            ag=st[w]; obs=ag.get("observation") or {}; act=ag.get("action")
            if not act or len(act)==60: continue
            sel=obs.get("select") or {}
            if sel.get("context")!=0: continue
            opts=sel.get("option") or []
            if len(opts)<2: continue
            try: ours=FN(obs)
            except Exception: continue
            if not ours or ours[0]>=len(opts) or act[0]>=len(opts): continue
            if list(ours)[:len(act)]==list(act): continue
            me=(obs.get("current") or {}).get("yourIndex",0)
            ot=opts[act[0]].get("type"); oo=opts[ours[0]].get("type")
            type_pairs[(OPT.get(ot,ot),OPT.get(oo,oo))]+=1
            ct=hand_card(obs,opts[act[0]],me); co=hand_card(obs,opts[ours[0]],me)
            if ct and co:
                tn=nm(ct.get("id")); on=nm(co.get("id"))
                play_pairs[(tn,on)]+=1; they_play[tn]+=1; we_play[on]+=1

print(f"same-archetype winners: {eps} episodes\n")
print("=== ctx 0: action TYPE they chose -> we chose ===")
for (t,o),n in type_pairs.most_common(10): print(f"   {n:>4}x  {t} -> {o}")
print("\n=== PLAY vs PLAY: which card ===")
for (t,o),n in play_pairs.most_common(12): print(f"   {n:>4}x  {t[:28]:<28} -> {o[:28]}")
print("\n=== net bias (positive = we OVER-play it) ===")
allc=set(we_play)|set(they_play)
for c in sorted(allc, key=lambda c: we_play[c]-they_play[c], reverse=True)[:8]:
    print(f"   {we_play[c]-they_play[c]:>+5}   {c[:34]:<34} (us {we_play[c]}, them {they_play[c]})")
for c in sorted(allc, key=lambda c: we_play[c]-they_play[c])[:5]:
    print(f"   {we_play[c]-they_play[c]:>+5}   {c[:34]:<34} (us {we_play[c]}, them {they_play[c]})")
