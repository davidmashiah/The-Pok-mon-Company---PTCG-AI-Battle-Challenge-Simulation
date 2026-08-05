"""Which early-return inside best_action is firing?"""
import json, os, sys
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class, SelectContext
from cg.game import battle_start, battle_select, battle_finish
import fsearch
from fsearch import _i

AG="v15_deepsearch"
full=os.path.join(WORK,"agents",AG); sys.path.insert(0,full)
cwd=os.getcwd(); os.chdir(full)
env={}; exec(compile(open("main.py",encoding="utf-8-sig").read(),"main.py","exec"),env)
os.chdir(cwd)
fn=[v for v in env.values() if callable(v)][-1]
mine=list(env.get("DECK") or env.get("my_deck"))

why=Counter()
orig=fsearch.best_action
def probe(obs, det, rollout, candidates, time_budget=1.0, max_candidates=8):
    if not fsearch.HAVE_SEARCH: why["no_search_api"]+=1
    elif obs.select is None or obs.current is None: why["no_select_or_current"]+=1
    elif _i(obs.select.context) != _i(SelectContext.MAIN): why["not_MAIN"]+=1
    elif len([c for c in candidates][:max_candidates]) < 2: why["fewer_than_2_candidates"]+=1
    elif det.build(obs) is None: why["determinization_failed"]+=1
    else: why["REACHED_SEARCH"]+=1
    return orig(obs, det, rollout, candidates, time_budget=time_budget,
                max_candidates=max_candidates)
fsearch.best_action=probe

store=json.load(open(os.path.join(WORK,"out","meta_decks.json"),encoding="utf-8"))
opp=next(t["deck"] for t in store["teams"].values() if abs(t.get("score",0)-1275.3)<0.05)
for g in range(6):
    d0,d1=(mine,opp) if g%2==0 else (opp,mine)
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
print("best_action early-return reasons:")
for k,v in why.most_common(): print(f"   {v:>5}  {k}")
