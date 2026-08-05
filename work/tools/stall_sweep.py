"""Sweep the whole deck library for non-terminating games.

A stalled game is a forfeit on the ladder (600 s budget, actTimeout=0) but is
silently dropped from win-rate samples. Completion rate must be tracked
separately or this class of bug stays invisible.
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)

def run(job):
    agent, mine, opp, n, seed, label = job
    sys.path.insert(0, os.path.join(WORK,"lib"))
    from cg.api import to_observation_class
    from cg.game import battle_start, battle_select, battle_finish
    full=os.path.join(WORK,"agents",agent); sys.path.insert(0,full)
    cwd=os.getcwd(); os.chdir(full)
    env={}; exec(compile(open("main.py",encoding="utf-8-sig").read(),"main.py","exec"),env)
    os.chdir(cwd)
    fn=[v for v in env.values() if callable(v)][-1]
    ok=stall=exc=0; worst=0
    for g in range(n):
        first=((seed+g)%2==0)
        d0,d1=(mine,opp) if first else (opp,mine)
        obs,_=battle_start(list(d0),list(d1))
        if obs is None: exc+=1; continue
        done=False
        try:
            for step in range(4000):
                o=to_observation_class(obs)
                if o.current is not None and o.current.result!=-1:
                    ok+=1; done=True; worst=max(worst,step); break
                who=o.current.yourIndex if o.current is not None else 0
                env["my_deck"]=list(d0 if who==0 else d1); env["DECK"]=env["my_deck"]
                obs=battle_select(list(fn(obs)))
            if not done: stall+=1
        except Exception:
            exc+=1
        finally:
            battle_finish()
    return label, ok, stall, exc, worst

if __name__=="__main__":
    agent=sys.argv[1] if len(sys.argv)>1 else "v2_lucario"
    per=int(sys.argv[2]) if len(sys.argv)>2 else 30
    full=os.path.join(WORK,"agents",agent)
    mine=[int(x.strip()) for x in open(os.path.join(full,"deck.csv")) if x.strip()][:60]
    store=json.load(open(os.path.join(WORK,"out","meta_decks.json"),encoding="utf-8"))
    seen=set(); jobs=[]
    for t in sorted(store["teams"].values(), key=lambda t:-t.get("score",0)):
        d=t.get("deck")
        if not d or len(d)!=60: continue
        k=tuple(sorted(d))
        if k in seen: continue
        seen.add(k)
        jobs.append((agent, mine, d, per, 11, f"{t['score']:.1f}"))
    print(f"agent={agent}  {len(jobs)} decks x {per} games")
    tot_ok=tot_stall=tot_exc=0; bad=[]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(run,j) for j in jobs]):
            label,ok,stall,exc,worst=f.result()
            tot_ok+=ok; tot_stall+=stall; tot_exc+=exc
            if stall or exc: bad.append((stall,exc,label,worst))
    n=tot_ok+tot_stall+tot_exc
    print(f"\ncompleted {tot_ok}/{n} = {tot_ok/n:.4f}   stalls={tot_stall}  exceptions={tot_exc}")
    if bad:
        print("decks with problems (stalls, exceptions, opponent score):")
        for s,e,l,w in sorted(bad, reverse=True):
            print(f"   stalls={s:<4} exc={e:<4} vs {l}")
    else:
        print("no stalls or exceptions anywhere.")
