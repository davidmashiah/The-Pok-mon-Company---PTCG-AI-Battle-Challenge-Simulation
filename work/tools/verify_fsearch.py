"""Does OUR search_begin call actually work, where the public one throws?"""
import os, sys, time
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class, SelectContext, search_begin, search_step
from cg.game import battle_start, battle_select, battle_finish
import fsearch, policy

deck=[int(x.strip()) for x in open(os.path.join(WORK,"agents","v2_lucario","deck.csv")) if x.strip()][:60]
det=fsearch.Determinizer(deck)

ok=fail=0; errs={}; lethals=0; mains=0
t0=time.time()
for g in range(3):
    obs,_=battle_start(list(deck),list(deck))
    for _ in range(2000):
        o=to_observation_class(obs)
        if o.current is not None and o.current.result!=-1: break
        if o.select is not None and int(o.select.context)==int(SelectContext.MAIN):
            mains+=1
            det.note_opponent(o)
            kw=det.build(o)
            if kw is None:
                fail+=1; errs["build_none"]=errs.get("build_none",0)+1
            else:
                try:
                    st=search_begin(o, manual_coin=False, **kw)
                    if st is None: fail+=1; errs["state_none"]=errs.get("state_none",0)+1
                    else:
                        ok+=1
                        # can we actually step it?
                        try: search_step(st.searchId,[0])
                        except Exception as e:
                            errs["step:"+type(e).__name__]=errs.get("step:"+type(e).__name__,0)+1
                except Exception as e:
                    fail+=1
                    k=type(e).__name__+": "+str(e)[:70]
                    errs[k]=errs.get(k,0)+1
            lt=fsearch.find_lethal(o,det,time_budget=0.5)
            if lt is not None: lethals+=1
        obs=battle_select(list(policy.act(obs,deck)))
    battle_finish()

print(f"MAIN selections     : {mains}")
print(f"search_begin OK     : {ok}")
print(f"search_begin FAILED : {fail}")
print(f"proven lethals found: {lethals}")
print(f"errors              : {errs}")
print(f"elapsed             : {time.time()-t0:.1f}s")
print()
print("==> WORKS" if ok>0 and fail==0 else "==> STILL BROKEN")
