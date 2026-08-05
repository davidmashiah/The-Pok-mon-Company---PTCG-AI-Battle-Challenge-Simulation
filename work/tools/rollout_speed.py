"""Can we afford Monte Carlo rollouts to TERMINAL states inside the 600s budget?"""
import os, sys, time, random
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class, SelectContext
from cg.game import battle_start, battle_select, battle_finish
import fsearch, policy
from fsearch import search_begin, search_step, search_end

deck=[int(x.strip()) for x in open(os.path.join(WORK,"agents","v14_search_noloop2","deck.csv")) if x.strip()][:60]
det=fsearch.Determinizer(deck)
obs,_=battle_start(list(deck),list(deck))

frame=None
for _ in range(400):
    o=to_observation_class(obs)
    if o.current is not None and o.current.result!=-1: break
    det.observe(o)
    if (o.current is not None and o.current.turn>=6 and o.select is not None
            and int(o.select.context)==int(SelectContext.MAIN)
            and len(o.select.option)>=3):
        frame=o; break
    obs=battle_select(list(policy.act(obs,deck)))

if frame is None:
    print("could not reach a branching mid-game MAIN frame"); raise SystemExit(1)
det.note_opponent(frame); kw=det.build(frame)
print(f"frame: turn {frame.current.turn} | {len(frame.select.option)} options")

rng=random.Random(0)
times=[]; depths=[]; terminal=0
t_all=time.time()
for _ in range(30):
    t0=time.time(); d=0
    try:
        cur=search_begin(frame, manual_coin=False, **kw)
        while cur is not None and d<600:
            sobs=cur.observation; c=sobs.current
            if c is not None and c.result!=-1: terminal+=1; break
            sel=sobs.select
            if sel is None: break
            n=len(sel.option)
            if n==0: break
            k=min(max(1,sel.minCount), n)
            cur=search_step(cur.searchId, rng.sample(range(n),k))
            d+=1
    except Exception:
        pass
    finally:
        try: search_end()
        except Exception: pass
    times.append(time.time()-t0); depths.append(d)
battle_finish()
tot=time.time()-t_all
times.sort(); depths.sort()
rps=len(times)/max(tot,1e-9)
print(f"rollouts          : {len(times)}   reached terminal: {terminal}")
print(f"median depth      : {depths[len(depths)//2]} engine steps")
print(f"median time       : {times[len(times)//2]*1000:.0f} ms")
print(f"throughput        : {rps:.1f} rollouts/s")
print()
print(f"400s budget       -> {int(400*rps)} rollouts/episode")
print(f"~60 decisions     -> {int(400*rps/60)} rollouts per decision")
