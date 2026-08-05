"""How fast does OUR agent land Mega Lucario ex, vs the #1's turn 3-4?"""
import os, sys
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish

AG=sys.argv[1] if len(sys.argv)>1 else "v2_lucario"
N=int(sys.argv[2]) if len(sys.argv)>2 else 40
full=os.path.join(WORK,"agents",AG); sys.path.insert(0,full)
cwd=os.getcwd(); os.chdir(full)
env={}; exec(compile(open("main.py",encoding="utf-8-sig").read(),"main.py","exec"), env)
os.chdir(cwd)
fn=[v for v in env.values() if callable(v)][-1]
deck=list(env.get("DECK") or env.get("my_deck"))
MEGA=678; CAPE=1159

megaturns=[]; capeon=Counter(); nocape=0
for g in range(N):
    obs,_=battle_start(list(deck),list(deck))
    seen=None; capped=False
    for _ in range(3000):
        o=to_observation_class(obs)
        if o.current is not None and o.current.result!=-1: break
        cur=o.current
        if cur is not None and seen is None:
            for p in (cur.players[0], cur.players[1]):
                for mon in list(p.active or [])+list(p.bench or []):
                    if mon is not None and mon.id==MEGA:
                        seen=cur.turn; break
                if seen: break
        if cur is not None and not capped:
            for p in (cur.players[0], cur.players[1]):
                for mon in list(p.active or [])+list(p.bench or []):
                    if mon is None: continue
                    for t in (mon.tools or []):
                        if t is not None and t.id==CAPE:
                            capeon[mon.id]+=1; capped=True
        obs=battle_select(list(fn(obs)))
    battle_finish()
    if seen: megaturns.append(seen)
    if not capped: nocape+=1

megaturns.sort()
print(f"agent={AG}  games={N}")
if megaturns:
    print(f"Mega Lucario ex first on field: n={len(megaturns)} "
          f"min={megaturns[0]} median={megaturns[len(megaturns)//2]} "
          f"max={megaturns[-1]}  (rank-1 agent: median 4)")
else:
    print("Mega Lucario ex NEVER reached the field!")
names={677:'Riolu',678:'Mega Lucario ex'}
print("Hero's Cape attached to:", {names.get(k,k):v for k,v in capeon.items()},
      f" | games with no cape: {nocape}/{N}")
