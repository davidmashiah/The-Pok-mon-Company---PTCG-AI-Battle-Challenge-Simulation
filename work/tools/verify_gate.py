"""Did gating cut cost without losing lethals?

Compares gated vs ungated find_lethal on the SAME frames.
"""
import os, sys, time
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class, SelectContext
from cg.game import battle_start, battle_select, battle_finish
import fsearch, policy

deck=[int(x.strip()) for x in open(os.path.join(WORK,"agents","v5_topdeck","deck.csv")) if x.strip()][:60]

mains=0; gated_in=0
t_gated=0.0; t_ungated=0.0
lethal_gated=0; lethal_ungated=0
missed=[]

for g in range(2):
    det_g=fsearch.Determinizer(deck); det_u=fsearch.Determinizer(deck)
    obs,_=battle_start(list(deck),list(deck))
    for _ in range(3000):
        o=to_observation_class(obs)
        if o.current is not None and o.current.result!=-1: break
        if o.select is not None and int(o.select.context)==int(SelectContext.MAIN):
            mains+=1
            det_g.observe(o); det_g.note_opponent(o)
            det_u.observe(o); det_u.note_opponent(o)
            plaus = fsearch.lethal_plausible(o)
            if plaus: gated_in+=1
            # gated path
            t0=time.time()
            lg = fsearch.find_lethal(o,det_g,time_budget=1.0) if plaus else None
            t_gated += time.time()-t0
            if lg is not None: lethal_gated+=1
            # ungated path (bypass the gate by calling the inner work directly)
            t0=time.time()
            _save=fsearch.lethal_plausible
            fsearch.lethal_plausible=lambda _o: True
            lu = fsearch.find_lethal(o,det_u,time_budget=1.0)
            fsearch.lethal_plausible=_save
            t_ungated += time.time()-t0
            if lu is not None:
                lethal_ungated+=1
                if lg is None: missed.append(mains)
        obs=battle_select(list(policy.act(obs,deck)))
    battle_finish()

print(f"MAIN frames            : {mains}")
print(f"  passed the gate      : {gated_in}  ({100*gated_in/max(mains,1):.0f}%)")
print(f"lethals found  gated   : {lethal_gated}")
print(f"lethals found  ungated : {lethal_ungated}")
print(f"  lethals the gate LOST: {len(missed)}")
print(f"time  gated            : {t_gated:.1f}s")
print(f"time  ungated          : {t_ungated:.1f}s   (speedup {t_ungated/max(t_gated,1e-9):.1f}x)")
print()
if missed: print("==> GATE LOSES LETHALS. Loosen it.")
else: print("==> gate loses nothing and is cheaper. Keep.")
