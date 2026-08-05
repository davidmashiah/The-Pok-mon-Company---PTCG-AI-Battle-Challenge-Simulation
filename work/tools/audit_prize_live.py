"""Does the PrizeTracker's exact split actually reach search_begin?

The tracker is verified correct (0 violations) and verified to fire. But
Determinizer.build() only USES it when the inferred multiset is consistent with
the remaining unseen cards. If that consistency check never passes, the tracker
is correct and useless -- the same 'present but not executing' failure that has
already appeared three times.
"""
import json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class, SelectContext
from cg.game import battle_start, battle_select, battle_finish
import fsearch, policy

deck=[int(x.strip()) for x in open(os.path.join(WORK,"agents","v14_search_noloop2","deck.csv")) if x.strip()][:60]
store=json.load(open(os.path.join(WORK,"out","meta_decks.json"),encoding="utf-8"))
HARD={1010.8,1109.6,1060.3,1034.9,1275.3,1063.4,1104.6}
opps=[];seen=set()
for t in sorted(store["teams"].values(), key=lambda t:-t.get("score",0)):
    d=t.get("deck")
    if not d or len(d)!=60: continue
    k=tuple(sorted(d))
    if k in seen or round(t.get("score",0),1) not in HARD: continue
    seen.add(k); opps.append(d)

tot_exact=tot_guess=0; tot_match=tot_unmatch=0
for g in range(10):
    opp=opps[g%len(opps)]
    det=fsearch.Determinizer(deck)
    obs,_=battle_start(list(deck),list(opp))
    if obs is None: continue
    for _ in range(2500):
        o=to_observation_class(obs)
        if o.current is not None and o.current.result!=-1: break
        if o.current is not None and o.current.yourIndex==0:
            det.observe(o); det.note_opponent(o)
            if o.select is not None and int(o.select.context)==int(SelectContext.MAIN):
                det.build(o)          # exercises the real path
        obs=battle_select(list(policy.act(obs,deck)))
    battle_finish()
    tot_exact+=det.exact_prizes; tot_guess+=det.guessed_prizes
    tot_match+=det.matched_opp;  tot_unmatch+=det.unmatched_opp

n=tot_exact+tot_guess
print(f"determinizations built : {n}")
print(f"  OUR prizes exact     : {tot_exact} ({tot_exact/max(n,1):.1%})")
print(f"  OUR prizes guessed   : {tot_guess}")
m=tot_match+tot_unmatch
print(f"  OPP deck matched     : {tot_match} ({tot_match/max(m,1):.1%})")
print(f"  OPP deck filler      : {tot_unmatch}")
print()
if tot_exact==0:
    print("==> prize tracker NEVER reaches search_begin: correct but useless.")
else:
    print(f"==> prize knowledge reaches search on {tot_exact/max(n,1):.1%} of determinizations.")
