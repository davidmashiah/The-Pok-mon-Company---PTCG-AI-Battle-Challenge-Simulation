"""Does opponent-deck matching actually fire, and how often is it right?

Ground truth is available because we choose the opponent's deck, so we can
compare the matched list against the real one.
"""
import json, os, sys
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import to_observation_class, SelectContext
from cg.game import battle_start, battle_select, battle_finish
import fsearch, policy

print("meta decklists loaded:", len(fsearch.meta_decks()))
store=json.load(open(os.path.join(WORK,"out","meta_decks.json"),encoding="utf-8"))
HARD={1010.8,1109.6,1060.3,1034.9,1275.3,1063.4,1104.6}
opps=[];seen=set()
for t in sorted(store["teams"].values(), key=lambda t:-t.get("score",0)):
    d=t.get("deck")
    if not d or len(d)!=60: continue
    k=tuple(sorted(d))
    if k in seen or round(t.get("score",0),1) not in HARD: continue
    seen.add(k); opps.append(d)

mine=[int(x.strip()) for x in open(os.path.join(WORK,"agents","v14_search_noloop2","deck.csv")) if x.strip()][:60]
matched=unmatched=correct=wrong=0
for g in range(14):
    opp=opps[g%len(opps)]
    det=fsearch.Determinizer(mine)
    obs,_=battle_start(list(mine),list(opp))
    if obs is None: continue
    for _ in range(2500):
        o=to_observation_class(obs)
        if o.current is not None and o.current.result!=-1: break
        if o.current is not None and o.current.yourIndex==0:
            det.observe(o); det.note_opponent(o)
            m=fsearch.match_opponent_deck(det._seen_opp)
            if m is None: unmatched+=1
            else:
                matched+=1
                if sorted(m)==sorted(opp): correct+=1
                else: wrong+=1
        obs=battle_select(list(policy.act(obs,mine)))
    battle_finish()
tot=matched+unmatched
print(f"frames where we tried to match : {tot}")
print(f"  matched a known decklist     : {matched} ({matched/max(tot,1):.1%})")
print(f"     -> matched the RIGHT deck : {correct}")
print(f"     -> matched the WRONG deck : {wrong}")
print(f"  no match (fell back to filler): {unmatched}")
if matched: print(f"\nprecision when it fires: {correct/matched:.1%}")
