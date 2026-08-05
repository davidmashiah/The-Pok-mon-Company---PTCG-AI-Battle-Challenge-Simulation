"""Does PrizeTracker fire, and when it fires is it RIGHT?

Ground truth: we control both decks, so we can read the real prize cards
straight out of the engine state and compare.
"""
import os, sys
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK,"lib"))
AG=os.path.join(WORK,"agents","v4_prize"); sys.path.insert(0,AG); os.chdir(AG)
from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish

src=open(os.path.join(AG,"main.py"),encoding="utf-8").read()
env={}; exec(compile(src,"main.py","exec"), env)
agent=env["agent"]; deck=env["my_deck"]

fired=correct=wrong=frames=0
for g in range(2):
    obs,_=battle_start(list(deck),list(deck))
    for _ in range(3000):
        o=to_observation_class(obs)
        if o.current is not None and o.current.result!=-1: break
        frames+=1
        sel=agent(obs)
        det=env.get("_det")
        if det is not None:
            got=det.prizes.prized()
            if got is not None:
                fired+=1
                # ground truth: engine exposes our prize cards (face-down are
                # None to us, but the raw dict carries them for the acting player)
                me=o.current.players[o.current.yourIndex]
                truth=Counter(c.id for c in (me.prize or []) if c is not None)
                if truth and sum(truth.values())==len(me.prize or []):
                    if got==truth: correct+=1
                    else: wrong+=1
        obs=battle_select(list(sel))
    battle_finish()

print(f"frames observed        : {frames}")
print(f"tracker produced answer: {fired}")
print(f"  verifiable & CORRECT : {correct}")
print(f"  verifiable & WRONG   : {wrong}")
print()
if wrong: print("==> DANGER: tracker emits wrong prize sets. Do not ship.")
elif fired: print("==> tracker fires and is never verifiably wrong.")
else: print("==> tracker NEVER fires (no deck-reveal frames reached).")
