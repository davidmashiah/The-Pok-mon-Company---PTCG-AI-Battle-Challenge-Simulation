"""Which DECKS actually win, measured from real games between strong agents?

Every deck test in this project so far used our own ~700 policy to pilot both
sides, which measures "which deck our weak policy handles best", not deck
strength. This uses the host's top-episode dataset instead: real games, real
agents (median rating ~1085), real outcomes. Our code never touches the play.

Confound stated up front: deck win rate here still mixes deck strength with
pilot skill. But every pilot in this dataset is far stronger than ours and
drawn from a similar band, so it is a far fairer comparison than self-play.
"""
import json, os, sys, zipfile
from collections import Counter, defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE); ROOT=os.path.dirname(WORK)
ZIP=os.path.join(ROOT,"data","episodes","d0802","pokemon-tcg-ai-battle-episodes-2026-08-02.zip")
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import all_card_data
C={c.cardId:c for c in all_card_data()}
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 900

def archetype(deck):
    """Name a deck by its ex / Mega-ex line."""
    cnt=Counter(deck)
    ex=[]
    for cid,n in cnt.items():
        c=C.get(cid)
        if c is not None and int(c.cardType)==0 and (getattr(c,'megaEx',False) or getattr(c,'ex',False)):
            ex.append((n,c.name))
    if not ex: return "(no ex line)"
    ex.sort(reverse=True)
    return " + ".join(n for _,n in ex[:2])

wins=Counter(); games=Counter(); seen=0
with zipfile.ZipFile(ZIP) as zf:
    for name in [n for n in zf.namelist() if n.endswith(".json")]:
        if seen>=LIMIT: break
        try: d=json.loads(zf.open(name).read().decode("utf-8"))
        except Exception: continue
        rw=d.get("rewards") or []
        if len(rw)!=2 or 1 not in rw: continue
        decks={}
        for st in d.get("steps",[]):
            for ai in (0,1):
                if ai<len(st) and ai not in decks:
                    a=st[ai].get("action") or []
                    if len(a)==60: decks[ai]=a
            if len(decks)==2: break
        if len(decks)!=2: continue
        seen+=1
        w=rw.index(1)
        for ai in (0,1):
            arch=archetype(decks[ai])
            games[arch]+=1
            if ai==w: wins[arch]+=1

print(f"games analysed: {seen}  (real agents, ~1085 median rating)\n")
print(f"{'archetype':<52}{'games':>7}{'winrate':>9}")
print("-"*70)
rows=[(wins[a]/games[a], games[a], a) for a in games if games[a]>=8]
for wr,n,a in sorted(rows, reverse=True):
    star = "  <-- OURS" if "Lucario" in a else ""
    print(f"{a[:50]:<52}{n:>7}{wr:>9.3f}{star}")
print()
low=[ (wins[a]/games[a],games[a],a) for a in games if games[a]>=8 ]
print(f"archetypes with >=25 games: {len(low)}")
