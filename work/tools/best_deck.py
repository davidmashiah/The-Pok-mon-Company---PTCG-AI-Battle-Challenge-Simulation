"""Pull the highest-performing decklist for a target archetype from real games."""
import json, os, sys, zipfile
from collections import Counter, defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE); ROOT=os.path.dirname(WORK)
ZIP=os.path.join(ROOT,"data","episodes","d0802","pokemon-tcg-ai-battle-episodes-2026-08-02.zip")
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import all_card_data
C={c.cardId:c for c in all_card_data()}
CT={0:'Pokemon',1:'Item',2:'Tool',3:'Supporter',4:'Stadium',5:'B.Energy',6:'S.Energy'}
MARK={int(x) for x in sys.argv[1].split(",")}
LIMIT=int(sys.argv[2]) if len(sys.argv)>2 else 900

rec=defaultdict(lambda:[0,0])   # decklist -> [wins, games]
seen=0
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
        seen+=1; w=rw.index(1)
        for ai in (0,1):
            if not (set(decks[ai]) & MARK): continue
            key=tuple(sorted(decks[ai]))
            rec[key][1]+=1
            if ai==w: rec[key][0]+=1

rows=[(wins/games, games, wins, k) for k,(wins,games) in rec.items() if games>=4]
rows.sort(key=lambda r:(-r[0]*min(r[1],20), -r[1]))
print(f"episodes scanned {seen}; distinct lists with >=4 games: {len(rows)}\n")
for wr,g,w,k in rows[:5]:
    print(f"--- winrate {wr:.3f} over {g} games ({w} wins) ---")
for wr,g,w,k in rows[:1]:
    cnt=Counter(k)
    print(f"\n=== BEST LIST (winrate {wr:.3f}, {g} games) ===")
    for grp in (0,3,1,2,4,6,5):
        rs=[(c,n) for c,n in cnt.items() if int(C[c].cardType)==grp]
        if not rs: continue
        print(f"-- {CT[grp]} ({sum(n for _,n in rs)})")
        for c,n in sorted(rs,key=lambda r:-r[1]):
            print(f"   x{n} [{c}] {C[c].name}")
    open(os.path.join(WORK,"out","best_lopunny_deck.csv"),"w").write("\n".join(str(x) for x in k)+"\n")
    print("\nwrote work/out/best_lopunny_deck.csv")
