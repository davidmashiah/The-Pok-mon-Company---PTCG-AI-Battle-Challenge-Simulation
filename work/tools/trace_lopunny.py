"""Trace a winning Lopunny game turn-by-turn: energy, retreats, attacks.

My hand-written pilot lost 386 of 396 games. It was written from card text.
This reads what winners actually DO instead.
"""
import json, os, sys, zipfile
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE); ROOT=os.path.dirname(WORK)
ZIP=os.path.join(ROOT,"data","episodes","d0802","pokemon-tcg-ai-battle-episodes-2026-08-02.zip")
sys.path.insert(0, os.path.join(WORK,"lib"))
from cg.api import all_card_data
C={c.cardId:c for c in all_card_data()}
def nm(i): return C[i].name if i in C else str(i)
LOP=849; N_GAMES=int(sys.argv[1]) if len(sys.argv)>1 else 2

# LogType: 4 DRAW 6 MOVE_CARD 8 SWITCH 10 PLAY 11 ATTACH 12 EVOLVE 15 ATTACK 16 HP_CHANGE
shown=0
with zipfile.ZipFile(ZIP) as zf:
    for name in [n for n in zf.namelist() if n.endswith(".json")]:
        if shown>=N_GAMES: break
        try: d=json.loads(zf.open(name).read().decode("utf-8"))
        except Exception: continue
        rw=d.get("rewards") or []
        if 1 not in rw: continue
        w=rw.index(1); wd=None
        for st in d.get("steps",[]):
            if w<len(st):
                a=st[w].get("action") or []
                if len(a)==60: wd=a; break
        if wd is None or LOP not in wd: continue
        shown+=1
        print(f"\n{'='*72}\nWINNING LOPUNNY GAME  ({os.path.basename(name)})\n{'='*72}")
        turn=0; per=defaultdict(list)
        for st in d.get("steps",[]):
            if w>=len(st): continue
            obs=st[w].get("observation") or {}
            cur=obs.get("current") or {}
            if cur.get("turn"): turn=max(turn,cur["turn"])
            for lg in (obs.get("logs") or []):
                if lg.get("playerIndex")!=w: continue
                t=lg.get("type")
                if t==11:
                    per[turn].append(f"ATTACH {nm(lg.get('cardId'))} -> {nm(lg.get('cardIdTarget'))}")
                elif t==12:
                    per[turn].append(f"EVOLVE -> {nm(lg.get('cardId'))}")
                elif t==10:
                    per[turn].append(f"PLAY {nm(lg.get('cardId'))}")
                elif t==8:
                    per[turn].append(f"SWITCH: bench {nm(lg.get('cardIdBench'))} -> ACTIVE (was {nm(lg.get('cardIdActive'))})")
                elif t==15:
                    per[turn].append(f"** ATTACK with {nm(lg.get('cardId'))} (atk {lg.get('attackId')})")
                elif t==16 and lg.get("value") and abs(lg["value"])>=100:
                    per[turn].append(f"     dealt/took {lg['value']}")
        for tn in sorted(per):
            if tn>14: break
            print(f"\n-- turn {tn} --")
            for e in per[tn][:14]: print("   ", e)
