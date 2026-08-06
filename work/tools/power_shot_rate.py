"""Power Shots per game -- the one number the Decidueye deck lives or dies on.

Win rate cannot steer this build at any sample size we can afford: the deck
either assembles Decidueye + attached energy + a Grass in hand, or it does
nothing at all. Counting the attack itself is nearly deterministic given the
policy and moves immediately, the same reason damage_model_audit counted attacks
instead of games.
"""
import argparse, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
AGENTS=os.path.join(WORK,'agents')

def _load(full):
    if full not in sys.path: sys.path.insert(0,full)
    cwd=os.getcwd()
    try:
        os.chdir(full); env={}
        exec(compile(open(os.path.join(full,'main.py'),encoding='utf-8-sig').read(),'main.py','exec'),env)
        fn=[v for v in env.values() if callable(v)][-1]
        d=fn({'current':None,'select':None})
    finally: os.chdir(cwd)
    if not (isinstance(d,(list,tuple)) and len(d)==60):
        d=[int(x) for x in open(os.path.join(full,'deck.csv'),encoding='utf-8').read().split() if x.strip()]
    return fn,[int(x) for x in d]

def _worker(job):
    agent,opp,n,seed0=job
    sys.path.insert(0,os.path.join(WORK,'lib'))
    from cg.api import to_observation_class, all_attack
    from cg.game import battle_start,battle_select,battle_finish
    A={a.attackId:a for a in all_attack()}
    fa,da=_load(os.path.join(AGENTS,agent)); fb,db=_load(os.path.join(AGENTS,opp))
    atk=Counter(); wins=0; games=0; prizes=[]
    for g in range(n):
        a_first=((seed0+g)%2==0)
        p0,p1=(fa,fb) if a_first else (fb,fa)
        d0,d1=(da,db) if a_first else (db,da)
        a_idx=0 if a_first else 1
        for f in (p0,p1):
            try: f({'current':None,'select':None})
            except Exception: pass
        obs,_=battle_start(list(d0),list(d1))
        if obs is None: continue
        seen=set(); res=None; st=None
        try:
            for _ in range(4000):
                o=to_observation_class(obs)
                if o.current is not None and o.current.result!=-1:
                    res=o.current.result; st=o.current; break
                st=o.current; who=st.yourIndex
                for lg in (o.logs or []):
                    if lg.type!=15: continue
                    k=(st.turn,getattr(lg,'serial',None),getattr(lg,'attackId',None),getattr(lg,'playerIndex',None))
                    if k in seen: continue
                    seen.add(k)
                    if getattr(lg,'playerIndex',None)==a_idx:
                        atk[getattr(A.get(getattr(lg,'attackId',None)),'name','?')]+=1
                obs=battle_select(list((p0 if who==0 else p1)(obs)))
        except Exception: pass
        finally: battle_finish()
        games+=1
        if res==a_idx: wins+=1
        try: prizes.append(6-len(st.players[a_idx].prize or []))
        except Exception: pass
    return {'atk':atk,'wins':wins,'games':games,'prizes':prizes}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--agent',required=True); ap.add_argument('--opponent',default='w5_grimmsnarl')
    ap.add_argument('-n','--games',type=int,default=60); ap.add_argument('--workers',type=int,default=6)
    a=ap.parse_args()
    per=max(1,a.games//a.workers)
    jobs=[(a.agent,a.opponent,per,w*7919) for w in range(a.workers)]
    atk=Counter(); wins=games=0; prizes=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_worker,jobs):
            atk.update(r['atk']); wins+=r['wins']; games+=r['games']; prizes+=r['prizes']
    print(f"\n{a.agent} vs {a.opponent}: {games} games")
    for k,v in atk.most_common(): print(f"   {k[:26]:26s} {v:4d}  ({v/max(games,1):.2f}/game)")
    print(f"   win rate {wins}/{games} = {wins/max(games,1):.3f}")
    if prizes: print(f"   prizes taken: mean {sum(prizes)/len(prizes):.2f}, six-prize games {sum(1 for p in prizes if p>=6)}")
    return 0
if __name__=='__main__': sys.exit(main())
