import policy_features as pf
from policies.handwritten_v26 import main as _sub
import cape_guard
DECK=list(pf.DECK)
def _legal(a,s,n):
    try:
        lo=int(s.get('minCount',0) or 0); hi=int(s.get('maxCount',0) or 0)
        return (lo<=len(a)<=max(hi,lo) and len(a)==len(set(a))
                and all(isinstance(i,int) and 0<=i<n for i in a))
    except Exception:
        return False
def agent(obs):
    if not obs or obs.get('select') is None:
        try: _sub.agent({})
        except Exception: pass
        return list(DECK)
    s=obs.get('select') or {}
    n=len(s.get('option') or [])
    try:
        base=list(_sub.agent(obs))
    except Exception:
        lo=int(s.get('minCount',0) or 0); hi=min(n,int(s.get('maxCount',0) or 0))
        base=list(range(max(lo,hi)))
    try:
        ov=cape_guard.choose(obs,base)
    except Exception:
        ov=None
    if ov is not None and _legal(ov,s,n):
        return ov
    return base
def w40_cape_entry(obs):
    return agent(obs)
