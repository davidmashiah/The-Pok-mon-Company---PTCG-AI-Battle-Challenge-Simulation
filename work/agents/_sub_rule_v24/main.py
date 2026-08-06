import policy_features as pf
from policies.rule_v24 import main as _sub
DECK=list(pf.DECK)
def agent(obs):
    if not obs or obs.get('select') is None:
        try: _sub.agent({})
        except Exception: pass
        return list(DECK)
    try:
        return list(_sub.agent(obs))
    except Exception:
        s=obs.get('select') or {}
        n=len(s.get('option') or [])
        mn=int(s.get('minCount',0) or 0); mx=min(n,int(s.get('maxCount',0) or 0))
        return list(range(max(mn,mx)))
def sub_rule_v24_entry(obs):
    return agent(obs)
