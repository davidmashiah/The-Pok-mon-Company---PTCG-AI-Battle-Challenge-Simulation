"""Audit: which capabilities are silently disabled at runtime in the SHIPPED bundle?

Twice now a feature has been dead in production while present in the source:
 - the public agents' search_begin (wrong signature, TypeError swallowed)
 - our meta_decks import (file not bundled, ImportError swallowed)
Both were guarded by a bare `except` and neither announced itself. This asserts
each capability is genuinely live inside an unpacked tarball.
"""
import os, subprocess, sys, tarfile, tempfile

HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.dirname(HERE)
tar=sys.argv[1] if len(sys.argv)>1 else os.path.join(WORK,"out","v14_search_noloop2.tar.gz")

d=tempfile.mkdtemp(prefix="audit_")
with tarfile.open(tar,"r:gz") as tf: tf.extractall(d)
print(f"auditing {os.path.basename(tar)}\n  unpacked to {d}\n")

PROBE = r'''
import os, sys, json
sys.path.insert(0, os.getcwd())
res = {}
# 1. engine search API importable?
try:
    from cg.api import search_begin, search_step, search_end
    res["search_api_importable"] = True
except Exception as e:
    res["search_api_importable"] = "FAIL: %s" % e
# 2. our search module loads and reports search available?
try:
    import fsearch
    res["fsearch_HAVE_SEARCH"] = bool(fsearch.HAVE_SEARCH)
    res["meta_decklists"] = len(fsearch.meta_decks())
except Exception as e:
    res["fsearch"] = "FAIL: %s" % e
# 3. agent module loads the way the harness loads it (exec, no __file__)
try:
    env = {}
    exec(compile(open("main.py", encoding="utf-8-sig").read(), "main.py", "exec"), env)
    fns = [v for v in env.values() if callable(v)]
    res["last_callable"] = getattr(fns[-1], "__name__", None)
    res["deck_len"] = len(env.get("DECK") or env.get("my_deck") or [])
    res["has_lethal_search"] = "_lethal_line" in env
    res["has_fullturn_search"] = "_best_line" in env
    # agents name the guard differently; checking one name reported "null"
    # for an agent that HAD the guard -- a false negative from the very tool
    # meant to catch silently-missing features.
    res["ability_turn_limit"] = (env.get("ABILITY_REPEAT_LIMIT")
                                 or env.get("ABILITY_TURN_LIMIT"))
    res["ability_game_limit"] = env.get("ABILITY_GAME_LIMIT")
    res["guard_actually_wraps"] = "_INNER" in env or "ability_uses" in env
except Exception as e:
    res["agent_load"] = "FAIL: %s" % e
print(json.dumps(res, indent=1))
'''
p=subprocess.run([sys.executable,"-c",PROBE], capture_output=True, text=True, cwd=d, timeout=300)
print(p.stdout or p.stderr)
