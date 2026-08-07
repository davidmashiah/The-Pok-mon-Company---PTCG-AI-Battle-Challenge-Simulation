"""Put the search layer on the BETTER base: the hand-written v26 policy.

The harness fix exposed something the old measurements hid completely. Against
the same opponent, w5_grimmsnarl:

                          h3 (shared modules)   h4 (clean)
  w8_grimm_tuned                0.5297            0.4247
  _sub_handwritten_v26          0.5269            0.5523

Under h3 the two looked identical, so every search variant this session was
built on w8 -- which under clean measurement is the WEAKER policy by 0.13 in the
mirror, the matchup carrying ~47% of the panel weight. The hand-written policy
reaches 0.5523 with no search at all, nearly what w34 gets (0.5676) with it.

This builds the same validated search on top of that policy instead. The pieces
are reused unchanged from build_search_layer.py -- the 27-deck opponent model,
the knockout-aware rollout, paired determinizations, margin-plus-majority
override -- because those are the parts that measured positive.

One structural difference: this base has no learned model, so there is no
per-option ranking to draw candidates from. Candidates are the policy's own pick
plus the remaining options in index order, and the model-score gate is inert.

  python work/tools/build_hw_search.py --name w50_hwsearch
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
sys.path.insert(0, HERE)

import build_search_layer as bsl  # noqa: E402

BASE = "_sub_handwritten_v26"

MAIN = r'''import policy_features as pf
from policies.handwritten_v26 import main as _sub
import search_validator
DECK = list(pf.DECK)


def _legal(a, s, n):
    try:
        lo = int(s.get('minCount', 0) or 0)
        hi = int(s.get('maxCount', 0) or 0)
        return (lo <= len(a) <= max(hi, lo) and len(a) == len(set(a))
                and all(isinstance(i, int) and 0 <= i < n for i in a))
    except Exception:
        return False


def agent(obs):
    if not obs or obs.get('select') is None:
        try:
            _sub.agent({})
        except Exception:
            pass
        return list(DECK)
    s = obs.get('select') or {}
    n = len(s.get('option') or [])
    search_validator.reset_decision()
    try:
        base = list(_sub.agent(obs))
    except Exception:
        lo = int(s.get('minCount', 0) or 0)
        hi = min(n, int(s.get('maxCount', 0) or 0))
        base = list(range(max(lo, hi)))
    try:
        ov = search_validator.validate(obs, base, DECK)
    except Exception:
        ov = None
    if ov is not None and _legal(ov, s, n):
        return ov
    return base


def w50_hwsearch_entry(obs):
    return agent(obs)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="w50_hwsearch")
    ap.add_argument("--budget", type=float, default=0.9)
    ap.add_argument("--det", type=int, default=3)
    ap.add_argument("--cands", type=int, default=3)
    ap.add_argument("--margin", type=float, default=1000.0)
    a = ap.parse_args()

    src = os.path.join(AGENTS, BASE)
    dst = os.path.join(AGENTS, a.name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))

    # Reuse the validator verbatim. nonmain=1 because without a learned model
    # there is no ranking, and the validator's "no ranking" path is the same one
    # that serves non-MAIN prompts.
    body = (bsl.VALIDATOR
            .replace("__BUDGET__", repr(a.budget))
            .replace("__DET__", repr(a.det))
            .replace("__CANDS__", repr(a.cands))
            .replace("__MARGIN__", repr(a.margin))
            .replace("__GATE__", repr(0.0))
            .replace("__NONMAIN__", repr(1))
            .replace("__MIRRORMAIN__", repr(0))
            .replace("__POLICYROLL__", repr(0))
            .replace("__OPPPOLICY__", repr(0))
            .replace("__THREATW__", repr(0.0)))
    for tok in ("__BUDGET__", "__DET__", "__CANDS__", "__MARGIN__", "__GATE__",
                "__NONMAIN__", "__MIRRORMAIN__", "__POLICYROLL__",
                "__OPPPOLICY__", "__THREATW__"):
        if tok in body:
            raise SystemExit(f"placeholder {tok} not substituted")
    compile(body, "search_validator.py", "exec")
    with open(os.path.join(dst, "search_validator.py"), "w",
              encoding="utf-8") as f:
        f.write(body)

    # the real top-50 decklists, same source as the w8-based builds
    lib = []
    try:
        import json
        store = json.load(open(os.path.join(WORK, "out", "top_decks.json"),
                               encoding="utf-8"))
        seen = set()
        for team in store.values():
            for deck in (team.get("decks") or []):
                if deck and len(deck) == 60:
                    key = tuple(sorted(deck))
                    if key not in seen:
                        seen.add(key)
                        lib.append(list(key))
    except Exception as exc:
        print(f"  WARNING: no opponent library ({type(exc).__name__})")
    with open(os.path.join(dst, "opp_library.py"), "w", encoding="utf-8") as f:
        f.write('"""Real top-50 decklists. Generated -- do not hand-edit."""\n')
        f.write(f"DECKS = {lib!r}\n")

    compile(MAIN, "main.py", "exec")
    with open(os.path.join(dst, "main.py"), "w", encoding="utf-8") as f:
        f.write(MAIN)

    print(f"built work/agents/{a.name} from {BASE}")
    print(f"  budget={a.budget}s det={a.det} cands={a.cands} "
          f"margin={a.margin} (baked in)")
    print(f"  + search_validator.py, opp_library.py ({len(lib)} decklists)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
