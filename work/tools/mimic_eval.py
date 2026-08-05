"""Measure our policy against a top-rated agent's ACTUAL decisions.

Why this exists: every other local metric here is self-play, which measures how
well we beat ourselves. Replays store, for every step, the exact observation an
agent saw AND the action it chose. So we can replay a 1275-rated agent's games
through our policy and count how often we would have played differently.

This is a validation signal that does not require a submission and does not
involve our own policy on both sides.

Caveats, stated rather than buried:
  * Disagreement is not automatically our error -- we could be right and they
    wrong. It is evidence, weighted by the 325-point rating gap between us.
  * Only meaningful where they play the same archetype we do; otherwise their
    correct play is for a deck we are not holding.
  * MAIN decisions are the interesting ones. Forced/trivial selections (a
    single legal option) are excluded, since agreeing there is free.

Usage:
  python work/tools/mimic_eval.py --agent v2_lucario --team Majkel1337
"""
import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
REPLAYS = os.path.join(WORK, "out", "replays")


def load_agent(name):
    full = os.path.join(WORK, "agents", name)
    if full not in sys.path:
        sys.path.insert(0, full)
    cwd = os.getcwd()
    try:
        os.chdir(full)
        with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
            src = fh.read()
        env = {}
        exec(compile(src, "main.py", "exec"), env)
    finally:
        os.chdir(cwd)
    return [v for v in env.values() if callable(v)][-1], env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="v2_lucario")
    ap.add_argument("--team", default="Majkel1337")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import all_card_data, to_observation_class
    cards = {c.cardId: c for c in all_card_data()}
    fn, env = load_agent(args.agent)

    OPT = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD",
           5: "ENERGY_CARD", 6: "ENERGY", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE",
           10: "ABILITY", 11: "DISCARD", 12: "RETREAT", 13: "ATTACK",
           14: "END", 15: "SKILL", 16: "SPECIAL_CONDITION"}

    files = sorted(f for f in os.listdir(REPLAYS) if f.endswith("-replay.json"))
    total = agree = 0
    by_ctx = Counter()
    ctx_total = Counter()
    disagree_kind = Counter()
    them_pick = Counter()
    our_pick = Counter()
    errors = 0

    for fname in files:
        with open(os.path.join(REPLAYS, fname), encoding="utf-8") as f:
            d = json.load(f)
        names = d.get("info", {}).get("TeamNames") or []
        tgt = None
        for i, n in enumerate(names):
            if args.team.lower() in (n or "").lower():
                tgt = i
        if tgt is None:
            continue
        # give our agent the deck they actually played, so card-specific rules
        # are evaluated on the list they are written against
        for step in d["steps"]:
            a = (step[tgt].get("action") or [])
            if len(a) == 60:
                env["my_deck"] = list(a)
                env["DECK"] = list(a)
                break

        for step in d["steps"]:
            ag = step[tgt]
            obs = ag.get("observation") or {}
            act = ag.get("action")
            if not act or not isinstance(act, list) or len(act) == 60:
                continue
            sel = obs.get("select")
            if not sel:
                continue
            opts = sel.get("option") or []
            if len(opts) < 2:
                continue                      # forced move, agreeing is free
            try:
                ours = fn(obs)
            except Exception:
                errors += 1
                continue
            if not isinstance(ours, list) or not ours:
                errors += 1
                continue
            ctx = sel.get("context")
            ctx_total[ctx] += 1
            total += 1
            if list(ours)[:len(act)] == list(act):
                agree += 1
                by_ctx[ctx] += 1
            else:
                t_them = opts[act[0]].get("type") if act[0] < len(opts) else None
                t_ours = opts[ours[0]].get("type") if ours[0] < len(opts) else None
                disagree_kind[(OPT.get(t_them, t_them), OPT.get(t_ours, t_ours))] += 1
                them_pick[OPT.get(t_them, t_them)] += 1
                our_pick[OPT.get(t_ours, t_ours)] += 1

    if total == 0:
        raise SystemExit(f"no comparable decisions found for team {args.team!r}")
    print(f"agent={args.agent}  imitating={args.team}  replays={len(files)}")
    print(f"non-trivial decisions compared : {total}")
    print(f"agreement                      : {agree}/{total} = {agree/total:.3f}")
    if errors:
        print(f"agent errored on               : {errors}")
    print("\nagreement by select context (context: agreed/total):")
    for c, n in ctx_total.most_common(10):
        print(f"   ctx {c:>3}: {by_ctx[c]:>4}/{n:<4} = {by_ctx[c]/n:.3f}")
    print("\nmost common disagreements (they chose -> we chose):")
    for (them, ours), n in disagree_kind.most_common(10):
        print(f"   {n:>4}x  {them} -> {ours}")


if __name__ == "__main__":
    main()
