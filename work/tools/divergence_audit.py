"""Where do we disagree with a 1100+ pilot running OUR EXACT DECK?

The whole remaining gap is decisions. 13 of the 16 top-50 Grimmsnarl teams run a
byte-identical 60 to ours, they score 1044-1197, we score ~925, and they do not
publish code. But their games are public, and a replay records the observation
handed to the agent AND the option it chose -- so we can put our own policy in
front of their positions and count where it does something different.

This is not cloning. Cloning their moves scored 389.3 here. This is a
DIAGNOSTIC: it produces a ranked list of the specific decisions where a
top pilot and ours part company, so the disagreements can be inspected one at a
time and the real ones fixed by hand.

Only games where the opponent played our exact decklist are used, so a
disagreement is never explained away by a different 60.

  python work/tools/divergence_audit.py --agent _sub_handwritten_v26
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import all_card_data  # noqa: E402

CARDS = {c.cardId: c for c in all_card_data()}
OPT = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD",
       5: "ENERGY_CARD", 6: "ENERGY", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE",
       10: "ABILITY", 11: "DISCARD", 12: "RETREAT", 13: "ATTACK", 14: "END",
       15: "SKILL", 16: "SPECIAL_CONDITION"}


def load(name):
    full = os.path.join(AGENTS, name)
    if full not in sys.path:
        sys.path.insert(0, full)
    cwd = os.getcwd()
    try:
        os.chdir(full)
        env = {}
        exec(compile(open(os.path.join(full, "main.py"),
                          encoding="utf-8-sig").read(), "main.py", "exec"), env)
        fn = [v for v in env.values() if callable(v)][-1]
        try:
            fn({"current": None, "select": None, "logs": []})
        except Exception:
            pass
    finally:
        os.chdir(cwd)
    deck = [int(x) for x in open(os.path.join(full, "deck.csv"),
                                 encoding="utf-8").read().split() if x.strip()]
    return fn, sorted(deck)


def opt_label(obs, o):
    """A readable name for the option: type, plus the card it acts on."""
    t = int(o.get("type", -1) if o.get("type") is not None else -1)
    name = OPT.get(t, str(t))
    cid = None
    try:
        cur = obs.get("current") or {}
        me = (cur.get("players") or [])[int(cur.get("yourIndex", 0) or 0)]
        area = int(o.get("area", 0) or 0)
        idx = int(o.get("index", -1))
        zone = {2: me.get("hand"), 4: me.get("active"),
                5: me.get("bench"), 3: me.get("discard")}.get(area)
        if zone and 0 <= idx < len(zone) and zone[idx]:
            cid = int((zone[idx] or {}).get("id", 0) or 0)
    except Exception:
        pass
    if cid:
        name += f":{getattr(CARDS.get(cid), 'name', cid)}"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="_sub_handwritten_v26")
    ap.add_argument("--replays", default=os.path.join(WORK, "out",
                                                      "top_replays"))
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()

    fn, our_deck = load(a.agent)
    print(f"{a.agent}: comparing against pilots running our exact 60\n")

    files = sorted(glob.glob(os.path.join(a.replays, "*.json")))[:a.limit]
    games = decisions = agree = 0
    by_type = collections.Counter()
    dis_type = collections.Counter()
    examples = collections.defaultdict(list)

    for path in files:
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        steps = d.get("steps") or []
        if not steps:
            continue
        # which side, if either, played OUR exact list?
        sides = []
        for pi in range(2):
            for st in steps:
                if pi < len(st):
                    act = st[pi].get("action")
                    if isinstance(act, list) and len(act) == 60:
                        if sorted(int(x) for x in act) == our_deck:
                            sides.append(pi)
                        break
        if not sides:
            continue
        games += 1
        for pi in sides:
            for st in steps:
                if pi >= len(st):
                    continue
                ag = st[pi]
                obs = ag.get("observation") or {}
                act = ag.get("action")
                sel = obs.get("select")
                if not sel or not act or not isinstance(act, list):
                    continue
                if len(act) == 60:
                    continue                 # the setup frame
                opts = sel.get("option") or []
                if len(opts) < 2:
                    continue                 # no real choice
                try:
                    ours = fn(obs)
                except Exception:
                    continue
                if not isinstance(ours, list) or not ours:
                    continue
                decisions += 1
                theirs = act[0]
                mine = ours[0]
                lab = opt_label(obs, opts[theirs]) if 0 <= theirs < len(opts) else "?"
                by_type[lab] += 1
                if mine == theirs:
                    agree += 1
                else:
                    dis_type[lab] += 1
                    if len(examples[lab]) < 2 and 0 <= mine < len(opts):
                        examples[lab].append(
                            (lab, opt_label(obs, opts[mine])))

    if not decisions:
        print("no comparable decisions found")
        return 0
    print(f"{games} games where a top pilot ran our exact deck")
    print(f"{decisions} decisions compared, agreement "
          f"{agree/decisions:.3f} ({agree}/{decisions})\n")

    print("where we disagree most (by what THEY chose):")
    print(f"{'their choice':44s} {'n':>5} {'disagree':>9} {'rate':>7}")
    print("-" * 70)
    rows = [(lab, by_type[lab], dis_type[lab], dis_type[lab] / by_type[lab])
            for lab in by_type if by_type[lab] >= 8]
    rows.sort(key=lambda r: -r[2])
    for lab, n, dis, rate in rows[:22]:
        print(f"{lab[:44]:44s} {n:5d} {dis:9d} {rate:7.3f}")
        for th, mn in examples.get(lab, [])[:1]:
            print(f"    they chose {th[:32]:32s} -> we chose {mn[:32]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
