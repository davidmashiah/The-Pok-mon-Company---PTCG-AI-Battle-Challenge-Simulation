"""Watch the games we LOSE, and say concretely how we lost them.

Every idea tested today came from a hypothesis I invented and then measured.
This inverts that: play the matchup, keep the losses, and report what actually
went wrong in them -- so the next change is aimed at a failure that exists
rather than one I imagined.

Per loss it records the things that distinguish the ways this deck dies:

  prizes taken by each side      a 6-0 is a different disease from a 5-6
  turn the game ended            fast losses are setup failures, slow ones are
                                 attrition or deck-out
  turn our Grimmsnarl ex landed  the whole deck is a Stage 2; if it never
                                 arrives nothing else matters
  attacks we made                zero attacks is a different bug from losing a
                                 damage race
  Rare Candy / Poffin used       did the assembly engine even fire
  cards left in deck             deck-out shows up here and nowhere else
  our board at the end           what we were holding when we died

Prints the same stats for WINS beside them, because a number is only diagnostic
if it differs between the two.

  python work/tools/loss_lab.py --agent w34_koroll --opponent w5_grimmsnarl \
      --games 60
"""
import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import all_card_data, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

CARDS = {c.cardId: c for c in all_card_data()}
GRIMM, MORGREM, IMPIDIMP = 648, 647, 646
RARE_CANDY, POFFIN = 1079, 1086


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
            d = fn({"current": None, "select": None, "logs": []})
        except Exception:
            d = None
    finally:
        os.chdir(cwd)
        for nm, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None) or ""
            if f.startswith(full + os.sep):
                del sys.modules[nm]
        while full in sys.path:
            sys.path.remove(full)
    if not (isinstance(d, (list, tuple)) and len(d) == 60):
        d = [int(x) for x in open(os.path.join(full, "deck.csv"),
                                  encoding="utf-8").read().split() if x.strip()]
    return fn, [int(x) for x in d]


def play(fa, da, fb, db, seed):
    """One game. Returns a dict describing how it went for agent A."""
    a_first = (seed % 2 == 0)
    p0, p1 = (fa, fb) if a_first else (fb, fa)
    d0, d1 = (da, db) if a_first else (db, da)
    me = 0 if a_first else 1
    for f in (fa, fb):
        try:
            f({"current": None, "select": None, "logs": []})
        except Exception:
            pass
    obs, _ = battle_start(list(d0), list(d1))
    if obs is None:
        return None
    r = {"won": False, "turns": 0, "grimm_turn": None, "attacks": 0,
         "prizes_me": 0, "prizes_opp": 0, "deck_left": 0, "played": Counter(),
         "end_active": None, "went_first": a_first}
    seen = set()
    last = None
    try:
        for _ in range(4000):
            o = to_observation_class(obs)
            st = o.current
            if st is None:
                break
            last = st
            if st.result != -1:
                r["won"] = (st.result == me)
                break
            r["turns"] = max(r["turns"], int(st.turn or 0))
            mine = st.players[me]
            # our Grimmsnarl ex in play?
            if r["grimm_turn"] is None:
                for p in ([mine.active[0]] if (mine.active and mine.active[0])
                          else []) + list(mine.bench or []):
                    if p is not None and p.id == GRIMM:
                        r["grimm_turn"] = int(st.turn or 0)
                        break
            for lg in (o.logs or []):
                key = (int(st.turn or 0), getattr(lg, "serial", None),
                       int(getattr(lg, "type", -1) or -1),
                       getattr(lg, "cardId", None))
                if key in seen:
                    continue
                seen.add(key)
                t = int(getattr(lg, "type", -1) or -1)
                pl = getattr(lg, "playerIndex", None)
                if pl != me:
                    continue
                if t == 15:
                    r["attacks"] += 1
                elif t == 10:
                    cid = getattr(lg, "cardId", None)
                    if cid:
                        r["played"][int(cid)] += 1
            who = st.yourIndex
            obs = battle_select(list((p0 if who == 0 else p1)(obs)))
    except Exception:
        pass
    finally:
        battle_finish()
    if last is not None:
        try:
            mine = last.players[me]
            opp = last.players[1 - me]
            r["prizes_me"] = 6 - len(mine.prize or [])
            r["prizes_opp"] = 6 - len(opp.prize or [])
            r["deck_left"] = int(mine.deckCount or 0)
            a = mine.active[0] if (mine.active and mine.active[0]) else None
            r["end_active"] = getattr(CARDS.get(getattr(a, "id", None)),
                                      "name", None)
        except Exception:
            pass
    return r


def summarise(rows, label):
    if not rows:
        print(f"  ({label}: none)")
        return
    n = len(rows)

    def avg(k):
        return sum(x[k] or 0 for x in rows) / n
    got = [x for x in rows if x["grimm_turn"] is not None]
    print(f"  {label:8s} n={n:3d}  prizes {avg('prizes_me'):.2f}-"
          f"{avg('prizes_opp'):.2f}  turns {avg('turns'):.1f}  "
          f"attacks {avg('attacks'):.2f}  deck_left {avg('deck_left'):.1f}")
    print(f"           Grimmsnarl ex in play: {len(got)}/{n} "
          f"({100*len(got)/n:.0f}%)"
          + (f", median turn {sorted(x['grimm_turn'] for x in got)[len(got)//2]}"
             if got else ""))
    print(f"           went first: {sum(1 for x in rows if x['went_first'])}/{n}")
    ends = Counter(x["end_active"] for x in rows if x["end_active"])
    if ends:
        print("           ended holding: " + ", ".join(
            f"{k} x{v}" for k, v in ends.most_common(4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="w34_koroll")
    ap.add_argument("--opponent", default="w5_grimmsnarl")
    ap.add_argument("-n", "--games", type=int, default=60)
    a = ap.parse_args()

    fa, da = load(a.agent)
    fb, db = load(a.opponent)
    rows = []
    for g in range(a.games):
        r = play(fa, da, fb, db, g)
        if r:
            rows.append(r)
    wins = [x for x in rows if x["won"]]
    losses = [x for x in rows if not x["won"]]
    print(f"\n{a.agent} vs {a.opponent}: {len(wins)}/{len(rows)} "
          f"= {len(wins)/max(1,len(rows)):.3f}\n")
    summarise(wins, "WINS")
    print()
    summarise(losses, "LOSSES")

    print("\nwhat we PLAY per game, wins vs losses (top differences):")
    def rate(rs, cid):
        return sum(x["played"][cid] for x in rs) / max(1, len(rs))
    cids = set()
    for x in rows:
        cids |= set(x["played"])
    diffs = []
    for cid in cids:
        w, l = rate(wins, cid), rate(losses, cid)
        if max(w, l) >= 0.3:
            diffs.append((abs(w - l), cid, w, l))
    diffs.sort(reverse=True)
    for _, cid, w, l in diffs[:12]:
        nm = getattr(CARDS.get(cid), "name", cid)
        print(f"   {str(nm)[:30]:30s} wins {w:5.2f}/game   losses {l:5.2f}/game"
              f"   delta {w-l:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
