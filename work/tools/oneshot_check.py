"""What fraction of Marnie's Grimmsnarl ex knockouts do we take in ONE attack?

This is the whole matchup. Grimmsnarl ex is ~53% of the field and gives up 2
prizes; our Mega Lucario ex gives up 3. Trading knockout for knockout we LOSE
the prize race, so the two-attack line is not a slower win, it is a loss. The
only way the matchup is favourable is one-shotting them:

    Mega Brave 270
      + Premium Power Pro x1 = 300  >= 290 (Grimmsnarl under Gravity Mountain)
      + Premium Power Pro x2 = 330  >= 320 (Grimmsnarl at full HP, no Stadium)

Raw counts of 300/330 damage events are confounded: a stronger agent ends games
sooner and therefore attacks fewer times. This normalises -- of the Grimmsnarl
knockouts we actually took, how many came from full HP.

Method: watch each opposing Pokemon by serial across our own observation
frames. A serial that leaves the board having never been SEEN damaged was taken
from full HP in a single attack. Our frames bracket our own attack, so a
two-attack kill is always visible as a damaged sighting in between.

Usage:
  python work/tools/oneshot_check.py --agents v32_ppp,v37_combo --games 40
"""
import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
OUT = os.path.join(WORK, "out")
GRIMMSNARL = 648


def _load(agent_dir, deck=None):
    full = os.path.join(WORK, "agents", agent_dir)
    if full not in sys.path:
        sys.path.insert(0, full)
    with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
        src = fh.read()
    env = {}
    exec(compile(src, "main.py", "exec"), env)
    if deck is not None:
        env["my_deck"][:] = list(deck)
    return [v for v in env.values() if callable(v)][-1], list(env["my_deck"])


def _worker(job):
    agent_dir, opp_deck, n, seed0 = job
    sys.path.insert(0, os.path.join(WORK, "lib"))
    os.chdir(tempfile.mkdtemp(prefix="oneshot_"))
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    fa, my_deck = _load(agent_dir)
    fb, _ = _load(agent_dir, opp_deck)
    st = Counter()

    for g in range(n):
        first = ((seed0 + g) % 2 == 0)
        d0, d1 = (my_deck, opp_deck) if first else (opp_deck, my_deck)
        p0, p1 = (fa, fb) if first else (fb, fa)
        me = 0 if first else 1
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        seen = {}          # serial -> [cardId, ever_damaged, last_hp]
        turns = 0
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                c = o.current
                if c is not None and c.result != -1:
                    st["wins" if c.result == me else "losses"] += 1
                    turns = c.turn
                    break
                if c is not None and c.yourIndex == me:
                    opp = c.players[1 - me]
                    alive = set()
                    for p in list(opp.active or []) + list(opp.bench or []):
                        if p is None:
                            continue
                        alive.add(p.serial)
                        rec = seen.setdefault(p.serial, [p.id, False, p.hp])
                        if p.hp < p.maxHp:
                            rec[1] = True
                        rec[2] = p.hp
                    # anything that was on the board and is now gone left it
                    for s, rec in list(seen.items()):
                        if s in alive or rec[0] is None:
                            continue
                        cid, dmgd, _hp = rec
                        if cid == GRIMMSNARL:
                            st["grimm_KOs"] += 1
                            st["grimm_ONE_SHOT" if not dmgd else "grimm_multi"] += 1
                        st["all_KOs"] += 1
                        rec[0] = None          # count each departure once
                who = c.yourIndex if c is not None else 0
                obs = battle_select(list((p0 if who == 0 else p1)(obs)))
        except Exception:
            st["err"] += 1
        finally:
            battle_finish()
        st["games"] += 1
        st["total_turns"] += turns
    return dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="v32_ppp,v37_combo")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    md = json.load(open(os.path.join(OUT, "meta_decks.json"), encoding="utf-8"))
    grim = next(v["deck"] for _, v in
                sorted(md["teams"].items(), key=lambda kv: -(kv[1].get("score") or 0))
                if GRIMMSNARL in (v.get("deck") or []))

    print(f"opponent: a real leaderboard Grimmsnarl list, {a.games} games each")
    print(f"{'agent':<16} {'games':>6} {'Grimm KOs':>10} {'one-shot':>9} "
          f"{'one-shot %':>11} {'turns/game':>11} {'win rate':>9}")
    print("-" * 80)
    for ag in a.agents.split(","):
        ag = ag.strip()
        per = max(1, a.games // a.workers)
        jobs = [(ag, grim, per, i * 4211) for i in range(a.workers)]
        tot = Counter()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for f in as_completed([ex.submit(_worker, j) for j in jobs]):
                tot.update(f.result())
        ko = tot["grimm_KOs"]
        one = tot["grimm_ONE_SHOT"]
        gm = max(1, tot["games"])
        print(f"{ag:<16} {tot['games']:>6} {ko:>10} {one:>9} "
              f"{100*one/max(1,ko):>10.1f}% {tot['total_turns']/gm:>11.1f} "
              f"{tot['wins']/gm:>9.3f}")
    print("\nnote: win rate here is our policy piloting THEIR deck on the other")
    print("side, so it is optimistic and NOT ladder-predictive (handoff S1).")
    print("The one-shot percentage is the mechanical claim being tested.")


if __name__ == "__main__":
    main()
