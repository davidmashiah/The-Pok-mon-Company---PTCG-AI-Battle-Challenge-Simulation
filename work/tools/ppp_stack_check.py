"""Do two Premium Power Pro stack to +60?

The card reads "During this turn, attacks used by your {F} Pokemon do 30 more
damage to your opponent's Active Pokemon". Nothing on it says "you can play only
one", so two copies should be +60 -- but that is a rules inference, and the
engine is the only authority that matters. It decides real damage:

    Mega Brave 270 + 30 = 300   one-shots Grimmsnarl ex only under Gravity
                                Mountain (290)
    Mega Brave 270 + 60 = 330   one-shots it at FULL 320 HP, no Stadium needed

The agent's planner counts the buff as a flat +30 no matter how many copies it
holds, so if stacking is real it is underestimating our damage by 30 in exactly
the matchup that is ~53% of the field.

Method: play real games, count PLAY(Premium Power Pro) events per turn, and
attribute the next damage dealt to the opponent's Active to that count. Reports
the observed damage distribution split by how many copies were live.

Usage: python work/tools/ppp_stack_check.py --games 30
"""
import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
OUT = os.path.join(WORK, "out")

PPP = 1141
GRIMMSNARL = 648


def _load(agent_dir, deck):
    full = os.path.join(WORK, "agents", agent_dir)
    if full not in sys.path:
        sys.path.insert(0, full)
    with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
        src = fh.read()
    env = {}
    exec(compile(src, "main.py", "exec"), env)
    env["my_deck"][:] = list(deck)
    return [v for v in env.values() if callable(v)][-1]


def _worker(job):
    agent_dir, my_deck, opp_deck, n, seed0 = job
    sys.path.insert(0, os.path.join(WORK, "lib"))
    os.chdir(tempfile.mkdtemp(prefix="pppstack_"))
    from cg.api import LogType, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    fa = _load(agent_dir, my_deck)
    fb = _load(agent_dir, opp_deck)
    # damage dealt to their ACTIVE, keyed by how many PPP we had played that turn
    by_count = defaultdict(Counter)
    st = Counter()

    for g in range(n):
        first = ((seed0 + g) % 2 == 0)
        d0, d1 = (my_deck, opp_deck) if first else (opp_deck, my_deck)
        p0, p1 = (fa, fb) if first else (fb, fa)
        me = 0 if first else 1
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        ppp_this_turn = 0
        cur_turn = -1
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                c = o.current
                if c is not None and c.result != -1:
                    break
                if c is not None and c.turn != cur_turn:
                    cur_turn = c.turn
                    ppp_this_turn = 0
                for lg in (o.logs or []):
                    t = int(lg.type)
                    if t in (int(LogType.TURN_START), int(LogType.TURN_END)):
                        ppp_this_turn = 0
                    elif (t == int(LogType.PLAY) and lg.cardId == PPP
                          and lg.playerIndex == me):
                        ppp_this_turn += 1
                        st["ppp_plays"] += 1
                        st["turns_with_%d_ppp" % ppp_this_turn] += 1
                    elif (t == int(LogType.HP_CHANGE) and lg.value
                          and lg.playerIndex is not None and lg.playerIndex != me):
                        by_count[ppp_this_turn][abs(lg.value)] += 1
                who = c.yourIndex if c is not None else 0
                obs = battle_select(list((p0 if who == 0 else p1)(obs)))
        except Exception:
            st["err"] += 1
        finally:
            battle_finish()
        st["games"] += 1
    return {"st": dict(st),
            "by": {str(k): dict(v) for k, v in by_count.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="v37_combo")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    my_deck = [int(x) for x in
               open(os.path.join(WORK, "agents", a.agent, "deck.csv")).read().split()]
    md = json.load(open(os.path.join(OUT, "meta_decks.json"), encoding="utf-8"))
    grim = next(v["deck"] for _, v in
                sorted(md["teams"].items(), key=lambda kv: -(kv[1].get("score") or 0))
                if GRIMMSNARL in (v.get("deck") or []))

    per = max(1, a.games // a.workers)
    jobs = [(a.agent, my_deck, grim, per, i * 313) for i in range(a.workers)]
    st = Counter()
    by = defaultdict(Counter)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(_worker, j) for j in jobs]):
            r = f.result()
            st.update(r["st"])
            for k, v in r["by"].items():
                by[int(k)].update({int(kk): vv for kk, vv in v.items()})

    print(f"\n{st['games']} games vs Grimmsnarl in {time.time()-t0:.0f}s")
    print(f"  total Premium Power Pro plays by us: {st['ppp_plays']}")
    for k in sorted(st):
        if k.startswith("turns_with_"):
            print(f"    turns reaching {k.split('_')[2]} copies: {st[k]}")
    print("\n  damage dealt to their ACTIVE, by copies of PPP live that turn")
    for cnt in sorted(by):
        big = {d: n for d, n in by[cnt].items() if d >= 250}
        print(f"    {cnt} copies: >=250 damage events -> "
              f"{dict(sorted(big.items())) or '(none)'}")
    print("\n  reading: Mega Brave is 270 base. 300 = one copy, 330 = TWO copies")
    print("  stacking. 330 >= 320 one-shots Grimmsnarl ex with no Stadium.")


if __name__ == "__main__":
    main()
