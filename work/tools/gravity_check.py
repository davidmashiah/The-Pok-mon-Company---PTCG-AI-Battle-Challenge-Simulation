"""Does Gravity Mountain actually put Marnie's Grimmsnarl ex inside Mega Brave's range?

The whole v34 thesis is one arithmetic claim:

    Grimmsnarl ex 320 HP  --Gravity Mountain-->  290
    Mega Brave 270  --Premium Power Pro-->      300   >= 290   = one-shot KO

Nothing in the repo has ever checked that the ENGINE agrees. It could apply the
-30 to maxHp only, or refuse to lower current HP below the damage already dealt,
or apply it on entry only. Any of those breaks the combo.

This plays real games against a real leaderboard Grimmsnarl decklist and reports
what the engine actually did:
  * observed maxHp of opponent Stage 2 Pokemon, split by whether OUR Gravity
    Mountain was the active stadium at the time
  * every damage value we dealt to their Active (the 300 tier is the combo)
  * how often Gravity Mountain was in play at all -- with 1 copy in 60 cards
    the effect can be arithmetically perfect and still never happen

Usage:
  python work/tools/gravity_check.py --deck v34_stadium --games 30
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
ROOT = os.path.dirname(WORK)
OUT = os.path.join(WORK, "out")

GRAVITY_MOUNTAIN = 1252
GRIMMSNARL = 648
PPP = 1141


def _load(agent_dir, deck_override=None):
    full = os.path.join(WORK, "agents", agent_dir)
    if full not in sys.path:
        sys.path.insert(0, full)
    with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
        src = fh.read()
    env = {}
    exec(compile(src, "main.py", "exec"), env)
    deck = list(deck_override) if deck_override else list(env["my_deck"])
    if deck_override:
        env["my_deck"][:] = list(deck_override)
    return [v for v in env.values() if callable(v)][-1], deck


def _worker(job):
    agent_dir, my_deck, opp_deck, n, seed0 = job
    sys.path.insert(0, os.path.join(WORK, "lib"))
    os.chdir(tempfile.mkdtemp(prefix="grav_"))
    from cg.api import LogType, all_card_data, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    cards = {c.cardId: c for c in all_card_data()}
    fa, _ = _load(agent_dir, my_deck)
    fb, _ = _load(agent_dir, opp_deck)

    st = Counter()
    hp_with = Counter()      # observed maxHp of their Stage 2, GM in play
    hp_without = Counter()
    dmg = Counter()
    wins = 0

    for g in range(n):
        first = ((seed0 + g) % 2 == 0)
        d0, d1 = (my_deck, opp_deck) if first else (opp_deck, my_deck)
        p0, p1 = (fa, fb) if first else (fb, fa)
        me = 0 if first else 1
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        gm_turns = set()
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                cur = o.current
                if cur is not None and cur.result != -1:
                    if cur.result == me:
                        wins += 1
                    break
                if cur is not None:
                    stad = cur.stadium[0] if cur.stadium else None
                    gm = stad is not None and stad.id == GRAVITY_MOUNTAIN
                    if gm:
                        gm_turns.add(cur.turn)
                    opp = cur.players[1 - me]
                    for p in list(opp.active or []) + list(opp.bench or []):
                        if p is None:
                            continue
                        c = cards.get(p.id)
                        if c is not None and c.stage2:
                            (hp_with if gm else hp_without)[(p.id, p.maxHp)] += 1
                # damage WE dealt to THEM
                for lg in (o.logs or []):
                    if int(lg.type) == int(LogType.HP_CHANGE) and lg.value:
                        if lg.playerIndex is not None and lg.playerIndex != me:
                            dmg[abs(lg.value)] += 1
                who = cur.yourIndex if cur is not None else 0
                obs = battle_select(list((p0 if who == 0 else p1)(obs)))
        except Exception:
            st["game_error"] += 1
        finally:
            battle_finish()
        st["games"] += 1
        if gm_turns:
            st["games_with_GM_in_play"] += 1
        st["turns_with_GM_in_play"] += len(gm_turns)

    st["wins"] = wins
    return {"st": dict(st), "hp_with": {str(k): v for k, v in hp_with.items()},
            "hp_without": {str(k): v for k, v in hp_without.items()},
            "dmg": dict(dmg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="v32_ppp")
    ap.add_argument("--deck", default="v34_stadium",
                    help="agent dir whose deck.csv supplies OUR decklist")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    my_deck = [int(x) for x in
               open(os.path.join(WORK, "agents", a.deck, "deck.csv")).read().split()]
    md = json.load(open(os.path.join(OUT, "meta_decks.json"), encoding="utf-8"))
    grim = None
    for k, v in sorted(md["teams"].items(),
                       key=lambda kv: -(kv[1].get("score") or 0)):
        if GRIMMSNARL in (v.get("deck") or []):
            grim = v["deck"]
            print(f"opponent: rank {v.get('rank')} {v.get('name')} "
                  f"score {v.get('score')} (Grimmsnarl)")
            break
    assert grim, "no Grimmsnarl deck in meta_decks.json"
    print(f"our deck from {a.deck}/deck.csv: "
          f"{my_deck.count(GRAVITY_MOUNTAIN)}x Gravity Mountain, "
          f"{my_deck.count(PPP)}x Premium Power Pro")

    per = max(1, a.games // a.workers)
    jobs = [(a.agent, my_deck, grim, per, i * 977) for i in range(a.workers)]
    st, hw, hwo, dmg = Counter(), Counter(), Counter(), Counter()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(_worker, j) for j in jobs]):
            r = f.result()
            st.update(r["st"]); hw.update(r["hp_with"])
            hwo.update(r["hp_without"]); dmg.update({int(k): v for k, v in r["dmg"].items()})

    print(f"\n{st['games']} games in {time.time()-t0:.0f}s   "
          f"win rate {st['wins']}/{st['games']} = {st['wins']/max(1,st['games']):.3f}")
    print(f"  games where Gravity Mountain was ever in play : "
          f"{st['games_with_GM_in_play']}/{st['games']}")
    print(f"  distinct turns with it in play                : "
          f"{st['turns_with_GM_in_play']}")
    print("\n  opponent Stage 2 (cardId, maxHp) sightings")
    print("    WITH our Gravity Mountain out :",
          dict(hw) if hw else "(none)")
    print("    WITHOUT it                    :",
          dict(hwo) if hwo else "(none)")
    print("\n  damage values we dealt to them (top 12):")
    for v, n in dmg.most_common(12):
        tag = ""
        if v == 300:
            tag = "  <-- Mega Brave + Premium Power Pro (the combo)"
        elif v == 270:
            tag = "  <-- Mega Brave, unboosted"
        print(f"    {v:>4} dmg : {n:>5}{tag}")
    if st.get("game_error"):
        print(f"  (game errors: {st['game_error']})")


if __name__ == "__main__":
    main()
