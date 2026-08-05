"""Smoke test: load the cg engine, dump the card pool, play a full random game.

Run:  .venv\Scripts\python.exe work\tools\smoke.py
"""
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import (  # noqa: E402
    Observation, all_attack, all_card_data, to_observation_class,
)
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

OUT = os.path.join(WORK, "out")
os.makedirs(OUT, exist_ok=True)


def read_deck(path):
    with open(path) as f:
        rows = [ln.strip() for ln in f if ln.strip()]
    return [int(r) for r in rows[:60]]


def main():
    t0 = time.time()

    # ---- 1. card pool -------------------------------------------------
    cards = all_card_data()
    attacks = all_attack()
    print(f"[ok] engine loaded in {time.time()-t0:.2f}s")
    print(f"[ok] all_card_data(): {len(cards)} cards")
    print(f"[ok] all_attack():    {len(attacks)} attacks")

    with open(os.path.join(OUT, "cards.json"), "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in cards], f, ensure_ascii=False, default=str, indent=1)
    with open(os.path.join(OUT, "attacks.json"), "w", encoding="utf-8") as f:
        json.dump([a.__dict__ for a in attacks], f, ensure_ascii=False, default=str, indent=1)
    print(f"[ok] wrote out/cards.json, out/attacks.json")

    # composition of the pool
    from collections import Counter
    print("  cardType histogram:", dict(Counter(int(c.cardType) for c in cards)))
    print("  ex/megaEx/aceSpec/tera:",
          sum(c.ex for c in cards), sum(c.megaEx for c in cards),
          sum(c.aceSpec for c in cards), sum(c.tera for c in cards))

    # ---- 2. play one full random game --------------------------------
    deck = read_deck(os.path.join(WORK, "lib", "sample_deck.csv"))
    print(f"\n[ok] sample deck: {len(deck)} cards, {len(set(deck))} distinct")

    rng = random.Random(0)
    t1 = time.time()
    obs_dict, start = battle_start(deck, list(deck))
    if obs_dict is None:
        print(f"[FAIL] battle_start returned None "
              f"(errorPlayer={start.errorPlayer}, errorType={start.errorType})")
        return 1

    steps = 0
    select_ctx = {}
    while True:
        obs: Observation = to_observation_class(obs_dict)
        if obs.current is not None and obs.current.result != -1:
            break
        if obs.select is None:
            print("[FAIL] select is None mid-game")
            return 1
        sd = obs.select
        select_ctx[int(sd.context)] = select_ctx.get(int(sd.context), 0) + 1
        n = len(sd.option)
        k = rng.randint(sd.minCount, min(sd.maxCount, n))
        pick = rng.sample(range(n), k) if k > 0 else []
        obs_dict = battle_select(pick)
        steps += 1
        if steps > 20000:
            print("[FAIL] runaway game")
            return 1

    obs = to_observation_class(obs_dict)
    dt = time.time() - t1
    print(f"[ok] random game finished: result={obs.current.result} "
          f"turn={obs.current.turn} steps={steps} in {dt:.2f}s "
          f"({steps/dt:.0f} selections/s)")
    print("  select contexts hit:", dict(sorted(select_ctx.items(), key=lambda kv: -kv[1])))
    battle_finish()

    # ---- 3. throughput: how many full games per second? --------------
    t2, games = time.time(), 0
    while time.time() - t2 < 5.0:
        o, s = battle_start(deck, list(deck))
        if o is None:
            break
        while True:
            ob = to_observation_class(o)
            if ob.current is not None and ob.current.result != -1:
                break
            sd = ob.select
            n = len(sd.option)
            k = rng.randint(sd.minCount, min(sd.maxCount, n))
            o = battle_select(rng.sample(range(n), k) if k > 0 else [])
        battle_finish()
        games += 1
    print(f"\n[ok] throughput: {games} full random games in 5s "
          f"({games/5.0:.1f} games/s)")
    print(f"\n[DONE] total {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
