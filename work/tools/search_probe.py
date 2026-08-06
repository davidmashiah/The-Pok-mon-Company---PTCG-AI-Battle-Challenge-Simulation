"""Does the engine's native determinized search actually work from OUR harness?

Worth probing rather than assuming. v57_pimc_full calls
`search_begin(obs, your_deck=yd)`, which does not match the signature in
work/lib/cg/api.py at all (that one takes seven required arguments). Either v57
silently fell back to its heuristic for every decision it ever made, or there
are two different api.py versions in play. A search layer built on a guess here
would be the fifth silently-broken component in this repo.

Answers, in order:
  1. does an in-game observation carry `search_begin_input`?
  2. what does search_begin accept, positionally and by keyword?
  3. what comes back -- SearchState, or an ApiResult with .state/.error?
  4. can we step it, and does the observation advance?
  5. how fast is one determinized playout, in decisions/second?
"""
import inspect
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg import api  # noqa: E402
from cg.api import (SelectContext, search_begin, search_step,  # noqa: E402
                    search_end, search_release, to_observation_class)
from cg.game import battle_finish, battle_start, battle_select  # noqa: E402

AGENTS = os.path.join(WORK, "agents")


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
        d = fn({"current": None, "select": None})
    finally:
        os.chdir(cwd)
    if not (isinstance(d, (list, tuple)) and len(d) == 60):
        d = [int(x) for x in open(os.path.join(full, "deck.csv"),
                                  encoding="utf-8").read().split() if x.strip()]
    return fn, [int(x) for x in d]


print("search_begin signature:")
print("   ", inspect.signature(search_begin))

fa, da = load("w8_grimm_tuned")
fb, db = load("w5_grimmsnarl")
for f in (fa, fb):
    f({"current": None, "select": None})

obs, _ = battle_start(list(da), list(db))
raw = obs
o = to_observation_class(obs)

# advance to a real MAIN decision with several options
steps = 0
while steps < 400:
    o = to_observation_class(raw)
    if o.current is not None and o.current.result != -1:
        break
    if (o.select is not None and o.select.context == SelectContext.MAIN
            and len(o.select.option) >= 3 and o.current.yourIndex == 0
            and o.current.turn >= 2):
        break
    who = o.current.yourIndex
    raw = battle_select(list((fa if who == 0 else fb)(raw)))
    steps += 1

print(f"\nreached turn {o.current.turn}, yourIndex {o.current.yourIndex}, "
      f"{len(o.select.option)} options, context {o.select.context}")
print("observation has search_begin_input:",
      getattr(o, "search_begin_input", None) is not None)

me = o.current.players[o.current.yourIndex]
opp = o.current.players[1 - o.current.yourIndex]
print(f"my deckCount={me.deckCount} prize={len(me.prize)} | "
      f"opp deckCount={opp.deckCount} prize={len(opp.prize)} "
      f"hand={opp.handCount}")

# A determinization: guess the hidden cards. Sizes must match exactly or
# search_begin raises -- that is the contract, and it is checked per field.
your_deck = random.sample(da, me.deckCount) if me.deckCount <= len(da) else list(da)
your_prize = [da[0]] * len(me.prize)
opp_deck = random.sample(db, opp.deckCount) if opp.deckCount <= len(db) else list(db)
opp_prize = [db[0]] * len(opp.prize)
opp_hand = [db[0]] * opp.handCount

print("\n-- search_begin, full positional contract --")
try:
    st = search_begin(o, your_deck, your_prize, opp_deck, opp_prize,
                      opp_hand, [], True)
    print("   OK ->", type(st).__name__,
          "searchId =", getattr(st, "searchId", None))
    print("   has .state attr:", hasattr(st, "state"),
          "| has .error attr:", hasattr(st, "error"))
    sid = st.searchId
    so = st.observation
    print("   returned observation options:",
          len(so.select.option) if so.select else None)

    print("\n-- search_step: take option 0 repeatedly, one turn --")
    t0 = time.time()
    n = 0
    cur, cid = so, sid
    while n < 25:
        if cur.current is not None and cur.current.result != -1:
            print("   game ended in search at step", n)
            break
        if cur.select is None or not cur.select.option:
            break
        k = max(1, cur.select.minCount)
        nxt = search_step(cid, list(range(min(k, len(cur.select.option)))))
        cur, cid = nxt.observation, nxt.searchId
        n += 1
    dt = time.time() - t0
    print(f"   stepped {n} decisions in {dt*1000:.1f} ms "
          f"({n/max(dt,1e-9):.0f} decisions/s)")
    search_release(sid)
    search_end()
    print("   released cleanly")
except Exception as exc:
    print("   FAILED:", type(exc).__name__, exc)

print("\n-- the v57 call shape, for comparison --")
try:
    st2 = search_begin(o, your_deck=your_deck)
    print("   OK (so v57's shape is valid) ->", type(st2).__name__)
    search_end()
except Exception as exc:
    print("   FAILED:", type(exc).__name__, exc)

battle_finish()
