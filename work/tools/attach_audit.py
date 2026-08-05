"""Do we waste our once-per-turn energy attachment? Measured in real games.

Conditional on an ATTACH option being offered, we take it 15.0% of the time
against a 1265-rated pilot's 41.9%. But that alone proves nothing: declining an
attachment at one decision is correct if we attach later in the same turn, and
the replay cannot follow our agent forward. The same confound made an earlier
"816 missed KOs" number evaporate to zero.

So this plays actual games and asks the unambiguous question: when our turn
ends, had we attached an energy, and did we have one in hand to attach?

Energy attachment is once per turn and Mega Brave needs two of them, so a wasted
attachment is a directly lost tempo -- and unlike a win-rate proxy, this is a
fact about our own play that no opponent model can distort.

Usage: python work/tools/attach_audit.py <agent> [games]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
LIB = os.path.join(WORK, "lib")
sys.path.insert(0, LIB)

from cg.api import OptionType, SelectContext, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

AGENT = sys.argv[1] if len(sys.argv) > 1 else "v32_ppp"
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
BASIC_F = 6


def load_agent(name):
    ag = os.path.join(WORK, "agents", name)
    for p in (ag, LIB):
        if p not in sys.path:
            sys.path.insert(0, p)
    root = os.path.dirname(WORK)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        with open(os.path.join(ag, "main.py"), encoding="utf-8") as fh:
            src = fh.read()
        env = {}
        exec(compile(src, "main.py", "exec"), env)
    finally:
        os.chdir(cwd)
    fns = [v for k, v in env.items() if callable(v)]
    deck = env.get("DECK") or env.get("my_deck")
    return fns[-1], list(deck)


def main():
    fn, deck = load_agent(AGENT)
    turns = wasted = had_energy = 0
    seen_turn = {}
    for g in range(GAMES):
        obs, _ = battle_start(list(deck), list(deck))
        prev = None
        for _ in range(4000):
            o = to_observation_class(obs)
            # the game is over when result != -1; testing `select is None`
            # instead kept calling into a finished battle and raised IndexError
            if o.current is not None and o.current.result != -1:
                break
            if o.select is None:
                break
            cur = obs.get("current") or {}
            me = cur.get("yourIndex", 0)
            t = cur.get("turn")
            pls = cur.get("players") or []
            if len(pls) >= 2 and me == 0:          # audit ONE side only
                mine = pls[me]
                hand = mine.get("hand") or []
                n_e = sum(1 for c in hand
                          if isinstance(c, dict) and c.get("id") == BASIC_F)
                # An attachment can only be "wasted" if the engine actually
                # OFFERED one this turn. Counting turns that merely had energy
                # in hand conflates real waste with setup turns and turns with
                # no legal target -- the same artifact that turned an earlier
                # "816 missed KOs" into zero.
                offered = any(o.get("type") == int(OptionType.ATTACH)
                              for o in ((obs.get("select") or {}).get("option") or []))
                key = (g, t)
                prev_off = seen_turn.get(key, (False, 0, False))[2]
                seen_turn[key] = (bool(cur.get("energyAttached")), n_e,
                                  prev_off or offered)
            sel = fn(obs)
            obs = battle_select(sel)
        battle_finish()
        # tally this game's turns
        for (gg, t), (attached, n_e, offered) in list(seen_turn.items()):
            if gg != g:
                continue
            turns += 1
            if offered:
                had_energy += 1
                if not attached:
                    wasted += 1
        seen_turn = {k: v for k, v in seen_turn.items() if k[0] != g}

    print(f"agent: {AGENT}   games: {GAMES}")
    print(f"our turns observed                              : {turns}")
    print(f"  turns where the engine OFFERED an attachment  : {had_energy}")
    print(f"  of those, turn ENDED with NO energy attached  : {wasted}")
    if had_energy:
        print(f"  -> wasted the once-per-turn attachment on "
              f"{100*wasted/had_energy:.1f}% of turns where it was possible")


if __name__ == "__main__":
    main()
