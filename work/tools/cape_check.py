"""Does Hero's Cape actually give +100 HP in THIS engine, and does it move a KO?

Card text is not behaviour. This repo has already paid to learn that both ways:
Premium Power Pro really does stack to +60, Gravity Mountain really is -30, and
Enriching Energy is not legal in this format at all despite reading fine. So
before a deck is rebuilt around a Tool, watch the engine apply it.

Method with no inference: w20_luc1084 (mined from a public notebook) already
runs 1 Hero's Cape, so we just play it and watch. Any Pokemon whose maxHp
exceeds its CardData hp is the Cape working; the delta is the size of the buff.

Why it matters here: Marnie's Grimmsnarl ex is 320 HP and the mirror's Shadow
Bullet does 180, so two hits (360) kill it. At 420 it takes three. In a mirror
where both sides race for the same knockouts, that is a whole extra turn of
attacking per attacker.

  python work/tools/cape_check.py --games 12
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

CAPE = 1159
CARDS = {c.cardId: c for c in all_card_data()}


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
        # Some bundles call to_observation_class on the setup frame, which
        # needs `logs` present. Send the full shape, and fall back to deck.csv
        # if the agent still refuses it.
        try:
            d = fn({"current": None, "select": None, "logs": []})
        except Exception:
            d = None
    finally:
        os.chdir(cwd)
        # Evict everything this bundle imported. Two agents in this repo both
        # ship a module called `policy_features`, and the SECOND agent's
        # `import policy_features` is a silent no-op once the first one is in
        # sys.modules -- so it binds the other agent's 60-card deck. Loading
        # w40_cape then w5_grimmsnarl raised "fixed 60-card deck changed",
        # which was the lucky version; the quiet version measures one agent
        # playing another agent's decklist.
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None) or ""
            if f.startswith(full + os.sep) or f.startswith(full + "/"):
                del sys.modules[name]
        if full in sys.path:
            sys.path.remove(full)
    if not (isinstance(d, (list, tuple)) and len(d) == 60):
        d = [int(x) for x in open(os.path.join(full, "deck.csv"),
                                  encoding="utf-8").read().split() if x.strip()]
    return fn, [int(x) for x in d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="w20_luc1084")
    ap.add_argument("--opponent", default="w5_grimmsnarl")
    ap.add_argument("-n", "--games", type=int, default=12)
    a = ap.parse_args()

    fa, da = load(a.agent)
    fb, db = load(a.opponent)
    print(f"{a.agent} runs {da.count(CAPE)}x Hero's Cape")

    seen = Counter()
    offers = []
    buffs = Counter()
    attached = 0
    for g in range(a.games):
        for f in (fa, fb):
            try:
                f({"current": None, "select": None, "logs": []})
            except Exception:
                pass
        obs, _ = battle_start(list(da), list(db))
        if obs is None:
            continue
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                if o.current is not None and o.current.result != -1:
                    break
                st = o.current
                # How is a Tool actually OFFERED? Guessing between PLAY(7) and
                # ATTACH(8) would decide the whole shape of the guard, so watch
                # a real option instead.
                if o.select is not None and len(offers) < 8:
                    me = st.players[st.yourIndex]
                    for opt in (o.select.option or []):
                        oid = None
                        try:
                            ix = int(opt.index)
                            if me.hand and 0 <= ix < len(me.hand):
                                oid = me.hand[ix].id
                        except Exception:
                            pass
                        if oid == CAPE:
                            tgt = "?"
                            try:
                                ipa = int(getattr(opt, "inPlayArea", 0) or 0)
                                ipi = int(getattr(opt, "inPlayIndex", -1))
                                zone = (me.active if ipa == 4 else me.bench)
                                if zone and 0 <= ipi < len(zone) and zone[ipi]:
                                    tgt = CARDS.get(zone[ipi].id).name
                            except Exception:
                                pass
                            offers.append(
                                f"type={int(opt.type)} area={opt.area} "
                                f"index={opt.index} "
                                f"inPlayArea={getattr(opt,'inPlayArea',None)} "
                                f"inPlayIndex={getattr(opt,'inPlayIndex',None)}"
                                f"  -> target {tgt}")
                for p in st.players:
                    for mon in ([p.active[0]] if (p.active and p.active[0])
                                else []) + list(p.bench or []):
                        if mon is None:
                            continue
                        tools = [t.id for t in (mon.tools or [])]
                        if CAPE in tools:
                            base = getattr(CARDS.get(mon.id), "hp", None)
                            if base:
                                seen[mon.id] += 1
                                buffs[mon.maxHp - base] += 1
                who = st.yourIndex
                obs = battle_select(list((fa if who == 0 else fb)(obs)))
        except Exception as exc:
            print("  game error:", type(exc).__name__, exc)
        finally:
            battle_finish()

    print("\nhow the engine OFFERS a Hero's Cape (7=PLAY, 8=ATTACH):")
    for s in offers[:8]:
        print("   ", s)
    if not offers:
        print("    (none captured)")

    print(f"\nobservations of a Caped Pokemon: {sum(seen.values())}")
    if not seen:
        print("  NEVER ATTACHED -- the agent holding the card did not play it.")
        print("  That is a policy result, not an engine result: a Tool in a "
              "deck is worth\n  nothing unless the pilot is taught to attach "
              "it.")
        return 0
    for cid, n in seen.most_common():
        c = CARDS.get(cid)
        print(f"  {n:5d} x {getattr(c,'name',cid)} (base HP {getattr(c,'hp','?')})")
    print("\nmaxHp - base HP, i.e. what the Cape is actually worth:")
    for d, n in buffs.most_common():
        print(f"  {d:+5d} HP  in {n} observations"
              f"{'   <-- as printed on the card' if d == 100 else ''}")

    grimm = CARDS.get(648)
    if grimm:
        print(f"\nMarnie's Grimmsnarl ex base {grimm.hp} HP; Shadow Bullet 180.")
        for hp in (grimm.hp, grimm.hp + 100):
            hits = -(-hp // 180)
            print(f"  at {hp} HP it survives {hits - 1} Shadow Bullet(s), "
                  f"dies to {hits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
