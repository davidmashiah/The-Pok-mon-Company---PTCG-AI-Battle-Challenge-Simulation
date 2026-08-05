"""How often is our main attack simply unavailable, and could a second attacker fix it?

Mega Brave reads "During your next turn, this Pokemon can't use Mega Brave." So
one Mega Lucario ex can only throw 270 every OTHER turn; on the off turns we hit
for 130 with Aura Jab. Meanwhile Archaludon ex hits 220 EVERY turn and Marnie's
Grimmsnarl hits 180 every turn. Against Archaludon that alone loses the prize
race, since it costs them 2 prizes to kill and us 3.

The deck already contains the answer, if the policy uses it. Aura Jab reads
"Attach up to 3 Basic {F} Energy cards from your discard pile to your BENCHED
Pokemon" -- it is a charge move for a SECOND Mega Lucario. With two charged
Mega Lucarios we can alternate and throw 270 every turn.

This measures the opportunity, not the theory:

  locked          our Active is a Mega Lucario with 2+ energy and Mega Brave is
                  NOT on offer (it attacked last turn)
  backup_ready    ...and a BENCHED Mega Lucario already has 2+ energy
  we_switched     ...and we actually moved to it this turn

If backup_ready is near zero the fix is upstream (we never charge a second one).
If backup_ready is high but we_switched is low, the policy is leaving 140 damage
a turn on the table and the fix is in the scoring.

Usage:
  python work/tools/megabrave_uptime.py --agent v43_judge2x --opp w2_archaludon --games 40
"""
import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
MEGA_LUCARIO = 678
MEGA_BRAVE = 983


def _load(agent_dir):
    full = os.path.join(WORK, "agents", agent_dir)
    if full not in sys.path:
        sys.path.insert(0, full)
    with open(os.path.join(full, "main.py"), encoding="utf-8-sig") as fh:
        src = fh.read()
    env = {}
    cwd = os.getcwd()
    try:
        os.chdir(full)
        exec(compile(src, "main.py", "exec"), env)
    finally:
        os.chdir(cwd)
    d = env.get("my_deck") or env.get("DECK")
    if not d:
        with open(os.path.join(full, "deck.csv"), encoding="utf-8") as fh:
            d = [int(x) for x in fh.read().split() if x.strip()]
    return [v for v in env.values() if callable(v)][-1], list(d)


def _worker(job):
    agent_dir, opp_dir, n, seed0 = job
    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import OptionType, SelectContext, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    fa_raw, da = _load(agent_dir)
    fb, db = _load(opp_dir)
    st = Counter()
    seen_turn = {"t": -1, "locked": False, "backup": False}

    def fa(obs_dict):
        sel = fa_raw(obs_dict)
        try:
            o = to_observation_class(obs_dict)
            c, s = o.current, o.select
            if c is None or s is None or s.context != SelectContext.MAIN:
                return sel
            me = c.players[c.yourIndex]
            act = (me.active or [None])[0]
            if act is None or act.id != MEGA_LUCARIO or len(act.energies) < 2:
                return sel
            can_mb = any(op.type == OptionType.ATTACK and op.attackId == MEGA_BRAVE
                         for op in s.option)
            has_attack = any(op.type == OptionType.ATTACK for op in s.option)
            if not has_attack:
                return sel
            backup = any(p is not None and p.id == MEGA_LUCARIO
                         and len(p.energies) >= 2 for p in (me.bench or []))
            # count each TURN once, not each frame
            if c.turn != seen_turn["t"]:
                seen_turn.update(t=c.turn, locked=False, backup=False)
                if not can_mb:
                    st["locked_turns"] += 1
                    seen_turn["locked"] = True
                    if backup:
                        st["locked_with_backup_ready"] += 1
                        seen_turn["backup"] = True
                else:
                    st["mega_brave_available_turns"] += 1
            if seen_turn["locked"] and seen_turn["backup"] and sel:
                op = s.option[sel[0]]
                if op.type in (OptionType.RETREAT,) or (
                        op.type == OptionType.PLAY and getattr(op, "cardId", None) == 1123):
                    st["we_switched"] += 1
                    seen_turn["backup"] = False
        except Exception:
            st["wrap_err"] += 1
        return sel

    for g in range(n):
        first = ((seed0 + g) % 2 == 0)
        d0, d1 = (da, db) if first else (db, da)
        p0, p1 = (fa, fb) if first else (fb, fa)
        me_i = 0 if first else 1
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        seen_turn.update(t=-1, locked=False, backup=False)
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                c = o.current
                if c is not None and c.result != -1:
                    st["wins" if c.result == me_i else "losses"] += 1
                    break
                who = c.yourIndex if c is not None else 0
                obs = battle_select(list((p0 if who == 0 else p1)(obs)))
        except Exception:
            st["err"] += 1
        finally:
            battle_finish()
        st["games"] += 1
    return dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="v43_judge2x")
    ap.add_argument("--opp", default="w2_archaludon")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    per = max(1, a.games // a.workers)
    jobs = [(a.agent, a.opp, per, i * 991) for i in range(a.workers)]
    st = Counter()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(_worker, j) for j in jobs]):
            st.update(f.result())

    lock = st["locked_turns"]
    avail = st["mega_brave_available_turns"]
    tot = max(1, lock + avail)
    print(f"\n{a.agent} vs {a.opp}: {st['games']} games "
          f"(win {st['wins']/max(1,st['games']):.3f}) in {time.time()-t0:.0f}s\n")
    print(f"  turns with Mega Lucario active & charged   : {lock+avail}")
    print(f"    Mega Brave AVAILABLE                     : {avail}  ({100*avail/tot:.0f}%)")
    print(f"    Mega Brave LOCKED (used it last turn)    : {lock}  ({100*lock/tot:.0f}%)")
    print(f"      ...with a charged Mega Lucario on bench: "
          f"{st['locked_with_backup_ready']}  "
          f"({100*st['locked_with_backup_ready']/max(1,lock):.0f}% of locked turns)")
    print(f"      ...and we actually switched to it      : {st['we_switched']}")
    print("\n  reading: locked turns are turns we hit for 130 instead of 270.")
    print("  A charged bench Mega Lucario converts one back into a 270.")


if __name__ == "__main__":
    main()
