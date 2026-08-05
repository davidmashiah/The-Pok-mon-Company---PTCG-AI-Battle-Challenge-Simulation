"""How exactly do we lose a matchup, against a REAL opponent agent?

Win rates say we lose; they never say why. This plays our agent against a
published opponent bundle and reconstructs the prize race:

  * every attack WE made: attacker, target, damage dealt, and whether it KO'd
  * every attack THEY made against us, and what it killed
  * prizes taken by each side, and how many we still held when we lost
  * for a named key target (e.g. Archaludon ex at 300 HP), how often we had it
    in one-shot range and whether we converted

Why this matters for Archaludon specifically: Archaludon ex is 300 HP and gives
up 2 prizes; our Mega Lucario ex gives up 3. Trading knockout for knockout we
LOSE the race, exactly as with Grimmsnarl. Mega Brave 270 + one Premium Power
Pro = 300, which is precisely lethal -- so the whole matchup turns on whether we
actually convert that, and this tool answers that rather than guessing.

Usage:
  python work/tools/matchup_autopsy.py --agent v43_judge2x --opp w2_archaludon \
      --games 60 --target 190
"""
import argparse
import os
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
PPP = 1141


def _load(agent_dir):
    """exec-load with cwd at the agent's own dir (public agents read deck.csv
    relative to cwd and fall back to a /kaggle_simulations path that does not
    exist off-Kaggle)."""
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
    agent_dir, opp_dir, target, n, seed0 = job
    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import LogType, all_card_data, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    cards = {c.cardId: c for c in all_card_data()}
    fa_raw, da = _load(agent_dir)
    fb, db = _load(opp_dir)
    st = Counter()
    _ctx = {}

    def fa(obs_dict):
        """Wrap our agent so we can see WHICH option it picked. An ATTACK made
        while the target is their Active is the event this tool exists to
        measure -- the damage histogram alone cannot tell a 270 aimed at a 130 HP
        Duraludon (an overkill knockout) from a 270 aimed at a 300 HP Archaludon
        ex (a whiff that hands them the prize race)."""
        sel = fa_raw(obs_dict)
        try:
            from cg.api import OptionType, SelectContext
            o = to_observation_class(obs_dict)
            if (o.select is not None and o.current is not None and sel
                    and o.select.context == SelectContext.MAIN):
                opt = o.select.option[sel[0]]
                if opt.type == OptionType.ATTACK:
                    st_ = o.current
                    opp_ = st_.players[1 - st_.yourIndex]
                    me_ = st_.players[st_.yourIndex]
                    act_ = (opp_.active or [None])[0]
                    if act_ is not None and act_.id == _ctx["target"]:
                        _ctx["pending"].clear()
                        _ctx["pending"].update(
                            serial=act_.serial, hp=act_.hp, maxhp=act_.maxHp,
                            had_ppp=any(x.id == PPP for x in (me_.hand or [])))
                        st["attacks_at_target"] += 1
        except Exception:
            st["wrap_err"] += 1
        return sel
    our_dmg = Counter()
    their_dmg = Counter()

    for g in range(n):
        first = ((seed0 + g) % 2 == 0)
        d0, d1 = (da, db) if first else (db, da)
        p0, p1 = (fa, fb) if first else (fb, fa)
        me = 0 if first else 1
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        prev_target = {}      # serial -> (id, hp) as we last saw it
        pending = {}          # an attack we launched at the target
        _ctx["target"] = target; _ctx["pending"] = pending
        won = None
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                c = o.current
                if c is not None and c.result != -1:
                    won = (c.result == me)
                    st["wins" if won else "losses"] += 1
                    pls = c.players
                    if not won:
                        st["lost_with_%d_prizes_left" % len(pls[me].prize or [])] += 1
                    st["prizes_we_took"] += 6 - len(pls[me].prize or [])
                    st["prizes_they_took"] += 6 - len(pls[1 - me].prize or [])
                    break
                # resolve an attack we launched at the target on a previous frame
                if pending and c is not None:
                    opp = c.players[1 - me]
                    still = None
                    for p in list(opp.active or []) + list(opp.bench or []):
                        if p is not None and p.serial == pending["serial"]:
                            still = p
                            break
                    if still is None:
                        st["TARGET_KOd"] += 1
                        if pending["hp"] == pending["maxhp"]:
                            st["TARGET_KOd_from_full"] += 1
                    else:
                        dealt = pending["hp"] - still.hp
                        st["TARGET_survived"] += 1
                        if dealt >= 250:
                            st["survived_after_big_hit"] += 1
                        if pending["had_ppp"]:
                            st["survived_while_we_HELD_ppp"] += 1
                    pending.clear()
                if c is not None and c.yourIndex == me:
                    opp = c.players[1 - me]
                    act = (opp.active or [None])[0]
                    if act is not None:
                        prev_target[act.serial] = (act.id, act.hp)
                        if act.id == target:
                            st["frames_target_active"] += 1
                            # could Mega Brave + one PPP finish it right now?
                            me_p = c.players[me]
                            mine = (me_p.active or [None])[0]
                            npp = sum(1 for x in (me_p.hand or []) if x.id == PPP)
                            if mine is not None and mine.id == 678 and len(mine.energies) >= 2:
                                st["target_active_and_mega_ready"] += 1
                                if 270 + 30 * min(2, npp) >= act.hp:
                                    st["target_in_oneshot_range"] += 1
                for lg in (o.logs or []):
                    if int(lg.type) == int(LogType.HP_CHANGE) and lg.value:
                        if lg.playerIndex is not None and lg.playerIndex != me:
                            our_dmg[abs(lg.value)] += 1
                        elif lg.playerIndex == me:
                            their_dmg[abs(lg.value)] += 1
                who = c.yourIndex if c is not None else 0
                obs = battle_select(list((p0 if who == 0 else p1)(obs)))
        except Exception:
            st["err"] += 1
        finally:
            battle_finish()
        st["games"] += 1
    return {"st": dict(st), "our": dict(our_dmg), "their": dict(their_dmg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="v43_judge2x")
    ap.add_argument("--opp", default="w2_archaludon")
    ap.add_argument("--target", type=int, default=190)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import all_card_data
    cards = {c.cardId: c for c in all_card_data()}
    tname = cards[a.target].name if a.target in cards else str(a.target)

    per = max(1, a.games // a.workers)
    jobs = [(a.agent, a.opp, a.target, per, i * 3301) for i in range(a.workers)]
    st, our, their = Counter(), Counter(), Counter()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(_worker, j) for j in jobs]):
            r = f.result()
            st.update(r["st"])
            our.update({int(k): v for k, v in r["our"].items()})
            their.update({int(k): v for k, v in r["their"].items()})

    g = max(1, st["games"])
    print(f"\n{a.agent}  vs  {a.opp}   {st['games']} games in {time.time()-t0:.0f}s")
    print(f"  win rate {st['wins']/g:.3f}")
    print(f"  prizes we took  {st['prizes_we_took']/g:.2f}/game")
    print(f"  prizes they took {st['prizes_they_took']/g:.2f}/game")
    print(f"\n  target = {tname} (id {a.target})")
    print(f"    frames it was their Active                 : {st['frames_target_active']}")
    print(f"    ...with our Mega Lucario ready (2+ energy) : {st['target_active_and_mega_ready']}")
    print(f"    ...and in one-shot range w/ PPP we HOLD    : {st['target_in_oneshot_range']}")
    print("\n  damage WE dealt (top 10):")
    for v, n in our.most_common(10):
        print(f"    {v:>4} x{n}")
    print("\n  damage THEY dealt to us (top 10):")
    for v, n in their.most_common(10):
        print(f"    {v:>4} x{n}")
    at = max(1, st["attacks_at_target"])
    print(f"\n  attacks we actually LAUNCHED at {tname}:")
    print(f"    total                                      : {st['attacks_at_target']}")
    print(f"    knocked it out                             : {st['TARGET_KOd']} "
          f"({100*st['TARGET_KOd']/at:.0f}%)")
    print(f"      ...of those, from FULL HP (true one-shot) : {st['TARGET_KOd_from_full']}")
    print(f"    it SURVIVED our attack                     : {st['TARGET_survived']} "
          f"({100*st['TARGET_survived']/at:.0f}%)")
    print(f"      ...while a Premium Power Pro sat UNPLAYED : "
          f"{st['survived_while_we_HELD_ppp']}  <-- unconverted one-shots")
    print("\n  when we lost, prizes we still held:")
    for k in sorted(k for k in st if k.startswith("lost_with_")):
        print(f"    {k.split('_')[2]} left : {st[k]}")


if __name__ == "__main__":
    main()
