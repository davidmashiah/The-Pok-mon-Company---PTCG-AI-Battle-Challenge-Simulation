"""Is Judge actually shrinking the Alakazam player's hand, or did we get lucky?

v39 beat the 5th-place Alakazam agent 0.695 where v32 managed 0.420. That is a
win-rate result, and this project has been burned by win-rate results. The
causal claim is narrow and mechanical, so measure it directly:

    Alakazam's Powerful Hand places 2 damage counters on our Active for EACH
    CARD IN THEIR HAND -- 20 damage a card, as counters, so Weakness and our
    340 HP Mega ex are irrelevant.

If Judge is doing the work, the opponent's hand size AT THE MOMENT THEY ATTACK
must drop. If their hand is the same size and we are just winning for some other
reason, the card is not the mechanism and the deck slot is unjustified.

Reports, per agent, against the real published Alakazam agent:
  * distribution of their hand size when they attacked
  * implied Powerful Hand damage, and how often that was lethal on our Mega
    Lucario ex (340 HP)
  * how many Judges we actually played, and their hand size just before/after

Usage: python work/tools/hand_disruption_check.py --agents v32_ppp,v39_judge --games 40
"""
import argparse
import os
import statistics
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
JUDGE = 1213
MEGA_LUCARIO_HP = 340
ALAKAZAM_LINE = {741, 742, 743}


def _load(agent_dir):
    """exec-load with cwd set to the agent's own directory.

    Public agents (w1_alakazam, w2_archaludon) read "deck.csv" relative to cwd
    and fall back to /kaggle_simulations/agent/deck.csv, which does not exist
    off-Kaggle -- loading them from a private temp cwd raises FileNotFoundError.
    """
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
    os.chdir(tempfile.mkdtemp(prefix="handdis_"))
    from cg.api import LogType, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    fa, da = _load(agent_dir)
    fb, db = _load(opp_dir)
    st = Counter()
    hands = []            # their hand size when they attacked

    for g in range(n):
        first = ((seed0 + g) % 2 == 0)
        d0, d1 = (da, db) if first else (db, da)
        p0, p1 = (fa, fb) if first else (fb, fa)
        me = 0 if first else 1
        obs, _ = battle_start(list(d0), list(d1))
        if obs is None:
            continue
        last_opp_hand = 0
        try:
            for _ in range(4000):
                o = to_observation_class(obs)
                c = o.current
                if c is not None and c.result != -1:
                    st["wins" if c.result == me else "losses"] += 1
                    break
                if c is not None:
                    opp = c.players[1 - me]
                    last_opp_hand = opp.handCount or 0
                for lg in (o.logs or []):
                    t = int(lg.type)
                    if t == int(LogType.ATTACK) and lg.playerIndex is not None \
                            and lg.playerIndex != me:
                        # their hand as we last saw it, i.e. going into the attack
                        hands.append(last_opp_hand)
                        st["their_attacks"] += 1
                        if 20 * last_opp_hand >= MEGA_LUCARIO_HP:
                            st["their_hand_was_lethal_on_mega"] += 1
                    elif t == int(LogType.PLAY) and lg.cardId == JUDGE \
                            and lg.playerIndex == me:
                        st["judges_we_played"] += 1
                who = c.yourIndex if c is not None else 0
                obs = battle_select(list((p0 if who == 0 else p1)(obs)))
        except Exception:
            st["err"] += 1
        finally:
            battle_finish()
        st["games"] += 1
    return {"st": dict(st), "hands": hands}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="v32_ppp,v39_judge")
    ap.add_argument("--opp", default="w1_alakazam")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    print(f"vs {a.opp} (a real published agent), {a.games} games each\n")
    print(f"{'agent':<14} {'games':>6} {'win':>6} {'Judges':>7} {'their attacks':>14} "
          f"{'their hand @ attack':>21} {'lethal-hand attacks':>21}")
    print("-" * 96)
    for ag in a.agents.split(","):
        ag = ag.strip()
        per = max(1, a.games // a.workers)
        jobs = [(ag, a.opp, per, i * 5501) for i in range(a.workers)]
        st = Counter()
        hands = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for f in as_completed([ex.submit(_worker, j) for j in jobs]):
                r = f.result()
                st.update(r["st"])
                hands += r["hands"]
        gm = max(1, st["games"])
        att = max(1, st["their_attacks"])
        med = statistics.median(hands) if hands else 0
        mean = statistics.mean(hands) if hands else 0
        print(f"{ag:<14} {st['games']:>6} {st['wins']/gm:>6.3f} "
              f"{st['judges_we_played']:>7} {st['their_attacks']:>14} "
              f"{('mean %.1f / med %.0f' % (mean, med)):>21} "
              f"{('%d (%.1f%%)' % (st['their_hand_was_lethal_on_mega'], 100*st['their_hand_was_lethal_on_mega']/att)):>21}")
    print("\nPowerful Hand = 20 damage per card in THEIR hand. 17 cards one-shots")
    print("our Mega Lucario ex (340 HP). If Judge is the mechanism, the last two")
    print("columns must fall; if they do not, the win rate came from elsewhere.")


if __name__ == "__main__":
    main()
