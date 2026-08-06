"""Why do we lose to REAL leaderboard teams? Autopsy our own ladder replays.

Split by opponent, our recent submissions win 0.677 against teams that are not
on the leaderboard and **0.339 against teams that are**. The blended 0.55 is
flattering noise; against actual competitors we win one game in three. No local
metric in this repo measures that, because every local opponent is our own
policy piloting a scraped decklist.

These are OUR OWN games, against the real field, so there is no
determinization, no proxy pilot and no anti-correlated validation. Reports:

  * opponent archetype (from the deck they submitted on the setup step)
  * our win rate per archetype, and per opponent rating band
  * HOW the game ended: prizes taken by each side, turn count, whether we were
    decked out, and how many prizes we were still holding when we lost

Usage:
  python work/tools/loss_autopsy.py --fetch      # download replays (paced)
  python work/tools/loss_autopsy.py --report
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
OUT = os.path.join(WORK, "out")
STORE = os.path.join(OUT, "our_replays")
COMP = "pokemon-tcg-ai-battle"
PACE = 1.2
_last = [0.0]

# 55274352 is v61_codex_safe, the adopted base and current champion. The two
# v51 draws are kept because the archetype mix we are matched into shifts with
# our rating, and comparing the two rating bands is the point.
OUR_SUBS = ["55299973", "55305926"]   # the two live w8_grimm_tuned draws

# archetype signatures: a card that only that deck plays
ARCHETYPES = [
    (743, "Alakazam"),
    (756, "Mega Kangaskhan ex"),
    (58, "Great Tusk"),
    (116, "Okidogi"),
    (648, "Marnie's Grimmsnarl ex"),
    (678, "Mega Lucario ex (ours)"),
    (723, "Mega Abomasnow ex"),
    (121, "Dragapult ex"),
    (849, "Mega Lopunny ex"),
    (345, "Crustle wall"),
    (143, "Snorlax stall"),
]


def kag(args, cwd=None):
    wait = PACE - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.run([sys.executable, "-m", "kaggle"] + args,
                       capture_output=True, text=True, cwd=cwd, timeout=300,
                       encoding="utf-8", errors="replace", env=env)
    return (p.stdout or "") + (p.stderr or "")


def fetch():
    os.makedirs(STORE, exist_ok=True)
    lb = json.load(open(os.path.join(OUT, "lb_full.json"), encoding="utf-8"))
    rated = {r["teamId"]: float(r["score"]) for r in lb if r.get("score")}
    want = []
    for ref in OUR_SUBS:
        p = os.path.join(OUT, f"ep_{ref}.json")
        if not os.path.exists(p):
            continue
        for e in json.load(open(p, encoding="utf-8"))["episodes"]:
            me = [a for a in e["agents"] if str(a.get("submissionId")) == ref]
            op = [a for a in e["agents"] if str(a.get("submissionId")) != ref]
            if not me or not op:
                continue
            want.append({
                "episode": str(e["id"]),
                "sub": ref,
                "won": me[0].get("rewardNullable") == 1,
                "my_index": me[0].get("index"),
                "opp_team": op[0].get("teamName"),
                "opp_rating": rated.get(op[0].get("teamId")),
            })
    json.dump(want, open(os.path.join(OUT, "our_episodes.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(want)} episodes to fetch "
          f"({sum(1 for w in want if w['opp_rating'] is not None)} vs ranked teams)")
    tmpd = tempfile.mkdtemp(prefix="ptcg_ourreplay_")
    got = 0
    for i, w in enumerate(want):
        dst = os.path.join(STORE, f"{w['episode']}.json")
        if os.path.exists(dst):
            got += 1
            continue
        kag(["competitions", "replay", w["episode"]], cwd=tmpd)
        src = os.path.join(tmpd, f"episode-{w['episode']}-replay.json")
        if os.path.exists(src):
            os.replace(src, dst)
            got += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(want)} ... {got} on disk")
    print(f"done: {got}/{len(want)} replays on disk in {STORE}")


def classify(deck):
    c = Counter(deck)
    for cid, name in ARCHETYPES:
        if c.get(cid):
            return name
    return "other"


def analyse():
    meta = json.load(open(os.path.join(OUT, "our_episodes.json"), encoding="utf-8"))
    by_arch = defaultdict(lambda: [0, 0])
    by_band = defaultdict(lambda: [0, 0])
    endings = Counter()
    prize_left = Counter()
    turns = []
    n = 0
    for w in meta:
        p = os.path.join(STORE, f"{w['episode']}.json")
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        steps = d.get("steps") or []
        if not steps:
            continue
        mi = w["my_index"]
        oi = 1 - mi
        # opponent decklist = their action on the deck-selection step
        opp_deck = None
        for st in steps[:3]:
            if oi < len(st) and isinstance(st[oi].get("action"), list) \
                    and len(st[oi]["action"]) == 60:
                opp_deck = st[oi]["action"]
                break
        arch = classify(opp_deck) if opp_deck else "unknown"
        n += 1
        by_arch[arch][0] += w["won"]
        by_arch[arch][1] += 1
        r = w["opp_rating"]
        band = ("unranked" if r is None else "<800" if r < 800 else
                "800-900" if r < 900 else "900-1000" if r < 1000 else "1000+")
        by_band[band][0] += w["won"]
        by_band[band][1] += 1
        # final state: prizes remaining on each side
        last = None
        for st in reversed(steps):
            for a in st:
                o = (a or {}).get("observation") or {}
                cur = o.get("current") or {}
                if cur.get("players"):
                    last = cur
                    break
            if last:
                break
        if last:
            turns.append(last.get("turn") or 0)
            pls = last["players"]
            if mi < len(pls) and oi < len(pls):
                mp = len(pls[mi].get("prize") or [])
                op = len(pls[oi].get("prize") or [])
                if not w["won"]:
                    prize_left[mp] += 1
                    endings["we_had_%d_prizes_left" % mp] += 1
                if (pls[mi].get("deckCount") or 0) == 0:
                    endings["our_deck_empty"] += 1

    print(f"\nanalysed {n} of our own ladder replays\n")
    print(f"{'opponent archetype':<28} {'games':>6} {'wins':>5} {'win rate':>9}")
    print("-" * 52)
    for k, (w, t) in sorted(by_arch.items(), key=lambda kv: -kv[1][1]):
        print(f"{k:<28} {t:>6} {w:>5} {w/max(1,t):>9.3f}")
    print(f"\n{'opponent rating band':<28} {'games':>6} {'wins':>5} {'win rate':>9}")
    print("-" * 52)
    for k in ["unranked", "<800", "800-900", "900-1000", "1000+"]:
        if by_band[k][1]:
            w, t = by_band[k]
            print(f"{k:<28} {t:>6} {w:>5} {w/max(1,t):>9.3f}")
    if turns:
        print(f"\nmean game length: {sum(turns)/len(turns):.1f} turns")
    print("\nwhen we LOST, prizes we still had left (6 = we took none):")
    for k in sorted(prize_left):
        print(f"   {k} prizes left : {prize_left[k]:>3}")
    if endings.get("our_deck_empty"):
        print(f"\n  games ending with OUR deck empty: {endings['our_deck_empty']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    if a.report or not a.fetch:
        analyse()


if __name__ == "__main__":
    main()
