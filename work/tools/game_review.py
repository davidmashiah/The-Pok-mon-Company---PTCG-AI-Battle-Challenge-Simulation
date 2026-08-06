"""Per-game post-mortem of a live submission: what beat us, and how.

The field test says which ARCHETYPES we lose to. It cannot say HOW, because its
opponents are our local panel. This reads the real ladder replays one game at a
time and reconstructs the ending, so the next change is aimed at a mechanism
rather than a matchup label.

For every episode it recovers, from the replay itself and not by inference:
  * the opponent's decklist, taken from their action on the deck-selection frame
  * their team rating, joined from the leaderboard
  * how the game ENDED -- prizes each side still held, turn count, and whether
    either deck hit zero, which is the difference between losing a race and
    milling ourselves out
  * our remaining prizes when we lost, which separates "close" from "blown out"

Run it repeatedly; it caches replays and only fetches what is new.

  python work/tools/game_review.py --sub 55299973 --fetch
  python work/tools/game_review.py --sub 55299973
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
OUT = os.path.join(WORK, "out")
STORE = os.path.join(OUT, "review_replays")
PACE = 1.4
_last = [0.0]

SIGNATURES = [
    (648, "Grimmsnarl"), (743, "Alakazam"), (678, "Mega Lucario"),
    (849, "Mega Lopunny"), (345, "Crustle"), (121, "Dragapult"),
    (756, "Mega Kangaskhan"), (58, "Great Tusk"), (116, "Okidogi"),
    (723, "Mega Abomasnow"), (143, "Snorlax"), (163, "Slowking"),
    (1071, "Meowth ex"), (184, "Latias ex"),
]


def classify(deck):
    if not deck:
        return "unknown"
    have = set(deck)
    hits = [n for cid, n in SIGNATURES if cid in have]
    return "+".join(hits[:2]) if hits else "unknown"


def kag(args):
    wait = PACE - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.run([sys.executable, "-m", "kaggle"] + args,
                       capture_output=True, text=True, timeout=300,
                       encoding="utf-8", errors="replace", env=env)
    return p.returncode


def episodes(sub):
    import urllib.request
    tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
    req = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/competitions/submissions/{sub}/episodes",
        headers={"Authorization": f"Bearer {tok}"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read()).get(
        "episodes", [])


def ratings():
    try:
        lb = json.load(open(os.path.join(OUT, "lb_full.json"), encoding="utf-8"))
        return {str(r["TeamId"]): float(r["Score"]) for r in lb if r.get("Score")}
    except Exception:
        return {}


def ending(steps, mi):
    """Reconstruct the final position from the last frame that carries one."""
    oi = 1 - mi
    for st in reversed(steps):
        for who in (mi, oi):
            try:
                cur = (st[who].get("observation") or {}).get("current") or {}
                pls = cur.get("players") or []
                if len(pls) == 2:
                    me, op = pls[mi], pls[oi]
                    return {
                        "turn": cur.get("turn"),
                        "my_prizes": len(me.get("prize") or []),
                        "op_prizes": len(op.get("prize") or []),
                        "my_deck": me.get("deckCount"),
                        "op_deck": op.get("deckCount"),
                    }
            except Exception:
                continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    os.makedirs(STORE, exist_ok=True)
    rate = ratings()

    eps = episodes(args.sub)
    rows = []
    for e in eps:
        me = [a for a in e["agents"] if str(a.get("submissionId")) == args.sub]
        op = [a for a in e["agents"] if str(a.get("submissionId")) != args.sub]
        if not me or not op:
            continue
        r, orr = me[0].get("reward"), op[0].get("reward")
        if r is None or orr is None:
            continue
        rows.append({"ep": str(e["id"]), "mi": int(me[0].get("index", 0)),
                     "won": r > orr, "opp_team": op[0].get("teamName"),
                     "opp_rating": rate.get(str(op[0].get("teamId")))})

    if args.fetch:
        got = 0
        for w in rows:
            p = os.path.join(STORE, f"episode-{w['ep']}-replay.json")
            if os.path.exists(p):
                got += 1
                continue
            if kag(["competitions", "replay", w["ep"], "-p", STORE, "-q"]) == 0:
                got += 1
        print(f"replays on disk: {got}/{len(rows)}")

    print(f"\n{'ep':>10} {'res':>3} {'opp rating':>10} {'archetype':16s} "
          f"{'turns':>5} {'prizes':>7} {'decks':>9}  opponent")
    print("-" * 104)
    by_arch = defaultdict(lambda: [0, 0])
    by_band = defaultdict(lambda: [0, 0])
    loss_prizes = Counter()
    deckouts = [0, 0]
    n_detail = 0
    for w in sorted(rows, key=lambda x: -(x["opp_rating"] or 0)):
        p = os.path.join(STORE, f"episode-{w['ep']}-replay.json")
        arch, end = "unknown", None
        if os.path.exists(p):
            try:
                d = json.load(open(p, encoding="utf-8"))
                steps = d.get("steps") or []
                oi = 1 - w["mi"]
                for st in steps[:4]:
                    a = st[oi].get("action") if oi < len(st) else None
                    if isinstance(a, list) and len(a) == 60:
                        arch = classify(a)
                        break
                end = ending(steps, w["mi"])
                n_detail += 1
            except Exception:
                pass
        res = "W" if w["won"] else "L"
        by_arch[arch][0] += w["won"]
        by_arch[arch][1] += 1
        if w["opp_rating"]:
            b = int(w["opp_rating"] // 100) * 100
            by_band[b][0] += w["won"]
            by_band[b][1] += 1
        pz = dz = ""
        if end:
            pz = f"{end['my_prizes']}-{end['op_prizes']}"
            dz = f"{end['my_deck']}-{end['op_deck']}"
            if not w["won"]:
                loss_prizes[end["my_prizes"]] += 1
                if (end["my_deck"] or 0) == 0:
                    deckouts[0] += 1
            if (end["my_deck"] or 0) == 0:
                deckouts[1] += 1
        print(f"{w['ep']:>10} {res:>3} {str(w['opp_rating'] or '?'):>10} "
              f"{arch[:16]:16s} {str(end['turn']) if end else '?':>5} "
              f"{pz:>7} {dz:>9}  {str(w['opp_team'])[:26]}")

    wins = sum(1 for w in rows if w["won"])
    print(f"\noverall {wins}-{len(rows)-wins} = {wins/max(len(rows),1):.3f} "
          f"over {len(rows)} episodes ({n_detail} with replay detail)")

    print(f"\n{'opponent archetype':22s} {'games':>6} {'wins':>5} {'winrate':>8}")
    for a, (wn, g) in sorted(by_arch.items(), key=lambda kv: -kv[1][1]):
        print(f"{a[:22]:22s} {g:6d} {wn:5d} {wn/g:8.3f}")

    print(f"\n{'opponent rating band':22s} {'games':>6} {'wins':>5} {'winrate':>8}")
    for b in sorted(by_band):
        wn, g = by_band[b]
        print(f"{b}-{b+99:<17d} {g:6d} {wn:5d} {wn/g:8.3f}")

    if loss_prizes:
        print("\nwhen we LOST, prizes we still held (6 = took none):")
        for k in sorted(loss_prizes):
            print(f"   {k} left : {loss_prizes[k]}")
    print(f"\ngames where OUR deck hit 0: {deckouts[1]} "
          f"(of which losses: {deckouts[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
