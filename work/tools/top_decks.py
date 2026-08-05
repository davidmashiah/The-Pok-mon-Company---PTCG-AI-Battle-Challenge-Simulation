"""What deck does the TOP of the leaderboard actually play, right now?

Why this and not `deck_choice.py`: that tool weights archetype-vs-archetype win
rates over 13,444 games from late July, across the WHOLE ladder. Most of that
population is below the median. The question that matters for us is narrower and
more current -- of the teams above 1000, what are they piloting today? A deck
that wins the whole ladder and a deck that wins the top of it are not the same
deck, and our adopted base brought its author's Alakazam list with it.

Method, with no inference: every episode opens with a setup frame where the
agent returns its 60 card ids, so the replay records each side's decklist
verbatim. We read it out rather than guessing an archetype from what got played.

  python work/tools/top_decks.py --teams 40 --per-team 2
  python work/tools/top_decks.py --report
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
STORE = os.path.join(OUT, "top_decks.json")
REPLAYS = os.path.join(OUT, "top_replays")
PACE = 1.5
_last = [0.0]

# A card that only that archetype plays. Read together with the printed
# decklist -- decks whose engine is a plain Stage 2 hide under whatever ex they
# splash, which is exactly the caveat that made the old index misleading.
SIGNATURES = [
    (743, "Alakazam"),
    (678, "Mega Lucario ex"),
    (648, "Marnie's Grimmsnarl ex"),
    (849, "Mega Lopunny ex"),
    (345, "Crustle"),
    (121, "Dragapult ex"),
    (723, "Mega Abomasnow ex"),
    (143, "Snorlax stall"),
    (58, "Great Tusk"),
]


def kag(args):
    wait = PACE - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.run([sys.executable, "-m", "kaggle"] + args,
                       capture_output=True, text=True, timeout=300,
                       encoding="utf-8", errors="replace", env=env)
    return (p.stdout or "") + (p.stderr or ""), p.returncode


def classify(deck):
    have = set(deck)
    hits = [name for cid, name in SIGNATURES if cid in have]
    return " + ".join(hits) if hits else "unknown"


def decks_from_replay(path):
    """-> {agent_index: [60 card ids]} read from the setup frame."""
    with open(path, encoding="utf-8") as fh:
        rep = json.load(fh)
    steps = rep.get("steps") or []
    if not steps:
        return {}
    out = {}
    # The deck-selection action does not always land on step 0 -- scan the first
    # few, the way loss_autopsy.py already does. Reading only step 0 found
    # nothing and looked like "no top team plays a deck".
    for st in steps[:4]:
        for i, agent in enumerate(st):
            if i in out:
                continue
            act = agent.get("action")
            if isinstance(act, list) and len(act) == 60 and all(
                    isinstance(x, int) for x in act):
                out[i] = act
    return out


def fetch(n_teams, per_team):
    lb = json.load(open(os.path.join(OUT, "lb_full.json"), encoding="utf-8"))
    lb = [r for r in lb if r.get("Score")]
    lb.sort(key=lambda r: -float(r["Score"]))
    store = {}
    if os.path.exists(STORE):
        try:
            store = json.load(open(STORE, encoding="utf-8"))
        except Exception:
            store = {}
    os.makedirs(REPLAYS, exist_ok=True)

    for rank, row in enumerate(lb[:n_teams], 1):
        tid, name, score = str(row["TeamId"]), row["TeamName"], row["Score"]
        if tid in store and len(store[tid].get("decks", [])) >= per_team:
            continue
        txt, rc = kag(["competitions", "team-submissions", tid, "--format", "json"])
        subs = []
        try:
            for s in json.loads(txt[txt.index("["):txt.rindex("]") + 1]):
                if s.get("publicScore"):
                    subs.append((float(s["publicScore"]), str(s["id"])))
        except Exception:
            print(f"  rank {rank} {name}: cannot list submissions")
            continue
        if not subs:
            continue
        subs.sort(reverse=True)
        best_sub = subs[0][1]

        # REST, not the CLI: the CLI's episode listing omits the agents block,
        # and without it we cannot tell WHICH side of the replay is this team's.
        # Taking both sides would report the field, not the top of it.
        eps = []
        try:
            import urllib.request
            token = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
            req = urllib.request.Request(
                "https://www.kaggle.com/api/v1/competitions/submissions/"
                f"{best_sub}/episodes",
                headers={"Authorization": f"Bearer {token}"})
            data = json.loads(urllib.request.urlopen(req, timeout=60).read())
            for e in data.get("episodes", []):
                mine = [a for a in e.get("agents", [])
                        if str(a.get("submissionId")) == str(best_sub)]
                if mine:
                    eps.append((str(e["id"]), int(mine[0].get("index", 0))))
        except Exception as exc:
            print(f"  rank {rank} {name}: cannot list episodes ({exc})")
            continue
        if not eps:
            continue

        found = []
        for ep, my_index in eps[:per_team * 3]:
            if len(found) >= per_team:
                break
            # the CLI names it episode-<id>-replay.json, not <id>.json; guessing
            # wrong made this loop skip every replay it had just downloaded
            path = os.path.join(REPLAYS, f"episode-{ep}-replay.json")
            if not os.path.exists(path):
                _t, rc = kag(["competitions", "replay", ep, "-p", REPLAYS, "-q"])
                if rc != 0:
                    continue
            if not os.path.exists(path):
                print(f"    replay {ep}: not written to {path}")
                continue
            try:
                d = decks_from_replay(path)
            except Exception:
                continue
            if my_index in d:
                found.append(d[my_index])
        if found:
            store[tid] = {"rank": rank, "name": name, "score": score,
                          "decks": found[:per_team * 2]}
            json.dump(store, open(STORE, "w"), indent=1)
            print(f"  rank {rank:3d} {score:>8} {name[:26]:26s} -> "
                  + ", ".join(sorted({classify(x) for x in found})))
    return 0


def report():
    store = json.load(open(STORE, encoding="utf-8"))
    per_arch = Counter()
    by_rank = defaultdict(Counter)
    for tid, rec in store.items():
        arch = Counter(classify(d) for d in rec["decks"]).most_common(1)[0][0]
        per_arch[arch] += 1
        band = "top25" if rec["rank"] <= 25 else ("26-50" if rec["rank"] <= 50
                                                  else "51+")
        by_rank[band][arch] += 1
    print(f"\n{'archetype':34s} {'teams':>6}")
    print("-" * 42)
    for a, c in per_arch.most_common():
        print(f"{a[:34]:34s} {c:>6}")
    for band in ("top25", "26-50", "51+"):
        if by_rank[band]:
            print(f"\n{band}: " + ", ".join(f"{a}={c}"
                                            for a, c in by_rank[band].most_common()))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=int, default=40)
    ap.add_argument("--per-team", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    return fetch(args.teams, args.per_team)


if __name__ == "__main__":
    sys.exit(main())
