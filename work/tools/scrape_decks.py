"""Scrape the actual 60-card decks of top leaderboard teams from public replays.

The competition publishes episode replays and the deck is submitted as the
agent's action on the deck-selection step, so every team's list is recoverable.
This builds the archetype library that lets our forward search determinize the
OPPONENT's hidden zones with a real decklist instead of filler.

Polite by construction: one request per PACE seconds, no retry on 429, and a
resumable checkpoint so an interrupted run does not re-fetch what it already has.

Usage:
  python work/tools/scrape_decks.py --top 60
  python work/tools/scrape_decks.py --report
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
OUT = os.path.join(WORK, "out")
STORE = os.path.join(OUT, "meta_decks.json")
COMP = "pokemon-tcg-ai-battle"
PACE = 1.1
_last = [0.0]


def kag(args, cwd=None):
    """Run the kaggle CLI, paced."""
    wait = PACE - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    # Team names are CJK; the default console codepage cannot decode them and
    # the reader thread dies mid-parse. Force UTF-8 both ways.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.run([sys.executable, "-m", "kaggle"] + args,
                       capture_output=True, text=True, cwd=cwd, timeout=300,
                       encoding="utf-8", errors="replace", env=env)
    out = (p.stdout or "") + (p.stderr or "")
    if "429" in out or "Too Many Requests" in out:
        return None, "429"
    return out, None


def parse_table(text):
    """Parse the CLI's fixed-width table into list-of-lists."""
    rows = []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    started = False
    for ln in lines:
        if set(ln.strip()) <= set("- "):
            started = True
            continue
        if not started:
            continue
        if ln.startswith("Use ") or ln.startswith("Next Page"):
            continue
        parts = [p for p in ln.split("  ") if p.strip()]
        rows.append([p.strip() for p in parts])
    return rows


def load():
    if os.path.exists(STORE):
        try:
            with open(STORE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"teams": {}, "failed": {}}


def save(s):
    os.makedirs(OUT, exist_ok=True)
    tmp = STORE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1, ensure_ascii=False)
    os.replace(tmp, STORE)


def leaderboard(top):
    """Page through the leaderboard. Default page size is 20, max 200."""
    out, token = [], None
    while len(out) < top:
        args = ["competitions", "leaderboard", COMP, "-s",
                "--page-size", str(min(200, max(20, top - len(out)))),
                "--format", "csv"]
        if token:
            args += ["--page-token", token]
        txt, err = kag(args)
        if err or not txt:
            break
        token, added = None, 0
        for ln in txt.splitlines():
            if ln.startswith("Next Page Token = "):
                token = ln.split("= ", 1)[1].strip()
                continue
            parts = ln.split(",")
            if len(parts) < 4 or not parts[0].strip().isdigit():
                continue
            try:
                out.append((parts[0].strip(), ",".join(parts[1:-2]).strip(),
                            float(parts[-1])))
                added += 1
            except ValueError:
                continue
        if not token or added == 0:
            break
    return out[:top]


def deck_from_replay(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    names = d.get("info", {}).get("TeamNames") or []
    found = {}
    for step in d.get("steps", []):
        for ai, ag in enumerate(step):
            act = ag.get("action") or []
            if len(act) == 60 and ai not in found:
                found[ai] = list(act)
        if len(found) == 2:
            break
    overage = {}
    for ai in (0, 1):
        vals = []
        for step in d.get("steps", []):
            if ai < len(step):
                o = step[ai].get("observation") or {}
                if "remainingOverageTime" in o:
                    vals.append(o["remainingOverageTime"])
        if vals:
            overage[ai] = round(vals[0] - vals[-1], 1)
    return names, found, overage, d.get("rewards")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    store = load()
    if args.report:
        report(store)
        return 0

    lb = leaderboard(args.top)
    print(f"leaderboard: {len(lb)} teams, {lb[0][2]:.1f} .. {lb[-1][2]:.1f}")
    tmpd = tempfile.mkdtemp(prefix="ptcg_replay_")

    for rank, (tid, tname, score) in enumerate(lb, 1):
        if tid in store["teams"]:
            print(f"[{rank:>3}] {score:>7.1f}  cached")
            continue
        try:
            txt, err = kag(["competitions", "team-submissions", tid])
            if err:
                print(f"[{rank:>3}] {score:>7.1f}  429, skipping")
                store["failed"][tid] = "429"
                save(store)
                continue
            subs = parse_table(txt or "")
            best = None
            for r in subs:
                if len(r) >= 3 and r[0].isdigit():
                    try:
                        best = (r[0], float(r[2]))
                        break
                    except ValueError:
                        continue
            if not best:
                store["failed"][tid] = "no-submission"
                save(store)
                print(f"[{rank:>3}] {score:>7.1f}  no submission")
                continue

            txt, err = kag(["competitions", "episodes", best[0]])
            if err:
                store["failed"][tid] = "429"
                save(store)
                continue
            eps = [r[0] for r in parse_table(txt or "")
                   if r and r[0].isdigit()]
            eps = [e for e in eps][:3]
            got = False
            for ep in eps:
                txt, err = kag(["competitions", "replay", ep], cwd=tmpd)
                if err:
                    break
                path = os.path.join(tmpd, f"episode-{ep}-replay.json")
                if not os.path.exists(path):
                    continue
                try:
                    names, decks, overage, rewards = deck_from_replay(path)
                finally:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                # which agent index is this team?
                idx = None
                for i, n in enumerate(names):
                    if n.strip() == tname.strip():
                        idx = i
                if idx is None or idx not in decks:
                    continue
                store["teams"][tid] = {
                    "rank": rank, "name": tname, "score": score,
                    "deck": decks[idx], "overage_used_s": overage.get(idx),
                    "episode": ep,
                }
                save(store)
                print(f"[{rank:>3}] {score:>7.1f}  {tname[:24]:<24} "
                      f"deck OK, used {overage.get(idx)}s of 600")
                got = True
                break
            if not got:
                store["failed"][tid] = "no-deck"
                save(store)
                print(f"[{rank:>3}] {score:>7.1f}  no deck recovered")
        except Exception as e:
            store["failed"][tid] = f"{type(e).__name__}: {e}"[:120]
            save(store)
            print(f"[{rank:>3}] error {type(e).__name__}: {e}")

    report(store)
    return 0


def report(store):
    sys.path.insert(0, os.path.join(WORK, "lib"))
    from cg.api import all_card_data
    cards = {c.cardId: c for c in all_card_data()}
    teams = store["teams"]
    print(f"\n{len(teams)} decks recovered, {len(store['failed'])} failed\n")

    # archetype = the ex/megaEx Pokemon in the list
    arch = Counter()
    times = []
    for t in teams.values():
        cnt = Counter(t["deck"])
        key = []
        for cid, n in cnt.items():
            c = cards.get(cid)
            if c and int(c.cardType) == 0 and (c.megaEx or c.ex):
                key.append(f"{c.name}x{n}")
        arch[" + ".join(sorted(key)) or "(no ex)"] += 1
        if t.get("overage_used_s") is not None:
            times.append(t["overage_used_s"])
    print("ARCHETYPES (by ex/Mega-ex line):")
    for k, v in arch.most_common(20):
        print(f"  {v:>3}x  {k}")
    if times:
        times.sort()
        print(f"\nOVERAGE USED of 600 s: min {times[0]:.1f}  "
              f"median {times[len(times)//2]:.1f}  max {times[-1]:.1f}")
        print(f"  teams using <60 s: {sum(1 for t in times if t < 60)}/{len(times)}")


if __name__ == "__main__":
    sys.exit(main())
