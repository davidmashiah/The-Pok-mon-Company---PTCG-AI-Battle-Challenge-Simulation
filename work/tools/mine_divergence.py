"""Mine where our policy disagrees with WINNING play, at scale.

The host publishes ~4,500 top-rated episodes per day (median participant rating
~1085 versus our ~700). Each replay stores, for every step, the exact
observation an agent saw and the action it chose. Replaying the WINNER's
decisions through our policy tells us precisely where we differ from play that
is measurably better than ours.

This replaces hand-reasoning about what good play looks like with counting.

Streams the zip rather than extracting 700 MB, and never stores observations --
it runs our policy inline and tallies.

Usage:
  python work/tools/mine_divergence.py <agent> <n_episodes>
"""
import io
import json
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)
ZIP = os.path.join(ROOT, "data", "episodes", "d0802",
                   "pokemon-tcg-ai-battle-episodes-2026-08-02.zip")

AGENT = sys.argv[1] if len(sys.argv) > 1 else "v14_search_noloop2"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 300
# card ids that identify OUR archetype; only score episodes the winner played
# with the same core, so disagreement reflects play quality not deck choice.
MARKERS = set()
if len(sys.argv) > 3 and sys.argv[3] != "-":
    MARKERS = {int(x) for x in sys.argv[3].split(",")}

sys.path.insert(0, os.path.join(WORK, "lib"))
from cg.api import all_card_data  # noqa: E402
CARDS = {c.cardId: c for c in all_card_data()}

full = os.path.join(WORK, "agents", AGENT)
sys.path.insert(0, full)
cwd = os.getcwd()
os.chdir(full)
env = {}
exec(compile(open("main.py", encoding="utf-8-sig").read(), "main.py", "exec"), env)
os.chdir(cwd)
FN = [v for v in env.items() if callable(v[1])][-1][1]

OPT = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD",
       5: "ENERGY_CARD", 6: "ENERGY", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE",
       10: "ABILITY", 11: "DISCARD", 12: "RETREAT", 13: "ATTACK",
       14: "END", 15: "SKILL", 16: "SPECIAL_CONDITION"}

total = agree = 0
ctx_tot = Counter()
ctx_ok = Counter()
pair = Counter()
ctx_pair = defaultdict(Counter)
episodes = 0
t0 = time.time()

with zipfile.ZipFile(ZIP) as zf:
    names = [n for n in zf.namelist() if n.endswith(".json")]
    for name in names:
        if episodes >= LIMIT:
            break
        try:
            with zf.open(name) as fh:
                d = json.loads(fh.read().decode("utf-8"))
        except Exception:
            continue
        rewards = d.get("rewards") or []
        if 1 not in rewards:
            continue                       # no clear winner
        win = rewards.index(1)
        # Comparing across archetypes measures DECK DIFFERENCE, not skill: a
        # Lucario policy asked "which Pokemon do you damage?" in a Dragapult
        # mirror has no right answer to agree with. Control run: a 5th-place
        # agent scores 25.9% here versus our 27.6%, and 4.3% vs our 3.9% on the
        # damage context -- i.e. the low numbers are a floor, not a defect.
        # Only same-archetype episodes carry signal.
        if MARKERS:
            wd = None
            for step in d.get("steps", []):
                if win < len(step):
                    a = step[win].get("action") or []
                    if len(a) == 60:
                        wd = set(a)
                        break
            if wd is None or not (wd & MARKERS):
                continue
        episodes += 1
        # give our policy the winner's own decklist so its card rules apply
        for step in d.get("steps", []):
            if win < len(step):
                a = step[win].get("action") or []
                if len(a) == 60:
                    env["my_deck"] = list(a)
                    env["DECK"] = list(a)
                    break
        for step in d.get("steps", []):
            if win >= len(step):
                continue
            ag = step[win]
            obs = ag.get("observation") or {}
            act = ag.get("action")
            if not act or not isinstance(act, list) or len(act) == 60:
                continue
            sel = obs.get("select")
            if not sel:
                continue
            opts = sel.get("option") or []
            if len(opts) < 2:
                continue                   # forced move
            try:
                ours = FN(obs)
            except Exception:
                continue
            if not ours:
                continue
            ctx = sel.get("context")
            total += 1
            ctx_tot[ctx] += 1
            if list(ours)[:len(act)] == list(act):
                agree += 1
                ctx_ok[ctx] += 1
            else:
                tt = opts[act[0]].get("type") if act[0] < len(opts) else None
                to = opts[ours[0]].get("type") if ours[0] < len(opts) else None
                k = (OPT.get(tt, tt), OPT.get(to, to))
                pair[k] += 1
                ctx_pair[ctx][k] += 1

dt = time.time() - t0
print(f"agent={AGENT}  episodes={episodes}  decisions={total}  ({dt:.0f}s)")
if total == 0:
    raise SystemExit("no decisions compared")
print(f"AGREEMENT WITH WINNERS: {agree}/{total} = {agree/total:.4f}\n")

print("worst contexts by volume x disagreement (fix these first):")
rows = []
for c, n in ctx_tot.items():
    if n < 50:
        continue
    a = ctx_ok[c]
    rows.append((n - a, c, n, a / n))
rows.sort(reverse=True)
for lost, c, n, rate in rows[:10]:
    top = ctx_pair[c].most_common(2)
    tops = "  ".join(f"{t[0]}->{t[1]}:{v}" for t, v in top)
    print(f"  ctx {c:>3}: {n:>6} decisions, agree {rate:.3f}, "
          f"{lost:>6} wrong   {tops}")

print("\nmost common disagreements overall (winner chose -> we chose):")
for (t, o), n in pair.most_common(12):
    print(f"  {n:>6}x  {t} -> {o}")
