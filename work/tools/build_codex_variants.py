"""Derive improvement variants from the adopted public base `p1_codex`.

`p1_codex` is jazivxt's "Codex Sol Eclipse Alakazam v22", extracted verbatim
from their public notebook. Their leaderboard rank is 121/6321 (988.8) where
ours is 2343 (697.7), and the bundle beats our champion `v51_roman_safe`
0.7583 over 240 games. It is the new base.

Variants are produced by SOURCE PATCHING rather than by hand-editing a copy, so
that (a) every difference from the base is stated in one place and reviewable,
and (b) re-deriving from an updated base is one command. Each patch asserts its
anchor text was found -- a silently-missed patch would ship the base under a new
name and pool its games with the variant's cell in the gauntlet store, which is
the one failure this project has paid for repeatedly.

  python work/tools/build_codex_variants.py --variant meta
  python work/tools/build_codex_variants.py --variant budget --episode-s 300
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
AGENTS = os.path.join(WORK, "agents")
BASE = "p1_codex"


def patch(src, old, new, what):
    """Textual replace that refuses to no-op."""
    if old not in src:
        raise SystemExit(f"PATCH FAILED ({what}): anchor not found:\n{old[:200]}")
    if src.count(old) != 1:
        raise SystemExit(f"PATCH FAILED ({what}): anchor found "
                         f"{src.count(old)}x, expected exactly 1")
    return src.replace(old, new)


# --------------------------------------------------------------------- meta
# The base loads opponent decklists from a `top20_decks/` directory that is not
# in the bundle, so `_TEMPLATES` is empty; `_TEMPLATE_SIG` is then rebuilt on
# EVERY agent call and contains a template only when a Grimmsnarl or Great Tusk
# line is already visible. Against everything else -- the Alakazam mirror,
# Archaludon, Crustle, Mega Lucario, which together are most of the field --
# `_sample_hidden` falls back to "most-seen visible card x30 + basic energy
# x30", i.e. it determinizes the opponent as a deck that cannot exist.
#
# This is the same defect we already found and fixed in our own search: the old
# opponent library had 0/31 Archaludon and 0/31 Cinderace lists, and replacing
# it moved `pimc_terminal` from 0.4182 to 0.5000 against v51. We already have
# the replacement -- work/lib/meta_decks.py, 70 lists built by
# build_meta_from_replays.py out of decks we have actually faced on the ladder.
#
# Shipped INLINE rather than as a data file: the base resolves `top20_decks`
# relative to cwd, and cwd-relative model loading has silently degraded an agent
# in this repo before (HANDOFF gotcha #8). A literal cannot fail to load, and it
# is covered by the gauntlet's content hash of main.py.
META_MATCHER = '''def _match_archetype(op_seen):
    """Pick the template most consistent with every card we have SEEN them play.

    Replaces an overlap count over Pokemon ids that accepted a template on a
    single shared Basic. Scoring the whole multiset and charging 2.5 for each
    copy the candidate cannot account for is the rule already validated in
    work/lib/fsearch.py::match_opponent_deck -- a list that cannot contain a
    card we have literally watched them play is not their list. Requiring a
    score of 4 before trusting a match keeps the old "unknown -> fall back"
    behaviour for the opening turns, when nothing is known yet.
    """
    if not op_seen or not _TEMPLATE_SIG:
        return None
    best, best_s = None, 0.0
    for _name, _sig, cnt, ids in _TEMPLATE_SIG:
        ok = miss = 0
        for cid, n in op_seen.items():
            avail = cnt.get(cid, 0)
            ok += min(n, avail)
            miss += max(0, n - avail)
        s = ok - 2.5 * miss
        if s > best_s:
            best, best_s = (cnt, ids), s
    return best if best_s >= 4.0 else None
'''


def build_meta(src):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "meta_decks", os.path.join(WORK, "lib", "meta_decks.py"))
    md = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(md)
    decks = [list(d) for d in md.DECKS if len(d) == 60]
    if len(decks) < 20:
        raise SystemExit(f"meta_decks.py only has {len(decks)} 60-card lists")

    lit = "_META_DECK_IDS = [\n"
    for d in decks:
        lit += "    " + repr(d) + ",\n"
    lit += "]\n"
    lit += ("_META_TEMPLATE_SIG = [\n"
            "    ('meta_%d' % _i, _pokemon_ids(Counter(_d)), Counter(_d), _d)\n"
            "    for _i, _d in enumerate(_META_DECK_IDS)\n"
            "]\n")

    # 1. replace the matcher
    old_matcher_start = 'def _match_archetype(op_seen):'
    i = src.index(old_matcher_start)
    j = src.index('\ndef _sample_hidden', i)
    src = src[:i] + META_MATCHER + src[j + 1:]

    # 2. inject the library right after the Tusk template block, which is the
    #    first point where _pokemon_ids and Counter are both already defined
    anchor = ("_GRIMM_LINE = {646, 647, 648}\n")
    src = patch(src, anchor, lit + "\n" + anchor, "inject meta templates")

    # 3. make the per-call rebuild include them
    src = patch(src,
                "    _TEMPLATE_SIG = templates\n    return _base_agent(obs_dict)",
                "    _TEMPLATE_SIG = templates + _META_TEMPLATE_SIG\n"
                "    return _base_agent(obs_dict)",
                "activate meta templates")
    return src


# --------------------------------------------------------------------- safe
# The base leaves `my_deck = []` at import and fills it only on the setup frame
# (select == None), resolving deck.csv from `__file__` (undefined under the
# harness's exec), then cwd, then /kaggle_simulations/agent/. On Kaggle the
# absolute path always exists, which is why the author's own submission is fine.
# Anywhere else -- our gauntlet, the gate's foreign-cwd probe, any tool that
# does not chdir into the bundle -- the fallback is `list(my_deck)`, i.e. an
# EMPTY deck, and `_sample_hidden` then determinizes our own deck as 60 filler
# energy for the whole game.
#
# So: resolve at import, bundle -> cwd -> inlined constant, all three agreeing.
# This is the same fix v51_roman_safe applied to the roman950 base, and it is
# the one that makes local measurement mean anything. Play on Kaggle is
# unchanged: the setup frame still returns the same 60 ids.
SAFE_LOADER = '''my_deck = []


def _codex_load_deck():
    """Bundle first, cwd second, constant last -- all three agree.

    Added on adoption. The base resolved the decklist only inside the setup
    frame, so every harness that does not chdir into the bundle ran it with an
    empty `my_deck`, and its own-deck determinization degraded to filler energy.
    """
    for _p in ("/kaggle_simulations/agent/deck.csv", "deck.csv"):
        try:
            with open(_p, "r", encoding="utf-8") as _f:
                _d = [int(_x) for _x in _f.read().split() if _x.strip()]
            if len(_d) == 60:
                return _d
        except Exception:
            pass
    return list(_CODEX_DECK)


_CODEX_DECK = {deck!r}
my_deck = _codex_load_deck()
'''


def build_safe(src):
    with open(os.path.join(AGENTS, BASE, "deck.csv"), encoding="utf-8") as fh:
        deck = [int(x) for x in fh.read().split() if x.strip()]
    if len(deck) != 60:
        raise SystemExit(f"base deck.csv has {len(deck)} cards, expected 60")
    return patch(src, "my_deck = []\n", SAFE_LOADER.format(deck=deck),
                 "import-time deck resolution")


# ------------------------------------------------------------------- budget
# The base spends TIME_BUDGET_S = 0.80 s per decision and ~6.2 s per episode.
# The harness allows 600 s per EPISODE (cabt.json actTimeout=0,
# remainingOverageTime=600), so 99% of the compute is idle. Budget scaling is
# verified on our own search: the same agent at 90 s beat an identical copy at
# 22 s, 0.600 over 60 games.
#
# The budget is WALL-CLOCK, so slower hardware simply performs fewer
# determinizations and can never overrun the limit. The `remaining / 12`
# throttle makes the spend decay geometrically, so the episode total is bounded
# by _EPISODE_BUDGET_S by construction no matter how many decisions occur.
BUDGET_GUARD = '''
# ---- episode budget guard (added by build_codex_variants.py) ----
_EPISODE_BUDGET_S = {episode_s}
_episode_spent = 0.0


def _episode_budget_reset():
    global _episode_spent
    _episode_spent = 0.0


def _episode_budget_slice():
    """Seconds this decision may spend. Geometric decay bounds the episode."""
    remain = _EPISODE_BUDGET_S - _episode_spent
    if remain <= 2.0:
        return 0.0
    return max(0.05, min(TIME_BUDGET_S, remain / 12.0))
'''


def build_budget(src, n_det, time_budget_s, episode_s):
    src = patch(src, "N_DET = 3  ", f"N_DET = {n_det}  ", "N_DET")
    src = patch(src, "TIME_BUDGET_S = 0.80",
                f"TIME_BUDGET_S = {time_budget_s}", "TIME_BUDGET_S")

    # place the guard just before _search_decide
    src = patch(src, "# ==================== 2-ply minimax ====================",
                BUDGET_GUARD.format(episode_s=episode_s)
                + "\n\n# ==================== 2-ply minimax ====================",
                "budget guard")

    # spend a slice, not the flat constant
    src = patch(src,
                "    t0 = time.monotonic()\n    deadline = t0 + TIME_BUDGET_S",
                "    _slice = _episode_budget_slice()\n"
                "    if _slice <= 0.0:\n"
                "        return None\n"
                "    t0 = time.monotonic()\n    deadline = t0 + _slice",
                "use budget slice")

    # account for what was actually spent
    src = patch(src,
                "        elapsed = time.monotonic() - t0\n        _stats[\"calls\"] += 1",
                "        elapsed = time.monotonic() - t0\n"
                "        global _episode_spent\n"
                "        _episode_spent += elapsed\n"
                "        _stats[\"calls\"] += 1",
                "account spend")

    # a crashed/aborted search still consumed wall clock; charge for it too, or
    # a pathological position could be retried every decision for free
    src = patch(src,
                "    except Exception as e:\n        _stats[\"fail\"] += 1",
                "    except Exception as e:\n"
                "        _episode_spent += time.monotonic() - t0\n"
                "        _stats[\"fail\"] += 1",
                "account spend on failure")

    # reset at the start of every episode -- the base already detects this
    src = patch(src,
                "        _search_ok = _SEARCH_IMPORT_OK  # new game\n        return my_deck",
                "        _search_ok = _SEARCH_IMPORT_OK  # new game\n"
                "        _episode_budget_reset()\n"
                "        return my_deck",
                "reset per episode")
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["safe", "meta", "budget", "meta_budget"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--n-det", type=int, default=24)
    ap.add_argument("--time-budget-s", type=float, default=6.0)
    ap.add_argument("--episode-s", type=float, default=300.0)
    args = ap.parse_args()

    with open(os.path.join(AGENTS, BASE, "main.py"), encoding="utf-8") as fh:
        src = fh.read()
    before = len(src)

    # Always applied: every variant is meant to be measurable and shippable,
    # and without it local play runs on an empty own-deck belief.
    src = build_safe(src)
    if "meta" in args.variant:
        src = build_meta(src)
    if "budget" in args.variant:
        src = build_budget(src, args.n_det, args.time_budget_s, args.episode_s)

    name = args.name or {"safe": "v61_codex_safe",
                         "meta": "v62_codex_meta",
                         "budget": "v63_codex_budget",
                         "meta_budget": "v64_codex_full"}[args.variant]
    out = os.path.join(AGENTS, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "main.py"), "w", encoding="utf-8") as fh:
        fh.write(src)
    shutil.copy(os.path.join(AGENTS, BASE, "deck.csv"),
                os.path.join(out, "deck.csv"))

    compile(src, "main.py", "exec")          # syntax gate before anything runs
    print(f"built work/agents/{name}  ({before} -> {len(src)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
