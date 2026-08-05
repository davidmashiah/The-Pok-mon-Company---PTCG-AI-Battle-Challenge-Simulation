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
_episode_last_turn = -1


def _episode_budget_reset():
    global _episode_spent, _episode_last_turn
    _episode_spent = 0.0
    _episode_last_turn = -1


def _episode_budget_note_turn(turn):
    """Second, independent new-episode detector: the turn counter went BACKWARDS.

    The primary one is the select == None setup frame, which is how
    kaggle_environments opens an episode. But that is a property of the harness,
    not of the game, and the first version of this guard trusted it alone. Under
    a harness that plays many episodes in one process the budget was consumed
    once and the agent then silently stopped searching for every later game --
    reporting 1.3 s of agent time against an allowance of 90 and a win rate that
    looked perfectly plausible. A turn number that decreases can only mean a new
    game, whoever is driving.
    """
    global _episode_last_turn
    try:
        t = int(turn)
    except (TypeError, ValueError):
        return
    if t < _episode_last_turn:
        _episode_budget_reset()
    _episode_last_turn = t


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
                "    _episode_budget_note_turn(getattr(st, 'turn', -1))\n"
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


# ---------------------------------------------------------------- statefix
# A real defect in the adopted base. `heuristic_scores` opens with
#
#     global pre_turn, ability_used_dudunsparce, ability_used_fezandipiti
#     if pre_turn != state.turn:
#         pre_turn = state.turn
#         ability_used_dudunsparce = False
#         ability_used_fezandipiti = False
#
# and the search calls it on every rollout step through `_greedy_pick`. A single
# decision runs hundreds of those, at SIMULATED turn numbers, so the real game
# resumes with `pre_turn` set to a turn that never happened and both
# once-per-turn ability flags clobbered by a hypothetical line. The flags exist
# to stop the agent re-attempting Dudunsparce's and Fezandipiti's abilities, so
# corrupting them costs real decisions in the real game.
#
# This is the same trap our own v57 hit and guarded against explicitly. The fix
# is the same: snapshot the three globals around the search and restore them in
# the finally, so simulated play cannot leak into the live position.
STATEFIX_SAVE = '''    global pre_turn, ability_used_dudunsparce, ability_used_fezandipiti
    _sf_saved = (pre_turn, ability_used_dudunsparce, ability_used_fezandipiti)
    began = False
    try:
'''
STATEFIX_RESTORE = '''    finally:
        (pre_turn, ability_used_dudunsparce,
         ability_used_fezandipiti) = _sf_saved
        if began:
'''


def build_statefix(src):
    src = patch(src, "    began = False\n    try:\n", STATEFIX_SAVE,
                "snapshot policy globals")
    return patch(src, "    finally:\n        if began:\n", STATEFIX_RESTORE,
                 "restore policy globals")


# ------------------------------------------------------------------ margin
# The base overrides the heuristic only when the searched value beats it by
# 500 -- half a prize on a scale where a prize is 1000. That threshold was
# chosen for a search averaging THREE determinizations, where the estimate is
# noisy enough that a small edge is usually sampling error. If more
# determinizations genuinely tighten the estimate, the same threshold then
# throws away real edges, so the two knobs have to move together: measure the
# margin change ONLY on top of a budget variant, never on the stock N_DET=3.
def build_margin(src, margin):
    return patch(src,
                 "        if avg[best] < avg[heur_top] + 500.0:",
                 f"        if avg[best] < avg[heur_top] + {margin}:",
                 "override margin")


# ------------------------------------------------------------------ attacks
# The base excludes ATTACK and END from the candidate list -- "terminal actions
# reached via greedy rollout so no need to branch on them". That is right for
# END and wrong for ATTACK whenever more than one attack is legal: the greedy
# heuristic alone decides WHICH attack, and that is the decision that takes
# prizes. Branching on attacks costs one extra rollout each and lets the search
# compare them by the same 2-ply value it uses for everything else.
def build_attacks(src):
    return patch(src,
                 "        if sel.option[i].type in (OptionType.ATTACK, OptionType.END):\n"
                 "            continue",
                 "        if sel.option[i].type == OptionType.END:\n"
                 "            continue",
                 "branch on attacks")


# ----------------------------------------------------------------- playout
# The base's search stops after the opponent's reply and hands the position to
# `_leaf_eval`, a hand-written sum of prizes, HP and energy. Every search this
# project has built failed at the EVALUATOR rather than at the depth --
# 1-ply + hand-written eval 0.3500, 2-ply + the same 0.0530, 1-ply + a learned
# value net 0.3667. A playout that runs to a TERMINAL state needs no evaluator:
# the engine reports the winner, which is exactly what we are optimising.
#
# We tried this before (v57) and it did not convert: 0.570 locally, 15-15 on the
# ladder. The stated reason it stalled was that both sides of every playout were
# piloted by our own ~700-level heuristic, so the search converged on "the best
# move assuming both players keep playing badly". That is precisely what changed
# -- this base's `heuristic_scores` is a far stronger rollout policy, and the
# whole compute budget is still unspent (8 s of 600 s used on Kaggle).
#
# Release discipline is not optional here. A playout is ~100 engine steps and a
# decision runs hundreds of playouts; deferring `search_release` once left tens
# of thousands of live search states and a single move took 1,089,510 ms.
PLAYOUT_BLOCK = '''
# ---- playout search to terminal states (added by build_codex_variants.py) ----
try:
    from cg.api import search_release as _search_release
except Exception:                                        # pragma: no cover
    _search_release = None

PLAYOUT_MAX_CAND = {max_cand}
PLAYOUT_MAX_STEPS = 700          # a full game is ~100 steps; this only bounds pathology
PLAYOUT_MIN_PLAYS = {min_plays}  # per candidate before it may override anything
PLAYOUT_MARGIN = {margin}        # win-rate edge required to overrule the heuristic
_pstats = {{"calls": 0, "ran": 0, "playouts": 0, "terminal": 0, "trunc": 0,
           "considered": 0, "overrides": 0, "fail": 0, "ms": 0.0,
           # Diagnostics, so the override threshold is set from the measured
           # distribution instead of guessed. The first build ran 12,929
           # playouts and overrode 0 of 107 eligible decisions -- without these
           # there is no way to tell "the heuristic really is right" from
           # "the samples are too few for the threshold to ever be met".
           "edge_n": 0, "edge_sum": 0.0, "edge_max": 0.0,
           "edge_ge02": 0, "edge_ge05": 0, "edge_ge08": 0,
           "plays_top_sum": 0, "plays_top_min": 10 ** 9}}


def _playout_once(root_sid, a, me_i, deadline):
    """Play one determinized game to the end after taking action `a`.

    Returns the engine's result, or None if it did not finish. Every search id
    this creates is released before returning -- see the note above.
    """
    mine = []
    res = None
    try:
        ss = search_step(root_sid, [a])
    except Exception:
        return None
    if ss is None:
        return None
    mine.append(ss.searchId)
    sid, cur = ss.searchId, ss.observation
    for _ in range(PLAYOUT_MAX_STEPS):
        if time.monotonic() > deadline:
            break
        cs = cur.current
        if cs is None:
            break
        if cs.result is not None and cs.result >= 0:
            res = cs.result
            break
        if cur.select is None:
            break
        ch, _order = _greedy_pick(cur)
        if not ch:
            break
        try:
            ss2 = search_step(sid, ch)
        except Exception:
            break
        if ss2 is None:
            break
        mine.append(ss2.searchId)
        sid, cur = ss2.searchId, ss2.observation
    if _search_release is not None:
        for _sid in mine:
            try:
                _search_release(_sid)
            except Exception:
                pass
    return res


def _playout_decide(obs, base_order, base_scores):
    """Rank candidates by simulated win rate. Returns an index, or None."""
    global pre_turn, ability_used_dudunsparce, ability_used_fezandipiti
    _pstats["calls"] += 1
    if not (USE_SEARCH and _search_ok) or _search_release is None:
        return None
    st, sel = obs.current, obs.select
    if st is None or sel is None or sel.context != SelectContext.MAIN:
        return None
    n = len(sel.option)
    if n < 2 or n > SEARCH_MAX_OPTS or st.turn < 2:
        return None
    if getattr(obs, "search_begin_input", None) is None:
        return None

    # Unlike the 2-ply search, ATTACK stays in the candidate set: which attack
    # to make is the decision that takes prizes, and a playout can actually
    # price it because it plays past the end of our turn either way.
    heur_top = base_order[0]
    cand = [heur_top]
    for i in base_order[1:]:
        if sel.option[i].type == OptionType.END:
            continue
        if base_scores[i] < 0:
            continue
        cand.append(i)
        if len(cand) >= PLAYOUT_MAX_CAND:
            break
    if len(cand) < 2:
        return None

    budget = _episode_budget_slice()
    if budget <= 0.0:
        return None

    me_i = st.yourIndex
    t0 = time.monotonic()
    deadline = t0 + budget
    wins = {{a: 0.0 for a in cand}}
    plays = {{a: 0 for a in cand}}
    # Simulated play calls heuristic_scores hundreds of times at hypothetical
    # turn numbers, and it rewrites pre_turn and both once-per-turn ability
    # flags as a side effect. Snapshot them or the live game resumes with state
    # from a line that never happened.
    _saved = (pre_turn, ability_used_dudunsparce, ability_used_fezandipiti)
    began = False
    try:
        while time.monotonic() < deadline:
            hidden = _sample_hidden(st, me_i)
            try:
                ss0 = search_begin(obs, **hidden)
            except Exception:
                return None
            if ss0 is None:
                return None
            began = True
            root_sid = ss0.searchId
            for a in cand:
                if time.monotonic() > deadline:
                    break
                res = _playout_once(root_sid, a, me_i, deadline)
                _pstats["playouts"] += 1
                if res is None:
                    _pstats["trunc"] += 1
                    continue          # an unfinished game tells us nothing
                _pstats["terminal"] += 1
                plays[a] += 1
                if res == me_i:
                    wins[a] += 1.0
                elif res == 2:
                    wins[a] += 0.5
            try:
                search_end()
            except Exception:
                pass
            began = False
    except Exception:
        _pstats["fail"] += 1
        return None
    finally:
        (pre_turn, ability_used_dudunsparce,
         ability_used_fezandipiti) = _saved
        if began:
            try:
                search_end()
            except Exception:
                pass
        _pstats["ms"] += (time.monotonic() - t0) * 1000.0
        # charge the shared episode pool HERE, inside finally: an early return
        # above must not hand back free wall-clock, or a pathological position
        # would be re-searched at full budget on every decision.
        _episode_budget_charge(time.monotonic() - t0)

    if plays.get(heur_top, 0) < PLAYOUT_MIN_PLAYS:
        return None               # no honest comparison to make
    _pstats["ran"] += 1
    rate = {{a: wins[a] / plays[a] for a in cand if plays[a] >= PLAYOUT_MIN_PLAYS}}
    if len(rate) < 2 or heur_top not in rate:
        return None
    _pstats["considered"] += 1
    _pstats["plays_top_sum"] += plays[heur_top]
    _pstats["plays_top_min"] = min(_pstats["plays_top_min"], plays[heur_top])
    best = max(rate, key=lambda i: (rate[i], -base_order.index(i)))
    _edge = rate[best] - rate[heur_top]
    _pstats["edge_n"] += 1
    _pstats["edge_sum"] += _edge
    _pstats["edge_max"] = max(_pstats["edge_max"], _edge)
    for _t, _k in ((0.02, "edge_ge02"), (0.05, "edge_ge05"), (0.08, "edge_ge08")):
        if _edge >= _t:
            _pstats[_k] += 1
    if best == heur_top:
        return None
    if rate[best] < rate[heur_top] + PLAYOUT_MARGIN:
        return None
    _pstats["overrides"] += 1
    return best
'''


def build_playout(src, max_cand, min_plays, margin):
    if "_episode_budget_slice" not in src:
        raise SystemExit("playout requires the budget guard: use a *_budget variant")
    # charge helper, so the playout and the 2-ply search share one episode pool
    src = patch(src,
                "def _episode_budget_slice():",
                "def _episode_budget_charge(seconds):\n"
                "    global _episode_spent\n"
                "    _episode_spent += max(0.0, seconds)\n"
                "\n"
                "\ndef _episode_budget_slice():",
                "budget charge helper")
    src = patch(src, "\ndef agent(obs_dict):\n",
                PLAYOUT_BLOCK.format(max_cand=max_cand, min_plays=min_plays,
                                     margin=margin)
                + "\n\ndef agent(obs_dict):\n",
                "insert playout search")
    return patch(src,
                 "    pick = _search_decide(obs, base_order, base_scores)\n"
                 "    if pick is None:\n"
                 "        return fallback",
                 "    pick = _playout_decide(obs, base_order, base_scores)\n"
                 "    if pick is None:\n"
                 "        pick = _search_decide(obs, base_order, base_scores)\n"
                 "    if pick is None:\n"
                 "        return fallback",
                 "wire playout ahead of the 2-ply search")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["safe", "meta", "budget", "meta_budget",
                             "budget_margin", "budget_attacks",
                             "budget_margin_attacks", "statefix",
                             "statefix_budget", "statefix_budget_attacks",
                             "statefix_budget_playout", "budget_playout"])
    ap.add_argument("--margin", type=float, default=150.0)
    ap.add_argument("--playout-max-cand", type=int, default=5)
    ap.add_argument("--playout-min-plays", type=int, default=10)
    ap.add_argument("--playout-margin", type=float, default=0.08)
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
    if "statefix" in args.variant:
        src = build_statefix(src)
    if "budget" in args.variant:
        src = build_budget(src, args.n_det, args.time_budget_s, args.episode_s)
    if "margin" in args.variant:
        src = build_margin(src, args.margin)
    if "attacks" in args.variant:
        src = build_attacks(src)
    if "playout" in args.variant:
        src = build_playout(src, args.playout_max_cand,
                            args.playout_min_plays, args.playout_margin)

    name = args.name or {"safe": "v61_codex_safe",
                         "meta": "v62_codex_meta",
                         "budget": "v63_codex_budget",
                         "meta_budget": "v64_codex_full"}.get(args.variant)
    if not name:
        raise SystemExit("--name is required for this variant")
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
