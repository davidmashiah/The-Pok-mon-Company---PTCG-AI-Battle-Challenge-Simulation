# HANDOFF — build the distilled-rollout search agent

**Competition:** `pokemon-tcg-ai-battle` (Kaggle simulation ladder). Deadline **2026-08-16**.
**Where we are:** best submission **780.0**, rank ~**1080 / 1360**. Leader **1244**. Top-20 needs ~**1130**.
**Your job:** the one remaining idea with a plausible route to +250 rather than +30. It is spelled
out in §5. Everything before that is what you need so you don't re-derive or re-refute it.

Read §1 and §2 first. They are what makes every number in this repo trustworthy or not.

---

## 1. THE MEASUREMENT INSTRUMENT (read before running anything)

**The historical failure:** every local metric in this project was our own ~700-level policy
piloting BOTH sides. Against opposition that weak, almost anything wins ~71% and real
differences are invisible or inverted. Two metrics pointed the *wrong way* on a submission that
scored 389 on the ladder.

**The fix, and it works:** benchmark against **real published agents** already extracted here:

| dir | what it is |
|---|---|
| `work/agents/z_roman950` | the public LB-950 baseline (`romanrozen v10`) our whole line was forked from |
| `work/agents/w1_alakazam` | the published 5th-place Alakazam agent |
| `work/agents/w2_archaludon` | published Archaludon agent |
| `work/agents/s_dragapult`, `s_mega` | sample agents, weaker |

More notebooks sit in `public/` (a 1300-rated Starmie writeup — **markdown only, no code**).

```bash
python work/tools/gauntlet.py --agents <candidate>,<opponent> --games 200 --workers 5
python work/tools/gauntlet.py --report
```

The store is content-hashed per bundle **including `fsearch.py` and `meta_decks.py`**, so editing
either starts a fresh cell instead of silently pooling old and new results. Do not remove that.

### Rules that are not negotiable
- **n≥200 for any head-to-head you intend to act on.** At n=45 the CI is ±0.14. `v57`'s headline
  0.600 is n=45 — it is *promising*, not established.
- **Confirmation on disjoint seeds, always.** `work/tools/evolve.py` rejected ~8 of every 10
  candidates that screened better. FINDINGS §11: a screen-only optimiser here reported 13
  improvements and all 13 were false.
- **≤6 python worker processes total** (8 cores). And note `pkill -f` does NOT work on this
  Windows/Git-Bash setup — use PowerShell `Stop-Process`, and verify with
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`. A forgotten background job silently
  starved a wall-clock-budgeted agent and produced a 0.250 reading that was really 0.550.
- **Anything time-budgeted must be benchmarked with the machine otherwise idle**, for the same reason.

### Ladder noise
Byte-identical bundles have read 64, 115 and 62 points apart. **<100 points is noise** unless
repeated. A submission means nothing under ~25 episodes. Two identical v51 draws currently sit at
**780.0** and **699.2**.

---

## 2. SUBMISSION MECHANICS

- **5/day**, resets 00:00 UTC. **Only the latest 2 are active**; a new one evicts the older.
- **STANDING USER INSTRUCTION: do not submit anything that is not a clear, large improvement.**
  The user was explicit — shipping more ~750-scoring agents is worthless. Submit only when a
  candidate beats `v51_roman_safe` convincingly at n≥200.

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
export PYTHONIOENCODING=utf-8      # the é in the repo path crashes the CLI otherwise
python work/tools/build_and_gate.py --agent <name> --games 10     # MANDATORY, 26 checks
./.venv/Scripts/kaggle.exe competitions submit -c pokemon-tcg-ai-battle \
    -f work/out/<name>.tar.gz -m "<description> [uid=...]"
```

The gate has caught real submission-killers, including a move that took **1,089,510 ms** and a
bundle that shipped one deck and played another. Never skip it.

---

## 3. CURRENT STATE

| submission | score | eps | win rate | what |
|---|---|---|---|---|
| v51_roman_safe draw 1 | **780.0** | 48 | 0.604 | **champion** |
| v51_roman_safe draw 2 | 699.2 | 44 | 0.568 | same bundle, second draw |
| v57_pimc_full (active) | 722.8 | 26 | 0.577 | playout search, still converging |

`v51_roman_safe` = the public LB-950 agent + exactly two safety fixes (a loose anti-hang Ability
backstop that never binds in normal play, and a deck-loader fix). **Everything this project added
between v14 and v43 was a net regression worth about −100 points.** Do not reintroduce it.

### Local head-to-heads vs v51 (the yardstick)
```
v57_pimc_full   0.6000  (27-18,   n=45)   <- playout search, 280 s budget
v55_evolved     0.5213  (208-191, n=399)  evolved weights+deck
v56_evolved2    0.5050  (202-198, n=400)  evolved with self-play panel
v53_pimc        0.5000  (30-30,   n=60)   playout search, 90 s budget
v54_dmg         0.4770  (114-125, n=239)  damage-model fixes
v52_order       0.4250  (85-115,  n=200)  abilities ranked below attach
v53_pimc(old)   0.4182  (92-128,  n=220)  playout search, BROKEN deck library
v59lo_halving   0.3833  (23-37,   n=60)   playout search + sequential halving
```

---

## 4. REFUTED — DO NOT REDO ANY OF THESE

Each was measured against real opponents with confirmation. Re-running them wastes days.

| idea | result |
|---|---|
| Deck archetype change | **Ours is the best deck in the field.** From 13,444 real player-vs-player games: Mega Lucario ex 0.669 whole-ladder, vs Lopunny 0.882/51, vs Ogerpon 0.842/19. Next best archetype 0.619. The #1 player (1244) plays it. `work/tools/deck_choice.py` |
| Deck composition tuning | ~50 rounds of confirmed hill-climbing found nothing (0.5050 head-to-head) |
| Policy weight tuning | same run, same answer. The 16 weights are at a local optimum |
| Self-play evolution (SELF on the panel) | panel fitness rose 0.476→0.512, head-to-head **0.5050**. The gain was selection noise |
| Sequential halving in the search | 0.3833. At 15 playouts the estimate is ±0.25, so it discards the true best |
| Lethal forward search | 0.4615 over **299** games, re-tested *after* the deck-library fix. Dead |
| Learned value net as search leaf | 0.3667. (The old test was a fake null — the model silently failed to load; fixed, then it lost honestly) |
| Behavioural cloning / DouZero action-scorer | 389.3 and 400.1 on the ladder |
| Damage-model correctness fixes | per-attack KO-prediction error 11.8%→4.2%, phantoms 0 — **and no win-rate gain** (0.4770). Correctness ≠ win rate here |
| Judge / Wally's Compassion tech cards | Judge helps only vs Alakazam and only on the weaker base (0.430 on the strong base). Wally's 0.509 with the branch confirmed firing |
| Ranking Abilities below the energy attach | 0.4250 |
| Two charged Mega Lucarios to beat the Mega Brave lockout | infeasible: locked on only 25% of charged turns, backup ready on 15% of those, and retreating a Mega discards 2 energy with only 2 Switch in deck |

### Two facts that overturned earlier assumptions — keep them
- **The meta model in the old handoff was wrong.** From our OWN replays: Grimmsnarl is 7.4% of our
  games and we win 0.818. The real spread is mirror 23.5% / Alakazam 18.8% / Crustle 15.4% /
  Archaludon 9.4%. Everything from v32–v37 optimised a matchup we already dominate.
  Re-run `work/tools/loss_autopsy.py --fetch` after any submission; the band shifts with the score.
- **Against teams actually on the leaderboard we win ~0.34**; the headline ~0.55 is inflated by
  unranked/inactive opponents.

---

## 5. THE TASK: distilled rollout policy

### Why this and not something else
`work/lib/fsearch.py::pimc_terminal()` ranks candidate actions by **playing the game out to a
terminal state** many times and counting real wins. No evaluator — the engine reports the winner.
This is the only thing that has beaten the champion (0.600, n=45), and it is the only approach left
whose strength scales with something we control.

Two bugs had to be fixed before it worked at all, both instructive:
1. **The opponent-deck library was fake.** `Determinizer` pads with filler when it cannot match the
   opponent, and the old library had **0/31 Archaludon and 0/31 Cinderace** lists. Every lookahead
   against the decks we lose to most simulated a game against a deck that does not exist. That
   single fix moved it 0.4182 → 0.5000. Rebuilt by `work/tools/build_meta_from_replays.py` → 70
   decks. **Re-run it whenever you download more replays.**
2. **Deferred cleanup.** Releasing playout search states only at the end of a call left tens of
   thousands live and the engine crawled — one move took 1,089,510 ms. Each playout's chain is now
   released as it finishes. Don't undo that.

### The actual ceiling, and the idea
Both sides of every playout are piloted by `AdvancedPolicy`, the ~700-level heuristic. So the search
converges on *"the best move assuming both players continue to play badly."* That is the ceiling.

**Make the rollout policy stronger than the heuristic, cheaply.** The searched agent is much better
than the heuristic — but you cannot call it recursively (67 engine steps per playout, 0.61 ms each).
So distil it:

1. **Generate data.** Run the search agent (`work/agents/v57_pimc_full`) and log, at every MAIN
   frame it searches, the full option list plus the search's final ranking. `work/lib/dzfeat.py` is
   the existing single featuriser and serves both replay dicts and engine objects — **do not fork
   it**. There is a working extraction pipeline to copy the shape of in `work/tools/dz_extract.py`.
2. **Train a small policy** to predict the search's top choice from the position. It must be
   numpy-only at runtime — `work/lib/dznp.py` / `vznp.py` show the pattern (torch is NOT available
   in the submission environment; weights ship as `.npz`).
3. **Budget check before you commit to it.** The rollout policy currently costs **0.09 ms/call**
   against the engine's 0.61 ms/step — it is 12% of playout time. A net that costs more than
   ~0.3 ms/call will cut playout throughput enough to lose more than it gains. Measure this first;
   it may kill the idea cheaply.
4. **Plug it in** as `rollout_policy` in `pimc_terminal`, keeping the heuristic as fallback.
5. **Iterate.** Search-with-net-rollouts is stronger than search-with-heuristic-rollouts → log it
   → retrain → repeat. This is the loop that removes the ceiling.

### Cheaper lever to try first (an hour, not a day)
Budget scales and is verified: the same agent at 90 s beats an identical copy at 22 s, **0.600/60**.
`v57` uses **280 s of a 600 s pool**; `v51` uses 0.4 s. The budget is wall-clock, so slower hardware
simply does fewer playouts and **can never overrun the limit**. Going to ~450 s is safe and is worth
maybe +0.04. Do this while the distillation work runs.

### How to validate anything you build
```bash
python work/tools/build_and_gate.py --agent <name> --games 6
python work/tools/gauntlet.py --agents <name>,v51_roman_safe --games 200 --workers 3
```
Budgeted agents are slow (~280 s/game). Use 3 workers, nothing else running. Expect ~2 h for n=200.
**Do not submit on n=45.**

---

## 6. TOOLING

| tool | what it does |
|---|---|
| `build_and_gate.py` | 26 pre-submission checks + tarball. Includes a deck-identity stage that exec-loads from a *foreign* cwd and fails if the deck shipped ≠ the deck played |
| `gauntlet.py` | accumulating, content-hashed head-to-head. The yardstick |
| `loss_autopsy.py` | downloads our own ladder replays, reports the meta we actually face by archetype and rating band. **This produced the two biggest findings in the project** |
| `deck_choice.py` | archetype-vs-archetype win rates from 13,444 real games |
| `build_meta_from_replays.py` | rebuilds the opponent-deck library from decks we actually face |
| `evolve.py` | hill-climbs weights + deck against a real-agent panel, with confirmation |
| `damage_model_audit.py` | per-attack KO-prediction accuracy (~500 attacks/120 games — resolves where win rates cannot) |
| `matchup_autopsy.py` | how a matchup is lost, prize by prize |
| `liveness_check.py` | asserts each new branch actually executed. **Five shipped components here were silently dead** |
| `megabrave_uptime.py`, `ppp_stack_check.py`, `gravity_check.py`, `hand_disruption_check.py` | mechanical checks |

### Verified card mechanics (in-engine, not inferred)
- **Premium Power Pro STACKS**: 0/1/2 copies → **270 / 300 / 330** damage. 330 one-shots
  Grimmsnarl ex at full 320 HP. Three copies still read 330 — treat +60 as the cap.
- **Gravity Mountain** really is −30: Grimmsnarl reads 290 with it out, 320 without.
- **Alakazam's Powerful Hand** places 2 damage counters per card in THEIR hand — 20 damage/card as
  counters, so Weakness and our 340 HP are irrelevant. In 19 of 28 games their hand alone was lethal.

---

## 7. GOTCHAS THAT COST REAL TIME

1. **`__file__` is undefined** — the harness `exec()`s `main.py`. The gate bans it.
2. **The last callable in the module dict wins** — `kaggle_environments` takes it by dict position.
3. **A game is over when `o.current.result != -1`**, not when `select is None`.
4. **`AdvancedPolicy.choose()` returns `ranked[:maxCount]`, and maxCount is 1 on MAIN.** Any search
   fed from it receives ONE candidate and silently does nothing. Rebuild the full ranking from
   `_score_option`. This has now bitten two separate components.
5. **Silent fallbacks are the recurring failure mode.** Assert every component actually ran.
6. **Give every parallel worker its own cwd** — older agents write `deck.csv` at import and clobber
   each other mid-read.
7. **`pkill -f` does not work here.** Use PowerShell `Stop-Process` and verify.
8. **Model weights must resolve from the repo root** — `dznp`/`vznp` look for `.npz` relative to cwd,
   so `chdir`-ing into the agent dir makes the model silently fail to load and the agent degrades to
   the plain heuristic. Assert the load flag.

---

## 8. GIT

Repo: https://github.com/davidmashiah/The-Pok-mon-Company---PTCG-AI-Battle-Challenge-Simulation
Commit as **DavidMashiah** (`user.name`/`user.email` are already set locally). **No co-author
trailers** — the user asked for their name only. `data/`, `public/`, `work/out/` and the engine
binaries are gitignored.
