# PTCG AI Battle — agent guide

Kaggle simulation ladder `pokemon-tcg-ai-battle`. Goal: raise our ladder rating.
Read this before running experiments. Rules first, rationale after — the
rationale is what lets you generalise when a case is not covered.

## 0. NEVER submit without explicit per-occasion approval

The user calls every submission. Local measurements are not grounds to submit.
5 submissions/day, resets 00:00 UTC. **The board shows the MAX of the two
ACTIVE submissions, and each new submission evicts the OLDER active.** So a good
draw survives only while you stop submitting — two more submissions destroy it.

## 1. Statistical power — the rule that invalidated a whole day of work

At n games per arm the smallest detectable win-rate difference (95%, 80% power)
is `sqrt(2*(1.96+0.84)^2*0.25/n)`:

| n per arm | smallest detectable effect |
|---|---|
| 186 | **0.145** |
| 400 | 0.099 |
| 1000 | 0.063 |
| 4356 | 0.030 |

**We need +0.114 win rate to reach a 1000 rating** (see §2). So:

* **Only test ideas that could plausibly deliver ≥0.10 win rate.** Anything
  smaller is both undetectable at sane sample sizes and insufficient to matter.
* **n ≥ 300 per arm** for any comparison you will act on. n=186 can only see
  effects of 0.145+.
* A "+0.03 improvement" at n=186 is indistinguishable from zero. Screening many
  variants at that n produces a false positive **by construction** — this
  happened: 46 preference-list mutations, one "accept" at 1.7 sigma, which then
  measured *worse* on a full panel.
* Screen and confirm must be **different measurements**, not the same one
  bigger. A larger rerun against the screening opponent reproduces the overfit.

## 2. Measure LIVE, not local. The local panel does not rank agents.

`work/tools/calibrate_instrument.py` correlates local field score against real
ladder results for every agent we have both for:

    Spearman(local field, live mean) = +0.50 over 5 agents   (not significant)
    Spearman(local field, live max)  = +0.10

It separates a *bad* agent from good ones and carries no demonstrated signal
among the ones we actually choose between. Concretely: local ranked `_sub_v28`
**+15.9 rating** over `w34_koroll`; across 144 real episodes they were 0.528 vs
0.500 — indistinguishable.

**`work/tools/live_winrate.py` is the arbiter.** It reads
`competitions.EpisodeService/ListEpisodes` (KGAT bearer token from
`~/.kaggle/access_token`) and gives the real win rate per submission, the rating
of opponents actually faced, and the trajectory. Current state:

    w34_koroll: 127 episodes, 0.5276, opponents median 899 -> implied ~918
    to be rated 1000 against that field we must win 0.641 -> **gap +0.114**

A fresh submission starts at 600 and climbs; ours read **1005.3** at ~10
episodes and settled near 900. **Under ~25 decided episodes a score means
nothing.**

## 3. Reading our own replays

`work/out/our_replays` spans **three different decks of ours** (91 Mega Lucario,
35 Alakazam, 29 Grimmsnarl). Always filter to the deck we currently ship:

    sorted(deck_of(steps, me)) == our_60()

`work/tools/our_field.py` now enforces this. Classifying only the OPPONENT
produced two confident false findings in one session ("local opponents are
strawmen", "our Grimmsnarl ex reaches play 38% vs their 97%" — really 92% vs
100%). **An impossible number is a broken measurement, not a discovery.**

## 4. Commands

```bash
.venv/Scripts/python.exe work/tools/gauntlet.py --agents A,B --games 400 --workers 6
.venv/Scripts/python.exe work/tools/field_now.py --agents A,B      # corrected weights, reads cells only
.venv/Scripts/python.exe work/tools/live_winrate.py --submission <id>
.venv/Scripts/python.exe work/tools/build_and_gate.py --agent A    # 27 checks, produces the tarball
.venv/Scripts/python.exe work/tools/our_field.py                   # real mix, filtered to our deck
```

* `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` — the repo path contains `é` and the
  console codepage will crash tools that print it.
* **≤6 workers** (8 cores). Never run two gauntlets on the same PAIR at once.
* Gauntlet cells are content-hashed per bundle, so a mutated bundle can never
  pool with its parent. `HARNESS` in `gauntlet.py` is part of the hash — bump it
  when the way we DRIVE agents changes, never otherwise.

## 5. Refuted — do not redo (each with a measurement)

| idea | result |
|---|---|
| Playout search, any leaf | Closed on a clone AND a non-clone opponent. Greedy leaf 0.3025, trained value-net leaf 0.4216, vs 0.521/0.5467 baselines. `MARGIN=1000` is a deliberate, correct muzzle |
| Cloning pilot ACTIONS | No skill: 0.2452 vs 0.2154 random on frames where the answer is not option 0. Holds with the author's own 262-feature pipeline |
| Preference-list tuning | 46 lists, 58 rounds, one accept — a false positive |
| Removing override layers | Both live layers load-bearing (-0.077, -0.175); the other five never fire |
| Matchup routing | +3.7 at best; a mirror-routed hybrid tied its parent |
| Majority vote over 3 policies | Mirror 0.4798 — worse than its best member (0.5530) |
| The bundle's dedicated mirror expert | 0.4866, worst of three |
| Hero's Cape / deck tech | ACE SPEC: exactly one per deck, and the slot is Unfair Stamp |

Value cloning ("is this position winning") DOES work — AUC 0.7430 vs a 0.6308
prize-difference baseline (`work/lib/valnet.py`) — it just fails as a search
leaf. It has never been tried inside the policy.

## 6. Architecture facts that cost hours to establish

* `w34_koroll`, `_sub_v28`, `_sub_handwritten_v26` and `w5_grimmsnarl` are **the
  same 187 files**, differing only in `main.py`. So every policy already ships in
  whatever we submit; combining them needs no vendoring and cannot collide.
* Therefore `w34_koroll` vs `w5_grimmsnarl` is **near self-play** — ~0.50 there
  is structural. `_sub_*` bundles bypass the ensemble, so their mirror cells are
  genuine.
* Two agents in one process collide: several bundles ship a module named
  `policy_features`, and the second `import` is a silent no-op that binds the
  FIRST agent's deck. Evict `sys.modules` entries under the agent dir between
  loads.
* The engine exposes **no seed**, so paired comparisons (common random numbers)
  are impossible — this is why variance is high and n must be large.
* The deck is solved: 13 of 16 top-50 Grimmsnarl teams run our byte-identical
  60. The gap is piloting, not cards.
