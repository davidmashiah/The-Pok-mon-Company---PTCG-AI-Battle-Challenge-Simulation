# PTCG AI Battle Challenge — Handoff

**Competition:** `pokemon-tcg-ai-battle` (Kaggle simulation ladder). Deadline **2026-08-16**.
**Goal:** raise ladder score. Current standing ~**700–780**. Leader (Majkel1337) **1265**. Field median ~**1085**.

Read §1 and §2 before doing anything. They are the two things that cost this project the most time.

---

## 1. THE MEASUREMENT PROBLEM (most important section)

**Every local metric built so far has failed at least once, and two pointed the *wrong way*.**
Ground truth is the ladder. Nothing else.

Proven by a controlled case: `v23_dz` scored **389.3** on the ladder while local metrics said:

| metric | verdict on v23_dz | reality |
|---|---|---|
| head-to-head vs v14 (438 games) | 0.5297 → "better" | **wrong direction** |
| meta_arena vs field decks (418 games) | 0.7105 vs v14's 0.7239 → "same" | **blind** |
| replay agreement w/ strong pilot | 0.2398 vs v14's 0.1853 → "better" | **wrong direction** |
| **ladder** | — | **0.407 win rate vs v14's 0.609** |

**Why they fail:** in every local test *our own ~700-level policy pilots the opponent*. Against weak
opposition almost anything wins ~71%, so differences that matter against the ladder's ~1085 field are
invisible or inverted.

### Ladder noise is huge
Byte-identical bundles submitted minutes apart:

| pair | scores | spread |
|---|---|---|
| v14 twin A / twin B | 670.3 / 734.2 | 64 |
| v32 draw 1 / draw 2 | 778.9 / 663.9 | **115** |
| v34 / v32 redraw (identical agents — see §3b) | 754.7 / 692.2 | 62 |

Early readings are worse: twin A read **791.9 → 882.8 → 637.3 → 670.3** as episodes accumulated.

**Rules:**
- A submission means nothing under ~25 episodes. Check episode count, not just score.
- Treat **<100 points** as noise unless supported by repeated draws.
- Always submit a **same-period control** alongside a candidate.
- Local runs of **20 games are noise** (v32's attach metric read 13.8% at 20 games, 7.2% at 40). Use ≥40, and prefer mechanical checks over win rates.

---

## 2. SUBMISSION MECHANICS (easy to get wrong)

- **5 submissions/day**, resets at **00:00 UTC**.
- **Only the LATEST 2 submissions are active.** A new submission evicts the older of the two. Plan
  what you're willing to knock off *before* submitting.
- Score shown is per-submission TrueSkill (μ₀ = 600), converging as episodes play.

### Working submit path
The MCP kaggle server **caches credentials at startup** → returns `Unauthenticated` for a token added
mid-session. Use the CLI or REST directly.

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
export PYTHONIOENCODING=utf-8      # console is cp1255; the é in the repo path crashes the CLI otherwise
./.venv/Scripts/kaggle.exe competitions submit -c pokemon-tcg-ai-battle \
    -f work/out/<agent>.tar.gz -m "<description with [uid=...]>"
```

List submissions / episode counts (bearer token works on REST):
```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/competitions/submissions/list/pokemon-tcg-ai-battle"
# episodes for one submission:
#   /api/v1/competitions/submissions/<ref>/episodes
```

**Never submit an ungated bundle:**
```bash
python work/tools/build_and_gate.py --agent <name> --games 10
```
The gate exec-loads `main.py` exactly as `kaggle_environments` does and runs 24+ checks. It has caught
real, submission-killing bugs.

---

## 3. CURRENT STATE

> **2026-08-04 update — read §3b first. `v34_stadium` never played the deck it
> shipped, so its ladder result is void and §8.1 below is obsolete.**

### Live now
| submission | score | episodes | what it is |
|---|---|---|---|
| **v34_stadium** (active) | 754.7 | 36 | **VOID** — shipped 3x Gravity Mountain, played v32's list |
| v32_ppp redraw (active) | 692.2 | 34 | second draw of v32 |
| v32_ppp draw 1 | 778.9 | 42 | best single reading we've had |
| v33_value | 697.4 | 16 | learned value net re-ranker — looked flat |
| v23_dz | 400.1 | 29 | learned action-scorer — **disaster** |
| v14 baseline | 703.4 / 670.3 / 734.2 / 697.6 | — | mean ~701 |

### Honest read
- **v32 (PPP damage fix): ~+20 over v14, INCONCLUSIVE.** I earlier claimed +81 from a single draw;
  its own redraw at 663.9 refuted that. Don't repeat the mistake.
- **v34 is the best active** but sits inside v32's own spread.
- Champion by evidence is still roughly "v14 + PPP fix".

---

## 3b. 2026-08-04: the deck we shipped was never the deck we played

`main.py` (inherited from the upstream notebook, in every agent v14→v34) did at import:

```python
Path("deck.csv").write_text("\n".join(map(str, DECK)))          # module level
DECK_PATH = "deck.csv" if os.path.exists("deck.csv") else "/kaggle_simulations/agent/deck.csv"
```

It **overwrote `deck.csv` from the hardcoded `DECK` constant and then read back the file
it had just written**, so a bundled decklist could never win. `v34_stadium`'s `main.py` is
**byte-identical to v32's** (md5 `31ca757a…`); only its `deck.csv` differed, and that file
lost. Reproduced directly: loading v34 the way `kaggle_environments` does yields a deck
with **1** Gravity Mountain, not 3.

**Consequences.**
- v34 (754.7) vs v32 (692.2) was a **62-point spread between two identical agents** — a
  third noise-floor datapoint next to the 64 and 115 already in §1. It is *not* evidence
  for the deck-counter direction, and §8.1's "resolve v34 vs v32" is unanswerable as posed.
- Local deck work (`deck_opt.py`, `best_deck.py`) was **unaffected** — those inject
  `env["my_deck"]` directly rather than going through the file.
- The write also **races**: parallel workers sharing a cwd clobber each other's `deck.csv`
  mid-read (`ValueError: The deck must contain 60 cards`). Harness tools need a private cwd.

**Fixed** in `v37_combo`: the write is gone; `_load_deck()` reads bundle → cwd → `DECK`
constant, all three in agreement. `build_and_gate.py` gained a **`gate_deck_identity`**
stage that exec-loads from a *foreign* cwd and fails if deck-played ≠ deck-shipped, or if
import writes anything into cwd. The old gate could not see this: its runner `chdir`-ed
into the bundle before loading, so it validated the file `main.py` had just overwritten.
Verified the new check fails v34 and passes v37.

## 3c. Verified card mechanics (in-engine, not inferred)

| claim | verdict |
|---|---|
| Gravity Mountain puts Grimmsnarl ex at 290 | **true** — `maxHp` reads 290 with it out, 320 without |
| Mega Brave 270 + Premium Power Pro = 300 | **true** |
| **Premium Power Pro STACKS** | **true** — 0/1/2 copies → **270 / 300 / 330** damage |
| 3 copies → 360 | **not observed** — seen twice, still read 330; model caps at +60 |

**330 ≥ 320 one-shots Grimmsnarl ex at full HP with no Stadium at all.** There are two
independent routes to the one-shot, and the planner knew about neither properly.

Stadium availability is purely a copy-count question (nothing searches one out): over 30
games, 1 copy is in play in **17/30 games (60 turns)**, 3 copies in **27/30 (155 turns)**.

## 3d. The damage model was wrong on 1 attack in 8 — and v32 made it *worse*

Win rates and one-shot rates count games and are hopelessly noisy here (two *identical*
agents read 12.3% vs 6.5% one-shot rate over 40 games). `work/tools/damage_model_audit.py`
counts **attacks** instead: for every attack at their Active, did `plan.remain_hp <= 0`
match what the engine actually did?

120 games vs a real leaderboard Grimmsnarl list:

| agent | attacks | MISSED | PHANTOM | error rate |
|---|---|---|---|---|
| v14 | 544 | 74 | 3 | 14.2% |
| **v32** | 515 | 54 | **10** | 12.4% |
| v38 (fixes, v32's 1× GM deck) | 528 | 20 | 4 | 4.5% |
| **v37** (fixes + 3× GM deck) | 489 | **16** | **0** | **3.3%** |

The split says the damage-model fixes do most of the work (12.4% → 4.5%) and the extra
Gravity Mountain adds the rest (4.5% → 3.3%) — consistent with the Stadium being in play
2.6× as often, which both puts targets in range and exercises the new lookahead.

- **MISSED** — predicted no KO, engine knocked it out anyway (understates our damage).
- **PHANTOM** — predicted a KO that never came. Strictly worse: we tap out the board for
  a line that does not exist.

**v32's celebrated PPP fix cut MISSED 74→54 but tripled PHANTOM 3→10**, netting almost
nothing (14.2% → 12.4%). It added the +30 whenever a copy sat in hand, then had branches
that decline to play that copy — including `if supporterPlayed and plan.remain_hp <= 0:
return -1`, which reads the very damage the card was going to provide and concludes the
card is unnecessary. That is a plausible reason its ladder result never separated from v14.

Reproduced at 60 games (v32 16.4% / v37 2.7%) and 120 games. ~500 attacks per row, so
unlike the win-rate metrics this one has the sample size to mean something.

### Agent lineage
`v14_search_noloop2` (baseline, ~701) → `v32_ppp` (PPP damage fix) → `v34_stadium` (deck) /
`v33_value` (value net) / `v35_lunar`, `v35c_lunar`, `v36_lunar` (Lunar Cycle guards, **built + gated,
never submitted**) → **`v37_combo`** (built + gated 26/26, **never submitted**).

`v37_combo` = v32 policy + four card-text-vs-code fixes, each verified in-engine:
1. **deck loader** — bundled `deck.csv` is authoritative; no import-time write (§3b). This
   is also what finally *ships* v34's 3× Gravity Mountain list.
2. **PPP persists after it is played** — tracked from `obs.logs`, since nothing in `State`
   exposes item effects. Was live on 393/2226 MAIN frames and unmodelled on 302 of them.
3. **PPP stacks** — count copies (cap 2, = +60), not a flat +30. Planner saw 2 copies
   available on ~17% of MAIN frames.
4. **Gravity Mountain in hand counts as −30** on Stage-2 targets, but only when the engine
   is actually offering the Stadium play this turn — so it can never invent a knockout.

Plus a minor one: Solrock's Cosmic Beam ignores Weakness/Resistance and was being doubled.

All four branches confirmed to execute in play (`work/tools/liveness_check.py`) — this
project has shipped five silently-dead components, so "it compiles" is not evidence.

---

## 3e. 2026-08-05: the meta model was wrong, and there is now a valid instrument

**Two things changed everything, both from data we already had.**

**(1) The field we actually face.** `work/tools/loss_autopsy.py` downloads our OWN ladder
replays and classifies the opponent. Over 149 games:

| opponent | share | our win rate |
|---|---|---|
| mirror (Mega Lucario) | 23.5% | 0.486 |
| **Alakazam** | **18.8%** | **0.357** |
| Crustle wall | 15.4% | 0.739 |
| **Archaludon ex** | **9.4%** | **0.286** |
| Marnie's Grimmsnarl ex | **7.4%** | **0.818** |

§5's "Grimmsnarl is ~53% of the field and our worst matchup" came from a scraped index of
the *whole* ladder, which is a different population from our own matchmaking band.
Grimmsnarl is 7% of our games and we already win 82%. **v32, v33, v34 and v37 all optimised
a matchup we dominate.** That is the simplest available explanation for why none of them moved.

Also: against teams **on** the leaderboard we win **0.339** (19/56); against unranked/inactive
submissions 0.677 (63/93). The blended 0.55 is flattering. The leaderboard is **1360 teams**,
median **826**, 95th pct 1017, max 1244 — so 1000 is roughly top 7%, not mid-field. No
episodes are lost to crashes or timeouts.

**(2) Real opponents.** `work/agents/w1_alakazam` (the published 5th-place agent) and
`w2_archaludon` were already in the repo, unused. Every metric in §1 failed because *our own
policy piloted the opponent*; benchmarking against independent published agents fixes exactly
that. It produced the project's first clean local separation — and caught three of the day's
own ideas before they shipped.

### Measured against real agents (gauntlet, Wilson 95%)

| build | vs w1_alakazam | vs w2_archaludon | head-to-head |
|---|---|---|---|
| v32_ppp (old champion) | 0.470 (n=300) | 0.359 (n=220) | — |
| v38_model (damage fixes only) | 0.550 (n=160) | 0.369 (n=160) | 0.488 vs v32 |
| **v43_judge2x (SUBMITTED)** | **0.694 (n=160)** | 0.325 (n=160) | 0.494 vs v32 |
| v42 (Judge x4) | 0.750 (n=160) | 0.250 (n=160) | **0.381** vs v32 |
| v44 (+Wally's Compassion) | 0.638 (n=160) | 0.344 (n=160) | 0.509 vs v43 |
| v45 (Lunar Cycle guard) | 0.594 (n=160) | 0.288 (n=160) | 0.488 vs v43 |

`v43_judge2x` = v32 + the damage-model fixes + **2x Judge** (−2 Dusk Ball). Alakazam's
Powerful Hand places 2 damage counters per card in THEIR hand — 20 damage a card, as
counters, so Weakness and our 340 HP are irrelevant; in 19 of 28 real games their hand alone
was lethal. Judge caps it. Mechanism verified, not inferred: their hand at attack falls
12.6 → 10.3 and lethal-hand attacks 28.5% → 17.6% (`work/tools/hand_disruption_check.py`).

**Dose matters more than the idea.** Judge x4 wins the Alakazam matchup harder (0.750) and
still LOSES overall (0.381 head-to-head) because 4 copies are dead cards in the other 81% of
games. Judge x2 keeps most of the gain at no measured cost.

### Refuted today, with real opponents (do not redo)
| idea | result |
|---|---|
| Judge x4 | 0.381 head-to-head. Too many dead cards |
| Judge scored above Lillie's Determination | stole the one-supporter slot for a worse draw; cost 0.38 head-to-head |
| **Wally's Compassion x2** (heal a Mega ex) | 0.509 head-to-head, no Archaludon gain. Branch confirmed LIVE (fired 20x/24 games). Healing costs the attack turn *and* the supporter slot — a treadmill against 220-per-turn |
| **Lunar Cycle guard** (v35/v36 idea, finally testable) | 0.488 head-to-head, and *worse* vs Alakazam (0.594 vs 0.694). The largest behavioural divergence from the 1265 pilot does not convert |
| Two charged Mega Lucarios to beat the Mega Brave lockout | Mega Brave is locked on only 25% of charged turns, a charged backup exists on 15% of those, and retreating a Mega discards 2 energy — 2 Switch cannot sustain it |

### Still open
- **Archaludon (9.4% @ ~0.29)** is structural, not a conversion failure: only 7 of 134
  survivals had an unplayed Premium Power Pro, so we play the card — we simply lack damage.
  They hit 220 every turn; Mega Brave locks itself out every other turn; they give up 2
  prizes and we give up 3.
- **The mirror (23.5% @ 0.486)** is untouched and is the largest single bucket.

## 4. REFUTED — DO NOT REDO THESE

Each cost hours. All measured, not guessed.

| idea | result |
|---|---|
| **Behavioral cloning of field winners** | field-trained net scores **0.3555 top-1 on our deck vs a 0.4088 always-first baseline** — worse than trivial |
| **Cloning the #1 player's deck + policy** (Majkel1337, 65k decisions) | **0.0217** vs v14 over 599 games — total collapse |
| **Deck-matched fine-tune** | 0.5175 vs v14; ladder-equivalent, no gain |
| **Learned action-scorer overruling the policy** | v23_dz → **389.3** on ladder |
| **1-ply search w/ hand-written `evaluate()`** | v30 → 0.3500 vs v14 (600 games) |
| **2-ply + PIMC search, same evaluator** | v31 → **0.0530** — deeper made it *worse* (wrong target) |
| **Old linear value net** | AUC 0.7121 vs **prize_diff alone 0.7122** — learned nothing |
| **"We miss 816 KOs"** | artifact; attacking ends the turn so setup-first is correct. **Unambiguous misses: 0** |
| **Use 1-prize Hariyama vs Grimmsnarl to save prizes** | the 1265 pilot uses the 3-prize Mega for **72%** of attacks there, Hariyama **0** |
| **Cornerstone Mask Ogerpon walls us** | no — Mega Lucario ex has no Ability |
| **Lunar guard reduces wasted attachments** | held at 20 games, **vanished at 40** (7.2 / 9.4 / 8.2%) |

**The pattern:** every *learned* component failed; the only changes that moved anything were
**mechanical fixes where the code didn't match the printed card text**.

---

## 5. KEY MEASURED FACTS

### The meta (from `work/out/matchups.json`, 13,444 real episodes, our code never touching play)
- **Marnie's Grimmsnarl ex is ~53% of the field** and is its *worst* performer (0.468 win rate).
- **Our deck (Mega Lucario ex): 0.697 overall, but only 0.509 vs Grimmsnarl** over 53 real games.
  That single matchup explains our ladder win rate: `0.53×0.509 + rest ≈ 0.62`, and we measure 0.609.
- **Grimmsnarl is weak to GRASS.** Teal Mask Ogerpon (a Grass deck) beats it **0.851 over 954 games**.
- Field-weighted expected win rate: ours **0.563**, best alternative (Teal Mask Ogerpon) **0.604** —
  only +0.04, and switching decks needs piloting that failed 3 times. Deck swap is *not* an easy win.

### The critical card interaction (our deck)
```
Marnie's Grimmsnarl ex : 320 HP, Stage 2, Darkness, weak to Grass
Gravity Mountain       : each Stage 2 in play gets -30 HP  ->  Grimmsnarl 290
                         (NONE of our Pokemon are Stage 2 — purely one-sided)
Mega Lucario ex        : Fighting, Stage 1, 340 HP, Mega Brave 270 dmg (2 {F})
Premium Power Pro      : +30 dmg from {F} attackers vs opponent's ACTIVE, BEFORE weakness
                         -> Mega Brave 300  >=  290  = ONE-SHOT KO
```
Without it: two-attack trade, and we give up **3 prizes** (Mega-ex) vs their **2** — we lose that race.
`v34` raises Gravity Mountain 1→3 because nothing in the deck can search for it.

### Rating ≠ win rate
Majkel1337: 0.606 win rate → **1265**. Us: 0.609 → **778**. Matchmaking pairs by rating, so climbing
means beating *stronger* opponents, not winning more games against weak ones.

### Behavioral divergence vs the 1265 pilot (2,170 identical positions — well sampled, still unexploited)
| option | offered | pilot takes | we take |
|---|---|---|---|
| ATTACH | 731 | 41.9% | **15.0%** |
| ABILITY | 608 | 12.5% | **41.8%** |
| EVOLVE | 289 | 22.5% | 14.5% |
| RETREAT | 574 | 4.2% | 0.9% |

We fire Lunatone's **Lunar Cycle 365× vs their 48×**. It *discards a Basic {F} Energy* to draw 3 —
our attack fuel. Abilities score **30000** in the policy vs attach's **~8000**, so the ability always
resolves first. Guards are built (`v35/v35c/v36`) but their benefit did **not** survive a 40-game check.
**This remains the single largest un-exploited signal.**

---

## 6. TOOLING

| tool | what it does |
|---|---|
| `work/tools/build_and_gate.py` | **build + 24 checks + tarball.** Mandatory before submit |
| `work/tools/matchup_query.py` | matchup win rates from real games (instant; index prebuilt) |
| `work/tools/index_matchups.py` | rebuild `matchups.json` (~30 min; only if new days downloaded) |
| `work/tools/cache_deck_games.py` | cache games featuring a card → `work/out/games_678.zip` (108 of our-deck games) |
| `work/tools/missed_ko.py` | objective: KOs available vs taken |
| `work/tools/attach_audit.py` | objective: wasted once-per-turn energy attachments |
| `work/tools/ppp_window.py` | positions where +30 flips no-KO → KO (PPP **in hand** only) |
| `work/tools/damage_model_audit.py` | **best mechanical metric so far.** Per-ATTACK: did the plan's KO prediction match the engine? ~500 attacks per 120 games, so it actually resolves |
| `work/tools/liveness_check.py` | asserts each new branch executed at least once. Single process |
| `work/tools/ppp_live_audit.py` | counts frames where the PPP buff is live but unmodelled |
| `work/tools/ppp_stack_check.py` | proves the +30 stacks (270/300/330 by copies played) |
| `work/tools/gravity_check.py` | proves Gravity Mountain reads 290 on Grimmsnarl; Stadium uptime by copy count |
| `work/tools/oneshot_check.py` | share of Grimmsnarl KOs taken from full HP. **Too noisy** — identical agents read 12.3% vs 6.5% at 40 games |
| `work/tools/gauntlet.py` | head-to-head, accumulating, content-hashed. **Not ladder-predictive** |
| `work/tools/meta_arena.py` | vs scraped field decks (`--contains 648` for Grimmsnarl). **Not ladder-predictive** |
| `work/tools/vz_extract.py` / `vz_train.py` / `vz_export.py` | value net pipeline |
| `work/tools/dz_*.py` | DouZero action-scorer pipeline (refuted, kept for reference) |
| `work/lib/dzfeat.py` | **single** featurizer — serves replay dicts AND engine objects. Do not fork it |
| `work/lib/vznp.py`, `dznp.py` | numpy-only inference (no torch at runtime) |

### Data on disk
- `data/episodes/{d0731,d0801,d0802}/*.zip` — 13,444 episodes (~2.2 GB)
- `work/out/matchups.json` — matchup index
- `work/out/games_678.zip` — 108 cached our-deck games
- `work/out/vz_big.npz` — 1.09M value-net positions from 7,000 games

### Value net (exists, unproven in play)
DouZero encoder (card embeddings + GRU over log), **game-split AUC 0.8097 vs 0.7194 prize-diff
baseline**. numpy port verified exact vs torch (1.2e-07). Wired into `v33_value` as the search's leaf
evaluator (policy proposes top-4 → 1-ply sim → net scores → lethal DFS overrides). Ladder read 697.4
@16 episodes — flat, but never given a full run.

---

## 7. GOTCHAS THAT COST REAL TIME

1. **`__file__` is undefined** — the harness `exec()`s `main.py`. An unguarded reference killed
   submission 55194301. The gate bans it.
2. **Last callable in the module dict wins** — `kaggle_environments` takes the last callable, *by dict
   position*. Rebinding `agent` leaves the original in place. Gate checks this.
3. **Engine loop:** a game is over when `o.current.result != -1`, **not** when `select is None`.
   Testing the wrong one raises `IndexError` from `battle_select`.
4. **exec-load agents from the REPO ROOT.** `dznp`/`vznp` resolve weights relative to cwd; chdir-ing
   into the agent dir makes the model silently fail to load and the agent degrades to plain v14.
   (This produced a fake null result once — assert `_DZ_OK`/model-loaded in any harness.)
5. **Silent fallbacks are the recurring failure mode** — 5+ components were dead in shipped bundles
   (`search_begin` wrong signature swallowed by bare `except`; `meta_decks` never bundled;
   `best_action` starved because `choose()` truncates to `maxCount`=1 on MAIN). **Always assert a
   component actually ran** (the gate now counts leaf evaluations / re-ranks / fire rates).
6. **≤6 worker processes total** (8 cores). Running 6+3 concurrently broke the process pool.
6b. **Give every parallel worker its own cwd.** Agents v14→v34 write `deck.csv` at import;
   workers sharing a cwd clobber it mid-read and `battle_start` dies with
   `ValueError: The deck must contain 60 cards`. Looks like a flaky engine, is not.
   `v37_combo` no longer writes at all, but older bundles still do.
7. **Don't pipe long runs through `head`** — SIGPIPE killed a training run at epoch 7.
8. **gauntlet.py hashes bundle content** including model files for model-using agents — so retrained
   weights start a fresh cell instead of silently pooling with old results.

---

## 8. SUGGESTED NEXT STEPS

Ranked by evidence, not enthusiasm.

1. ~~**Resolve v34 vs v32 properly.**~~ **OBSOLETE** — see §3b. The two bundles are the same
   agent; the question cannot be answered from that pair. `v37_combo` is the first build that
   actually ships the 3× Gravity Mountain list, so submit **v37_combo against a control** and
   let both reach 40+ episodes. Note v37 bundles the deck change *and* the damage-model fixes,
   so a positive result does not attribute between them; that is a deliberate trade, because at
   a ~100-point noise floor a single-change A/B costs more slots than it can resolve.
2. **Ladder-test a Lunar Cycle guard** (`v36_lunar` is the cleanest: only spend energy on cards once
   the turn's attachment is made). Local metrics can't judge it; the divergence (365 vs 48) is the
   best-sampled behavioral signal we have. Pair it against a control.
3. **Keep auditing card text vs code.** This is the only vein that has produced anything. Method:
   read a card's printed effect, then check the policy/planner implements it. The PPP bug was found
   this way (planner never added the +30 → 10.3% of PPP-in-hand decisions mis-scored as "no KO").
   Unaudited: opponent tools/abilities in *their* damage against us, `Brave Bangle` (+30 vs our ex,
   seen 2,327×), `Cynthia's Roserade` (+30, 657×), `Shaymin`/`Rabsca` bench protection.
4. **Give the value net one honest ladder run** before discarding it — it cleared its offline bar and
   was only ever read at 16 episodes.
5. **Don't** restart behavioral cloning, self-play RL, or deeper search on the hand-written evaluator.
   All three are refuted above with numbers.

### Ready to submit (2026-08-04, awaiting approval — nothing was submitted)

Today's 5 slots were already spent before this work started; they reset at 00:00 UTC.

Both bundles are built and gate 26/26:
- `work/out/v37_combo.tar.gz` — damage-model fixes **+** the 3× Gravity Mountain deck
- `work/out/v38_model.tar.gz` — the same fixes on v32's unchanged deck (lower-risk variant)

**Recommended pair: `v37_combo` + a fresh `v32_ppp` as the same-period control.** Rationale:
the open question is whether any of this moves the ladder at all, so pair the strongest
candidate against the current champion rather than against its own variant. Only 2 stay
active, so a three-way test is not available. If v37 clears the noise floor, `v38_model`
is the follow-up that attributes the gain between the deck and the model.

Do **not** read a single draw as a result: the noise floor is ~100 points and now has three
identical-agent datapoints behind it (64, 115, 62).

### Standing user instruction
**Never submit without explicit approval.** The user calls every submission. Slots are scarce and
only 2 stay active.
