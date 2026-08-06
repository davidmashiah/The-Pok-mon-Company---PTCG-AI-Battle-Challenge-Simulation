# HANDOFF — PTCG AI Battle Challenge

**Competition:** `pokemon-tcg-ai-battle` (Kaggle simulation ladder). Ladder deadline **2026-08-16**.
**State as of 2026-08-06 22:30 UTC.** Self-contained; you do not need the chat log.

---

## 0. WHERE WE ARE

| | |
|---|---|
| **Live score** | **880.9** (max of the two active draws) |
| Active A | `55299973` — `w8_grimm_tuned`, **880.9** |
| Active B | `55305926` — **byte-identical** to A, **825.3** |
| Slots | 2 of 5 used (UTC day resets 00:00) |
| Top-50 cutoff | **1043.6**. Leader 1188.9. 6466 teams, median 631.5 |

The two active submissions are the same bytes and read **55.6 points apart**, so this agent's
true value is roughly **853 ± 28**, not 880.9. Never quote the lucky twin.

**STANDING RULE: never submit without explicit approval.** The user calls every submission.

---

## 1. THE MEASUREMENT INSTRUMENT — READ FIRST

`work/tools/field_test.py` measures a weighted win rate across the archetypes the **top 50**
actually play, then converts to a rating through an anchor. It now has **three** calibration
points, one fully out-of-sample:

| agent | field | predicted | actual live | error |
|---|---|---|---|---|
| `v61_codex_safe` | 0.4914 | 726.1 (anchor) | 726.1 | — |
| **`w5_grimmsnarl`** (tetsutani's public bundle, we own nothing in it) | **0.6030** | **805** | **801.6** | **+3.4** |
| `w8_grimm_tuned` | 0.6376 | 830 | 853 (mean of twins) | −23 |

```bash
python work/tools/field_test.py --agent <name> --games 190 --workers 6
```

Trust it in the 700–900 band. The extrapolation to 1040 is **beyond its validated range**.

### 1a. Screen with the Grimmsnarl cap before spending a full field test
`work/tools/grimm_cap.py`. Grimmsnarl carries ~47% of the renormalised panel weight, so its cell
bounds everything: `field ≤ w_g·p_grimm + (1−w_g)·1.0`. Spot a candidate a perfect 1.000
everywhere else; if it still cannot beat 0.6376, it is dead. One pair instead of six.

### 1b. Size the lever before building it
`work/tools/what_is_it_worth.py`. Rating moves with the **logit** of the field rate, and at 0.64
that curve is flat:

| change | field | rating |
|---|---|---|
| **every** non-mirror matchup → 0.95, mirror unchanged | 0.753 | **926** |
| mirror alone → 0.95 | 0.835 | **1013** |
| mirror 0.530 → 0.580 | 0.661 | **+18** |

**Winning every other matchup 95% of the time does not reach the cutoff.** Reaching ~1040 needs
roughly mirror 0.67 **and** ~0.95 against everything else, simultaneously. That is a different
agent, not a tweak.

### Other non-negotiables
- **n≥200 for any head-to-head you act on.**
- **≤6 python workers** (8 cores). `pkill -f` does not work; use PowerShell `Stop-Process` and
  verify with `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`.
- A tuning search **cannot certify its own result** — re-measure with `field_test.py`.
- **Two agents in one process collide.** Several bundles ship a module named `policy_features`;
  the second `import` is a silent no-op and binds the FIRST agent's 60-card deck. Evict
  `sys.modules` entries whose `__file__` is under the agent dir between loads (see
  `work/tools/cape_check.py`). Caught when `w5_grimmsnarl` raised "fixed 60-card deck changed".

---

## 2. THE AGENT WE SHIP

**`w8_grimm_tuned`** — tetsutani's published Grimmsnarl coalition agent + a bundle-path fix.
Gate 27/27. Artefact `work/out/w8_grimm_tuned.tar.gz`.

**It cannot be re-decked, at all.** Three independent locks: `main.py` asserts a fixed 60; the
learned policy scores options through a **closed 180-entry intent vocabulary** keyed to card ids,
so a card outside the deck has no representation; and the feature schema is built over `DECK_IDS`.
Any deck idea needs a different base.

---

## 3. MINING PUBLIC CODE IS NOW EXHAUSTED (measured, not assumed)

The only lever that ever moved this score was adopting a published agent from a higher-ranked
author. On 2026-08-06 the seam was worked to the end: **all 100 notebooks, four sort orders,
every author joined to the live leaderboard** (`work/tools/mine_notebook.py`, and the join recipe
in §7). Three authors nobody had mined were found — ranks **11, 88, 97**. All extracted, all
measured:

| candidate | vs Grimmsnarl | field | projected |
|---|---|---|---|
| **w8_grimm_tuned (ours)** | **0.5297** | **0.6376** | **830** |
| w26_arist_prob (rank-97 author) | 0.4762 | 0.5009 | 733 |
| w20_luc1084 ("1084.5 Baseline") | 0.4381 | 0.4816 | 719 |
| w24_tientrum (rank-88, genuinely 1034.6 live on Jul 5) | 0.2525 | — | capped at 0.638 |
| w21_libout1208 ("Max Elo 1208") | 0.176 | — | rejected on the cap |

**Notebook titles are marketing.** "1084.5 Baseline" is by an author at rank 1632; "Max Elo 1208"
rank 4074; "1000 Fixed Agent" rank 5447. Only the author-rank join is real.

**The tientrum result is the important one.** That build honestly scored 1034.6 — and is now
capped at exactly our current 0.638, because Grimmsnarl grew into 32% of the top 50 after it was
live. Strength here is **not stationary**.

---

## 4. WHAT THE TOP 50 ACTUALLY PLAY (re-measured 2026-08-06)

| archetype | teams | panel weight |
|---|---|---|
| **Marnie's Grimmsnarl ex** | **16 (32%)** | 0.30 ✓ |
| Mega Lopunny ex | 9 (18%) | — no agent published anywhere |
| unknown | 8 (16%) | — |
| Alakazam | 7 (14%) | 0.16 ✓ |
| Crustle | 5 (10%) | 0.08 ✓ |
| Mega Lucario ex | 3 (6%) | 0.02 — **understated**, and our worst matchup (0.446) |
| Dragapult ex | 2 (4%) | 0.06 |
| Archaludon | **0** | 0.02 — **stale, no longer in the top 50** |

**16 of the top 50 play the exact deck we play — and 13 of those 16 play the byte-identical 60.**
`work/tools/mirror_diff.py` reads their real lists out of their own replays and diffs them against
ours: rank 2 (1197.3), rank 4 (1156.3), rank 5 (1135.5), rank 10 (1124.4), rank 21, rank 26 …
all *zero cards different*. The only deviation anywhere in the top 50 is 2× **Handheld Fan**
(1161, not ACE SPEC), run by 3 of the 16 — it moves an Energy off their attacking Grimmsnarl each
time it damages the holder.

**So the deck is solved and shared, and the entire 853 → 1197 spread is piloting.** Do not spend
another hour on decklists. Note also that the pilot we ship is a public bundle whose own author
sits at **801.6** — we are running an ~800-rated pilot on the consensus deck.

---

## 5. REFUTED — DO NOT REDO

Everything in the previous handoff's ledger still stands (coalition weights are a dead knob —
`coalition_expert` fired **0** times in 933 instrumented decisions; router table search; setup
speed; 4× determinizations; behavioural cloning 389.3; learned value nets ×3; deck/policy transfer
`p4_crustle_live` 0.1167; both published Grass agents lose to Grimmsnarl at 0.100 and 0.160).

Added 2026-08-06:

| idea | result |
|---|---|
| **Adopting any published agent** | exhausted, §3. Nothing beats w8 |
| **Deck tech on w8** | impossible, §2 |
| **Hero's Cape as a 3-of** | **illegal.** It is ACE SPEC and the format allows exactly ONE; `battle_start` returns `None` with no error. Our deck already spends the slot on Unfair Stamp |
| **Hero's Cape as the 1 ACE SPEC** (`w40_cape`) | **REFUTED, and it is a regression.** Against a matched control at the same n, under the same harness: control `_sub_handwritten_v26` **0.5523** (n=239), `w40_cape` **0.4417** (n=240), CIs [0.489,0.614] vs [0.380,0.505]. +100 HP is worth less than Unfair Stamp's post-knockout refuel — which is why 13 of the 16 top-50 Grimmsnarl teams decline it |

### The one thing that DID measure better: `w30_search`
Field **0.6519 → projected 841**, against w8's 0.6376 → 830. Per cell:

| | w8 | w30_search |
|---|---|---|
| Grimmsnarl | 0.530 | 0.538 |
| Alakazam | 0.747 | **0.780** |
| Crustle | 0.811 | **0.844** |
| Dragapult | 0.720 | 0.661 |
| **Mega Lucario** | 0.446 | **0.559** |
| Archaludon | 0.629 | 0.634 |

No single cell clears its own CI, but 5 of 6 moved up and the largest gain is on our worst
matchup — which the re-survey shows is 6% of the top 50, not the 2% the panel assumes. **+11
projected points is far inside the ladder's ±55–85 noise floor, so it cannot be confirmed live.**
Treat it as "not a regression, probably a small gain", never as a result.

### The v57 discovery — it invalidates an old ledger entry
`v57_pimc_full`'s playout search **never executed a single playout**. `_SEARCH_OK` is set from
whether the *import* succeeds; the call `search_begin(obs, your_deck=yd)` then raises `TypeError`
(this engine's signature takes seven required positional args) straight into
`except Exception: return None`. It played its entire 701.8-point ladder run as a pure heuristic.
Verified by signature (`work/tools/search_probe.py`) and by its own exception path. **So the
"playout search refuted" entry never tested playout search.** Fifth silently-broken component here.

The native search is fast and healthy: **2225 decisions/s**, ~0.45 ms/step, and `w30_search` runs
~110 playouts/game in ~5 s/episode against a 600 s allowance.

---

## 6. THE HONEST ARITHMETIC ON 1040

Live ~853. Cutoff 1043.6. That gap needs field 0.6376 → ~0.82–0.855, i.e. **mirror ~0.67 AND
~0.95 against everything else**. Per §1b, no single matchup gets there, and incremental pilot work
is worth ~+18 points per +0.05 of mirror win rate.

Three independent routes were tried today and all closed: adopt a better agent (exhausted),
change the deck (impossible on this base, one ACE SPEC slot on the other), improve the pilot with
search (inside noise). **Say this plainly to the user rather than promising a number.**

What is genuinely still open, in descending expected value:
1. **The Strategy competition** (§8) — $240,000, deadline **2026-09-13**, explicitly rewards
   mid-tier ladder teams with deep analysis. Our evidence there is unusually strong.
2. **Draw variance.** Identical bundles read 55–85 points apart. The LB takes the max of two
   active submissions and each new one evicts the *older*. Re-rolling the weaker slot while
   holding the good one is legitimate and could plausibly hold ~900–950. It is not a path to 1040.
3. A genuinely new agent, if someone has ~2 weeks. The Basic-only **Iron Leaves ex + Teal Mask
   Ogerpon ex** package (all Basics, 180 for [GGC] → 360 vs Grimmsnarl's Grass weakness,
   `Rapid Vernier` switches it in from the bench and consolidates energy onto it) is the best
   structure found — it avoids the Stage-2 assembly problem that killed `v200_decidueye` (0.017)
   and both published Grass agents. `work/tools/answer_pool.py` has the full survey.

---

## 7. TOOLING

| tool | what it does |
|---|---|
| `field_test.py` | **The yardstick.** Calibrated at three points, one out-of-sample |
| `grimm_cap.py` | Rejects a candidate from ONE pair using the weighted-field bound |
| `what_is_it_worth.py` | Rating value of any matchup improvement, before you build it |
| `mine_notebook.py` | .ipynb → runnable `work/agents/<name>/`. Handles `%%writefile`, DECK literals, base64 tar/zip |
| `search_probe.py` | Ground truth on the native search API — signature, speed, return shape |
| `build_search_layer.py` | w8 + determinized search-validation, paired determinizations, knobs baked in |
| `search_liveness.py` | Proves the search executes (playouts, overrides, seconds/episode) |
| `answer_pool.py` / `grass_pool.py` / `card_detail.py` | Whole-pool surveys: what answers a 320 HP Stage-2 ex, for how much energy |
| `deck_report.py` | Resolves any deck.csv against our engine + asserts `battle_start` accepts it |
| `cape_check.py` | In-engine verification of a Tool's effect; also the sys.modules eviction pattern |
| `build_and_gate.py` | 27 pre-submission checks + tarball. **Never skip** |
| `gauntlet.py` | Accumulating content-hashed head-to-head. Bump `HARNESS` if you change how agents are driven |
| `top_decks.py` | What the top N teams actually play, read from their replays |
| `field_calibration.json` | The anchor. Do not re-anchor without re-deriving the whole table |

Mining recipe (REST; the kaggle MCP server caches creds at startup and returns `Unauthenticated`):
```bash
TOKEN="$(cat ~/.kaggle/access_token)"
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/kernels/list?competition=pokemon-tcg-ai-battle&sortBy=voteCount&pageSize=100"
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/competitions/pokemon-tcg-ai-battle/leaderboard/download" -o lb.zip
# join TeamMemberUserNames -> rank. Rank the AUTHOR, never the title.
```

---

## 8. THE OTHER COMPETITION — $240,000, DEADLINE 2026-09-13

`pokemon-tcg-ai-battle-challenge-strategy`, a **separate** competition four weeks after the ladder
closes. **8 finalists × $30,000.** Submission is a Kaggle Writeup, max 2000 words, at
`/competitions/pokemon-tcg-ai-battle-challenge-strategy/projects` — **a saved draft that is never
explicitly Submitted is not judged.**

Scoring: Model **70%** (clarity; originality and technical soundness; consistency under repeated
matches; avoiding over-reliance on specific matchups) / Deck **20%** / Report **10%**. It states
that mid-tier ladder teams can still score highly through deep analysis.

`WRITEUP_DRAFT.md` holds the evidence inventory, now 11 items. The strongest additions from
2026-08-06 are §8 (out-of-sample calibration), §9 (a 1034.6 agent invalidated by meta drift alone),
§10 (logit sizing — why matchup work cannot reach the target) and §11 (ACE SPEC / silent engine
rejection).

---

## 9. IMMEDIATE STATE

- Verify nothing is running: `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`.
- `work/out/ft_w30.log` — full field test of `w30_search`.
- `w40_cape` — the hand-written v26 policy with Unfair Stamp → Hero's Cape, plus `cape_guard.py`.
  Guard verified live: the Cape lands on Grimmsnarl ex in 316 of 362 observations. A/B vs
  `w5_grimmsnarl` was queued against the base policy's 0.5269.
- **10+ commits are unpushed.** `git push` fails — Git Credential Manager's token stopped working
  and it cannot prompt non-interactively. **Someone must run `git push` from a real terminal, or
  `gh auth login`.** Nothing is lost.
- Mined notebooks and the bundles extracted from them are gitignored and rebuildable with
  `mine_notebook.py`; `w30_search` and `w40_cape` rebuild from their builders.
