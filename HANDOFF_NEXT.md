# HANDOFF — the base changed. Everything is now built on `v61_codex_safe`.

**Competition:** `pokemon-tcg-ai-battle` (Kaggle simulation ladder). Deadline **2026-08-16**.

**2026-08-05, 15:33 UTC — the project's biggest single jump, and it was not ours.**
`v61_codex_safe` beats the old champion `v51_roman_safe` **0.8375 over 240 games**. Nothing in
this repo's history has separated by more than 0.07; this is 0.34. It is a **published agent by a
higher-ranked author**, adopted whole, plus one safety fix.

Read §1 before anything else — it is why the previous 40 experiments failed and this one did not.

---

## 0. WHERE WE ARE

| | |
|---|---|
| Our rank | **2343 / 6321**, score 697.7 (LB median 632.6, p95 910.3) |
| Leader | Majkel1337 **1208.2**. Top-20 cutoff ~**1081** |
| Submitted 15:33 UTC | `v61_codex_safe` = **55274352**, μ₀ 600, converging |
| Also active | `v57_pimc_full` = 55267894 (694.4) — the same-period control |
| Slots | **0 left today.** 5/day, resets 00:00 UTC. Only the latest 2 stay active |

`v51_roman_safe` draw 1 (780.0) is no longer active. Our line's ladder history sits at 654–780
across a dozen submissions, i.e. **entirely inside the noise floor**.

---

## 1. THE LESSON THAT MATTERS MORE THAN ANY CODE HERE

Between v14 and v57 this project built ~40 increments: damage-model fixes, tech cards, ability
ordering, behavioural cloning, value nets, evolved weights, evolved decks, three kinds of search.
**Every single one landed inside the noise floor or below it.** The full refuted list is §5.

Twice, the score actually moved. Both times the move was **adopting a published agent from an
author ranked well above us** — `romanrozen` (rank 468) became `v51`, and now `jazivxt`
(rank **121**, 988.8) becomes `v61`.

So before building anything, spend fifteen minutes doing this:

```bash
TOKEN="$(cat ~/.kaggle/access_token)"
# 1. what has been published
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/kernels/list?competition=pokemon-tcg-ai-battle&sortBy=dateCreated&pageSize=60"
# 2. the FULL leaderboard  (leaderboard/view returns only the top 20)
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/competitions/pokemon-tcg-ai-battle/leaderboard/download" -o lb.zip
# 3. join on TeamMemberUserNames -> read only authors ranked above us
```

Vote count is a poor proxy for strength. The 85-vote Grimmsnarl notebook is rank 530; the
43-vote one is rank 121. **Rank the author, not the notebook.** No top-20 team publishes anything.

---

## 2. THE MEASUREMENT INSTRUMENT

Every local metric in this project failed at least once, and two pointed the *wrong way*, because
**our own ~700-level policy piloted both sides**. The fix, and it works, is to benchmark against
**real published agents**:

| dir | what it is | LB rank of author |
|---|---|---|
| `work/agents/p1_codex` | jazivxt's Codex Sol Eclipse Alakazam v22 — **the new base** | **121** (988.8) |
| `work/agents/z_roman950` | romanrozen v10, the old base | 468 (872.4) |
| `work/agents/w1_alakazam` | an earlier published Alakazam agent | — |
| `work/agents/w2_archaludon` | published Archaludon agent | — |
| `work/agents/s_dragapult`, `s_mega` | sample agents | — |

```bash
python work/tools/gauntlet.py --agents <candidate>,<opponent> --games 240 --workers 3
python work/tools/gauntlet.py --report
```

### Rules that are not negotiable
- **n≥200 for any head-to-head you act on.** v57's headline 0.600 at n=45 decayed to 0.570 at
  n=135 and to 15-15 on the ladder. n=45 means nothing.
- **≤6 python workers total** (8 cores). `pkill -f` does NOT work here — use PowerShell
  `Stop-Process` and verify with `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`.
- **Time-budgeted agents must be measured with the machine otherwise idle.** A forgotten
  background job once turned a 0.550 reading into 0.250.
- **The store is content-hashed** per bundle including `fsearch.py`, `meta_decks.py` **and a
  `HARNESS` constant**. Editing any of them starts a fresh cell instead of silently pooling.

### Ladder noise
Byte-identical bundles have read 64, 115 and 62 points apart. **<100 points is noise** unless
repeated, and a submission means nothing under ~25 episodes.

---

## 3. WHAT WAS ADOPTED, AND HOW IT WAS VERIFIED

`v61_codex_safe` = `p1_codex` verbatim + **one** change: resolve `deck.csv` at import
(bundle → cwd → inlined constant) instead of only on the setup frame. Built by
`work/tools/build_codex_variants.py --variant safe`, which patches the base and **asserts every
anchor was found** — a silently-missed patch would ship the base under a new name.

### Measured, n=240 each, harness 2
| opponent | v61_codex_safe | our whole line |
|---|---|---|
| `v51_roman_safe` (old champion) | **0.8375** [0.786, 0.879] | — |
| `w2_archaludon` | **0.8583** [0.809, 0.897] | 0.250 – 0.369 |
| `w1_alakazam` | **0.8917** [0.846, 0.925] | 0.450 – 0.513 |
| `s_dragapult` | **0.4500** (n=120) | v14 gets 0.647 — **a real hole** |

Archaludon was the matchup the previous handoff called *structural* — "they hit 220 every turn,
we simply lack damage". It was not structural. It was the policy.

### The falsification pass (do this before believing any number this large)
1. **Decks identical under both harnesses** for all 7 agents — probed directly, not assumed.
2. **Sides balanced** 120/240; **0 errors, 0 draws** in all 240 games. Nothing was forfeited.
3. **The yardstick was not damaged by the harness fix.** `v51` vs `v43_judge2x` reads 0.535
   (h1, n=200) and 0.550 (h2, n=400). The one control that *moved* — v51 vs z_roman950, 0.518 →
   0.434 — is 1.7σ at n≈200 between two near-identical agents, i.e. noise.
4. **The safety patch does not change play.** `v61` vs `p1_codex` = 0.439 (n=66), CI contains 0.5.
5. **The result holds under both harnesses**: 0.758 with the agent crippled, 0.838 with it fixed.
6. **Gate 27/27.** Worst move 805 ms; worst episode 19.2 s of the 600 s pool.
7. **No state leak**: `agent_drift.py` shows 1.08× agent-time drift over 8 games in one process,
   so the base's missing `search_release` is not the 1,089,510 ms hazard we hit before.

What remains unknowable locally: whether jazivxt's 988.8 comes from *this* notebook. Only the
ladder can answer that, which is what 55274352 is doing now.

---

## 4. THREE HARNESS BUGS FOUND ON THE WAY — ALL UNDERSTATED THE AGENT

1. **Neither harness issued THE SETUP CALL.** The contract (`work/lib/sample_main.py`) is that an
   episode opens with `select == None` and the agent returns its 60 card ids. `p1_codex` assigns
   `my_deck` there **and nowhere else**, so every local game ran it determinizing its OWN deck as
   60 filler energy — and it still won 0.758. Both harnesses now make the call.
   `gauntlet.py` gained `HARNESS = 2`, folded into the bundle hash.
2. **The gate required the last callable to be *named* `agent`.** Kaggle's `get_last_callable`
   binds by dict **position** regardless of name, and a unique final name is the correct pattern —
   the sample agent uses one. The rule failed a bundle whose author is 291 points above us.
   Replaced with a functional check: whatever we bind must answer the setup frame with 60 card
   ids. Strictly stronger, and it still catches the `agent = wrapper` hazard.
3. **`.gitignore`'s unanchored `deck.csv`** matched `work/agents/*/deck.csv`, so **no agent's
   decklist was ever tracked** and no bundle in this repo could be rebuilt from git. Anchored to
   the root scratch file it was written for; 47 decklists added.

---

## 5. REFUTED — DO NOT REDO ANY OF THESE

All measured against real opponents with confirmation. Re-running them wastes days.

| idea | result |
|---|---|
| Deck archetype change *for the old Lucario policy* | our deck was the best available **for that policy** |
| Deck composition tuning | ~50 rounds of confirmed hill-climbing found nothing (0.5050) |
| Policy weight tuning | same run, same answer — a local optimum |
| Self-play evolution | panel fitness 0.476→0.512, head-to-head 0.5050. The gain was selection noise |
| Sequential halving in the playout search | 0.3833. At 15 playouts the estimate is ±0.25 |
| Lethal forward search | 0.4615 over 299 games |
| Learned value net as search leaf | 0.3667 |
| Behavioural cloning / DouZero action-scorer | 389.3 and 400.1 on the ladder |
| Damage-model correctness fixes | KO-prediction error 11.8%→4.2% and **no win-rate gain** (0.4770) |
| Judge / Wally's Compassion tech cards | 0.430 / 0.509 |
| Ranking Abilities below the energy attach | 0.4250 |
| **Playout search to terminal states** (`v57`) | 0.570 local (n=135), **15-15 on the ladder**. The one idea the last handoff called the only route to +250. It was not |

**The pattern: every idea generated inside this project failed. Both wins came from outside it.**

---

## 6. EVERYTHING TRIED ON THE NEW BASE — ALL INSIDE THE NOISE

Built by `build_codex_variants.py`, each measured head-to-head against `v61_codex_safe`:

| agent | change | result vs v61 |
|---|---|---|
| `v62_codex_meta` | the base's opponent templates are **empty** unless a Grimmsnarl or Tusk line is visible, so it determinizes most of the field as a deck that cannot exist. Ships our 70 replay-built lists + the validated multiset matcher from `fsearch`. The matcher fires (30/266 decisions) | **0.4542** [0.392, 0.517] n=240 |
| `v65_codex_b12` | N_DET 3→12, 3.0 s/decision, 90 s/episode. The base uses **8 s of the 600 s pool** | **0.5375** [0.474, 0.599] n=240 |
| `v71_statefix` | the base's rollouts corrupt `pre_turn` and both once-per-turn ability flags in the live game; snapshot and restore | **0.5208** [0.458, 0.583] n=240 |
| `v70_playout` | playouts to TERMINAL states, no evaluator, using the base's own heuristic as the rollout policy. Runs correctly: 12,929 playouts, 99.0% terminal, drift 1.00×, overrides 13/100 decisions | **0.4625** [0.401, 0.526] n=240 |
| `v63`/`v64`/`v66`/`v67`/`v68`/`v69` | bigger budgets, lower override margin, branch on attacks | built, unmeasured |

**Nothing beats the base.** Four structurally different attacks — more compute, a correctness fix,
a different search target, and a better opponent model — all land inside the noise. Do not pick the
highest of four inconclusive readings and call it an improvement; that is the
screen-without-confirmation mistake that produced 13 false positives here before.

**What v65 and v70 tell you together, and it is the useful part:** quadrupling determinizations
changes nothing, which says the 2-ply search is limited by the BIAS of `_leaf_eval` rather than by
the variance of its estimate. Removing the evaluator entirely — which is exactly what a terminal
playout does — then makes it *worse*, not better. So the leaf evaluator is not the binding
constraint on this base either, and "search harder" in any form is spent. The remaining untried
lever is the policy's own 60+ tunable weights (§6b).

### 6b. THE GAP IS EXPLAINED — AND IT SETS THE CEILING ON ADOPTION

Win rate **by opponent rating band**, ours against the base author's live agent:

| opponent rating | ours (55274352) | jazivxt live (55255635) |
|---|---|---|
| <700 | 6/6 = 1.000 | 7/7 = 1.000 |
| 700–800 | 10/18 = 0.556 | 5/7 = 0.714 |
| **800–900** | **5/14 = 0.357** | **10/13 = 0.769** |
| 900–1000 | 0/2 | 18/32 = 0.562 |
| 1000+ | 0/1 | 4/10 = 0.400 |

It is not convergence. So: **pull their live submission's replay and read the decklist out of the
setup frame.** It is a **Crustle wall + Mega Kangaskhan ex + Cornerstone Mask Ogerpon ex** control
deck — not Alakazam. Their 960.8 agent is not the notebook we adopted, and it is **not published**:
its decklist matches neither of their two published notebooks.

Everything public was then tested against `v61`, n=240 each:

| agent | vs v61 |
|---|---|
| `p3_crustle` — their *published* Crustle agent (gated ML selector, v29) | **0.4500** |
| `p4_crustle_live` — that policy piloting their real live decklist | **0.1167** (28-212) |
| `w1_alakazam`, `w2_archaludon`, `z_roman950`, `w5_grimmsnarl` | all far below |

`p4` is the sharpest lesson available on deck/policy coupling: their own live decklist, copied
verbatim, **destroys** the published policy — 0.117. A deck is not transferable without the policy
it was tuned with.

**Conclusion: `v61_codex_safe` is the strongest agent obtainable from public code.** No top-20 team
publishes anything; the one rank-121 author who does publishes builds older than what they run.
Reaching 1000 (rank ~100) therefore requires beating every published agent in the competition, not
adopting one.

### The gap as it looked before that was resolved
Our `v61` and the base author's *published* agent are the same code, yet:

| submission | episodes | W-L | win rate | score |
|---|---|---|---|---|
| jazivxt 55255635 | 70 | 44-25 | 0.638 | **960.8** |
| **ours 55274352** | 37 | 18-18 | 0.500 | **765.3** |
| Raihan 55177269 (rank 2) | **712** | 422-289 | 0.594 | 1207.5 |

Ruled out so far: the bundled engine is **byte-identical** to the competition's current
`sample_submission/cg` (all five files hash-match); `group.txt`, which their build ships and ours
does not, appears in neither the sample submission nor the docs; our safety patch does not change
play (`v61` vs `p1_codex` 0.439, n=66, CI contains 0.5). Remaining candidates: episode count (Raihan
needed 712), TrueSkill path dependence, or their 2026-08-05 submission being a newer private
version than the 2026-07-30 notebook.

### The one known weakness
`s_dragapult` 0.450 (n=120) where our old `v14` gets 0.647 — the only opponent that beats the new
base. Not in the top 5 of the field we face, so it may not matter.

---

## 7. TOOLING

| tool | what it does |
|---|---|
| `build_and_gate.py` | 27 pre-submission checks + tarball. **Never skip it** |
| `build_codex_variants.py` | derives variants from `p1_codex` by asserted source patches |
| `gauntlet.py` | accumulating, content-hashed head-to-head. The yardstick |
| `agent_drift.py` | **new.** Does an agent slow down as its process lives? Catches search-state leaks |
| `loss_autopsy.py` | downloads our ladder replays; reports the meta we actually face |
| `deck_choice.py` | archetype-vs-archetype win rates from 13,444 real games |
| `build_meta_from_replays.py` | rebuilds the 70-deck opponent library |
| `liveness_check.py` | asserts each new branch actually executed. **Five shipped components were dead** |

---

## 8. GOTCHAS THAT COST REAL TIME

1. **`__file__` is undefined** — the harness `exec()`s `main.py`. Guarded use is fine
   (`globals().get("__file__")`); the gate distinguishes them.
2. **The last callable in the module dict wins**, by position, whatever it is named.
3. **A game is over when `o.current.result != -1`**, not when `select is None`.
4. **The setup frame (`select is None`) is part of the contract.** Agents initialise there.
5. **Silent fallbacks are the recurring failure mode.** Assert every component actually ran.
6. **Give every parallel worker its own cwd** — older agents write `deck.csv` at import.
7. **`pkill -f` does not work here.** PowerShell `Stop-Process`, then verify.
8. **The kaggle MCP server caches credentials at startup** and returns `Unauthenticated`. Use the
   CLI or REST with `Authorization: Bearer $(cat ~/.kaggle/access_token)`.
9. **`export PYTHONIOENCODING=utf-8`** — the é in the repo path crashes the CLI otherwise.

---

## 9. STANDING INSTRUCTION

**Never submit without explicit approval.** The user calls every submission. Slots are scarce
(5/day) and only 2 stay active.

Commit as **DavidMashiah**, no co-author trailers. `data/`, `public/`, `public_new/`,
`work/out/` and the engine binaries are gitignored.
