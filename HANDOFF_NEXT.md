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

## 6. WHAT IS IN FLIGHT

Built from the new base by `build_codex_variants.py`, all **untested or in progress**:

| agent | change | status |
|---|---|---|
| `v62_codex_meta` | the base's opponent templates are **empty** unless a Grimmsnarl or Tusk line is visible, so it determinizes most of the field as a deck that cannot exist. Ships our 70 replay-built lists + the validated multiset matcher from `fsearch`. Matcher fires; overrides 30/266 decisions | **measuring vs v61, n=240** |
| `v65_codex_b12` | N_DET 3→12, 3.0 s/decision, 90 s episode cap. The base uses **6 s of the 600 s pool** | **measuring vs v61, n=240** |
| `v63`/`v64` | the same at 300 s/episode | built, too slow to measure until the moderate one proves the direction |

### The one known weakness
`s_dragapult` 0.450 (n=120) where our old `v14` gets 0.647. Dragapult is not in the top-5 of the
field we actually face, so it may not matter — but it is the only opponent that beats the new base
and it is the obvious place to look for the next gain. Re-run `loss_autopsy.py --fetch` once
55274352 has episodes; the band we are matched into will shift as the score rises.

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
