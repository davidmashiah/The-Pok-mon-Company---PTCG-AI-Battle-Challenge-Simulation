# HANDOFF — PTCG AI Battle Challenge

**Competition:** `pokemon-tcg-ai-battle` (Kaggle simulation ladder). Ladder deadline **2026-08-16**.
**State as of 2026-08-06 20:31 UTC.** This file is self-contained; you do not need the chat log.

---

## 0. WHERE WE ARE RIGHT NOW

| | |
|---|---|
| **Live score** | **~849** (the higher of our two active draws) |
| Session start | 697.7 |
| Active sub A | `55299973` — `w8_grimm_tuned`, **848.8** |
| Active sub B | `55305926` — same bundle, byte-identical second draw, **843.5** |
| Slots | **2 of 5 used today** (UTC); resets 00:00 UTC |
| Top 50 cutoff | **1040.1**. Leader 1208.2. LB is 6321 teams, median 632.6 |

**The user's goal is top 50 (1040+). We are not there and the arithmetic in §5 says this base
cannot get there.** Read §5 before promising anything.

**STANDING RULE: never submit without explicit approval.** The user calls every submission.

---

## 1. THE MEASUREMENT INSTRUMENT — READ THIS FIRST

Everything in this repo has failed at measurement before it failed at ideas. Three rules, all
paid for:

### 1a. Head-to-head does NOT convert to ladder points
`v61` beat `v51` **0.8333 over 240 games** and gained about **+30** on the ladder. An Elo reading
of 0.83 predicts +300. The head-to-head was enormous because Alakazam beats Mega Lucario — a
**matchup**, and the ladder does not pay for matchups. Do not judge a candidate by a head-to-head.

### 1b. Use `field_test.py` — it is calibrated and it works
`work/tools/field_test.py` measures the weighted win rate across the archetypes the **top 50**
actually play, then converts to a rating through an anchor measured on our own submission.

```bash
python work/tools/field_test.py --agent <name> --games 190 --workers 6
```

It has predicted two live scores correctly:

| agent | field | projected | actual live |
|---|---|---|---|
| `v61_codex_safe` | 0.4914 | 726 | **726.1** |
| `w8_grimm_tuned` | 0.6376 | 830 | **829.5 → 848.8** |

Anchor is stored in `work/out/field_calibration.json`. **Do not re-anchor it on a different
agent without re-deriving the whole table.**

Panel and weights (top-50 shares, from `work/tools/top_decks.py` over 50 teams' own replays):

| archetype | share | opponent agent |
|---|---|---|
| Grimmsnarl | 0.30 | `w5_grimmsnarl` |
| Alakazam | 0.16 | `w1_alakazam` |
| Crustle | 0.08 | `p3_crustle` |
| Dragapult | 0.06 | `s_dragapult` |
| Mega Lucario | 0.02 | `z_roman950` |
| Archaludon | 0.02 | `w2_archaludon` |

**Known gap, stated not hidden:** Mega Lopunny is **20%** of the top 50 and no Lopunny agent is
published anywhere (checked 4 pages by vote count). "unknown" archetypes are another 18%. So the
panel covers **62%** of the top field and shares are renormalised over that. The 400-point Elo
slope is a convention, not this ladder's measured slope.

### 1c. A tuning search CANNOT certify its own result
Three searches produced "gains" that **all vanished** when measured on an independent code path:

| search | its own number | independent `field_test` |
|---|---|---|
| coalition run 1 | 0.7147 → 873 | **0.6376 → 830** |
| coalition run 2 | 0.6920 → 873 | **0.6114 → 811** |
| router | 0.6665 → 852 | **0.6228 → 819** |

Screen-plus-confirm on self-selected seeds is not enough at this noise level. **Always re-measure
a search result with `field_test.py` before believing it, and never submit on a searcher's own
number.**

### Other non-negotiables
- **n≥200 for any head-to-head you act on.** 60-game readings have produced three separate ghosts.
- **≤6 python workers total** (8 cores). `pkill -f` does NOT work here — use PowerShell
  `Stop-Process`, verify with `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`.
- **Ladder noise floor is real and measured on our own bundles**: the two active submissions are
  byte-identical and read **848.8 vs 843.5**, and earlier in the day were **856.5 vs 771.9** — an
  85-point spread. A submission means nothing under ~25 episodes.

---

## 2. THE AGENT WE SHIP

**`w8_grimm_tuned`** — tetsutani's published Grimmsnarl coalition agent + two changes of ours.
Gate: **27/27 PASSED**. `work/out/w8_grimm_tuned.tar.gz` is the submitted artefact.

Lineage, all reproducible:
```
w5_grimmsnarl     tetsutani's published bundle, unpacked from the notebook's base64 asset
  -> w7_grimm_safe   + bundle-path fix   (work/tools/build_grimm_safe.py)
  -> w8_grimm_tuned  + coalition weights (see §4 — this part turned out to be a no-op)
```

`build_grimm_safe.py` fixes a real submission-killer: the bundle resolved assets as
`__file__` → `/kaggle_simulations/agent` → **cwd**, and since the harness `exec()`s main.py the
first is undefined, so from any other directory import died on `models/feature_schema.pkl.gz`.
Fixed by *finding* the bundle via `sys.path`. Kaggle's absolute path stays first so ladder play is
bit-for-bit the author's.

---

## 3. HOW WE GOT FROM 697 TO 849 (so you don't re-derive it)

The only thing that has ever moved this score is **adopting a published agent from a
higher-ranked author**, twice. Method:

```bash
TOKEN="$(cat ~/.kaggle/access_token)"
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/kernels/list?competition=pokemon-tcg-ai-battle&sortBy=dateCreated&pageSize=60"
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/competitions/pokemon-tcg-ai-battle/leaderboard/download" -o lb.zip
# join on TeamMemberUserNames -> rank. leaderboard/view returns only the top 20.
```
Rank the **author**, not the notebook — the 85-vote notebook is rank 530, the 43-vote one rank 121.

**This is now exhausted.** Every public agent has been extracted and field-tested:

| agent | field | projected |
|---|---|---|
| **w8_grimm_tuned (ours, live)** | **0.6376** | **830** |
| `_sub_handwritten_v26` (a w8 sub-policy alone) | 0.6329 | 827 |
| `w12_routed` | 0.6228 | 819 |
| `w10_coal2` | 0.6114 | 811 |
| `w7_grimm_safe` | 0.6030 | 805 |
| `w9_fastsetup3` | 0.5969 | 800 |
| `v61_codex_safe` (jazivxt Alakazam) | 0.4914 | 726 |
| `s_mega` | 0.4503 | 697 |
| `w6_kangaskhan` | 0.1604 | 334 |

No top-20 team publishes anything. jazivxt (rank 121) publishes builds **older than what they
run** — their live 960.8 submission plays a Crustle deck matching neither of their notebooks
(verified by reading the decklist out of their own replay).

---

## 4. REFUTED — DO NOT REDO ANY OF THESE

Every entry measured at n≥240 unless noted. Re-running them wastes days.

### On the current Grimmsnarl base
| idea | result |
|---|---|
| **Coalition weight tuning** (`coalition_weights.json`) | **The coalition NEVER FIRES.** Instrumented over 933 live decisions vs Grimmsnarl: `matchup_router` 351 overrides, residual/tactical/development 1 each, advisor 0, **coalition_expert 0**. It is gated behind `profile == "grass_fast" and confidence >= 0.45`. Hours were spent tuning a dead knob |
| **Router table search** (which expert per archetype) | 0.6228 independent vs 0.6376 base. `work/tools/tune_router.py` works correctly; the shipped routing is already good |
| **Setup speed** (`w9_fastsetup`) | 0.5969. The "attacks by turn 4 → 2-0, turn 5+ → 1-7" split from live games is **CONFOUNDED** — attacking early indicates a good *draw* (Impidimp + Rare Candy + energy), not a better policy. Forcing it changes a symptom |
| Coalition-worse-than-its-members | A 60-game sample said 5 of 6 sub-policies beat the ensemble. At n=1116: 0.6329 vs 0.6376, a tie |

### On the previous Alakazam base (`v61`)
4× determinizations 0.5375 · live-state fix 0.5208 · terminal playouts 0.4625 · real opponent
decklists 0.4542 · deck-out margin 2 → 0.5333 / margin 4 → 0.4625 · **all of them stacked → 0.4467**
(the stack test is what proved the individual readings were noise).

### Deck / archetype
| idea | result |
|---|---|
| **Grass type-counter** | Grimmsnarl IS weak to Grass (verified, weakness=1), but both published Grass agents lose to it: `g1_venusaur` **0.100**, `g2_leafeon` **0.160**. Type advantage does not survive a weak pilot — which is why zero of the top 50 play Grass |
| **Their live decklist on the published policy** (`p4_crustle_live`) | **0.1167** (28-212). Sharpest deck/policy-coupling result in the repo: a deck is not transferable without the policy it was tuned with |
| Enriching Energy | **Not legal in this format** — the engine rejects the deck outright. 46 of 49 policy-named cards are legal; see `work/out/pool_legality.json` |

### Historical (previous sessions, still valid)
Behavioural cloning 389.3 · learned action-scorer 400.1 · 1-ply hand-written eval 0.3500 ·
2-ply PIMC 0.0530 · value-net leaf 0.3667 · damage-model correctness fixes: per-attack error
11.8%→4.2%, phantoms → 0, and **no win-rate gain** (0.4770) · policy-weight and deck hill-climbing
on the old base found nothing over ~50 confirmed rounds.

---

## 5. WHY THIS BASE CANNOT REACH 1040 — THE ARITHMETIC

Every agent in the repo measured against `w5_grimmsnarl`:

```
w8_grimm_tuned (ours)   0.530   <- the best anti-Grimmsnarl agent that exists, anywhere
_sub_handwritten_v26    0.527
s_mega                  0.467
z_roman950              0.425
w1_alakazam             0.308
p3_crustle              0.208
v61_codex_safe          0.207
g2_leafeon              0.160
g1_venusaur             0.100
```

Grimmsnarl is **30% of the top-50 field** and 0.530 is the ceiling of the response. So:

```
field = 0.30 x 0.530  +  0.70 x (everything else)
        rest 0.72 today   -> 0.638 -> 830   (measured, live-confirmed)
        rest 0.85         -> 0.754 -> 927
        rest 0.92         -> 0.798 -> 1000
```

**1040 requires ~0.92 against every non-Grimmsnarl archetype** while holding the mirror. We are at
0.72 there (Crustle 0.811, Alakazam 0.747, Dragapult 0.720, Archaludon 0.629, Lucario 0.446).
Tuning has been exhausted. The only structural way past this is a deck that **beats** Grimmsnarl
rather than coin-flipping it.

---

## 6. THE ONE LIVE LEAD: DECIDUEYE

**Proven in-engine, not theorised:**

```
Marnie's Grimmsnarl ex   320 HP, 2 prizes, Shadow Bullet 180 for 2 energy, weak to Grass
Decidueye (id 129)       150 HP, 1 PRIZE,  Power Shot   170 for 1 energy
                         cost: discard a Basic {G} Energy from hand
                         second attack: Stock Up on Feathers, free, draw until you hold 7
```

170 × 2 weakness = **340 ≥ 320**. Confirmed in a real game log:
`t9 US ATTACK Decidueye Power Shot base=170` → `t9 OPP DAMAGE Marnie's Grimmsnarl -340`.

A one-prize Pokemon one-shotting their two-prize attacker for one energy: **they need six
knockouts, we need three.** That is the only structure found that beats the 30% slice.

### Status: built, does not work yet
`work/tools/build_decidueye.py` → `work/agents/v200_decidueye`. Deck is legal (engine accepts it),
60 cards, 9 Basics, 15 Grass energy. **Win rate 0.017.**

Diagnosed with `work/tools/power_shot_rate.py` (Power Shots per game is the metric — win rate
cannot steer this build):

```
Power Shot             0.50 /game     <- needs ~2.5-3
Stock Up on Feathers   0.77 /game
prizes taken           0.65 /game

assembly chain over 20 games:
  Rowlet in play        17/20
  Rare Candy drawn      13/20
  DECIDUEYE in play     12/20
  Decidueye ACTIVE       1.50 turns/game
    ...armed (>=1E)      0.70/game
    ...and Grass in hand 0.80/game
```

**The bottleneck is Stage-2 consistency**, not the attacker. Three turns to build a 150 HP body
that dies to one Shadow Bullet, and no armed replacement ready when it dies.

Two policy bugs already found and fixed in it (both in `build_decidueye.py`, keep them):
1. Attacks scored above EVOLVE, so it swung Rowlet's 0-damage "Add On" for five turns —
   **attacking ends the turn**, so an attack must be either the play that takes a prize or the
   last action of a turn.
2. `AreaType.HAND` hardcoded in the ATTACH card lookup instead of `opt.area`, so energy never
   attached and Power Shot could never fire.

### What to try next on it
- More consistency, not more attackers: the deck already has 4 Poffin / 4 Ultra Ball / 4 Pokégear
  / 4 Bug Catching Set / 4 Lillie's. Verify the POLICY actually plays them (count card plays the
  way `power_shot_rate.py` counts attacks) before adding more.
- Iterate against **`power_shot_rate.py`**, not win rate. Target ≥2.5 Power Shots/game, then
  `field_test.py`. Win rate at n=60 cannot see progress here.
- Only 4 Decidueye exist in the deck and each survives ~1 turn; check whether that alone caps the
  attack count before blaming the engine.

---

## 7. TOOLING

| tool | what it does |
|---|---|
| `field_test.py` | **The yardstick.** Weighted top-band field win rate → projected rating. Calibrated, predicts live scores |
| `build_and_gate.py` | 27 pre-submission checks + tarball. **Never skip.** Handles nested bundles now |
| `gauntlet.py` | accumulating, content-hashed head-to-head. `HARNESS` constant is in the hash — bump it if you change how agents are driven |
| `game_review.py` | per-game post-mortem of a live submission: opponent decklist from their setup frame, rating, how the game ended |
| `top_decks.py` | what the top N teams actually play, read from their replays |
| `power_shot_rate.py` | attacks-per-game diagnostic for the Decidueye build |
| `setup_speed.py` | turn of our first attack, one number per game |
| `agent_drift.py` | does an agent slow down as its process lives? Catches engine search-state leaks |
| `tune_coalition.py` / `tune_router.py` / `tune_deck.py` / `tune_weights.py` | searchers. **All must be validated by `field_test.py`** — see §1c |
| `build_grimm_safe.py`, `build_decidueye.py`, `build_codex_variants.py` | reproducible agent builders; every patch asserts its anchor was found |
| `loss_autopsy.py` | downloads our ladder replays, reports the meta we actually face |

---

## 8. GOTCHAS THAT COST REAL TIME

1. **`__file__` is undefined** — the harness `exec()`s `main.py`. Guarded use is fine.
2. **The last callable in the module dict wins**, by dict POSITION, whatever it is named. Do not
   require it to be called `agent` — the gate used to and it rejected a rank-121 author's bundle.
3. **THE SETUP FRAME** (`select is None`) is part of the contract: the agent returns its 60 card
   ids there, and agents initialise per-episode state in it. Both harnesses now issue it **once
   per game**. Getting this wrong measured a crippled agent for a whole session.
4. **`from X import Y` is a no-op if X is already in `sys.modules`** — a searcher that rewrites a
   bundle's config file must evict every module loaded from an agent directory before re-loading,
   and must **assert the config actually bound** before counting a game. This silently invalidated
   an entire router search until caught.
5. **Assets must resolve from the bundle, never the cwd.** Three separate agents in this repo have
   shipped broken because of it.
6. **Give every parallel worker its own agent directory** — searchers write config into the bundle.
7. **The kaggle MCP server caches credentials at startup** and returns `Unauthenticated`. Use the
   CLI or REST with `Authorization: Bearer $(cat ~/.kaggle/access_token)`.
8. **`export PYTHONIOENCODING=utf-8`** — the é in the repo path crashes the CLI otherwise.
9. Replay logs **repeat verbatim across frames**; dedupe by `(turn, serial, type, cardId, value)`
   or you will double-count every event.

---

## 9. SUBMISSION MECHANICS

- **5/day**, resets 00:00 UTC. **Only the latest 2 stay active**, and the leaderboard score is
  the **max of the two**.
- **Each submission evicts the OLDER active.** This is a ratchet: if your good draw is the older
  one, a new submission kills it. Do not submit twice in a row while holding a good score.
- Draw variance is a real, legitimate lever — byte-identical bundles have read 85 points apart —
  but the ratchet means you cannot farm it freely.

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
export PYTHONIOENCODING=utf-8
python work/tools/build_and_gate.py --agent <name> --games 8      # MANDATORY
./.venv/Scripts/kaggle.exe competitions submit -c pokemon-tcg-ai-battle \
    -f work/out/<name>.tar.gz -m "<description> [uid=...]"
```

---

## 10. THE OTHER COMPETITION — $240,000, DEADLINE 2026-09-13

`pokemon-tcg-ai-battle-challenge-strategy` is a **separate competition**, four weeks after the
ladder closes. **8 finalists × $30,000.** Submission is a **Kaggle Writeup, max 2000 words**, at
`/competitions/pokemon-tcg-ai-battle-challenge-strategy/projects` — a saved draft that is never
explicitly *Submitted* is not judged.

Scoring: **Model 70%** (clarity, **originality and technical soundness**, consistency under
repeated matches, **avoiding over-reliance on specific matchups**) / **Deck 20%** / **Report 10%**.
It states explicitly that mid-tier ladder teams can still score highly through deep analysis.

`WRITEUP_DRAFT.md` indexes the evidence we own. The strongest material is the measurement work:
the calibrated field instrument, the noise floor measured on byte-identical twins, the
refuted-ideas ledger, and the four silently-broken-component bugs — that is genuinely original and
directly answers three of the five Model-score bullets.

---

## 11. IMMEDIATE STATE / LOOSE ENDS

- **Nothing is running** (verify: `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`).
- **Uncommitted:** `work/tools/power_shot_rate.py`, `work/agents/w12_routed/`, and edits to
  `work/tools/build_decidueye.py`. Commit or discard deliberately.
- **9 commits are unpushed.** `git push` fails: Git Credential Manager's GitHub token stopped
  working mid-session and it cannot prompt non-interactively
  (`could not read Username for 'https://github.com'`). **Someone must run `git push` once from a
  real terminal, or `gh auth login`.** Nothing is lost; everything is committed locally.
- Search state files, if you want to resume any of them: `work/out/tune_router.json`,
  `work/out/tune_coalition.json`, `work/out/tune_deck.json`, `work/out/tune_weights.json`.
- Scratch bundles `_route_w*`, `_coal_w*`, `_deck_w*`, `_sub_*`, `w9_fastsetup*` are gitignored
  and rebuildable; delete freely.

## 12. IF YOU DO ONE THING

Make the Decidueye build work (§6). It is the only structure found that beats the 30% of the field
that caps us, its damage is verified in-engine, and its failure is a well-localised consistency
problem with a low-variance metric (`power_shot_rate.py`) to steer by. Everything else on this base
is measured and closed.

Do not promise the user a number. Two honest calibration points exist (726 → 726.1, 830 → 829.5);
use them, and say plainly when a projection is below 1040.
