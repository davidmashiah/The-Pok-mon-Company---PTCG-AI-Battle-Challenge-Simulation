# Strategy Category write-up — working draft

**Target:** `pokemon-tcg-ai-battle-challenge-strategy`, deadline **2026-09-13**, $240,000
(8 finalists × $30,000). Submission is a **Kaggle Writeup, max 2000 words**, created at
`/competitions/pokemon-tcg-ai-battle-challenge-strategy/projects` and then explicitly **Submitted**
— a saved draft is not judged.

**Scoring (verbatim from the Evaluation page):**

| | weight | what it asks |
|---|---|---|
| Model | **70%** | clarity of approach and rationale; **originality and technical soundness**; **consistency under repeated matches**; **avoids over-reliance on specific initial states, matchups or situational advantages**; performance in the track |
| Deck | **20%** | deck concept articulated and aligned with the strategy; key cards justified |
| Report | **10%** | structure; figures, charts, tables |

The page states plainly that a high ladder rank "does not guarantee a strong result" and that
mid-tier participants "can still achieve high overall scores through deep analysis, originality,
and well-structured reporting."

---

## The thesis to write around

**Our original contribution is not an agent. It is a measurement instrument, and the ledger of
what it refuted.**

That is an honest reading of this project's record and it happens to line up with three of the
five Model-score bullets. Forty ideas were built and measured here; every one generated
internally landed inside the noise floor. What is genuinely rare — and defensible — is that we
can *prove* that, with confirmation protocols, sample sizes and controls that most entries will
not have.

The risk to manage: the agent we ship is adopted, not invented. Originality is scored explicitly.
So the write-up must be about the method, and the shipped agent must carry our own measured
additions (§ "What is ours").

---

## Evidence inventory (all measured, all reproducible from this repo)

### 1. Local self-play metrics do not merely fail here — they invert
A controlled case: `v23_dz` scored **389.3** on the ladder while three local metrics said it was
better or equal.

| metric | verdict | reality |
|---|---|---|
| head-to-head vs v14 (438 games) | 0.5297 → "better" | **wrong direction** |
| meta_arena vs field decks (418 games) | 0.7105 vs 0.7239 → "same" | **blind** |
| replay agreement with a strong pilot | 0.2398 vs 0.1853 → "better" | **wrong direction** |
| ladder | — | 0.407 win rate vs v14's 0.609 |

Cause: in every local test *our own ~700-level policy piloted the opponent*. Against opposition
that weak almost anything wins ~71%, so differences that matter against the real field are
invisible or inverted. **Figure: this table.**

### 2. We quantified the noise floor instead of assuming it
Byte-identical bundles, submitted minutes apart, scored **64, 115 and 62 points apart**. One twin
read 791.9 → 882.8 → 637.3 → 670.3 as episodes accumulated. Rule adopted: **<100 points is noise;
a submission means nothing under ~25 episodes.**
This is the direct answer to "how consistently does the model perform under repeated matches" —
we measured our own reproducibility rather than reporting a single lucky draw.
**Figure: score-vs-episode-count curve for the identical twins.**

### 3. The fix: benchmark against real published agents, with confirmation
Replacing the self-play opponent with independent published agents produced the project's first
clean separations. Paired with a rule that nothing is adopted on a screen alone — every candidate
must also win a confirmation run on disjoint seeds. **That stage rejected roughly 8 of every 10
candidates that screened better.**
**Figure: screen win-rate vs confirmation win-rate scatter, showing the 8-in-10 rejection.**

### 4. The meta we optimised for was the wrong one — found from our own replays
An earlier model said Marnie's Grimmsnarl ex was ~53% of the field and our worst matchup. Our own
149 downloaded ladder replays said it is **7.4% of our games and we win 0.818**. The real spread
was mirror 23.5% / Alakazam 18.8% / Crustle 15.4% / Archaludon 9.4%. Four consecutive agent
versions had optimised a matchup we already dominated.
Also: against teams *on* the leaderboard we won **0.339**; against unranked ones **0.677**. The
blended 0.55 was flattering.
**Figure: opponent-archetype share vs our win rate, split by opponent rating band.**

### 5. Correctness is not win rate — a result worth reporting because it is counter-intuitive
`damage_model_audit.py` counts *attacks*, not games: for every attack, did the planner's KO
prediction match the engine? Fixing the damage model took per-attack error **11.8% → 4.2%** with
phantom KOs to **0** — and produced **no win-rate gain at all** (0.4770, n=239).
**Figure: error-rate bars (MISSED / PHANTOM) next to the flat win-rate.**

### 6. Silent failure is the dominant failure mode in this environment
Five shipped components were dead code. Named, with mechanism:
- `search_begin(obs, your_deck=...)` — the call every public agent we inspected makes raises
  `TypeError` and is swallowed by a bare `except`. Their search is dead code.
- `AdvancedPolicy.choose()` returns `ranked[:maxCount]` and `maxCount` is 1 on MAIN, so any
  search fed from it received **one** candidate and silently did nothing.
- `main.py` wrote `deck.csv` from a hardcoded constant at import and read it back, so a bundled
  decklist could never win. One submission's entire result was void.
- Model weights resolved relative to cwd, so `chdir` made the net fail to load silently and the
  agent degraded to the plain heuristic — producing a *fake null result*.
- Deferred `search_release` left tens of thousands of engine states live; one move took
  **1,089,510 ms**.
Response: `liveness_check.py` asserts every new branch actually executed, and a 27-check gate runs
before any submission. **This section is the strongest originality material we have.**

### 7. Deck choice, grounded in 13,444 real player-vs-player games
`deck_choice.py` computes archetype-vs-archetype win rates from real episodes with our code never
touching play. Needed for the 20% Deck score — must be **recomputed for the deck we now ship**.

---

## What is ours (needed to defend the originality bullet)

The shipped agent's base is public. Our own measured contributions on top:
- the harness fixes (setup frame, entry-point binding, deck resolution) — each one a bug that
  silently understated or corrupted a measurement;
- the opponent-belief library: the base determinizes most of the field as a deck that **cannot
  exist**; we ship 70 decklists built from decks we actually faced, plus a multiset matcher that
  refuses a match it cannot support;
- spending the compute budget: the base uses ~6 s of the 600 s per-episode allowance;
- the instrument and the ledger above.

**TODO before submitting:** finalise which of these survived measurement at n≥240, and report the
ones that did not with the same prominence as the ones that did. The negative results are the
point.

---

## Structure (2000 words)

1. **The problem nobody states** — in an imperfect-information ladder with a ~100-point noise
   floor, the binding constraint is measurement, not policy. (~250 w)
2. **Building an instrument** — real opponents, confirmation on disjoint seeds, content-hashed
   accumulating store, mechanical checks that resolve where win rates cannot. (~450 w)
3. **What it refuted** — the ledger, including our own favourite ideas. (~400 w)
4. **The deck** — archetype evidence from 13,444 real games; key cards and the game plan. (~350 w)
5. **The agent** — what it does, what is adopted and credited, what is ours. (~350 w)
6. **What we would do next / limits** — the matchup we still lose, stated openly. (~200 w)

Crediting the adopted base explicitly is not a weakness to hide; a report that is candid about
provenance and rigorous about measurement is exactly what a 10%-weighted Report score and a
"technically sound" Model score reward.
