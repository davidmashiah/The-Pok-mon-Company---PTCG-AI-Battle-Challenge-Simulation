# PTCG AI Battle — research log

Running record of what we tried, what we measured, and what turned out to be
false. Kept in the order things were learned, because the negative results are
the load-bearing ones.

Convention: every claim is tagged **[measured]**, **[inferred]**, or
**[assumed]**. Superseded claims are struck through, not deleted.

---

## 1. The public forward-search agents do not search

**[measured]** Every public "search" agent we inspected calls

```python
sbi = search_begin(obs, your_deck=yd)
```

The real signature is

```python
search_begin(agent_observation, your_deck, your_prize, opponent_deck,
             opponent_prize, opponent_hand, opponent_active, manual_coin=False)
```

so the call raises `TypeError: missing 5 required positional arguments` on
**every** invocation. It is swallowed by a bare `except Exception: return None`,
which silently falls back to the heuristic.

Two independent confirmations:
- Called directly, uncaught, it raises as predicted.
- Instrumenting `simulate_action` over 5 full games: **0 successful calls**, and
  ~3 ms total spent inside `SEARCH_ALGO` per game — far too little to contain a
  simulation.

A widely-forked agent advertising `SEARCH_TIME_BUDGET=1.5`, beam width 3 and 15
MCTS iterations is therefore a **pure heuristic**. Its published score is
achievable with no search at all.

*Correct usage is in `work/lib/fsearch.py`: 179/179 `search_begin` calls
succeed with a real determinization.*

## 2. Working search is not the same as useful search

**[measured]** Having fixed it, search still did not clearly help in mirror
matches:

| variant | vs baseline | games | Wilson 95% |
|---|---|---|---|
| lethal-only search | 0.486 for baseline | 360 | [0.435, 0.538] |
| + full-turn re-ranking | 0.469 for baseline | 360 | [0.419, 0.521] |
| lethal + prize tracker | 0.420 for baseline | 119 | [0.335, 0.510] |

All inconclusive. Pooled, search wins ~53% (n=839) — suggestive, not
established. The heuristic already carries a large lethal bonus and attacks
greedily, so proven-lethal search rarely finds a knockout it missed.

~~Search is the decisive edge.~~ Superseded: search *works* but its value is
unproven; the mirror-match test that produced these numbers is itself suspect
(§4).

## 3. Copying the #1 player's deck made us measurably worse

**[measured]** Decks are recoverable from public replays (the deck is the
agent's action on the deck-selection step). We extracted the exact 60 cards of
the rank-1 team (1275.3) and swapped it in.

With policy held fixed and no search on either side, 298 games:

```
our original deck  61.1%   Wilson95% [0.554, 0.664]
```

CI entirely above 0.5 — the top deck is **worse in our hands**.

**[inferred]** A rule-based policy is tuned to its own list. Ours keys on Dusk
Ball / Carmine / Gravity Mountain; theirs wants Judge timing, Ultra Ball's
two-card discard and Wally's energy-stripping heal. The stronger list in
stronger hands is the weaker list in ours.

Process note: the first attempt changed deck, card rules and search budget
together and produced an uninterpretable result. Single-variable agents had to
be rebuilt to attribute it.

## 4. Mirror-match validation was the wrong instrument

**[measured]** Scraping 67 leaderboard teams yields 31 distinct decklists
(scores 988–1275). The field is roughly a third Marnie's Grimmsnarl ex; our
Mega Lucario ex archetype is played by exactly **one** team.

Every measurement in §2 was our deck vs our deck with our policy on both sides.
That optimises for beating yourself. `work/tools/meta_arena.py` replaces it
with play against the scraped field.

Field result for the current agent (per-archetype ranking is the signal; the
absolute number is optimistic because opponents are piloted by *our* policy and
therefore underplay their decks):

| matchup | win rate |
|---|---|
| Teal Mask Ogerpon ex | **0.592**, 0.658 |
| Fezandipiti ex | **0.667**, 0.667 |
| Mega Lucario ex (the #1 list) | 0.725 |
| Marnie's Grimmsnarl ex (~1/3 of field) | 0.958–0.992 |

**[measured]** This refuted a theory we were about to build on: Mega Lucario ex
is weak to Psychic and every Grimmsnarl list runs Munkidori ×4, so Grimmsnarl
looked like the matchup to fix. It is in fact our *best* matchup.

**[inferred]** The real losses come from engine denial, not damage races.
`Fezandipiti ex`'s `Cruel Arrow` does 100 damage to *any* Pokémon including the
bench, which kills an 80 HP Riolu before it can evolve into Mega Lucario ex —
despite Fezandipiti itself being weak to Fighting. `Teal Mask Ogerpon ex`'s
`Myriad Leaf Shower` gains +30 per Energy on **both** actives, so powering up
our attacker feeds their damage.

## 5. Imitating the rank-1 agent's tempo: hypothesis refuted

We and the rank-1 team (1275.3) play the **same archetype**, so the gap is
policy quality rather than deck choice. Replays expose the full action
sequence, so their play is directly measurable.

**[measured]** Across 9 of their episodes: Mega Lucario ex reaches the field on
turn 3–4 (median 4), and `Hero's Cape` is attached to **Riolu** in 3 of the 4
games where it was attached at all.

**[inferred]** Caping Riolu dominates caping Mega Lucario ex: Tools persist
through evolution, so it protects the fragile 80 HP link *and* still ends up on
a 440 HP attacker. The upstream policy scored Mega Lucario ex higher (+200 vs
+100), buying only the second half.

**[measured]** But the implied hypothesis — that we are slower and cape the
wrong target — is **false**. Our agent over 40 games: Mega Lucario ex on the
field at median turn **3** (faster than theirs), and Hero's Cape already goes
to Riolu 23/40 games versus Mega Lucario ex 14/40.

So tempo is not the gap, and the cape preference was already mostly correct.
Two plausible explanations for a 325-point difference, both eliminated by
measurement before any of it was built on.

## 6. Benchmarking against a top agent's actual decisions, without submitting

Every metric above is self-play in some form, which measures how well we beat
ourselves. But replays store, for every step, **the exact observation an agent
saw and the action it chose**. So a rated agent's games can be replayed through
our policy and the disagreements counted directly — a validation signal that
costs no submission and never puts our policy on both sides.
(`work/tools/mimic_eval.py`.)

**[measured]** Against the rank-1 agent (1275.3) over 9 of its episodes, 449
non-trivial decisions (options ≥ 2; forced moves excluded because agreeing
there is free):

```
agreement 106/449 = 0.236
```

Control for the obvious confound — that we were being judged on *their* deck,
which contains cards our policy has no rules for — by re-running with a variant
carrying their exact list *and* rules for Judge / Ultra Ball / Wally's
Compassion: **0.232**. The deck explains essentially none of the divergence.

**[inferred]** MAIN-context agreement (0.19) is *overstated as a problem*:
actions within a turn commute, so attach-then-ability versus
ability-then-attach counts as two disagreements for an identical turn. The
trustworthy figures are the order-independent contexts, and they are worse:

| context | agreement |
|---|---|
| DISCARD — which card to pitch | 1/23 = **0.043** |
| ATTACH_TO — which Pokémon gets the Energy | 0/10 = **0.000** |
| TO_HAND — what to search for | 32/103 = 0.311 |
| SWITCH — what to promote | 5/20 = 0.250 |

**[measured]** Breaking down the SWITCH disagreements by card gives a single
coherent story: **9 of 12 were them promoting a cheap 1-prize body**
(Makuhita ×5, Lunatone ×2, Solrock ×2, Hariyama ×1) **where we promoted Mega
Lucario ex.**

**[inferred]** This is prize economy, and our policy had it backwards. Knocking
out a Mega Evolution Pokémon ex awards the opponent **3** prizes; a Makuhita
awards 1. The upstream policy gave Mega Lucario ex the single largest promotion
bonus (`score += 20`), so it volunteered its 3-prize attacker into the Active
Spot even with no Energy on it — standing there purely as a target. Fixed by
penalising promotion of a high-prize Pokémon that cannot attack
(`45 × prize_count` when it holds fewer than the 2 Energy Mega Brave needs).

Caveat kept in view: agreement is a **proxy**, not the objective. Tuning
directly against it would be overfitting to imitation of one opponent, so
changes derived this way are still required to prove themselves on field win
rate before adoption.

## 7. The field evaluation saturated, and that explains the null results

**[measured]** Four structurally different agents, evaluated on the same 25
scraped decklists:

| agent | field win rate | n |
|---|---|---|
| v2 champion (pure heuristic) | 0.8943 [0.8827, 0.9050] | 2924 |
| v7 (+ working full-turn search) | 0.8922 [0.8731, 0.9087] | 1169 |
| v8 (+ matchup counters) | 0.8913 [0.8795, 0.9021] | 2926 |
| v10b (+ prize-aware promotion) | 0.8872 [0.8752, 0.8982] | 2926 |

Every interval overlaps every other. Adding a working forward search moves the
number by 0.2 percentage points.

**[inferred]** This is a property of the instrument, not of the agents. The
opponents are piloted by *our* Lucario-tuned policy, so they misplay their own
decks and we beat them ~89% of the time. Near that ceiling the remaining 11% is
dominated by mulligans and prize luck, and genuine skill differences compress
into a band narrower than the sampling noise. Roughly 2900 games per arm cannot
resolve a 0.7% spread.

The mirror-match tests had the opposite failure: they sit at 50% and so
discriminate well, but they measure how well an agent beats *itself*.

Evidence that real effects are present but hidden: v10b was designed to stop
exposing our 3-prize attacker, and it moved exactly the matchups it targeted —
Fezandipiti ex from 0.667 to 0.708 and 0.667 to 0.742 — while giving the gain
back on Ogerpon. The aggregate showed none of this.

**Correction to method:** evaluate on the subset where the champion is *not*
near ceiling (the 0.59–0.78 matchups: Teal Mask Ogerpon, Fezandipiti, Mega
Froslass/Starmie, the Lucario mirror). Right opponents, and enough headroom for
a change to register. `meta_arena.py --hard`.

Generalisable lesson: **before trusting a string of null results, check whether
the metric can express a difference at all.** Eight consecutive "no
improvement" verdicts were evidence about the ruler, not the agents.

## 8. The one adopted change: an infinite Ability loop that silently loses games

Chasing win rate produced eight null results. The change that actually mattered
was found by asking why games were failing to *finish*.

**[measured]** Playing the champion against one leaderboard deck (Mega
Kangaskhan ex / Mega Venusaur ex, 1055.6), **18 of 40 games never terminated**
— they hit a 4000-selection cap rather than raising an exception.

**[measured]** Instrumenting a stalled game shows a three-step cycle repeated
333 times:

```
333x  ctx=0   ABILITY
333x  ctx=33  SWITCH_ENERGY
333x  ctx=21  ATTACH_FROM
```

**[inferred]** `_score_ability` returns a flat **30000**, the highest score
anywhere in the policy. Any Ability that can be re-activated without changing
the game state is therefore selected forever. Upstream evidently hit this once
and patched it for a single card (`_remember_lunatone_ability`) instead of
generally.

This never appears in a win-rate table, because a stalled game is dropped from
the sample rather than scored. On the real ladder it is far worse than a
dropped sample: `actTimeout=0` with a 600 s per-episode budget means the loop
burns the entire allowance and **forfeits the game**.

**Fix:** count Ability activations per (card, serial) per turn and refuse one
that has already been used `ABILITY_REPEAT_LIMIT` times.

**[measured]** Tuning that limit is a real trade-off:

| limit | hard-subset win rate | stalls |
|---|---|---|
| champion (none) | 0.7176 [0.7005, 0.7340] | 18/40 |
| 2 | 0.6995 [0.6829, 0.7170] | 0/40 |
| **6** | **0.7128 [0.6957, 0.7294]** | **0/40** |

A limit of 2 stops the loop but blocks legitimate repeated Ability use and
costs measurable win rate. A limit of 6 still bounds the loop at ~18 steps
instead of ~999 while remaining statistically indistinguishable from the
champion.

**[measured]** Whole-field confirmation, 25 decks, identical seed:

| | champion | limit 6 |
|---|---|---|
| games completed | 2926 | **2994** |
| stalls vs the Kangaskhan deck | 66 of 120 | **0** |
| overall win rate | 0.8943 [0.8827, 0.9050] | 0.8955 [0.8840, 0.9059] |

Kangaskhan variants were ~10% of the scraped top field, and we were failing to
finish 55% of those games — roughly 5% of all games lost to timeout, invisible
to every win-rate metric because the affected games were excluded from the
denominator rather than counted as losses.

**[measured]** Sweeping the entire 31-deck library, 30 games each:

| | champion | limit 6 |
|---|---|---|
| games completed | 917/930 = 0.9860 | **929/930 = 0.9989** |
| stalls | 13 | **1** |

12 of 13 stalls eliminated. One residual stall remained against a different
opponent (1083.0).

**[measured]** That residual is a *second, distinct* loop, and the per-turn
guard is structurally unable to catch it:

```
STALL at turn 1312   decks p0=30 p1=7
  133x player0  ABILITY -> ctx9 TO_DECK -> END
  133x player1  ABILITY -> ctx9 TO_DECK -> END
```

Both agents activate the Ability **once per turn**, return a card to the deck,
and end. One activation per turn never reaches a per-turn limit of 6. The
Ability puts a card back and the draw takes it again, so the deck never
depletes and the deck-out loss condition is never reached — the game ran to
turn 1312.

**Fix:** a second cap on total activations per card per *game*
(`ABILITY_GAME_LIMIT = 25`), alongside the per-turn one.

**[measured]** With both guards, over the full 31-deck library, 30 games each:

| | champion (per-turn only) | both guards |
|---|---|---|
| completed | 929/930 = 0.9989 | **930/930 = 1.0000** |
| stalls | 1 | **0** |
| hard-subset win rate | 0.7128 [0.6957, 0.7294] | 0.7163 [0.6993, 0.7328] |

Zero stalls and zero exceptions across 930 games, at no win-rate cost.

Generalisable lesson: fixing one instance of a bug class does not close the
class. The first guard was keyed to the *shape* of the observed loop (many
activations in one turn) rather than to the invariant that actually matters
(unbounded activations without progress), so a loop with a different shape
walked straight through it.

Generalisable lesson: **an evaluation that drops failed runs cannot see the
failures.** Track completion rate alongside win rate, or the most expensive
class of bug stays invisible. Eight win-rate experiments missed this entirely;
it surfaced only from asking why games were not finishing.

## 9. Local validation disagrees with the ladder about search — neither is conclusive

**[measured]** Ladder, after ~40 episodes each. Same policy, same deck, one
variable:

| submission | episodes | score |
|---|---|---|
| v3 — policy + working lethal search | 44 | **728.6** |
| v2 — identical policy, no search | 40 | 606.3 |

Episode counts are comparable, so the +122 is not a convergence artifact of one
agent having played more.

**[measured]** Local, same comparison: mirror win rate 0.486 [0.435, 0.538] for
the no-search side over 360 games — i.e. search wins 51.4%. Full-field win rate
0.892 vs 0.894. Agreement with the rank-1 agent's real decisions: 0.238 vs
0.236.

Three local instruments, three "no difference". The ladder says +122.

**[inferred]** Both sides are weaker evidence than they look:
- 40 episodes is far too few for the TrueSkill σ to have tightened; ±50–80 is
  plausible, so "+122" and "+30" are both consistent with the ladder data.
- Every local instrument has a structural reason to miss it. Self-play pits our
  policy against *itself* piloting foreign decks, so opponents misplay and
  games are decided by deck matchup rather than by fine decisions. Search is
  gated to ~30% of frames and only overrides when it *proves* a lethal, so it
  changes roughly 5–10 decisions out of 449 — invisible to an agreement metric,
  and easily lost in the noise of a win-rate metric.

The honest summary is that a change touching very few decisions, but the
*decisive* ones, is close to unmeasurable by aggregate statistics on weak
opponents. That is a property of the measurement, not evidence of absence.

~~Search is worth about +130 on the ladder.~~ Superseded: the gap is +122 and
rests on ~40 noisy episodes; it is suggestive, not established.

**How this is being handled:** do not resolve it by argument. Keep the two
components that have independent evidence (working search; the Ability-repeat
guard), ship them together, and let ladder episodes accumulate — the ladder is
the only instrument here that samples strong, diverse opponents.

## 10. A deck "improvement" that was one lucky seed

**[measured]** Per-card usage over 90 games against the hard subset shows how
often each card is played relative to how often it is drawn:

| card | copies | used/drawn |
|---|---|---|
| Boss's Orders | 3 | **0.20** |
| Switch | 2 | 0.44 |
| Gravity Mountain | 1 | 0.56 |
| Carmine | 4 | 0.60 |
| ... | | |
| Makuhita | 2 | 3.35 |

(Ratios above 1 occur because cards also arrive by search, not only by draw, so
this measures "sits in hand", not literal play rate.) Boss's Orders is drawn
310 times and played 62 — three deck slots for a card idle 80% of the time.

Three single-card swaps were screened on the hard subset, 400 games each:

| variant | seed 909 |
|---|---|
| champion deck | 0.7163 [0.6993, 0.7328] |
| **−1 Boss's Orders +1 Poké Pad** | **0.7354 [0.7186, 0.7515]** |
| −1 Boss's Orders +1 Basic Energy | 0.7052 [0.6879, 0.7218] |
| −1 Boss's Orders +1 Premium Power Pro | *illegal deck, n=0* |

The Poké Pad swap looked like the first real play-quality gain of the project.

**[measured]** It is not. Re-run on two independent seeds:

| seed | champion | −Boss +Poké Pad |
|---|---|---|
| 909 (screening seed) | 0.7163 | 0.7354 |
| 4242 | 0.7140 | **0.7105** |
| 31337 | 0.7168 | **0.7064** |
| **spread across seeds** | **0.0028** | **0.0290** |

The gain existed only on the seed it was selected on. Taking the best of three
candidates and reporting its score is a **maximum over three draws**, not an
estimate of its value.

The variance was itself the warning: the champion's score moves 0.0028 across
seeds, the candidate's 0.0290 — **ten times the spread**. A configuration whose
score swings that much across nominally equivalent runs is not a better
configuration, it is a noisier one, and high variance tends to precede a bad
mean.

Also worth recording: the Premium Power Pro variant produced `n=0` because the
base list already runs the maximum 4 copies, so the swap built an illegal
5-copy deck and every `battle_start` returned None. The deck-construction
helper had no legality check. It failed loudly rather than silently, which is
the tolerable version of that mistake — a silent version would have been scored
as a real result.

## 11. Automated deck search finds nothing, and the two-stage gate is why

An automated hill-climber (`work/tools/deck_opt.py`) mutates one card at a
time from a 163-card pool drawn from scraped leaderboard lists, scores
candidates against the hard subset, and adopts only what survives an
independent confirmation run.

**[measured]** Full run — 25 rounds, 2469 s: 13 candidates passed the cheap
screen, **0 survived confirmation**. Representative:

```
screen 0.782 > champ 0.674  ->  confirm cand 0.7048 vs champ 0.7167   reject
screen 0.754 > champ 0.697  ->  confirm cand 0.7060 vs champ 0.7143   reject
screen 0.731 > champ 0.697  ->  confirm cand 0.6782 vs champ 0.7208   reject
screen 0.691 > champ 0.663  ->  confirm cand 0.6869 vs champ 0.7345   reject
```

The first line is the whole lesson in one row: a screened gain of **+10.8
points** that was, on independent seeds, a **1.2 point loss**.

Every apparent gain evaporated on independent seeds — the same failure that
produced the false Poké Pad result in §10, caught automatically here.

**[inferred]** Two conclusions. First, the list is at a local optimum with
respect to single-card swaps *for this policy*, which is consistent with §3:
decks and policies are tuned jointly, so a policy written around this list
resists changes to it. Second, and more useful methodologically: without the
confirmation stage this run would have reported **13 improvements, all false**.
A screen-only optimiser on a noisy objective is a machine for generating
spurious results at scale — and it generates them faster than a human can
generate them by hand, which is exactly what makes it dangerous.

## 12. The ladder is the only instrument that separates these agents

**[measured]** Ladder scores against the shared v2 baseline (606.3, 40
episodes):

| agent | change vs v2 | score | episodes |
|---|---|---|---|
| v3 | + working lethal search | 720.9 | 48 |
| champion | + Ability stall guard | 691.2 | 21 |

Both changes are worth roughly +85 to +115 on the ladder. **Neither was
detectable locally**: search measured 0.486/0.469/0.892 across three
instruments (all null), and the stall guard measured *slightly negative* on
hard-subset win rate (0.7128 vs 0.7176) because the deck that triggers the
stall is not in that subset.

**[measured]** Ladder ratings are themselves unstable at low episode counts.
The champion read 550.4 → 565.6 → 626.8 → 730.4 → 734.7 → **691.2** over 4 to
21 episodes. Any single reading below ~40 episodes is a draw from a wide
distribution, not a level. A conclusion drawn at the 734.7 peak would have been
wrong within an hour.

## 13. We shipped the same swallowed-import bug we criticised

§1 documents the public agents calling `search_begin` with the wrong signature
and swallowing the `TypeError` in a bare `except`, so their search silently
never ran. We then did the same thing.

Forward search determinizes the opponent's hidden zones. To do that well it
matches their visible cards against a library of decklists scraped from public
replays (`work/lib/meta_decks.py`, 31 lists), falling back to filler cards when
nothing matches. `fsearch.meta_decks()` imports that module inside a
`try/except`, because the fallback is legitimate.

**[measured]** `meta_decks.py` was never added to the build's shared-file list,
so it was absent from **every tarball shipped**. The import raised, the bare
`except` swallowed it, `_META` became `[]`, and `match_opponent_deck` returned
`None` on every call. The feature was dead code in all four submissions.

It produced no error, no log line, and no test failure. The only visible symptom
would have been search quality that was quietly worse than intended — which is
precisely the class of defect that is invisible to win-rate metrics.

**Fixes:**
1. Ship the file.
2. A gate check that runs `fsearch.meta_decks()` **from the staged bundle** and
   fails if it returns an empty list. Verified by removing the file and
   confirming the gate exits non-zero, then restoring it and confirming
   "31 decklists visible to fsearch at runtime".

**[measured]** With it actually loaded, over 690 decision frames:

| | |
|---|---|
| frames where a match was attempted | 690 |
| matched a known decklist | 498 (72.2%) |
| ...matched the **correct** deck | 453 |
| ...matched a **wrong** deck | 45 |
| no match, fell back to filler | 192 |
| **precision when it commits** | **91.0%** |

Caveat: every opponent in this test is *in* the library, so 72.2% is an upper
bound on coverage. Ladder opponents outside the 31 lists fall back to filler,
which is the safe path.

Generalisable lesson: **a feature guarded by `try/except ImportError` cannot
report its own absence.** Anything optional-by-design needs a positive
assertion that it is present and non-empty in the artifact that ships — not in
the source tree, which is where it was all along.

## 14. Auditing that the search is actually running

Having been caught twice by silently disabled code (§1, §13),
`work/tools/audit_search_live.py` instruments a real game and separates the
four ways `find_lethal` can return `None` — gate rejection, determinization
failure, `search_begin` failure, exception — which are otherwise
indistinguishable to the caller.

**[measured]** 20 games against the hard subset:

| | |
|---|---|
| `lethal_plausible` gate | 388 pass / 1104 reject (26.0% pass) |
| `search_begin` succeeded | 388 |
| `search_begin` failed | **0** |
| proved a winning line | 52 (13.4% of searched frames) |
| time in search | 7.66 s per game, of a 600 s budget |

The search is genuinely live: no swallowed failures, and when the gate admits a
frame it proves a game-ending line about one time in seven.

**[inferred]** This also quantifies why the effect is locally unmeasurable.
About **1.3 proven lethals per player per game** means search alters a handful
of decisions — but they are the decisions that end games. Aggregate win rate
over weak opponents cannot resolve a handful of decisions; the ladder, playing
strong and diverse opponents where those decisions are contested, can.

## 15. Third instance of the same bug — and it invalidates an earlier result

Applying the §14 audit to the *other* search component found the same failure a
third time.

**[measured]** `best_action` (full-turn re-ranking) over 14 games: **87 calls,
87 returned `None`, 0.0 s spent.** The feature was a complete no-op.

Two separate causes, both silent:

1. `AdvancedPolicy.choose()` returns `ranked[:select.maxCount]`, and `maxCount`
   is **1** for MAIN selections. The re-ranker was handed a single candidate
   and hit its own `len < 2` guard. Fixed by exposing the untruncated ranking.
2. That raised the call count to 435 — still 435 no-ops. The remaining blocker
   was the **upstream dead `SEARCH_ALGO`**: on MAIN frames it returns a
   *one-element* list (`if len(candidates) == 1: return [candidates[0]] + ...`),
   which the agent preferred over the real ranking. Fixed by dropping the dead
   call from the decision path entirely — it never searched anyway (§1), and
   removing it also stops `choose()` being invoked twice per frame, which was
   double-incrementing the Ability counters.

**[measured]** After both fixes, 12 games:

| | before | after |
|---|---|---|
| calls | 435 | 946 |
| returned a ranking | 0 | 581 |
| **changed the heuristic's top choice** | **0** | **213 (22.5%)** |
| time in search | 0.0 s | 9.1 s (0.76 s/game) |

~~Full-turn search does not help (0.469 mirror / 0.892 field, inconclusive).~~
**Superseded and withdrawn.** The agent that produced those numbers (v7) shares
the same `SEARCH_ALGO` structure, so its full-turn search was almost certainly
also a no-op. Those runs measured a function that returned `None` every time.
Full-turn search has never actually been evaluated; the earlier conclusion was
about nothing.

Generalisable lesson, now three times over: **a null result from a component
you have not proven is executing is not evidence about the component.** Before
concluding "X does not help", instrument X and confirm it ran and changed
behaviour. Two of this project's three adopted changes, and one retracted
conclusion, came from that check.

## 16. Full-turn search, once it actually runs, is harmful

§15 fixed the full-turn re-ranker so that it genuinely executes (22.5% of
decisions overridden, versus 0% before). Evaluating it properly for the first
time, on the hard subset:

| agent | hard-subset win rate | n | runtime |
|---|---|---|---|
| v14 (lethal search only) | 0.7086 [0.6888, 0.7276] | 2100 | 3377 s |
| v14 (independent rerun) | 0.7090 [0.6893, 0.7281] | 2100 | 3377 s |
| **v15 (+ full-turn re-ranking)** | **0.5962 [0.5750, 0.6170]** | 2100 | 5451 s |

**[measured]** v15 is 11.2 points worse with **non-overlapping intervals**, and
60% slower. The two v14 runs agreeing to within 0.0004 confirms the harness is
reproducible, so this is the effect, not drift.

**[inferred]** The likely cause is the rollout: `best_action` evaluates a
candidate by continuing the turn with the heuristic and scoring the resulting
position with a hand-written evaluation. Both halves are weak — the evaluation
is a linear guess at position value, and the rollout inherits every flaw of the
policy it is trying to improve. Overriding a tuned heuristic on 22.5% of
decisions using a cruder value estimate makes the agent worse. Proving a
*terminal* win (§14) needs no evaluation function at all, which is why lethal
search helps and this does not.

**This is why the result matters beyond the agent:** the more sophisticated
build lost decisively. Sophistication was not evidence of quality, and shipping
it on that basis would have been a significant regression.

### It also corrects §9

§9 argued that local validation was *anti-correlated* with the ladder. That was
the wrong diagnosis. The instrument has a **resolution floor**:

| change | decisions altered | locally detectable |
|---|---|---|
| lethal search | ~2% | no — ladder shows +107 |
| full-turn search | 22.5% | yes — local shows −11.2 points |
| Ability stall guard | rare, but decides whole games | no — needs completion rate, not win rate |

Local self-play resolves changes that move a large fraction of decisions and
cannot resolve changes that move a few decisive ones. Both facts were visible
in the same data; only the second looked like anti-correlation.

## 17. Rebinding a name does not move it in the module dict

`kaggle_environments` calls the **last callable in the exec'd module dict**.
Wrapping an existing agent the obvious way defeats that:

```python
_inner_agent = agent      # NEW key -> appended at the end of the dict
def agent(...):           # EXISTING key -> keeps its ORIGINAL position
    ...
```

The wrapper reads as the last definition in the source, but the last *callable
in the dict* is `_inner_agent` — the unwrapped original. The harness calls
that, and the wrapper never runs.

**[measured]** A stall guard grafted onto the public 5th-place agent this way
appeared to half-work — 15 stalls fell to 12 — while never executing at all.
The apparent improvement was noise, which is worse than no improvement, because
it looks like evidence.

**Fix:** keep the original in a non-callable container.

```python
_INNER = {"fn": agent}    # dicts are not callable
def agent(obs):
    ...
    sel = _INNER["fn"](obs)
```

**[measured]** With the guard genuinely running, the same agent goes from
760/775 (0.9806, 15 stalls) to **775/775 (1.0000, 0 stalls)**.

**This also defeated our own gate.** The gate checked *AST source order*, where
`def agent` legitimately is last. Source order and dict order disagree exactly
in this case. The check now binds the last callable by its **dict key** and
compares that to `"agent"`, because a rebound wrapper's inner function still
has `__name__ == "agent"` and would pass a `__name__` comparison too.

Generalisable lesson: when a host selects your entry point by a rule (last
callable, first match, alphabetical), **assert against that rule as the host
evaluates it at runtime** — not against a proxy that usually agrees with it.

## 18. ~700 is a class ceiling, not our agent's flaw

**[measured]** The strongest evidence of the project. A completely different
policy (the public 5th-place agent, written by another author) on a completely
different deck (Alakazam), with our stall guards added, was submitted as a
deliberate probe with reading bands registered in advance.

| agent | policy | deck | score |
|---|---|---|---|
| v3 | romanrozen fork | Lucario | 701.3 |
| champion | + stall guard | Lucario | 707.0 |
| v14 | + search + guards | Lucario | ~713 |
| **w3** | **5th-place author's** | **Alakazam** | **667.1** (35 ep) |

Two independent policies and two independent decks land in the same band.
**~700 is where hand-written rule-based agents sit against this field**, not a
property of our particular rules.

That single result explains every null of the session: rule tweaks, deck swaps
and policy rewrites all move an agent *within* its class, and the class is what
is capped.

## 19. Deck strength, measured without our own policy in the loop

**[measured]** Every earlier deck experiment used our ~700 policy to pilot both
sides, which measures "which deck our weak policy handles best". Using the
host's top-episode dataset instead — real games between ~1085-rated agents,
our code touching nothing:

| archetype | games | win rate |
|---|---|---|
| Mega Lopunny ex + Mega Froslass ex | 27 | **0.741** |
| Mega Lopunny ex + Lillie's Clefairy ex | 52 | 0.615 |
| Mega Lopunny ex | 90 | 0.600 |
| Teal Mask Ogerpon ex | 141 | 0.532 |
| **Marnie's Grimmsnarl ex** | **735** | **0.480** |
| Mega Kangaskhan ex | 43 | 0.419 |
| **Mega Lucario ex (ours)** | **<25 of 1400** | extinct at this level |

Two things worth keeping: the most-played deck in the field (Grimmsnarl, 53% of
observed decks) is a *losing* deck, and our own archetype is essentially not
played by strong agents at all. A 25-round automated deck optimiser had earlier
concluded our list was at a local optimum — using the broken instrument.

**[inferred]** Piloting difficulty dominates theoretical power on an AI ladder.
The winning deck's engine is one decision — bench Lopunny, promote it, hit for
230 off one Colourless energy. The losing decks require multi-turn arithmetic
(Mega Absol's Terminal Period needs the opponent on *exactly* 6 damage
counters) or coin flips. Humans execute those; bots do not.

## 20. Four ways to exploit the best deck, all worse than doing nothing

The rotation was reverse-engineered from turn-by-turn traces of winning games:
two Mega Lopunny ex, Air Balloon on *the attacker* so its retreat is free, and
retreating one promotes the other, which re-triggers Gale Thrust's +170 every
turn. A Dudunsparce draw engine (draw 3, then shuffles itself back into the
deck) is built first.

**[measured]** Against v14, 396 games each:

| build | Gale Thrust 230-tier rate | win rate vs v14 |
|---|---|---|
| tuned policy + deck, **no override** | — | **0.5076 [0.459, 0.557]** |
| v21 tuned policy + 2-decision override | **63.9%** | 0.4722 [0.424, 0.521] |
| v20 policy rewritten from traces | 45.2% | **0.0379** |
| v19 policy rewritten from card text | 24.0% | **0.0253** |

Combo execution rose from 13.9% to 63.9% across these builds and win rate fell
monotonically. Optimising the visible metric (how often the combo fires)
destroyed the invisible one (everything else a tuned policy does correctly).

**[inferred]** A policy tuned against the real ladder is worth on the order of
700 rating points; hand-picked scores written in an evening are worth ~30. A
deck edge cannot cross that gap, and *forcing* the deck's combo overrides the
tuned policy's judgement about when the enabling play is affordable.

Generalisable lesson: **when replacing part of a tuned system, the correct
default is to change nothing.** Doing nothing scored 0.5076; the most careful,
most minimal intervention scored 0.4722; the full rewrites scored 0.03.

## 21. Behavioural cloning on this data does not beat a trivial baseline

**[measured]** Winner decisions from the host dataset, framed as pointwise
ranking (every option a row, label = "was it chosen"), split by decision so no
decision straddles train/test:

| model | top-1 |
|---|---|
| gradient boosting + card identity | 0.3719 |
| gradient boosting | 0.3658 |
| logistic regression | 0.3255 |
| **"always pick option 0"** | **0.3665** |
| our hand-written policy | 0.2402 |
| random | 0.2017 |

Adding card identity moved it by +0.6 points. The model never separates from
the trivial baseline.

Two things this exposes. First, **the agreement metric leaks option ordering** —
any agent that respects the engine's natural ordering scores ~0.37 for free, so
a headline "BC nearly doubles our accuracy" would have been badly wrong without
the baseline printed alongside. Second, our hand-written policy scores *below*
always-picking-the-first-option, i.e. it systematically fights that ordering.

## 22. Operational facts that are not in the competition overview

**[measured]**
- The engine ships *inside* the competition data (`sample_submission/cg/`,
  prebuilt for Windows/Linux/macOS/arm64) and is bundled into your own
  submission. `pip install cabt` does not exist.
- `cabt.json`: `actTimeout = 0`, `observation.remainingOverageTime = 600`.
  There is **no per-move limit**; each agent has 600 s for the whole episode.
  A fixed per-move budget therefore blows up in long games.
- Replay-measured budget use varies enormously: rank 1 used 24.6 s, rank 2
  14.4 s, but two top-15 teams used **339.9 s** and **226.7 s**.
  ~~Top teams barely use their compute.~~ Superseded — that came from n=1.
- `kaggle_environments` loads `main.py` via `exec()`, so **`__file__` is
  undefined** and the harness takes the **last callable** in the module. A
  submission using `__file__` for path resolution dies at import. This cost us
  one submission before the gate was hardened to load agents the same way
  production does.

---

## Method notes

- Every comparison reports a Wilson 95% interval and is only acted on when the
  interval excludes 0.5.
- Comparisons are against *what we would otherwise ship*, never an older
  baseline, so corrections cannot be double-counted.
- Candidate selection is two-stage: a cheap screen proposes, an independent
  larger run disposes. Best-of-N on a noisy score is a maximum, not an estimate.
- `work/tools/build_and_gate.py` is a mechanical gate that exits non-zero;
  every past failure is re-checked in it by name, and it is verified by
  re-staging the known-bad build and confirming it fails.
