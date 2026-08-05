"""Working forward search on the cabt engine.

Why this file exists: every public agent I inspected calls

    search_begin(obs, your_deck=yd)

which raises TypeError (5 required positional args missing) on every call and
is swallowed by a bare `except Exception`. Their search is dead code. This is
the correct call, with an honest determinization of hidden information.

Two guarantees, both load-bearing:
  * Never raise out of this module. Callers fall back to heuristics.
  * Never claim a win we cannot prove. A hallucinated lethal is worse than no
    lethal: we would tap out our board for a line the real game cannot play.
"""
import time
import random as _rnd_mod
from collections import Counter

from cg.api import (
    CardType, Observation, OptionType, SelectContext, all_card_data,
)

try:
    from cg.api import search_begin, search_end, search_release, search_step
    HAVE_SEARCH = True
except Exception:                                    # pragma: no cover
    HAVE_SEARCH = False

_CARDS = None
_META = None


def meta_decks():
    """Known opponent decklists, scraped from public leaderboard replays.

    Shipped as meta_decks.py beside the agent. Absent -> we fall back to the
    old filler determinization, so this is strictly additive.
    """
    global _META
    if _META is None:
        try:
            import meta_decks
            _META = [list(d) for d in meta_decks.DECKS if len(d) == 60]
        except Exception:
            _META = []
    return _META


def match_opponent_deck(seen_counter):
    """Pick the known decklist most consistent with what we've seen them play.

    Score = number of observed cards the candidate can actually account for,
    minus a penalty for cards it cannot. A list that cannot contain something
    we have literally watched them play is not their list.
    """
    metas = meta_decks()
    if not metas or not seen_counter:
        return None
    best, best_s = None, 0.0
    for deck in metas:
        have = Counter(deck)
        ok = miss = 0
        for cid, n in seen_counter.items():
            avail = have.get(cid, 0)
            ok += min(n, avail)
            miss += max(0, n - avail)
        s = ok - 2.5 * miss
        if s > best_s:
            best, best_s = deck, s
    # require real evidence before trusting a match
    if best is None or best_s < 4:
        return None
    return best


def _cards():
    global _CARDS
    if _CARDS is None:
        _CARDS = {c.cardId: c for c in all_card_data()}
    return _CARDS


def _i(x, d=-1):
    try:
        return int(x)
    except (TypeError, ValueError):
        return d


def _filler_ids():
    """A Basic Pokemon id and a Basic Energy id, for padding unknown zones.

    search_begin requires the opponent deck to contain >=1 Basic Pokemon at
    setup, so the padding cannot be pure energy.
    """
    cards = _cards()
    basic_mon = next((c.cardId for c in cards.values()
                      if _i(c.cardType) == 0 and c.basic and not c.ex
                      and not c.megaEx), 722)
    basic_energy = next((c.cardId for c in cards.values()
                         if _i(c.cardType) == _i(CardType.BASIC_ENERGY)), 3)
    return basic_mon, basic_energy


# ---------------------------------------------------------------- prizes
class PrizeTracker:
    """Conservative deduction of which of OUR cards are prized.

    Adapted from the 1300+ writeup by masamikobayashi (public, competition
    forums). The governing rule there, and it is the right one:

        "A wrong prize inference is worse than no prize inference."

    Why it matters for us: search_begin takes your_deck and your_prize as
    separate lists. If we guess that split arbitrarily, the engine will happily
    find a winning line that draws a card which is really sitting under a
    prize, and we will tap out our board for a line the real game cannot play.
    Returning "unknown" and falling back to the heuristic is strictly better.

    Deduction is only possible on frames where the deck is fully revealed
    (obs.select.deck is not None), e.g. during a search-card effect.
    """

    def __init__(self, decklist):
        self._decklist = list(decklist)
        self._prized = None          # Counter, or None == unknown
        self._last_prize_count = None

    def update(self, obs: Observation):
        try:
            st = obs.current
            if st is None:
                return
            me = st.players[st.yourIndex]
            pc = len(me.prize or [])

            # prizes were taken -> our deduction is stale; drop it rather than
            # carry a wrong one forward
            if (self._prized is not None and self._last_prize_count is not None
                    and pc < self._last_prize_count):
                self._prized = None
            self._last_prize_count = pc

            if self._prized is not None:
                return
            sel = obs.select
            if sel is None or sel.deck is None:
                return
            if len(sel.deck) != me.deckCount:
                return

            rem = Counter(self._decklist)

            def sub(c):
                if c is not None:
                    rem[c.id] -= 1

            for c in sel.deck:
                sub(c)
            for c in (me.hand or []):
                sub(c)
            for p in list(me.active or []) + list(me.bench or []):
                if p is None:
                    continue
                sub(p)
                for grp in (p.preEvolution, p.energyCards, p.tools):
                    for c in (grp or []):
                        sub(c)
            for c in (me.discard or []):
                sub(c)
            for c in (st.stadium or []):
                if c is not None and getattr(c, "playerIndex", None) == st.yourIndex:
                    sub(c)
            # The card currently resolving its effect has left the hand but may
            # not be in the discard yet. Missing it shifts the count by one and
            # silently corrupts the whole deduction.
            eff = getattr(sel, "effect", None)
            if eff is not None and getattr(eff, "playerIndex", None) == st.yourIndex:
                if rem.get(eff.id, 0) > 0:
                    rem[eff.id] -= 1

            if any(v < 0 for v in rem.values()):
                return                                  # inconsistent -> unknown
            inferred = Counter({k: v for k, v in rem.items() if v > 0})
            if sum(inferred.values()) != pc:
                return                                  # ambiguous -> unknown
            self._prized = inferred
        except Exception:
            self._prized = None

    def prized(self):
        return self._prized.copy() if self._prized is not None else None


# ---------------------------------------------------------------- determinize
class Determinizer:
    """Turns an observation into the six card-id lists search_begin needs.

    Our own hidden cards are *deduced*: decklist minus everything we can see.
    The opponent's are *guessed*: what we have seen them use, padded with
    filler. That asymmetry is fine for same-turn lethal checking, which is
    what we use search for.
    """

    def __init__(self, decklist):
        self.decklist = list(decklist)
        self._seen_opp = Counter()
        self.prizes = PrizeTracker(decklist)
        self.exact_prizes = 0     # how often we shipped a known-exact split
        self.guessed_prizes = 0
        self.matched_opp = 0      # opponent identified as a known meta deck
        self.unmatched_opp = 0

    def observe(self, obs: Observation):
        """Call on EVERY frame. Prize deduction needs the deck-reveal frames,
        which are not the frames where we search."""
        self.prizes.update(obs)

    def note_opponent(self, obs: Observation):
        """Accumulate opponent cards we have observed, for a better guess."""
        try:
            st = obs.current
            opp = st.players[1 - st.yourIndex]
            for c in (opp.discard or []):
                if c is not None:
                    self._seen_opp[c.id] += 0  # presence only; counts reset below
            seen = Counter()
            for c in (opp.discard or []):
                if c is not None:
                    seen[c.id] += 1
            for p in list(opp.active or []) + list(opp.bench or []):
                if p is None:
                    continue
                seen[p.id] += 1
                for grp in (p.preEvolution, p.energyCards, p.tools):
                    for c in (grp or []):
                        if c is not None:
                            seen[c.id] += 1
            self._seen_opp = seen
        except Exception:
            pass

    def _my_unseen(self, obs: Observation):
        """decklist - (hand + discard + in play + our stadium). Never negative."""
        st = obs.current
        me = st.players[st.yourIndex]
        rem = Counter(self.decklist)

        def sub(c):
            if c is not None and c.id in rem:
                rem[c.id] -= 1

        for c in (me.hand or []):
            sub(c)
        for c in (me.discard or []):
            sub(c)
        for p in list(me.active or []) + list(me.bench or []):
            if p is None:
                continue
            sub(p)
            for grp in (p.preEvolution, p.energyCards, p.tools):
                for c in (grp or []):
                    sub(c)
        for c in (st.stadium or []):
            if c is not None and getattr(c, "playerIndex", None) == st.yourIndex:
                sub(c)
        # cards currently revealed from our deck by an effect
        try:
            if obs.select is not None and obs.select.deck:
                for c in obs.select.deck:
                    sub(c)
        except Exception:
            pass
        out = []
        for cid, n in rem.items():
            if n > 0:
                out.extend([cid] * n)
        return out

    def build(self, obs: Observation, shuffle=False):
        """Return the kwargs for search_begin, or None if we cannot.

        With shuffle=True the hidden pools are permuted, so repeated calls give
        DIFFERENT guesses at what is in the deck/prizes/opponent hand. Averaging
        over several such guesses is the whole point of PIMC -- without it,
        sampling N times just re-scores one fixed guess N times.
        """
        try:
            st = obs.current
            me = st.players[st.yourIndex]
            opp = st.players[1 - st.yourIndex]
            mon, energy = _filler_ids()

            unseen = self._my_unseen(obs)
            if shuffle:
                _rnd.shuffle(unseen)
            need_deck = me.deckCount
            need_prize = len(me.prize or [])

            # If the tracker knows exactly which cards are prized, honour it:
            # prized cards go to your_prize and are REMOVED from your_deck, so
            # search cannot draw a card that is really under a prize.
            known = self.prizes.prized()
            your_prize = []
            if known is not None and sum(known.values()) == need_prize:
                pool = Counter(unseen)
                if all(pool.get(cid, 0) >= n for cid, n in known.items()):
                    for cid, n in known.items():
                        your_prize.extend([cid] * n)
                        pool[cid] -= n
                    unseen = []
                    for cid, n in pool.items():
                        if n > 0:
                            unseen.extend([cid] * n)
                    self.exact_prizes += 1
                else:
                    known = None
            else:
                known = None
            if known is None:
                self.guessed_prizes += 1

            if shuffle:
                _rnd.shuffle(unseen)
            # pad generously: search_begin only errors when a list is TOO SHORT
            while len(unseen) < need_deck:
                unseen.append(energy)
            your_deck = unseen[:need_deck] if need_deck else []
            if not your_prize:
                your_prize = unseen[need_deck:need_deck + need_prize]
            while len(your_prize) < need_prize:
                your_prize.append(energy)

            # opponent: if their visible cards identify a known leaderboard
            # decklist, determinize their hidden zones with THAT list minus
            # what we have already seen. Otherwise fall back to filler.
            need_opp = (opp.deckCount + len(opp.prize or []) + opp.handCount)
            matched = match_opponent_deck(self._seen_opp)
            opp_pool = []
            if matched is not None:
                self.matched_opp += 1
                rest = Counter(matched)
                for cid, n in self._seen_opp.items():
                    rest[cid] -= n
                for cid, n in rest.items():
                    if n > 0:
                        opp_pool.extend([cid] * n)
            else:
                self.unmatched_opp += 1
                for cid, n in self._seen_opp.items():
                    opp_pool.extend([cid] * n)
            while len(opp_pool) < need_opp + 4:
                opp_pool.append(energy if len(opp_pool) % 3 else mon)
            # the engine requires a Basic Pokemon somewhere in the opp deck
            opp_deck = opp_pool[:opp.deckCount] if opp.deckCount else []
            if opp.deckCount and mon not in opp_deck:
                opp_deck[0] = mon
            k = opp.deckCount
            opp_prize = opp_pool[k:k + len(opp.prize or [])]
            while len(opp_prize) < len(opp.prize or []):
                opp_prize.append(energy)
            k += len(opp.prize or [])
            opp_hand = opp_pool[k:k + opp.handCount]
            while len(opp_hand) < opp.handCount:
                opp_hand.append(energy)

            opp_active = []
            act = opp.active or []
            if len(act) > 0 and act[0] is None:
                opp_active = [mon]

            return dict(your_deck=your_deck, your_prize=your_prize,
                        opponent_deck=opp_deck, opponent_prize=opp_prize,
                        opponent_hand=opp_hand, opponent_active=opp_active)
        except Exception:
            return None


# ---------------------------------------------------------------- gating
_ATTACKS = None


def _attacks():
    global _ATTACKS
    if _ATTACKS is None:
        from cg.api import all_attack
        _ATTACKS = {a.attackId: a for a in all_attack()}
    return _ATTACKS


def lethal_plausible(obs: Observation):
    """Is it worth spending search budget on this frame?

    Running the full search on every MAIN frame spends nearly all of it proving
    a negative: most turns simply have no winning line. Gating on a cheap upper
    bound of our damage means the budget goes to the frames that can actually
    end the game, and lets us search those much deeper.

    Deliberately generous -- a missed lethal costs a game, a wasted search
    costs milliseconds.
    """
    try:
        st = obs.current
        if st is None:
            return False
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]

        # find_lethal looks for an actual WIN, not merely a knockout. The engine
        # ends a game for exactly three reasons we can cause in one turn
        # (LogType.RESULT.reason): 1 = we take our last prize, 2 = they start a
        # turn with an empty deck, 3 = they have no Active Pokemon.
        #
        # An earlier version of this gate tested only the prize condition and
        # silently dropped 3 of 3 real lethals: KOing the last Pokemon of an
        # opponent with an empty bench wins on reason 3 at ANY prize count.
        my_prizes = len(me.prize or [])
        opp_bench = len(opp.bench or [])
        if opp.deckCount <= 1:
            return True                     # reason 2 in reach
        if opp_bench == 0:
            return True                     # reason 3 in reach
        # reason 1: best realistic single turn is a Mega ex KO (3) + a bench KO
        if my_prizes > 4:
            return False

        act = me.active[0] if (me.active and me.active[0]) else None
        if act is None:
            return False
        oact = opp.active[0] if (opp.active and opp.active[0]) else None
        if oact is None:
            return True                     # face-down/unknown: look anyway

        cards, atks = _cards(), _attacks()
        acard = cards.get(act.id)
        if acard is None:
            return False
        best = 0
        for aid in acard.attacks:
            a = atks.get(aid)
            if a and a.damage:
                best = max(best, a.damage)
        ocard = cards.get(oact.id)
        if (best and ocard is not None and ocard.weakness is not None
                and _i(ocard.weakness) == _i(acard.energyType)):
            best *= 2
        # headroom for damage buffs (Premium Power Pro etc.) and multi-step lines
        return oact.hp <= best * 1.6 + 80
    except Exception:
        return False


# ---------------------------------------------------------------- lethal hunt
def _won(sobs, my_index):
    try:
        cur = sobs.current
        return cur is not None and cur.result == my_index
    except Exception:
        return False


def _over(sobs):
    try:
        cur = sobs.current
        return cur is not None and cur.result != -1
    except Exception:
        return False


def find_lethal(obs: Observation, det: Determinizer, time_budget=0.8,
                node_budget=4000, max_width=8):
    """Search our own turn for a proven winning line.

    Returns the first option-index list to play, or None. Only returns a line
    that the engine itself played out to a win under this determinization.
    """
    if not HAVE_SEARCH:
        return None
    if obs.select is None or obs.current is None:
        return None
    if _i(obs.select.context) != _i(SelectContext.MAIN):
        return None
    if not lethal_plausible(obs):
        return None

    kw = det.build(obs)
    if kw is None:
        return None

    my_index = obs.current.yourIndex
    t0 = time.time()
    nodes = 0
    opened = []

    try:
        root = search_begin(obs, manual_coin=False, **kw)
    except Exception:
        return None
    if root is None:
        return None

    best = None

    def rec(state, depth, first_pick):
        nonlocal nodes, best
        if best is not None:
            return
        if time.time() - t0 > time_budget or nodes > node_budget or depth > 14:
            return
        sobs = state.observation
        if _over(sobs):
            if _won(sobs, my_index):
                best = first_pick
            return
        sel = sobs.select
        if sel is None:
            return
        # only expand OUR decisions
        if sobs.current is not None and sobs.current.yourIndex != my_index:
            return

        n = len(sel.option)
        if n == 0:
            return
        lo = max(1, sel.minCount)
        if lo > n:
            return

        # order: attacks first (they are what actually wins), then the rest
        order = sorted(range(n), key=lambda i: 0 if _i(sel.option[i].type)
                       == _i(OptionType.ATTACK) else 1)
        for i in order[:max_width]:
            if best is not None or time.time() - t0 > time_budget:
                return
            pick = [i] if sel.minCount <= 1 else list(range(lo))
            try:
                nxt = search_step(state.searchId, pick)
                nodes += 1
            except Exception:
                continue
            if nxt is None:
                continue
            opened.append(nxt.searchId)
            rec(nxt, depth + 1, first_pick if first_pick is not None else pick)

    try:
        rec(root, 0, None)
    except Exception:
        best = None
    finally:
        try:
            for sid in opened:
                try:
                    search_release(sid)
                except Exception:
                    pass
            search_end()
        except Exception:
            pass

    return best


# ---------------------------------------------------------------- evaluation
def evaluate(sobs, my_index):
    """Score a position from our point of view. Higher is better.

    Prizes dominate: the game is a race to take six, and every other term is
    only a proxy for future prizes. Keep the proxies small so they can break
    ties but never outvote an actual prize swing.
    """
    try:
        st = sobs.current
        if st is None:
            return 0.0
        if st.result == my_index:
            return 1e9
        if st.result == (1 - my_index):
            return -1e9
        if st.result == 2:
            return 0.0

        me = st.players[my_index]
        opp = st.players[1 - my_index]
        cards = _cards()
        s = 0.0

        # 1. prize race -- the actual win condition
        s += (len(opp.prize or []) - len(me.prize or [])) * 1000.0

        def board(p, sign):
            v = 0.0
            for mon in list(p.active or []) + list(p.bench or []):
                if mon is None:
                    continue
                c = cards.get(mon.id)
                v += mon.hp * 0.30                       # surviving HP
                v += len(mon.energies or []) * 12.0      # invested energy
                if c is not None:
                    # a damaged high-prize Pokemon is a liability
                    if c.megaEx:
                        v -= (mon.maxHp - mon.hp) * 0.22
                    elif c.ex:
                        v -= (mon.maxHp - mon.hp) * 0.16
            return v * sign

        s += board(me, 1.0)
        s += board(opp, -1.0)

        # 2. resources
        s += (me.handCount - opp.handCount) * 6.0
        s += (me.deckCount - opp.deckCount) * 0.5
        # decking out is a loss condition, so a nearly empty deck is dangerous
        if me.deckCount <= 3:
            s -= (4 - me.deckCount) * 250.0

        # 3. their active being nearly dead is worth real tempo
        oact = opp.active[0] if (opp.active and opp.active[0]) else None
        if oact is not None and oact.maxHp:
            s += (1.0 - oact.hp / float(oact.maxHp)) * 90.0
        else:
            s += 60.0                       # they have no active at all
        return s
    except Exception:
        return 0.0


def _rollout_to_turn_end(state, my_index, rollout_policy, opened, max_steps=80):
    """Play OUR remaining turn out with the heuristic, return the end position."""
    cur = state
    for _ in range(max_steps):
        sobs = cur.observation
        if _over(sobs):
            return sobs
        st = sobs.current
        if st is None or sobs.select is None:
            return sobs
        if st.yourIndex != my_index:
            return sobs                      # our turn is over
        try:
            sel = rollout_policy(sobs)
            nxt = search_step(cur.searchId, list(sel))
        except Exception:
            return cur.observation
        if nxt is None:
            return cur.observation
        opened.append(nxt.searchId)
        cur = nxt
    return cur.observation


_rnd = _rnd_mod.Random(12345)

SEARCH_STATS = {"calls": 0, "no_engine": 0, "not_main": 0, "cand_lt2": 0,
                "build_none": 0, "begin_none": 0, "scored_lt2": 0,
                "all_equal": 0, "ranked": 0}


def _rollout_turns(state, my_index, rollout_policy, opened, half_turns=1,
                   max_steps=200):
    """Play forward until the side to move has changed `half_turns` times.

    half_turns=1 stops at the end of OUR turn -- which is exactly the myopia
    that made the first version of this search lose: a line that dumps every
    resource looks wonderful at our own end of turn and is then punished on the
    reply. half_turns=2 plays the opponent's answer out too, so the position is
    scored after they have had their say.
    """
    cur = state
    switches = 0
    last = None
    for _ in range(max_steps):
        sobs = cur.observation
        if _over(sobs):
            return sobs
        st = sobs.current
        if st is None or sobs.select is None:
            return sobs
        if last is None:
            last = st.yourIndex
        elif st.yourIndex != last:
            switches += 1
            last = st.yourIndex
            if switches >= half_turns:
                return sobs
        try:
            sel = rollout_policy(sobs)
            nxt = search_step(cur.searchId, list(sel))
        except Exception:
            return cur.observation
        if nxt is None:
            return cur.observation
        opened.append(nxt.searchId)
        cur = nxt
    return cur.observation


def best_action2(obs: Observation, det: Determinizer, rollout_policy,
                 candidates, time_budget=2.0, max_candidates=8,
                 half_turns=2, samples=3, evaluator=None):
    """PIMC re-rank: average each candidate over several hidden-info guesses,
    scoring AFTER the opponent replies rather than at our own end of turn."""
    SEARCH_STATS["calls"] += 1
    if not HAVE_SEARCH or obs.select is None or obs.current is None:
        SEARCH_STATS["no_engine"] += 1
        return None
    if _i(obs.select.context) != _i(SelectContext.MAIN):
        SEARCH_STATS["not_main"] += 1
        return None
    cand = [c for c in candidates][:max_candidates]
    if len(cand) < 2:
        SEARCH_STATS["cand_lt2"] += 1
        return None

    my_index = obs.current.yourIndex
    t0 = time.time()
    totals = {a: 0.0 for a in cand}
    counts = {a: 0 for a in cand}
    opened = []
    try:
        for si in range(max(1, samples)):
            if time.time() - t0 > time_budget:
                break
            kw = det.build(obs, shuffle=(si > 0))
            if kw is None:
                SEARCH_STATS["build_none"] += 1
                return None
            root = search_begin(obs, manual_coin=False, **kw)
            if root is None:
                SEARCH_STATS["begin_none"] += 1
                break
            for a in cand:
                if time.time() - t0 > time_budget:
                    break
                try:
                    nxt = search_step(root.searchId, [a])
                except Exception:
                    continue
                if nxt is None:
                    continue
                opened.append(nxt.searchId)
                end = _rollout_turns(nxt, my_index, rollout_policy, opened,
                                     half_turns=half_turns)
                # A LEARNED leaf evaluator is the whole point: the hand-written
                # evaluate() is why deeper search scored WORSE (v30 0.3500,
                # v31 0.0530) -- more lookahead just optimised harder against a
                # wrong target. Fall back to it only if the net is unavailable.
                v = evaluator(end, my_index) if evaluator is not None else None
                totals[a] += evaluate(end, my_index) if v is None else v
                counts[a] += 1
    except Exception:
        return None
    finally:
        try:
            for sid in opened:
                try:
                    search_release(sid)
                except Exception:
                    pass
            search_end()
        except Exception:
            pass

    scored = [(totals[a] / counts[a], a) for a in cand if counts[a] > 0]
    if len(scored) < 2:
        SEARCH_STATS["scored_lt2"] += 1
        return None
    scored.sort(key=lambda r: -r[0])
    if abs(scored[0][0] - scored[-1][0]) < 1e-9:
        SEARCH_STATS["all_equal"] += 1
        return None
    SEARCH_STATS["ranked"] += 1
    ranked = [a for _, a in scored]
    return ranked + [c for c in candidates if c not in ranked]


PIMC_STATS = {"calls": 0, "playouts": 0, "terminal": 0, "truncated": 0,
              "ranked": 0, "no_time": 0, "begin_none": 0}


def pimc_terminal(obs: Observation, det: Determinizer, rollout_policy,
                  candidates, time_budget=6.0, samples=4, max_candidates=6,
                  max_steps=900):
    """Rank candidate actions by playing the game OUT TO THE END, many times.

    Why this and not the other two search functions in this file: both of them
    stop early and hand the position to an evaluator, and BOTH have been
    measured to make the agent worse --

        1-ply + hand-written evaluate()      v30  0.3500 vs v14
        2-ply + PIMC, same evaluator         v31  0.0530
        1-ply + learned value net as leaf    v33  0.3667 vs v43

    The common factor is the evaluator, not the depth. A playout to a terminal
    state needs no evaluator at all: the engine reports who won, and that is the
    ground truth we are actually optimising. Both sides are piloted by the same
    heuristic, so the comparison between candidates is fair even though the
    absolute play is only heuristic-strength.

    The budget this spends was simply sitting idle: the harness allows 600 s per
    EPISODE (cabt.json actTimeout=0, remainingOverageTime=600) and the current
    agents use 0.4 s (v51) to 12.9 s (v43) of it.

    Returns a re-ordered index list, or None to leave the heuristic alone.
    """
    PIMC_STATS["calls"] += 1
    if not HAVE_SEARCH or obs.select is None or obs.current is None:
        return None
    if _i(obs.select.context) != _i(SelectContext.MAIN):
        return None
    cand = [c for c in candidates][:max_candidates]
    if len(cand) < 2:
        return None

    my_index = obs.current.yourIndex
    t0 = time.time()
    wins = {a: 0.0 for a in cand}
    plays = {a: 0 for a in cand}
    opened = []
    try:
        for si in range(max(1, samples)):
            if time.time() - t0 > time_budget:
                break
            kw = det.build(obs, shuffle=(si > 0))
            if kw is None:
                return None
            try:
                root = search_begin(obs, manual_coin=False, **kw)
            except Exception:
                return None
            if root is None:
                PIMC_STATS["begin_none"] += 1
                break
            for a in cand:
                if time.time() - t0 > time_budget:
                    break
                try:
                    nxt = search_step(root.searchId, [a])
                except Exception:
                    continue
                if nxt is None:
                    continue
                # Release each playout's states AS SOON as it ends, not at the
                # end of the call. A playout is ~100 search_step calls and a
                # single call runs hundreds of playouts, so deferring cleanup
                # leaves tens of thousands of live search states alive at once;
                # the engine then slows to a crawl and one move took 18 minutes
                # (worst move 1,089,510 ms -- caught by the gate's per-move
                # timing, which is exactly what that check is for).
                mine = [nxt.searchId]
                cur = nxt
                res = None
                for _ in range(max_steps):
                    if time.time() - t0 > time_budget:
                        break
                    sobs = cur.observation
                    if _over(sobs):
                        res = sobs.current.result
                        break
                    if sobs.select is None or sobs.current is None:
                        break
                    try:
                        sel = rollout_policy(sobs)
                        nx2 = search_step(cur.searchId, list(sel))
                    except Exception:
                        break
                    if nx2 is None:
                        break
                    mine.append(nx2.searchId)
                    cur = nx2
                for sid in mine:
                    try:
                        search_release(sid)
                    except Exception:
                        pass
                PIMC_STATS["playouts"] += 1
                if res is None:
                    PIMC_STATS["truncated"] += 1
                    continue          # unfinished games tell us nothing
                PIMC_STATS["terminal"] += 1
                plays[a] += 1
                if res == my_index:
                    wins[a] += 1.0
                elif res == 2:
                    wins[a] += 0.5
    except Exception:
        return None
    finally:
        try:
            for sid in opened:
                try:
                    search_release(sid)
                except Exception:
                    pass
            search_end()
        except Exception:
            pass

    scored = [(wins[a] / plays[a], a) for a in cand if plays[a] > 0]
    if len(scored) < 2:
        PIMC_STATS["no_time"] += 1
        return None
    if abs(max(s for s, _ in scored) - min(s for s, _ in scored)) < 1e-9:
        return None
    scored.sort(key=lambda r: -r[0])
    PIMC_STATS["ranked"] += 1
    ranked = [a for _, a in scored]
    return ranked + [c for c in candidates if c not in ranked]


def best_action(obs: Observation, det: Determinizer, rollout_policy,
                candidates, time_budget=1.0, max_candidates=8):
    """Evaluate each candidate MAIN option by simulating the rest of our turn.

    `candidates` is the heuristic's own ranking; we re-rank its top few by what
    the engine says actually happens. Returns a re-ordered index list, or None
    to leave the heuristic ordering alone.
    """
    SEARCH_STATS["calls"] += 1
    if not HAVE_SEARCH or obs.select is None or obs.current is None:
        SEARCH_STATS["no_engine"] += 1
        return None
    if _i(obs.select.context) != _i(SelectContext.MAIN):
        SEARCH_STATS["not_main"] += 1
        return None
    cand = [c for c in candidates][:max_candidates]
    if len(cand) < 2:
        SEARCH_STATS["cand_lt2"] += 1
        return None
    kw = det.build(obs)
    if kw is None:
        SEARCH_STATS["build_none"] += 1
        return None

    my_index = obs.current.yourIndex
    t0 = time.time()
    opened = []
    scored = []
    try:
        root = search_begin(obs, manual_coin=False, **kw)
        if root is None:
            SEARCH_STATS["begin_none"] += 1
            return None
        for a in cand:
            if time.time() - t0 > time_budget:
                break
            try:
                nxt = search_step(root.searchId, [a])
            except Exception:
                continue
            if nxt is None:
                continue
            opened.append(nxt.searchId)
            end = _rollout_to_turn_end(nxt, my_index, rollout_policy, opened)
            scored.append((evaluate(end, my_index), a))
    except Exception:
        return None
    finally:
        try:
            for sid in opened:
                try:
                    search_release(sid)
                except Exception:
                    pass
            search_end()
        except Exception:
            pass

    if len(scored) < 2:
        SEARCH_STATS["scored_lt2"] += 1
        return None
    scored.sort(key=lambda r: -r[0])
    # if every candidate evaluates identically the search learned nothing
    if abs(scored[0][0] - scored[-1][0]) < 1e-9:
        SEARCH_STATS["all_equal"] += 1
        return None
    SEARCH_STATS["ranked"] += 1
    ranked = [a for _, a in scored]
    ranked += [c for c in candidates if c not in ranked]
    return ranked
