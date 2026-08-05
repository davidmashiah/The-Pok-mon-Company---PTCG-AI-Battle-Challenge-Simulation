"""FALSIFIABLE prize-tracker test.

The earlier test was vacuous: our own prize cards are face-down (None) to us,
so it never had ground truth and "never wrong" meant "never checked".

Real ground truth: when a prize is taken, the engine logs the move
PRIZE -> HAND with the card id. So every card revealed from a prize AFTER the
tracker made a claim must appear in that claim. If one doesn't, the tracker
lied and we would have let search plan around a card that was never available.
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "lib"))

from cg.api import AreaType, LogType, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
import fsearch  # noqa: E402
import policy  # noqa: E402

deck = [int(x.strip()) for x in
        open(os.path.join(WORK, "agents", "v2_lucario", "deck.csv"))
        if x.strip()][:60]

GAMES = 25
claims = 0
checked = 0
violations = []

for g in range(GAMES):
    det = fsearch.Determinizer(deck)
    obs, _ = battle_start(list(deck), list(deck))
    first_claim = None          # (Counter) first claim we saw, for player 0
    revealed_after = Counter()
    for _ in range(3000):
        o = to_observation_class(obs)
        if o.current is not None and o.current.result != -1:
            break
        me_idx = o.current.yourIndex if o.current else 0

        # only audit player 0's tracker (it is the one holding OUR decklist)
        if me_idx == 0:
            det.observe(o)
            got = det.prizes.prized()
            if got is not None and first_claim is None:
                first_claim = Counter(got)
                claims += 1

        # ground truth: cards moving PRIZE -> HAND for player 0
        if first_claim is not None:
            for lg in (o.logs or []):
                if (lg.playerIndex == 0
                        and lg.cardId is not None
                        and lg.fromArea is not None
                        and int(lg.fromArea) == int(AreaType.PRIZE)
                        and lg.toArea is not None
                        and int(lg.toArea) == int(AreaType.HAND)):
                    revealed_after[lg.cardId] += 1

        obs = battle_select(list(policy.act(obs, deck)))
    battle_finish()

    if first_claim is not None and revealed_after:
        checked += 1
        for cid, n in revealed_after.items():
            if first_claim.get(cid, 0) < n:
                violations.append(
                    (g, cid, n, first_claim.get(cid, 0)))

print(f"games                         : {GAMES}")
print(f"games where tracker claimed   : {claims}")
print(f"games with revealed prizes to check against : {checked}")
print(f"VIOLATIONS (claim contradicted by reality)  : {len(violations)}")
for v in violations[:10]:
    print(f"   game {v[0]}: card {v[1]} revealed x{v[2]} from prize "
          f"but claim said x{v[3]}")
print()
if checked == 0:
    print("==> INCONCLUSIVE: no game both claimed and revealed a prize. "
          "Test did not exercise the property.")
elif violations:
    print("==> TRACKER IS WRONG. Do not use it to constrain search.")
else:
    print(f"==> tracker's claims held on all {checked} checkable games.")
