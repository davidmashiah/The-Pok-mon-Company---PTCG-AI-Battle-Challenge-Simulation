"""Win-probability value net (linear, exported as plain weights).

Trained on both points of view of real games between ~1085-rated
agents, labels discounted toward 0.5 by distance from the end.
Pure Python so the agent needs no sklearn at inference.
"""
W = [0.022389, -0.018465, 0.030803, 0.000588, -0.00226, 0.000947, -0.009907, 0.008284, 0.000166, 0.098583, 3.4e-05, 9.6e-05, -0.009606, 0.015355, -0.000192, 6.8e-05, 0.02758, -0.000761, 0.000885, -0.000918, 0.000118, -5.1e-05, 0.000111, -0.000709, -0.007903, -0.022338, 0.016912, -0.015723, 0.014571, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.046801, 0.03539, -0.015509, -0.013222]
B = 0.507865
NAMES = ['my_prizes', 'op_prizes', 'prize_diff', 'my_hand', 'op_hand', 'hand_diff', 'my_deck', 'op_deck', 'turn', 'is_first', 'my_act_hp', 'my_act_maxhp', 'my_act_hpfrac', 'my_act_energy', 'op_act_hp', 'op_act_maxhp', 'op_act_hpfrac', 'op_act_energy', 'my_bench', 'op_bench', 'my_board_hp', 'op_board_hp', 'board_hp_diff', 'my_board_energy', 'op_board_energy', 'my_prize_liability', 'op_prize_liability', 'my_discard', 'op_discard', 'my_act_bestdmg', 'op_act_bestdmg', 'dmg_diff', 'my_poisoned', 'my_burned', 'my_asleep', 'my_paralyzed', 'my_confused', 'op_poisoned', 'op_burned', 'op_asleep', 'op_paralyzed', 'op_confused', 'my_act_is_ex', 'op_act_is_ex', 'supporter_played', 'energy_attached']
CENTER = 0.500317
TEMP = 0.163684


def score_raw(feats):
    """feats: list[float] in NAMES order -> RAW score. Use this to RANK.

    This is a Ridge regression, so its output is unbounded: on real data it
    spans about -21 to +35 with a median near 6. Clamping it into [0, 1] --
    which this module used to do, and call a probability -- sent 78.5% of
    positions to exactly 1.0 and 16% to exactly 0.0, collapsing AUC from 0.645
    to 0.552 (0.5 being random). Two search variants were built on the clamped
    version and both "refuted" the value net while actually ranking leaves with
    a near-constant. Rank with score_raw; never rank with a squashed score.
    """
    z = B
    for i, v in enumerate(feats):
        z += W[i] * v
    return z


def score(feats):
    """Monotone squash of score_raw into (0, 1), for display only.

    Order-preserving, so it is safe for ranking too, but score_raw is cheaper
    and has no saturation. TEMP is fitted from the training spread so typical
    positions land inside the informative part of the curve instead of pinned
    at the ends.
    """
    z = (score_raw(feats) - CENTER) / TEMP
    if z < -30.0:
        return 0.0
    if z > 30.0:
        return 1.0
    return 1.0 / (1.0 + 2.718281828459045 ** (-z))
