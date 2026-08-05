"""Win-probability value net (linear, exported as plain weights).

Trained on both points of view of real games between ~1085-rated
agents, labels discounted toward 0.5 by distance from the end.
Pure Python so the agent needs no sklearn at inference.
"""
W = [-0.035363, 0.019081, 0.058153, -0.002512, -0.001301, 0.00066, -0.053829, 0.007383, -0.010976, -0.007573, -0.016734, -0.018328, 0.129605, 0.018498, -0.027427, 0.046684, -0.060621, -0.002659, -0.008488, 0.02473, 0.002321, -0.005455, 0.008082, -0.011424, 0.000814, 0.006503, -0.01237, -0.080631, 0.019976, 0.037158, -0.016895, 0.053598, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.044693, 0.085987, -0.032895, 0.006216]
B = 0.780579
NAMES = ['my_prizes', 'op_prizes', 'prize_diff', 'my_hand', 'op_hand', 'hand_diff', 'my_deck', 'op_deck', 'turn', 'is_first', 'my_act_hp', 'my_act_maxhp', 'my_act_hpfrac', 'my_act_energy', 'op_act_hp', 'op_act_maxhp', 'op_act_hpfrac', 'op_act_energy', 'my_bench', 'op_bench', 'my_board_hp', 'op_board_hp', 'board_hp_diff', 'my_board_energy', 'op_board_energy', 'my_prize_liability', 'op_prize_liability', 'my_discard', 'op_discard', 'my_act_bestdmg', 'op_act_bestdmg', 'dmg_diff', 'my_poisoned', 'my_burned', 'my_asleep', 'my_paralyzed', 'my_confused', 'op_poisoned', 'op_burned', 'op_asleep', 'op_paralyzed', 'op_confused', 'my_act_is_ex', 'op_act_is_ex', 'supporter_played', 'energy_attached']


def score(feats):
    """feats: list[float] in NAMES order -> win probability in (0,1)."""
    z = B
    for i, v in enumerate(feats):
        z += W[i] * v
    if z < 0.0:
        return 0.0
    if z > 1.0:
        return 1.0
    return z
