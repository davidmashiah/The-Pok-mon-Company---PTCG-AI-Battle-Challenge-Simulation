from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from cg.api import (
    AreaType,
    CardType,
    SelectContext,
    OptionType,
    Card,
    Pokemon,
    all_card_data,
    to_observation_class,
)

"""
Mega Venusaur ex Deck V3: Controlled Healing Loop & Zarude Tech Agent.

Goals:
- Keep the CABT submission format stable.
- Return only legal option indexes and respect minCount/maxCount.
- Prioritize Bulbasaur -> Ivysaur -> Mega Venusaur ex setup through Vitality Forest.
- Use Teal Mask Ogerpon ex to push Grass Energy onto the board early.
- Treat Mega Venusaur ex as the main attacker and Zarude as a matchup tech attacker.
- Gate Mega Venusaur ex's Energy-moving ability to healing-loop and rebuild windows.
- Preserve Pokemon Switch unless it creates a real attack or Zarude line.
- Fall back safely if an unexpected observation shape appears.
"""

EMBEDDED_DECK = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    650, 650, 650, 650,
    651, 651, 651, 651,
    652, 652, 652,
    1071,
    178, 178,
    96, 96, 96, 96,
    1126,
    1122, 1122, 1122, 1122,
    1094, 1094, 1094, 1094,
    1121, 1121, 1121, 1121,
    1123, 1123, 1123,
    1229, 1229, 1229, 1229,
    1227, 1227, 1227, 1227,
    1182, 1182,
    1261, 1261, 1261, 1261,
]

if len(EMBEDDED_DECK) != 60:
    raise RuntimeError(f"Embedded deck must contain 60 cards, got {len(EMBEDDED_DECK)}.")


def load_deck() -> list[int]:
    candidates = [
        "deck.csv",
        os.path.join("/kaggle_simulations/agent", "deck.csv"),
    ]
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as file:
                deck = [int(line.strip()) for line in file if line.strip()]
            if len(deck) == 60:
                return deck
        except Exception:
            pass
    return EMBEDDED_DECK[:]


my_deck = load_deck()

try:
    all_card = all_card_data()
    card_table = {card.cardId: card for card in all_card}
except Exception:
    card_table = {}


BASIC_GRASS_ENERGY = 1

BULBASAUR = 650
IVYSAUR = 651
MEGA_VENUSAUR_EX = 652

MEOWTH_EX = 1071
ZARUDE = 178
TEAL_MASK_OGERPON_EX = 96

PRECIOUS_CARRY = 1126
POKEGEAR_3_0 = 1122
BUG_CATCHING_SET = 1094
ULTRA_BALL = 1121
POKEMON_SWITCH = 1123

MITSURU_CONSIDERATION = 1229
LILLIE_DETERMINATION = 1227
BOSS_ORDERS = 1182
VITALITY_FOREST = 1261

SEARCH_CARDS = {PRECIOUS_CARRY, POKEGEAR_3_0, BUG_CATCHING_SET, ULTRA_BALL}
DRAW_SUPPORTERS = {MITSURU_CONSIDERATION, LILLIE_DETERMINATION}

# Backward-compatible local aliases keep this conservative V3 patch small.
Basic_Grass_Energy = BASIC_GRASS_ENERGY
Bulbasaur = BULBASAUR
Ivysaur = IVYSAUR
Mega_Venusaur_ex = MEGA_VENUSAUR_EX
Meowth_ex = MEOWTH_EX
Zarude = ZARUDE
Teal_Mask_Ogerpon_ex = TEAL_MASK_OGERPON_EX
Precious_Carry = PRECIOUS_CARRY
Pokegear_3_0 = POKEGEAR_3_0
Bug_Catching_Set = BUG_CATCHING_SET
Ultra_Ball = ULTRA_BALL
Switch = POKEMON_SWITCH
Mitsuru_Consideration = MITSURU_CONSIDERATION
Lillie_Determination = LILLIE_DETERMINATION
Boss_Orders = BOSS_ORDERS
Vitality_Forest = VITALITY_FOREST


class AttackPlan:
    def __init__(self) -> None:
        self.attacker = -1
        self.target = -1
        self.attack_index = -1
        self.remain_hp = -1
        self.energy = False
        self.zarude = False
        self.venusaur_heal_loop = False
        self.venusaur_waiting_mitsuru = False
        self.venusaur_rebuild = False


plan = AttackPlan()
pre_turn = 0


def safe_len(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return 0


def zone_cards(value: Any) -> list[Any]:
    return list(value or [])


def card_data(card_id: int) -> Any:
    return card_table.get(card_id)


def max_hp(card_id: int) -> int:
    data = card_data(card_id)
    for attr in ("hp", "HP", "maxHp", "maxHP"):
        value = getattr(data, attr, None)
        if isinstance(value, int):
            return value
    if card_id == Mega_Venusaur_ex:
        return 330
    if card_id == ZARUDE:
        return 120
    return 0


def all_own_pokemon(my_state: Any) -> list[Pokemon]:
    cards = zone_cards(getattr(my_state, "active", [])) + zone_cards(getattr(my_state, "bench", []))
    return [card for card in cards if card is not None]


def active_pokemon(my_state: Any) -> Pokemon | None:
    active = zone_cards(getattr(my_state, "active", []))
    return active[0] if active else None


def pokemon_energy_count(pokemon: Pokemon | None) -> int:
    if pokemon is None:
        return 0
    return max(
        safe_len(getattr(pokemon, "energies", [])),
        safe_len(getattr(pokemon, "energyCards", [])),
    )


def pokemon_grass_energy_count(pokemon: Pokemon | None) -> int:
    if pokemon is None:
        return 0
    total = 0
    energy_cards = zone_cards(getattr(pokemon, "energyCards", [])) + zone_cards(getattr(pokemon, "energies", []))
    for energy in energy_cards:
        if getattr(energy, "id", None) == Basic_Grass_Energy:
            total += 1
    return total


def grass_energy_on_board(my_state: Any) -> int:
    total = 0
    for pokemon in all_own_pokemon(my_state):
        total += pokemon_grass_energy_count(pokemon)
    return total


def is_damaged(pokemon: Pokemon | None) -> bool:
    if pokemon is None:
        return False
    hp = getattr(pokemon, "hp", max_hp(pokemon.id))
    return hp < max_hp(pokemon.id)


def damaged_mega_venusaur_exists(my_state: Any) -> bool:
    for pokemon in all_own_pokemon(my_state):
        if pokemon.id == Mega_Venusaur_ex and is_damaged(pokemon):
            return True
    return False


def active_mega_venusaur_healing_loop_ready(my_state: Any, hand_counts: dict[int, int]) -> bool:
    active = active_pokemon(my_state)
    return (
        active is not None
        and active.id == Mega_Venusaur_ex
        and is_damaged(active)
        and pokemon_energy_count(active) > 0
        and hand_counts[Mitsuru_Consideration] > 0
    )


def active_mega_venusaur_waiting_for_mitsuru(my_state: Any, hand_counts: dict[int, int]) -> bool:
    active = active_pokemon(my_state)
    return (
        active is not None
        and active.id == Mega_Venusaur_ex
        and is_damaged(active)
        and pokemon_energy_count(active) == 0
        and hand_counts[Mitsuru_Consideration] > 0
    )


def active_mega_venusaur_rebuild_ready(my_state: Any) -> bool:
    active = active_pokemon(my_state)
    return (
        active is not None
        and active.id == Mega_Venusaur_ex
        and not is_damaged(active)
        and pokemon_energy_count(active) < 4
        and grass_energy_on_board(my_state) >= 4
    )


def mega_venusaur_ability_allowed(my_state: Any, hand_counts: dict[int, int]) -> bool:
    """Allow Mega Venusaur ex ability only inside the explicit healing-loop sequence."""
    return (
        active_mega_venusaur_healing_loop_ready(my_state, hand_counts)
        or active_mega_venusaur_rebuild_ready(my_state)
    )


def zarude_plan_ready(my_state: Any, field_counts: dict[int, int], hand_counts: dict[int, int]) -> bool:
    energy_access = grass_energy_on_board(my_state) + hand_counts[Basic_Grass_Energy]
    return (
        field_counts[Mega_Venusaur_ex] >= 1
        and field_counts[Teal_Mask_Ogerpon_ex] >= 2
        and (field_counts[ZARUDE] >= 1 or hand_counts[ZARUDE] >= 1)
        and energy_access >= 3
    )


def get_card(obs: Any, area: Any, index: int, player_index: int) -> Pokemon | Card | None:
    try:
        players = zone_cards(obs.current.players)
        player = players[player_index]
        if area == AreaType.DECK:
            return zone_cards(getattr(obs.select, "deck", []))[index]
        if area == AreaType.HAND:
            return zone_cards(player.hand)[index]
        if area == AreaType.DISCARD:
            return zone_cards(player.discard)[index]
        if area == AreaType.ACTIVE:
            return zone_cards(player.active)[index]
        if area == AreaType.BENCH:
            return zone_cards(player.bench)[index]
        if area == AreaType.PRIZE:
            return zone_cards(player.prize)[index]
        if area == AreaType.STADIUM:
            return zone_cards(obs.current.stadium)[index]
        if area == AreaType.LOOKING:
            return zone_cards(obs.current.looking)[index]
    except Exception:
        return None
    return None


def prize_count(pokemon: Pokemon) -> int:
    data = card_data(pokemon.id)
    if data is None:
        return 2 if getattr(pokemon, "id", 0) in (Mega_Venusaur_ex, Teal_Mask_Ogerpon_ex, Meowth_ex) else 1

    count = 3 if getattr(data, "megaEx", False) else 2 if getattr(data, "ex", False) else 1
    for card in zone_cards(getattr(pokemon, "energyCards", [])):
        if getattr(card, "id", None) == 12:
            count -= 1
    for card in zone_cards(getattr(pokemon, "tools", [])):
        if getattr(card, "id", None) == 1172 and "Lillie" in getattr(data, "name", ""):
            count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon | None) -> int:
    if pokemon is None:
        return -10_000

    data = card_data(pokemon.id)
    score = prize_count(pokemon) * 1000
    score += safe_len(getattr(pokemon, "energies", [])) * 150
    score += safe_len(getattr(pokemon, "tools", [])) * 100

    if data is not None:
        if getattr(data, "stage2", False):
            score += 250
        elif getattr(data, "stage1", False):
            score += 130

    if pokemon.id in (144, 322, 323, 337):
        score -= 200
    if pokemon.id == 112 and safe_len(getattr(pokemon, "energies", [])) >= 1:
        score += 300

    score += getattr(pokemon, "hp", 0)
    return score


def field_attacker_score(pokemon: Pokemon | None, active: bool = False) -> int:
    if pokemon is None:
        return -10_000

    energy_count = safe_len(getattr(pokemon, "energies", []))
    if pokemon.id == Mega_Venusaur_ex:
        score = 2500 + energy_count * 350
    elif pokemon.id == Ivysaur:
        score = 1300 + energy_count * 180
    elif pokemon.id == Bulbasaur:
        score = 650 + energy_count * 120
    elif pokemon.id == Teal_Mask_Ogerpon_ex:
        score = 1800 + energy_count * 250
    elif pokemon.id == ZARUDE:
        score = 1450 + energy_count * 230
        if energy_count >= 3:
            score += 700
    elif pokemon.id == Meowth_ex:
        score = 700 + energy_count * 120
    else:
        score = 100 + energy_count * 100

    if active:
        score += 120

    data = card_data(pokemon.id)
    if data is not None:
        if getattr(data, "megaEx", False):
            score -= 60
        elif getattr(data, "ex", False):
            score -= 25
    return score


def estimated_energy_goal(pokemon: Pokemon | None) -> int:
    if pokemon is None:
        return 0
    if pokemon.id == Mega_Venusaur_ex:
        return 4
    if pokemon.id == Ivysaur:
        return 3
    if pokemon.id in (Bulbasaur, Teal_Mask_Ogerpon_ex):
        return 2
    if pokemon.id == ZARUDE:
        return 3
    if pokemon.id == Meowth_ex:
        return 1
    return 1


def energy_score(pokemon: Pokemon | None, active: bool) -> int:
    if pokemon is None:
        return -1

    energy_count = safe_len(getattr(pokemon, "energies", []))
    goal = estimated_energy_goal(pokemon)

    score = 8000 + (25 if active else 0)
    if pokemon.id == Mega_Venusaur_ex:
        score += 500
    elif pokemon.id == Ivysaur:
        score += 360
    elif pokemon.id == Bulbasaur:
        score += 280
    elif pokemon.id == Teal_Mask_Ogerpon_ex:
        score += 430
    elif pokemon.id == ZARUDE:
        score += 340
    elif pokemon.id == Meowth_ex:
        score += 120
    else:
        score -= 100

    if energy_count < goal:
        score += (goal - energy_count) * 120
    else:
        score -= 200 + energy_count * 20
    return score


def setup_active_score(card: Card | Pokemon | None, first_player: bool) -> int:
    if card is None:
        return 0
    if card.id == Teal_Mask_Ogerpon_ex:
        return 50
    if card.id == Meowth_ex:
        return 35
    if card.id == Bulbasaur:
        return 30
    if card.id == ZARUDE:
        return 25
    return 5


def bench_play_score(card_id: int, field_counts: dict[int, int]) -> int:
    if card_id == Bulbasaur:
        if field_counts[Bulbasaur] + field_counts[Ivysaur] + field_counts[Mega_Venusaur_ex] >= 3:
            return -1
        return 20000
    if card_id == Teal_Mask_Ogerpon_ex:
        if field_counts[Teal_Mask_Ogerpon_ex] >= 3:
            return -1
        return 19800
    if card_id == ZARUDE:
        if field_counts[ZARUDE] >= 1:
            return -1
        return 11600
    if card_id == Meowth_ex:
        if field_counts[Meowth_ex] >= 1:
            return -1
        return 12000
    return 10000


def to_hand_score(
    card: Card | Pokemon | None,
    field_counts: dict[int, int],
    hand_counts: dict[int, int],
    discard_counts: dict[int, int],
    state: Any,
) -> int:
    if card is None:
        return 0

    score = 200 - hand_counts[card.id] * 40
    line_count = field_counts[Bulbasaur] + field_counts[Ivysaur] + field_counts[Mega_Venusaur_ex]

    if card.id == Bulbasaur:
        if line_count == 0:
            score += 170
        elif line_count < 2:
            score += 80
        else:
            score -= 120
    elif card.id == Ivysaur:
        score += 170 if field_counts[Bulbasaur] >= 1 else -80
    elif card.id == Mega_Venusaur_ex:
        if field_counts[Ivysaur] >= 1:
            score += 220
        elif field_counts[Bulbasaur] >= 1:
            score += 80
        else:
            score -= 100
    elif card.id == Teal_Mask_Ogerpon_ex:
        if field_counts[Teal_Mask_Ogerpon_ex] == 0:
            score += 180
        elif field_counts[Teal_Mask_Ogerpon_ex] == 1:
            score += 120
        elif field_counts[Teal_Mask_Ogerpon_ex] == 2:
            score += 45
        else:
            score -= 120
    elif card.id == ZARUDE:
        if field_counts[ZARUDE] == 0 and field_counts[Mega_Venusaur_ex] >= 1 and field_counts[Teal_Mask_Ogerpon_ex] >= 2:
            score += 150
        elif field_counts[ZARUDE] == 0:
            score += 35
        else:
            score -= 110
    elif card.id == Meowth_ex:
        score += 40 if field_counts[Meowth_ex] == 0 else -120
    elif card.id == Basic_Grass_Energy:
        score += 100 if not getattr(state, "energyAttached", False) else 20
    elif card.id in SEARCH_CARDS:
        if card.id == Bug_Catching_Set:
            score += 140
        elif card.id == Ultra_Ball:
            score += 100
        else:
            score += 60
    elif card.id in DRAW_SUPPORTERS:
        bonus = 130 if card.id == Lillie_Determination and getattr(state, "turn", 99) <= 2 else 80
        score += bonus if not getattr(state, "supporterPlayed", False) else -80
    elif card.id == Boss_Orders:
        score += 60 if not getattr(state, "supporterPlayed", False) else -80
    elif card.id == Switch:
        score += 25
    elif card.id == Vitality_Forest:
        score += 50

    return score


def discard_score(card: Card | Pokemon | None, field_counts: dict[int, int], hand_counts: dict[int, int]) -> int:
    if card is None:
        return 0

    if card.id == Basic_Grass_Energy:
        return 180 if hand_counts[Basic_Grass_Energy] >= 2 else 30
    if card.id == Vitality_Forest:
        return 150 if hand_counts[Vitality_Forest] >= 2 else 40
    if card.id == Switch:
        return 90 if hand_counts[Switch] >= 2 else -10
    if card.id == Pokegear_3_0:
        return 120
    if card.id == Bug_Catching_Set:
        return 100
    if card.id == Precious_Carry:
        return 60
    if card.id == Ultra_Ball:
        return 70
    if card.id == Mega_Venusaur_ex:
        return -220
    if card.id == Ivysaur:
        return -180
    if card.id == Bulbasaur:
        return -120
    if card.id == Teal_Mask_Ogerpon_ex:
        return -80 if field_counts[Teal_Mask_Ogerpon_ex] == 0 else 40
    if card.id == ZARUDE:
        return -90 if field_counts[ZARUDE] == 0 else 30
    if card.id == Meowth_ex:
        return 30
    if card.id in DRAW_SUPPORTERS:
        return 80
    if card.id == Boss_Orders:
        return 20
    return 0


def venusaur_healing_loop_card_score(
    card: Card | Pokemon | None,
    option: Any,
    player_index: int,
    my_index: int,
    my_state: Any,
    hand_counts: dict[int, int],
    context: Any,
) -> int | None:
    if card is None or player_index != my_index:
        return None

    option_area = getattr(option, "area", None)
    option_index = getattr(option, "index", -1)
    is_active_option = option_area == AreaType.ACTIVE and option_index == 0
    is_active_mega = is_active_option and getattr(card, "id", None) == Mega_Venusaur_ex

    if active_mega_venusaur_healing_loop_ready(my_state, hand_counts):
        if context == SelectContext.ATTACH_FROM:
            # First select the damaged active Mega Venusaur ex as the energy source.
            return 62000 if is_active_mega else 100
        # Then choose any other Pokemon as the temporary energy bank.
        if is_active_option:
            return -5000
        if isinstance(card, Pokemon):
            return 61000 + field_attacker_score(card, active=False)
        return None

    if active_mega_venusaur_waiting_for_mitsuru(my_state, hand_counts):
        # After the energy has been moved away, Mitsuru should choose the active Mega Venusaur ex.
        return 63000 if is_active_mega else None

    if active_mega_venusaur_rebuild_ready(my_state):
        if context == SelectContext.ATTACH_FROM:
            # After healing, pull Grass Energy back from the best available source.
            if is_active_option:
                return -5000
            return 62000 + pokemon_grass_energy_count(card) * 500
        # Then choose the active Mega Venusaur ex as the destination.
        return 63000 if is_active_mega else None

    return None


def context_name(context: Any) -> str:
    return getattr(context, "name", str(context))


def best_bench_attacker_index(my_state: Any, active_score: int) -> int:
    best_index = -1
    best_score = active_score
    for idx, pokemon in enumerate(zone_cards(getattr(my_state, "bench", []))):
        if pokemon is None:
            continue
        score = field_attacker_score(pokemon, active=False)
        if score > best_score + 220:
            best_score = score
            best_index = idx + 1
    return best_index


def option_count(select: Any) -> int:
    return safe_len(getattr(select, "option", []))


def count_attr(select: Any, name: str, default: int) -> int:
    try:
        value = getattr(select, name)
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def legalize_ranked_indices(ranked_indices: list[int], scores: list[int], select: Any) -> list[int]:
    total = option_count(select)
    if total <= 0:
        return []

    max_count = min(max(0, count_attr(select, "maxCount", 1)), total)
    min_count = min(max(0, count_attr(select, "minCount", 1)), max_count)
    if max_count <= 0:
        return []

    score_by_index = {idx: scores[idx] for idx in range(min(len(scores), total))}
    selected: list[int] = []
    seen: set[int] = set()

    for idx in ranked_indices:
        if not isinstance(idx, int) or idx < 0 or idx >= total or idx in seen:
            continue
        if len(selected) < min_count or score_by_index.get(idx, 0) > 0:
            selected.append(idx)
            seen.add(idx)
        if len(selected) >= max_count:
            break

    if len(selected) < min_count:
        for idx in range(total):
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
            if len(selected) >= min_count:
                break

    return selected[:max_count]


def safe_fallback(select: Any) -> list[int]:
    return legalize_ranked_indices(list(range(option_count(select))), [0] * option_count(select), select)


def raw_fallback(obs_dict: dict) -> list[int]:
    try:
        if not isinstance(obs_dict, dict):
            return [0]
        raw_select = obs_dict.get("select")
        if raw_select is None:
            return my_deck
        if isinstance(raw_select, dict):
            options = raw_select.get("option") or []
            max_count = min(max(0, int(raw_select.get("maxCount", 1))), len(options))
            min_count = min(max(0, int(raw_select.get("minCount", 1))), max_count)
        else:
            options = getattr(raw_select, "option", []) or []
            max_count = min(max(0, int(getattr(raw_select, "maxCount", 1))), len(options))
            min_count = min(max(0, int(getattr(raw_select, "minCount", 1))), max_count)
        return list(range(min_count))
    except Exception:
        return [0]


def score_play_option(
    card: Card | Pokemon | None,
    state: Any,
    field_counts: dict[int, int],
    hand_counts: dict[int, int],
    stadium_id: int,
    my_state: Any | None = None,
) -> int:
    if card is None:
        return -1

    data = card_data(card.id)
    if data is not None and getattr(data, "cardType", None) == CardType.POKEMON:
        return bench_play_score(card.id, field_counts)

    score = 10000
    if card.id == Switch:
        return 7600 if plan.attacker >= 1 and plan.zarude else 6400 if plan.attacker >= 1 else -1
    if card.id == Boss_Orders:
        return 4300 if plan.target >= 1 else -1
    if card.id == Vitality_Forest:
        if stadium_id == Vitality_Forest:
            return -1
        if field_counts[Bulbasaur] >= 1 or field_counts[Ivysaur] >= 1:
            return 7200
        return 5200
    if card.id == Precious_Carry:
        return 8300 if getattr(state, "turn", 99) <= 2 else 4800
    if card.id == Ultra_Ball:
        if field_counts[Bulbasaur] >= 1 and field_counts[Ivysaur] >= 1 and field_counts[Mega_Venusaur_ex] >= 1:
            return 5800
        return 7300
    if card.id == Bug_Catching_Set:
        return 7900
    if card.id == Pokegear_3_0:
        return 5900 if not getattr(state, "supporterPlayed", False) else 2600
    if card.id == Mitsuru_Consideration:
        if getattr(state, "supporterPlayed", False):
            return -1
        if my_state is not None and active_mega_venusaur_healing_loop_ready(my_state, hand_counts):
            return -1
        if my_state is not None and active_mega_venusaur_waiting_for_mitsuru(my_state, hand_counts):
            return 66000
        return 7200 if my_state is not None and damaged_mega_venusaur_exists(my_state) else 3200
    if card.id == Lillie_Determination:
        if getattr(state, "supporterPlayed", False):
            return -1
        return 6400 if getattr(state, "turn", 99) <= 2 else 3600
    return score


def agent_impl(obs: Any) -> list[int]:
    if getattr(obs, "select", None) is None:
        return my_deck

    state = getattr(obs, "current", None)
    select = obs.select
    if state is None or option_count(select) == 0:
        return safe_fallback(select)

    context = getattr(select, "context", None)
    ctx_name = context_name(context)

    players = zone_cards(getattr(state, "players", []))
    my_index = getattr(state, "yourIndex", 0)
    if my_index < 0 or my_index >= len(players):
        return safe_fallback(select)

    op_index = 1 - my_index if len(players) > 1 else my_index
    my_state = players[my_index]
    op_state = players[op_index]

    global plan
    global pre_turn
    turn = getattr(state, "turn", 0)
    if pre_turn != turn:
        pre_turn = turn
        plan = AttackPlan()

    field_counts: dict[int, int] = defaultdict(int)
    hand_counts: dict[int, int] = defaultdict(int)
    discard_counts: dict[int, int] = defaultdict(int)

    for card in zone_cards(getattr(my_state, "active", [])) + zone_cards(getattr(my_state, "bench", [])):
        if card is not None:
            field_counts[card.id] += 1
    for card in zone_cards(getattr(my_state, "hand", [])):
        if card is not None:
            hand_counts[card.id] += 1
    for card in zone_cards(getattr(my_state, "discard", [])):
        if card is not None:
            discard_counts[card.id] += 1

    stadium_id = 0
    for card in zone_cards(getattr(state, "stadium", [])):
        if card is not None:
            stadium_id = card.id

    plan.venusaur_heal_loop = active_mega_venusaur_healing_loop_ready(my_state, hand_counts)
    plan.venusaur_waiting_mitsuru = active_mega_venusaur_waiting_for_mitsuru(my_state, hand_counts)
    plan.venusaur_rebuild = active_mega_venusaur_rebuild_ready(my_state)

    if context == SelectContext.MAIN:
        can_switch = False
        can_op_switch = False
        for option in zone_cards(getattr(select, "option", [])):
            option_type = getattr(option, "type", None)
            if option_type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, getattr(option, "index", 0), my_index)
                if card is None:
                    continue
                if card.id == Switch:
                    can_switch = True
                elif card.id == Boss_Orders:
                    can_op_switch = True
            elif option_type == OptionType.RETREAT:
                can_switch = True

        plan.zarude = zarude_plan_ready(my_state, field_counts, hand_counts)
        active_cards = zone_cards(getattr(my_state, "active", []))
        active = active_cards[0] if active_cards else None
        active_score = field_attacker_score(active, active=True)
        bench_attacker = best_bench_attacker_index(my_state, active_score)
        plan.attacker = 0
        bench_cards = zone_cards(getattr(my_state, "bench", []))
        if can_switch and bench_attacker >= 1 and bench_attacker - 1 < len(bench_cards):
            bench_pokemon = bench_cards[bench_attacker - 1]
            if bench_pokemon is not None and (bench_pokemon.id == Mega_Venusaur_ex or (plan.zarude and bench_pokemon.id == ZARUDE)):
                plan.attacker = bench_attacker

        op_cards = []
        op_active = zone_cards(getattr(op_state, "active", []))
        op_cards.append(op_active[0] if op_active else None)
        op_cards.extend(zone_cards(getattr(op_state, "bench", [])))

        best_target_score = -1
        best_target_index = 0
        for target_index, op_pokemon in enumerate(op_cards):
            if op_pokemon is None:
                continue
            if target_index != 0 and not can_op_switch:
                continue
            score = pokemon_score(op_pokemon) + (180 if target_index == 0 else 0)
            if best_target_score < score:
                best_target_score = score
                best_target_index = target_index
        plan.target = best_target_index

        planned_pokemon = active
        if plan.attacker >= 1 and plan.attacker - 1 < len(bench_cards):
            planned_pokemon = bench_cards[plan.attacker - 1]
        if planned_pokemon is not None:
            plan.energy = safe_len(getattr(planned_pokemon, "energies", [])) < estimated_energy_goal(planned_pokemon)

    scores: list[int] = []
    for option_index, option in enumerate(zone_cards(getattr(select, "option", []))):
        try:
            option_type = getattr(option, "type", None)
            score = 0

            if option_type == OptionType.NUMBER:
                score = getattr(option, "number", 0)
            elif option_type == OptionType.YES:
                score = 1
            elif option_type == OptionType.CARD:
                player_index = getattr(option, "playerIndex", my_index)
                card = get_card(obs, getattr(option, "area", None), getattr(option, "index", 0), player_index)
                if card is not None:
                    loop_score = venusaur_healing_loop_card_score(
                        card,
                        option,
                        player_index,
                        my_index,
                        my_state,
                        hand_counts,
                        context,
                    )
                    if loop_score is not None:
                        score = loop_score
                    elif context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                        if player_index == my_index:
                            score = field_attacker_score(card, active=True)
                            if getattr(option, "index", -1) == plan.attacker - 1:
                                score += 5000
                        else:
                            score = pokemon_score(card)
                            if getattr(option, "index", -1) == plan.target - 1:
                                score += 5000
                    elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                        score = setup_active_score(card, getattr(state, "firstPlayer", -1) == my_index)
                    elif context == SelectContext.TO_HAND:
                        score = to_hand_score(card, field_counts, hand_counts, discard_counts, state)
                    elif context == SelectContext.ATTACH_FROM:
                        score = energy_score(card, getattr(option, "area", None) == AreaType.ACTIVE)
                    elif "DISCARD" in ctx_name or "TRASH" in ctx_name:
                        score = discard_score(card, field_counts, hand_counts)
                    elif isinstance(card, Pokemon) and player_index != my_index:
                        score = pokemon_score(card)
                    else:
                        score = to_hand_score(card, field_counts, hand_counts, discard_counts, state)
            elif option_type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, getattr(option, "index", 0), my_index)
                score = score_play_option(card, state, field_counts, hand_counts, stadium_id, my_state)
            elif option_type == OptionType.ATTACH:
                pokemon = get_card(
                    obs,
                    getattr(option, "inPlayArea", AreaType.ACTIVE),
                    getattr(option, "inPlayIndex", 0),
                    my_index,
                )
                score = energy_score(pokemon, getattr(option, "inPlayArea", None) == AreaType.ACTIVE)
                target_index = 0 if getattr(option, "inPlayArea", None) == AreaType.ACTIVE else 1 + getattr(option, "inPlayIndex", 0)
                if plan.venusaur_heal_loop:
                    if target_index == 0:
                        score = -5000
                    else:
                        score = 62500 + field_attacker_score(pokemon, active=False)
                elif plan.venusaur_rebuild and target_index == 0 and pokemon is not None and pokemon.id == Mega_Venusaur_ex:
                    score = 65500
                if target_index == plan.attacker and plan.energy:
                    score += 500
            elif option_type == OptionType.EVOLVE:
                pokemon = get_card(obs, getattr(option, "inPlayArea", AreaType.ACTIVE), getattr(option, "inPlayIndex", 0), my_index)
                evolve_card = get_card(obs, AreaType.HAND, getattr(option, "index", 0), my_index)
                score = 9000 + safe_len(getattr(pokemon, "energies", [])) * 50
                if evolve_card is not None:
                    if evolve_card.id == Mega_Venusaur_ex:
                        score += 1400
                    elif evolve_card.id == Ivysaur:
                        score += 900
                    if stadium_id == Vitality_Forest and evolve_card.id in (Ivysaur, Mega_Venusaur_ex):
                        score += 1700
            elif option_type == OptionType.ABILITY:
                card = get_card(obs, getattr(option, "area", AreaType.ACTIVE), getattr(option, "index", 0), my_index)
                if card is not None and card.id == Teal_Mask_Ogerpon_ex:
                    score = 34500
                elif card is not None and card.id == Mega_Venusaur_ex:
                    if mega_venusaur_ability_allowed(my_state, hand_counts):
                        score = 70000
                    else:
                        score = -1
                elif card is not None and card.id == Vitality_Forest:
                    score = 1600 if field_counts[Bulbasaur] >= 1 or field_counts[Ivysaur] >= 1 else 500
                else:
                    score = 30000
            elif option_type == OptionType.RETREAT:
                score = 4100 if plan.attacker >= 1 else -1
            elif option_type == OptionType.ATTACK:
                active_cards = zone_cards(getattr(my_state, "active", []))
                active = active_cards[0] if active_cards else None
                if active is not None and active.id == Mega_Venusaur_ex:
                    score = 24500 + option_index
                elif active is not None and active.id == ZARUDE and plan.zarude:
                    score = 23800 + option_index
                else:
                    score = 20000 + option_index

        except Exception:
            score = 0
        scores.append(score)

    ranked = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
    return legalize_ranked_indices(ranked, scores, select)


def agent(obs_dict: dict) -> list[int]:
    obs = None
    try:
        obs = to_observation_class(obs_dict)
        return agent_impl(obs)
    except Exception:
        if obs is not None and getattr(obs, "select", None) is not None:
            return safe_fallback(obs.select)
        return raw_fallback(obs_dict)
