from __future__ import annotations

import os
from collections import defaultdict

from cg.api import (
    AreaType,
    Card,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    SelectType,
    all_card_data,
    to_observation_class,
)


class C:
    BASIC_GRASS_ENERGY = 1
    LEGACY_ENERGY = 12

    LEAFEON = 26
    POLTCHAGEIST = 28
    SINISTCHA = 94
    EEVEE_ASCENSION = 43
    EEVEE_COLORFUL = 145
    CHANSEY = 124
    BUDEW = 235
    LEAFEON_EX = 236
    EEVEE_EX = 249
    EEVEE_BOOSTED = 317
    SYLVEON = 330
    DWEBBLE = 344
    CRUSTLE = 345
    TEAL_MASK_OGERPON = 349
    SHAYMIN_SEND_FLOWERS = 1013

    BUDDY_BUDDY_POFFIN = 1086
    BUG_CATCHING_SET = 1094
    NIGHT_STRETCHER = 1097
    ULTRA_BALL = 1121
    POKEGEAR_30 = 1122
    SWITCH = 1123
    TERA_ORB = 1127
    SURVIVAL_BRACE = 1155
    MAXIMUM_BELT = 1158
    POKE_PAD = 1152
    COUNTER_GAIN = 1168
    BOSS_ORDERS = 1182
    SALVATORE = 1189
    BROCKS_SCOUTING = 1210
    HILDA = 1225
    LILLIES_DETERMINATION = 1227
    BATTLE_CAGE = 1264


ATTACK_LEAFLET_BLESSINGS = 10
ATTACK_SOLAR_BEAM = 11
ATTACK_HOOK = 14
ATTACK_ASCENSION = 39
ATTACK_CURSED_DROP = 116
ATTACK_SPILL_THE_TEA = 117
ATTACK_COLORFUL_CATCH = 189
ATTACK_LUCKY_ATTACHMENT = 157
ATTACK_BOUNDLESS_POWER = 158
ATTACK_ITCHY_POLLEN = 323
ATTACK_VERDANT_STORM = 324
ATTACK_MOSS_AGATE = 325
ATTACK_MAGICAL_SHOT = 459
ATTACK_DWEBBLE_ASCENSION = 478
ATTACK_SUPERB_SCISSORS = 479
ATTACK_GRASS_KAGURA = 484
ATTACK_OGRES_HAMMER = 485
ATTACK_SEND_FLOWERS = 1463
ATTACK_LEAF_STEP = 1464

EEVEE_IDS = {C.EEVEE_BOOSTED}
MAIN_LINE = EEVEE_IDS | {C.LEAFEON_EX}
GRASS_BENCH_TARGETS = {C.LEAFEON_EX, C.EEVEE_BOOSTED, C.DWEBBLE, C.CRUSTLE, C.SHAYMIN_SEND_FLOWERS, C.POLTCHAGEIST, C.SINISTCHA}
WALL_IDS = {C.SYLVEON, C.CRUSTLE, C.DWEBBLE, C.SHAYMIN_SEND_FLOWERS, C.CHANSEY}
BENCH_BASICS = EEVEE_IDS | {C.DWEBBLE, C.SHAYMIN_SEND_FLOWERS, C.CHANSEY, C.POLTCHAGEIST}
SEARCH_ITEMS = {
    C.BUDDY_BUDDY_POFFIN,
    C.BUG_CATCHING_SET,
    C.ULTRA_BALL,
    C.POKEGEAR_30,
    C.TERA_ORB,
    C.POKE_PAD,
}
SUPPORTERS = {
    C.BOSS_ORDERS,
    C.SALVATORE,
    C.BROCKS_SCOUTING,
    C.HILDA,
    C.LILLIES_DETERMINATION,
}
RECOVERY_IDS = MAIN_LINE | {C.SYLVEON, C.DWEBBLE, C.CRUSTLE, C.SHAYMIN_SEND_FLOWERS, C.CHANSEY, C.POLTCHAGEIST, C.SINISTCHA, C.BASIC_GRASS_ENERGY}

PRIORITY_TARGET_IDS = {
    677: 2600,  # Riolu
    678: 2100,  # Mega Lucario ex
    673: 1900,  # Makuhita
    674: 1200,  # Hariyama
    675: 900,  # Lunatone
    676: 900,  # Solrock
    741: 2300,  # Abra
    742: 1900,  # Kadabra
    743: 1300,  # Alakazam
    119: 2100,  # Dreepy
    120: 2100,  # Drakloak
    121: 1500,  # Dragapult ex
    1030: 2200,  # Staryu
    360: 1900,  # Misty's Staryu
    104: 1700,  # Froslass
    861: 1500,  # Mega Froslass ex
    169: 1700,  # Duraludon
    190: 1400,  # Archaludon ex
    665: 1900,  # Scorbunny
    666: 1700,  # Cinderace
    344: 1700,  # Dwebble
    345: 900,  # Crustle
    305: 1200,  # Dunsparce
}

DECK_PATH = "deck.csv"
if not os.path.exists(DECK_PATH):
    DECK_PATH = "/kaggle_simulations/agent/deck.csv"
with open(DECK_PATH, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]

card_table = {card.cardId: card for card in all_card_data()}
SINISTCHA_ENABLED = C.POLTCHAGEIST in my_deck or C.SINISTCHA in my_deck
_DIAG = {"calls": 0, "fallback": 0, "last_error": ""}


def get_card(obs: Observation, area, index: int | None, player_index: int | None) -> Pokemon | Card | None:
    if obs.current is None or area is None or index is None:
        return None
    try:
        area = AreaType(area)
    except Exception:
        return None
    try:
        player = obs.current.players[obs.current.yourIndex if player_index is None else player_index]
        if area == AreaType.DECK:
            return obs.select.deck[index] if obs.select and obs.select.deck else None
        if area == AreaType.HAND:
            return player.hand[index] if player.hand else None
        if area == AreaType.DISCARD:
            return player.discard[index]
        if area == AreaType.ACTIVE:
            return player.active[index]
        if area == AreaType.BENCH:
            return player.bench[index]
        if area == AreaType.PRIZE:
            return player.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index] if obs.current.looking else None
    except Exception:
        return None
    return None


def field_pokemon(player_state) -> list[tuple[int, Pokemon]]:
    out: list[tuple[int, Pokemon]] = []
    for pokemon in getattr(player_state, "active", []) or []:
        if pokemon is not None:
            out.append((0, pokemon))
    for idx, pokemon in enumerate(getattr(player_state, "bench", []) or []):
        if pokemon is not None:
            out.append((idx + 1, pokemon))
    return out


def card_id(card: Card | Pokemon | None) -> int:
    return int(getattr(card, "id", 0) or 0)


def data_for(card: Card | Pokemon | None):
    return card_table.get(card_id(card))


def is_energy(card: Card | Pokemon | None) -> bool:
    data = data_for(card)
    return bool(data and data.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY))


def energy_cards(pokemon: Pokemon | None) -> list[Card]:
    return list(getattr(pokemon, "energyCards", None) or [])


def energy_count(pokemon: Pokemon | None) -> int:
    if pokemon is None:
        return 0
    cards = energy_cards(pokemon)
    if cards:
        return len(cards)
    return len(getattr(pokemon, "energies", None) or [])


def typed_energy_count(pokemon: Pokemon | None, energy_id: int) -> int:
    if pokemon is None:
        return 0
    cards = energy_cards(pokemon)
    if cards:
        return sum(1 for card in cards if card.id == energy_id)
    return sum(1 for energy in (getattr(pokemon, "energies", None) or []) if int(energy) == energy_id)


def has_tool(pokemon: Pokemon | None, tool_id: int) -> bool:
    return any(card.id == tool_id for card in (getattr(pokemon, "tools", None) or []))


def has_any_tool(pokemon: Pokemon | None) -> bool:
    return bool(getattr(pokemon, "tools", None) or [])


def damage_on(pokemon: Pokemon | None) -> int:
    if pokemon is None:
        return 0
    return max(0, int(getattr(pokemon, "maxHp", 0) or 0) - int(getattr(pokemon, "hp", 0) or 0))


def prize_count(pokemon: Pokemon | None) -> int:
    data = data_for(pokemon)
    if not data:
        return 1
    return 3 if data.megaEx else 2 if data.ex else 1


def adjusted_grass_damage(raw_damage: int, target: Pokemon | None) -> int:
    data = data_for(target)
    if target is None or not data:
        return raw_damage
    damage = raw_damage
    if data.weakness == EnergyType.GRASS:
        damage *= 2
    if data.resistance == EnergyType.GRASS:
        damage = max(0, damage - 30)
    return damage


class LeafeonPolicy:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.active = self.me.active[0] if self.me.active else None
        self.op_active = self.opponent.active[0] if self.opponent.active else None
        self.deck_count = int(getattr(self.me, "deckCount", 0) or 0)

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.hand_grass = 0
        self.hand_energy = 0
        self.opponent_total_energy = 0
        self._count_cards()

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []
        scores = [self._score_option(option) for option in self.select.option]
        ranked = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        if self.select.minCount == 0:
            ranked = [idx for idx in ranked if scores[idx] > 0]
        chosen = ranked[: self.select.maxCount]
        if len(chosen) < self.select.minCount:
            for idx in range(len(self.select.option)):
                if idx not in chosen:
                    chosen.append(idx)
                if len(chosen) >= self.select.minCount:
                    break
        return chosen[: self.select.maxCount]

    def _count_cards(self) -> None:
        for _, pokemon in field_pokemon(self.me):
            self.field_counts[pokemon.id] += 1
        for _, pokemon in field_pokemon(self.opponent):
            self.opponent_total_energy += energy_count(pokemon)
        for card in getattr(self.me, "hand", []) or []:
            self.hand_counts[card.id] += 1
            if is_energy(card):
                self.hand_energy += 1
            if card.id == C.BASIC_GRASS_ENERGY:
                self.hand_grass += 1
        for card in getattr(self.me, "discard", []) or []:
            self.discard_counts[card.id] += 1

    def _my_board(self) -> list[Pokemon]:
        return [pokemon for _, pokemon in field_pokemon(self.me)]

    def _opponent_board(self) -> list[tuple[int, Pokemon]]:
        return field_pokemon(self.opponent)

    def _my_prizes_remaining(self) -> int:
        return len(getattr(self.me, "prize", None) or [])

    def _opponent_prizes_remaining(self) -> int:
        return len(getattr(self.opponent, "prize", None) or [])

    def _opponent_prizes_taken(self) -> int:
        remaining = self._opponent_prizes_remaining()
        return max(0, 6 - remaining) if remaining else 0

    def _behind_on_prizes(self) -> bool:
        mine = self._my_prizes_remaining()
        theirs = self._opponent_prizes_remaining()
        return bool(mine and theirs and mine > theirs)

    def _late_game_recovery_window(self) -> bool:
        return self._behind_on_prizes() or self._opponent_prizes_taken() >= 3 or self.opponent_total_energy >= 5

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER:
            return float(option.number or 0)
        if option.type == OptionType.YES:
            return 120 if self.context == SelectContext.IS_FIRST else 80
        if option.type == OptionType.NO:
            return 100 if self.context == SelectContext.MULLIGAN else 0
        if option.type == OptionType.CARD:
            return self._score_card_option(option)
        if option.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
            return self._score_energy_option(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.EVOLVE:
            return self._score_evolve(option)
        if option.type == OptionType.ABILITY:
            return 7200
        if option.type == OptionType.RETREAT:
            return 13500 if self._should_retreat() else -1
        if option.type == OptionType.ATTACK:
            return self._score_attack(option)
        if option.type == OptionType.END:
            return 1
        return 0

    def _score_card_option(self, option) -> float:
        owner = self.my_index if option.playerIndex is None else option.playerIndex
        card = get_card(self.obs, option.area, option.index, owner)
        if card is None:
            return 0

        if owner == self.op_index and isinstance(card, Pokemon):
            return self._target_value(card, active=self._option_board_index(option) == 0)

        if self.context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            if isinstance(card, Pokemon):
                return self._to_active_score(card)
            return 0
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._setup_active_score(card)
        if self.context in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            return self._bench_score(card)
        if self.context == SelectContext.TO_HAND:
            return self._to_hand_score(card)
        if self.context in (SelectContext.EVOLVES_TO, SelectContext.EVOLVE):
            return self._evolution_card_score(card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._attach_target_score(card, option.area == AreaType.ACTIVE)
        if self.context in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
            return self._discard_score(card)
        if self.context in (SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
            return self._to_deck_score(card)
        if self.context in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY, SelectContext.EFFECT_TARGET):
            if isinstance(card, Pokemon):
                return self._target_value(card, active=self._option_board_index(option) == 0)
        if self.context in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
            if isinstance(card, Pokemon):
                return damage_on(card) + self._leafeon_like(card) * 300
        return self._to_hand_score(card) * 0.1

    def _score_energy_option(self, option) -> float:
        pokemon = get_card(self.obs, getattr(option, "area", None), getattr(option, "index", None), getattr(option, "playerIndex", self.my_index))
        if self.context in (SelectContext.DISCARD_ENERGY_CARD, SelectContext.DISCARD_ENERGY):
            if isinstance(pokemon, Pokemon) and pokemon.id == C.LEAFEON_EX and energy_count(pokemon) <= 2:
                return -700
            return 500
        if self.context in (SelectContext.TO_HAND_ENERGY, SelectContext.TO_DECK_ENERGY):
            return 900
        return 120

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        if card is None:
            return 0
        data = data_for(card)
        if data and data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card: Card | Pokemon) -> float:
        bench_free = int(getattr(self.me, "benchMax", 5) or 5) - len(self.me.bench or [])
        if bench_free <= 0:
            return -1
        return 20000 + self._bench_score(card)

    def _score_play_trainer(self, card: Card | Pokemon) -> float:
        cid = card.id
        if cid == C.BUDDY_BUDDY_POFFIN:
            return 18800 if self._need_early_wall_basic() else 900
        if cid == C.POKE_PAD:
            if self.deck_count <= 7:
                return 150
            if self._need_sinistcha_card():
                return 17400
            if self._need_crustle_card() or self._need_sylveon_card():
                return 17200
            if self._need_early_wall_basic():
                return 15000
            return 4200
        if cid == C.TERA_ORB:
            return 17800 if self._need_leafeon_card() and self._late_game_recovery_window() else 1800
        if cid == C.ULTRA_BALL:
            if self.deck_count <= 8:
                return 200
            if self._need_sinistcha_card():
                return 17900
            if self._need_crustle_card() and not self._has_ex_wall():
                return 17850
            if self._need_leafeon_card() and self._late_game_recovery_window():
                return 17600
            if self._need_sylveon_card() or self._need_early_wall_basic():
                return 14800
            return 4700
        if cid == C.BUG_CATCHING_SET:
            if self.deck_count <= 7:
                return 120
            return 14200 if self.hand_grass == 0 or self._need_crustle_card() or (self._need_leafeon_card() and self._late_game_recovery_window()) else 3800
        if cid == C.POKEGEAR_30:
            if self.deck_count <= 7:
                return 100
            return 11600 if not self.state.supporterPlayed else 500
        if cid == C.NIGHT_STRETCHER:
            return 12800 if self._has_useful_discard() else 220
        if cid == C.SWITCH:
            return 13200 if self._should_switch_item() else 90
        if cid == C.BATTLE_CAGE:
            return 10100 if self._bench_damage_prevention_matters() else 1300
        if cid == C.COUNTER_GAIN:
            return 16800 if self._best_counter_gain_target() is not None else 300
        if cid == C.MAXIMUM_BELT:
            return 16600 if self._best_maximum_belt_target() is not None else 300
        if cid in (C.MAXIMUM_BELT, C.SURVIVAL_BRACE):
            return 300
        if cid in SUPPORTERS:
            return self._supporter_score(cid)
        return 800

    def _supporter_score(self, cid: int) -> float:
        if self.state.supporterPlayed:
            return -1
        if cid == C.SALVATORE:
            return 18400 if self._has_evolve_source() and self._late_game_recovery_window() else 250
        if cid == C.HILDA:
            if self._need_sinistcha_card():
                return 17600
            if self._need_crustle_card() or self._need_sylveon_card():
                return 17400
            if (self._need_leafeon_card() and self._late_game_recovery_window()) or self.hand_grass == 0:
                return 17000
            return 4300
        if cid == C.BROCKS_SCOUTING:
            if self._need_early_wall_basic():
                return 16000
            if self._need_leafeon_card() and self._late_game_recovery_window():
                return 12000
            return 2400
        if cid == C.LILLIES_DETERMINATION:
            if self.deck_count <= 8:
                return -1
            if len(self.me.hand or []) <= 3:
                return 15800
            if self.field_counts[C.LEAFEON_EX] >= 1 and self.hand_energy == 0:
                return 13600
            return 3200
        if cid == C.BOSS_ORDERS:
            target, score = self._best_boss_target()
            return 16500 + score * 0.05 if target is not None and score > 0 and self._best_available_front_damage() > 0 else -1
        return 1000

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None or pokemon is None:
            return 0
        if card.id in (C.BASIC_GRASS_ENERGY, C.LEGACY_ENERGY):
            legacy_bonus = 2200 if card.id == C.LEGACY_ENERGY and self._valuable_legacy_target(pokemon, option.inPlayArea == AreaType.ACTIVE) else 0
            if isinstance(pokemon, Pokemon) and pokemon.id == C.LEAFEON_EX and not self._leafeon_ready(pokemon):
                return 26000 + legacy_bonus + self._attach_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
            if isinstance(pokemon, Pokemon) and pokemon.id in EEVEE_IDS and self._late_game_recovery_window() and self.field_counts[C.LEAFEON_EX] == 0:
                return 20600 + legacy_bonus + self._attach_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
            if isinstance(pokemon, Pokemon) and pokemon.id in (C.CRUSTLE, C.DWEBBLE, C.SHAYMIN_SEND_FLOWERS, C.CHANSEY, C.POLTCHAGEIST, C.SINISTCHA):
                return 11800 + legacy_bonus + self._attach_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
            return 9000 + legacy_bonus + self._attach_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        if card.id == C.COUNTER_GAIN:
            return 8600 + self._tool_target_score(pokemon)
        if card.id == C.MAXIMUM_BELT:
            return 8800 + self._maximum_belt_target_score(pokemon)
        if card.id in (C.MAXIMUM_BELT, C.SURVIVAL_BRACE):
            return 200
        return 120

    def _score_evolve(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        target = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None:
            return 0
        score = 11600 + self._evolution_card_score(card)
        if card.id == C.LEAFEON_EX and isinstance(target, Pokemon) and target.id == C.EEVEE_BOOSTED and option.inPlayArea == AreaType.ACTIVE and self._late_game_recovery_window():
            score += 1200
        return score

    def _score_attack(self, option) -> float:
        active = self.active
        if active is None:
            return 0
        if option.attackId == ATTACK_VERDANT_STORM:
            raw = self._verdant_raw_damage(active)
            damage = adjusted_grass_damage(raw, self.op_active)
            score = 5200 + raw * 16 + self.opponent_total_energy * 1200
            if self.op_active is not None and damage >= self.op_active.hp:
                score += 11000 + prize_count(self.op_active) * 1800
            if self.opponent_total_energy >= 5:
                score += 3500
            if not self._late_game_recovery_window() and self.opponent_total_energy < 5:
                score -= 3500
            return score
        if option.attackId == ATTACK_MOSS_AGATE:
            damage = adjusted_grass_damage(230, self.op_active)
            score = 3800 + damage * 8 + self._bench_damage_total() * 2
            if self.op_active is not None and damage >= self.op_active.hp:
                score += 9200 + prize_count(self.op_active) * 1500
            return score
        if option.attackId == ATTACK_ASCENSION:
            return 17000 if self._need_leafeon_card() or self.field_counts[C.LEAFEON_EX] == 0 else 2800
        if option.attackId == ATTACK_DWEBBLE_ASCENSION:
            return 19000 if self.field_counts[C.CRUSTLE] < 2 else 4200
        if option.attackId == ATTACK_SEND_FLOWERS:
            return 18600 if self._best_grass_accel_target() is not None else 2500
        if option.attackId == ATTACK_LUCKY_ATTACHMENT:
            return 17600 if self.hand_energy > 0 and self._best_accel_target_from_hand() is not None else 2400
        if option.attackId == ATTACK_SUPERB_SCISSORS:
            damage = adjusted_grass_damage(120, self.op_active)
            score = 7600 + damage * 8
            if self.op_active is not None and damage >= self.op_active.hp:
                score += 9000 + prize_count(self.op_active) * 1300
            if self._opponent_ex_pressure():
                score += 1400
            return score
        if option.attackId == ATTACK_CURSED_DROP:
            score = 11200
            if self._opponent_non_ex_pressure():
                score += 2400
            if self.op_active is not None and self.op_active.hp <= 40:
                score += 6000 + prize_count(self.op_active) * 1100
            return score
        if option.attackId == ATTACK_SPILL_THE_TEA:
            attached_grass = sum(typed_energy_count(pokemon, C.BASIC_GRASS_ENERGY) for pokemon in self._my_board())
            damage = adjusted_grass_damage(70 * min(3, attached_grass), self.op_active)
            score = 3600 + damage * 7
            if self.op_active is not None and damage >= self.op_active.hp:
                score += 7000 + prize_count(self.op_active) * 1200
            return score
        if option.attackId == ATTACK_LEAF_STEP:
            return 4200
        if option.attackId == ATTACK_BOUNDLESS_POWER:
            return 3600
        if option.attackId == ATTACK_GRASS_KAGURA:
            return 15400 if self._needs_grass_kagura() else 2200
        if option.attackId == ATTACK_ITCHY_POLLEN:
            return 11800 if not any(self._leafeon_ready(pokemon) for pokemon in self._my_board()) else 1400
        if option.attackId == ATTACK_OGRES_HAMMER:
            damage = adjusted_grass_damage(120, self.op_active)
            score = 4400 + damage * 7
            if self.op_active is not None and damage >= self.op_active.hp:
                score += 7200
            return score
        if option.attackId == ATTACK_COLORFUL_CATCH:
            return 8200 if self.hand_grass == 0 else 900
        if option.attackId == ATTACK_MAGICAL_SHOT:
            return 5000
        return 1600

    def _setup_active_score(self, card: Card | Pokemon) -> float:
        if card.id == C.DWEBBLE:
            return 2300
        if card.id == C.SHAYMIN_SEND_FLOWERS:
            return 2050
        if card.id == C.CHANSEY:
            return 1750
        if card.id == C.POLTCHAGEIST:
            return 680
        if card.id == C.EEVEE_BOOSTED:
            return 1100
        if card.id == C.EEVEE_ASCENSION:
            return 1500
        if card.id == C.TEAL_MASK_OGERPON:
            return 900
        if card.id == C.BUDEW:
            return 760
        if card.id == C.EEVEE_EX:
            return 720
        if card.id == C.EEVEE_COLORFUL:
            return 780
        return 80

    def _bench_score(self, card: Card | Pokemon) -> float:
        if card.id == C.DWEBBLE:
            return 1900 if self.field_counts[C.DWEBBLE] + self.field_counts[C.CRUSTLE] < 2 else 700
        if card.id == C.SHAYMIN_SEND_FLOWERS:
            return 1760 if self.field_counts[C.SHAYMIN_SEND_FLOWERS] < 2 else 380
        if card.id == C.CHANSEY:
            return 1320 if self.field_counts[C.CHANSEY] < 1 else 260
        if card.id == C.POLTCHAGEIST:
            return 1420 if self.field_counts[C.POLTCHAGEIST] + self.field_counts[C.SINISTCHA] < 2 else 220
        if card.id in EEVEE_IDS:
            line_count = self._eevee_line_count()
            if card.id == C.EEVEE_BOOSTED:
                if self._late_game_recovery_window():
                    return 1780 if line_count < 2 else 680
                return 1180 if line_count < 1 else 220
            if card.id == C.EEVEE_ASCENSION:
                return 1500 if line_count < 2 else 720 if line_count < 3 else 140
            return 960 if line_count < 2 else 360
        if card.id == C.TEAL_MASK_OGERPON:
            return 920 if self.field_counts[C.TEAL_MASK_OGERPON] < 1 else 260
        if card.id == C.BUDEW:
            return 760 if self.field_counts[C.BUDEW] < 1 and self._eevee_line_count() >= 2 else 80
        return 80

    def _to_hand_score(self, card: Card | Pokemon) -> float:
        cid = card.id
        base = 100 - self.hand_counts[cid] * 20
        if cid == C.LEAFEON_EX:
            return base + (1900 if self._need_leafeon_card() and self._late_game_recovery_window() else 520)
        if cid == C.SYLVEON:
            return base + (1800 if self._need_sylveon_card() else 420 if self._opponent_ex_pressure() else 80)
        if cid == C.CRUSTLE:
            return base + (2050 if self._need_crustle_card() else 480)
        if cid == C.DWEBBLE:
            return base + (1780 if self.field_counts[C.DWEBBLE] + self.field_counts[C.CRUSTLE] < 2 else 280)
        if cid == C.SHAYMIN_SEND_FLOWERS:
            return base + (1600 if self.field_counts[C.SHAYMIN_SEND_FLOWERS] < 2 else 240)
        if cid == C.CHANSEY:
            return base + (1220 if self.field_counts[C.CHANSEY] < 1 else 160)
        if cid == C.POLTCHAGEIST:
            return base + (1350 if self.field_counts[C.POLTCHAGEIST] + self.field_counts[C.SINISTCHA] < 2 else 160)
        if cid == C.SINISTCHA:
            return base + (1900 if self._need_sinistcha_card() else 360)
        if cid in EEVEE_IDS:
            return base + (1550 if self._need_more_eevee() and self._late_game_recovery_window() else 420)
        if cid == C.TEAL_MASK_OGERPON:
            return base + (700 if self.field_counts[C.TEAL_MASK_OGERPON] < 1 else 120)
        if cid == C.BUDEW:
            return base + (500 if self.field_counts[C.BUDEW] < 1 and self._eevee_line_count() >= 2 else 80)
        if cid == C.BASIC_GRASS_ENERGY:
            return base + (1450 if self.hand_grass == 0 else 620 if self._needs_attack_energy() else 180)
        if cid == C.LEGACY_ENERGY:
            return base + 1200
        if cid == C.COUNTER_GAIN:
            return base + 1250
        if cid in (C.MAXIMUM_BELT, C.SURVIVAL_BRACE):
            return base + 80
        if cid in SEARCH_ITEMS:
            return base + 760
        if cid == C.NIGHT_STRETCHER:
            return base + (760 if self._has_useful_discard() else 130)
        if cid == C.SWITCH:
            return base + (520 if self._should_switch_item() else 80)
        if cid == C.BATTLE_CAGE:
            return base + (420 if self._bench_damage_prevention_matters() else 90)
        if cid in SUPPORTERS:
            return base + 620
        return base

    def _evolution_card_score(self, card: Card | Pokemon) -> float:
        if card.id == C.LEAFEON_EX:
            return 2450 if self._late_game_recovery_window() else 500
        if card.id == C.SYLVEON:
            return 2150 if self._need_sylveon_card() else 650 if self._opponent_ex_pressure() else 120
        if card.id == C.CRUSTLE:
            return 2450 if self._need_crustle_card() else 720
        if card.id == C.SINISTCHA:
            return 2100 if self._need_sinistcha_card() or self._opponent_non_ex_pressure() else 420
        return self._to_hand_score(card)

    def _discard_score(self, card: Card | Pokemon) -> float:
        cid = card.id
        if cid == C.BASIC_GRASS_ENERGY:
            return 360 if self.hand_grass > 2 else -600
        if cid == C.LEGACY_ENERGY:
            return -1450
        if cid == C.LEAFEON_EX and self.field_counts[C.LEAFEON_EX] < 2:
            return -1200
        if cid in EEVEE_IDS and self._eevee_line_count() < 3:
            return -950
        if cid in (C.COUNTER_GAIN,):
            return -1400
        if cid == C.MAXIMUM_BELT:
            return -1400
        if cid == C.SURVIVAL_BRACE:
            return 700
        if cid == C.SYLVEON:
            return 480 if not self._opponent_ex_pressure() else -260
        if cid == C.CRUSTLE and self.field_counts[C.CRUSTLE] < 2:
            return -900
        if cid == C.DWEBBLE and self.field_counts[C.DWEBBLE] + self.field_counts[C.CRUSTLE] < 2:
            return -820
        if cid == C.SHAYMIN_SEND_FLOWERS and self.field_counts[C.SHAYMIN_SEND_FLOWERS] < 1:
            return -500
        if cid == C.CHANSEY and self.field_counts[C.CHANSEY] < 1:
            return -320
        if cid == C.POLTCHAGEIST and self.field_counts[C.POLTCHAGEIST] + self.field_counts[C.SINISTCHA] < 1:
            return -420
        if cid == C.SINISTCHA and self.field_counts[C.SINISTCHA] < 1 and self._opponent_non_ex_pressure():
            return -620
        if cid == C.TEAL_MASK_OGERPON and self.field_counts[C.TEAL_MASK_OGERPON] >= 1:
            return 420
        if cid == C.BUDEW and self.field_counts[C.BUDEW] >= 1:
            return 440
        if cid in SUPPORTERS and self.hand_counts[cid] > 1:
            return 600
        if cid in SEARCH_ITEMS and not self._need_more_eevee() and not self._need_leafeon_card():
            return 520
        if cid == C.BATTLE_CAGE and not self._bench_damage_prevention_matters():
            return 430
        return 120

    def _to_deck_score(self, card: Card | Pokemon) -> float:
        if card.id in RECOVERY_IDS:
            return 800
        return 120

    def _attach_target_score(self, pokemon: Pokemon | None, active: bool) -> float:
        if pokemon is None:
            return -100
        score = -80
        if pokemon.id == C.LEAFEON_EX:
            score = 2600 if not self._leafeon_ready(pokemon) else 260
            if typed_energy_count(pokemon, C.BASIC_GRASS_ENERGY) == 0:
                score += 900
        elif pokemon.id == C.EEVEE_EX:
            score = 1600 if self.field_counts[C.LEAFEON_EX] < 2 else 260
            if typed_energy_count(pokemon, C.BASIC_GRASS_ENERGY) == 0:
                score += 500
        elif pokemon.id in EEVEE_IDS:
            score = 2050 if self._late_game_recovery_window() and self.field_counts[C.LEAFEON_EX] < self._eevee_line_count() + 1 else 420
            if typed_energy_count(pokemon, C.BASIC_GRASS_ENERGY) == 0:
                score += 660
        elif pokemon.id == C.CRUSTLE:
            score = 1680 if energy_count(pokemon) < 3 else 300
            if typed_energy_count(pokemon, C.BASIC_GRASS_ENERGY) == 0:
                score += 620
        elif pokemon.id == C.DWEBBLE:
            score = 1500 if self.field_counts[C.CRUSTLE] < 2 else 340
            if typed_energy_count(pokemon, C.BASIC_GRASS_ENERGY) == 0:
                score += 540
        elif pokemon.id == C.SHAYMIN_SEND_FLOWERS:
            score = 1320 if active and energy_count(pokemon) == 0 else 240
        elif pokemon.id == C.CHANSEY:
            score = 900 if active and energy_count(pokemon) < 1 else 260
        elif pokemon.id == C.SINISTCHA:
            score = 980 if energy_count(pokemon) < 1 else 220
        elif pokemon.id == C.POLTCHAGEIST:
            score = 740 if energy_count(pokemon) < 1 and self._need_sinistcha_card() else 120
        elif pokemon.id == C.TEAL_MASK_OGERPON:
            score = 1250 if self._needs_grass_kagura() else 260
        elif pokemon.id == C.SYLVEON:
            score = 520 if self._opponent_ex_pressure() else 180
        if active:
            score += 320
        return score

    def _tool_target_score(self, pokemon: Pokemon | None) -> float:
        if pokemon is None or has_any_tool(pokemon):
            return -500
        if pokemon.id == C.LEAFEON_EX and self._behind_on_prizes():
            return 2600 + energy_count(pokemon) * 260
        if pokemon.id == C.CRUSTLE and self._behind_on_prizes():
            return 1700 + energy_count(pokemon) * 180
        if pokemon.id in EEVEE_IDS and self._late_game_recovery_window():
            return 1100
        if pokemon.id == C.LEAFEON_EX:
            return 900 + energy_count(pokemon) * 180
        if pokemon.id in EEVEE_IDS:
            return 360
        return -220

    def _to_active_score(self, pokemon: Pokemon | None) -> float:
        if pokemon is None:
            return 0
        if self._leafeon_ready(pokemon):
            if self._late_game_recovery_window() or self._can_take_active_ko_with_leafeon(pokemon):
                return 3300 + int(has_tool(pokemon, C.COUNTER_GAIN)) * 350
            return 900
        if self._is_ex_wall(pokemon):
            return 2900 + energy_count(pokemon) * 130
        if pokemon.id == C.DWEBBLE:
            return 2500 if self.field_counts[C.CRUSTLE] < 2 else 900
        if pokemon.id == C.SHAYMIN_SEND_FLOWERS:
            return 2300 if self._best_grass_accel_target() is not None else 1050
        if pokemon.id == C.CHANSEY:
            return 1950 if self._best_accel_target_from_hand() is not None else 880
        if pokemon.id == C.SINISTCHA:
            return 2100 if self._opponent_non_ex_pressure() else 1250
        if pokemon.id == C.POLTCHAGEIST:
            return 420
        if pokemon.id == C.LEAFEON_EX:
            return 2200 + energy_count(pokemon) * 240
        if pokemon.id == C.EEVEE_BOOSTED and self._need_leafeon_card():
            return 1700 if self._late_game_recovery_window() else 760
        if pokemon.id == C.EEVEE_EX and self._need_leafeon_card():
            return 1500 + energy_count(pokemon) * 220
        if pokemon.id == C.EEVEE_ASCENSION and self.field_counts[C.LEAFEON_EX] == 0:
            return 1650
        if pokemon.id == C.BUDEW and not any(self._leafeon_ready(p) for p in self._my_board()):
            return 1200
        if pokemon.id == C.TEAL_MASK_OGERPON and self._needs_grass_kagura():
            return 1550
        if pokemon.id == C.SYLVEON and self._opponent_ex_pressure():
            return 1200
        return 120 + energy_count(pokemon) * 180

    def _target_value(self, pokemon: Pokemon | None, active: bool = False) -> float:
        if pokemon is None:
            return 0
        data = data_for(pokemon)
        hp = max(0, int(getattr(pokemon, "hp", 0) or 0))
        damage = self._best_available_front_damage()
        score = 120 + min(520, hp * 1.1) + energy_count(pokemon) * 240
        score += PRIORITY_TARGET_IDS.get(pokemon.id, 0)
        if data:
            if data.megaEx:
                score += 1500
            elif data.ex:
                score += 950
            if data.stage2:
                score += 560
            elif data.stage1:
                score += 340
        if hp and damage >= hp:
            score += 3600 + prize_count(pokemon) * 1200
        if active:
            score += 260
        return score

    def _best_boss_target(self) -> tuple[Pokemon | None, float]:
        best = (None, -1.0)
        active_damage = self._best_available_front_damage()
        for board_index, pokemon in self._opponent_board():
            if board_index == 0:
                continue
            hp = max(0, int(getattr(pokemon, "hp", 0) or 0))
            score = self._target_value(pokemon)
            if hp and active_damage >= hp:
                score += 2600
            if energy_count(pokemon) >= 2:
                score += 900
            if score > best[1]:
                best = (pokemon, score)
        return best

    def _best_belt_target(self) -> Pokemon | None:
        candidates = [
            pokemon
            for pokemon in self._my_board()
            if pokemon.id == C.LEAFEON_EX and not has_any_tool(pokemon)
        ]
        if not candidates:
            candidates = [
                pokemon
                for pokemon in self._my_board()
                if pokemon.id in EEVEE_IDS and not has_any_tool(pokemon)
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda pokemon: (pokemon.id == C.LEAFEON_EX, energy_count(pokemon), pokemon.hp))

    def _best_available_front_damage(self) -> int:
        best = 0
        for pokemon in self._my_board():
            if self._leafeon_ready(pokemon):
                best = max(best, adjusted_grass_damage(self._verdant_raw_damage(pokemon), self.op_active))
            if pokemon.id == C.CRUSTLE and self._crustle_ready(pokemon):
                best = max(best, adjusted_grass_damage(120, self.op_active))
            if pokemon.id == C.TEAL_MASK_OGERPON and energy_count(pokemon) >= 3:
                best = max(best, adjusted_grass_damage(120, self.op_active))
        return best

    def _verdant_raw_damage(self, attacker: Pokemon | None) -> int:
        raw = self.opponent_total_energy * 60
        if attacker is not None and has_tool(attacker, C.MAXIMUM_BELT):
            data = data_for(self.op_active)
            if data and (data.ex or data.megaEx):
                raw += 50
        return raw

    def _leafeon_ready(self, pokemon: Pokemon | None) -> bool:
        if pokemon is None or pokemon.id != C.LEAFEON_EX or typed_energy_count(pokemon, C.BASIC_GRASS_ENERGY) < 1:
            return False
        needed = 1 if has_tool(pokemon, C.COUNTER_GAIN) and self._behind_on_prizes() else 2
        return energy_count(pokemon) >= needed

    def _crustle_ready(self, pokemon: Pokemon | None) -> bool:
        if pokemon is None or pokemon.id != C.CRUSTLE or typed_energy_count(pokemon, C.BASIC_GRASS_ENERGY) < 1:
            return False
        needed = 2 if has_tool(pokemon, C.COUNTER_GAIN) and self._behind_on_prizes() else 3
        return energy_count(pokemon) >= needed

    def _leafeon_like(self, pokemon: Pokemon | None) -> int:
        if pokemon is None:
            return 0
        return int(pokemon.id == C.LEAFEON_EX or pokemon.id in EEVEE_IDS)

    def _eevee_line_count(self) -> int:
        return self.field_counts[C.LEAFEON_EX] + self.field_counts[C.SYLVEON] + sum(self.field_counts[eid] for eid in EEVEE_IDS)

    def _need_more_eevee(self) -> bool:
        target = 2 if self._late_game_recovery_window() else 1
        return self._eevee_line_count() < target

    def _need_leafeon_card(self) -> bool:
        eevee_total = sum(self.field_counts[eid] for eid in EEVEE_IDS)
        if not self._late_game_recovery_window():
            return False
        return self.field_counts[C.LEAFEON_EX] + self.hand_counts[C.LEAFEON_EX] < max(1, min(eevee_total, 2))

    def _need_early_wall_basic(self) -> bool:
        if len(self.me.bench or []) >= int(getattr(self.me, "benchMax", 5) or 5):
            return False
        wall_basics = self.field_counts[C.DWEBBLE] + self.field_counts[C.SHAYMIN_SEND_FLOWERS] + self.field_counts[C.CHANSEY]
        if wall_basics < 2:
            return True
        return self._need_sinistcha_basic() or (self._need_more_eevee() and self._late_game_recovery_window())

    def _need_crustle_card(self) -> bool:
        dwebble_total = self.field_counts[C.DWEBBLE] + self.field_counts[C.CRUSTLE]
        if dwebble_total == 0:
            return False
        return self.field_counts[C.CRUSTLE] + self.hand_counts[C.CRUSTLE] < min(2, dwebble_total)

    def _need_sylveon_card(self) -> bool:
        if not self._opponent_ex_pressure():
            return False
        eevee_total = sum(self.field_counts[eid] for eid in EEVEE_IDS)
        if eevee_total == 0:
            return False
        return self.field_counts[C.SYLVEON] + self.hand_counts[C.SYLVEON] < 1 and not self._has_ex_wall()

    def _need_sinistcha_basic(self) -> bool:
        if not SINISTCHA_ENABLED:
            return False
        if not self._opponent_non_ex_pressure() and self.field_counts[C.POLTCHAGEIST] + self.field_counts[C.SINISTCHA] >= 1:
            return False
        return self.field_counts[C.POLTCHAGEIST] + self.field_counts[C.SINISTCHA] < 2

    def _need_sinistcha_card(self) -> bool:
        if not SINISTCHA_ENABLED:
            return False
        polt_total = self.field_counts[C.POLTCHAGEIST] + self.field_counts[C.SINISTCHA]
        if polt_total == 0:
            return False
        return self.field_counts[C.SINISTCHA] + self.hand_counts[C.SINISTCHA] < min(2, polt_total)

    def _opponent_non_ex_pressure(self) -> bool:
        has_non_ex_attacker = False
        for _, pokemon in self._opponent_board():
            data = data_for(pokemon)
            if data and not data.ex and not data.megaEx and pokemon.id not in {C.BASIC_GRASS_ENERGY}:
                has_non_ex_attacker = True
        return has_non_ex_attacker and not self._opponent_ex_pressure()

    def _is_ex_wall(self, pokemon: Pokemon | None) -> bool:
        return pokemon is not None and pokemon.id in (C.CRUSTLE, C.SYLVEON) and self._opponent_ex_pressure()

    def _has_ex_wall(self) -> bool:
        return any(self._is_ex_wall(pokemon) for pokemon in self._my_board())

    def _best_grass_accel_target(self) -> Pokemon | None:
        candidates = [
            pokemon
            for pokemon in self._my_board()
            if pokemon.id in GRASS_BENCH_TARGETS and pokemon is not self.active
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda p: (
                p.id == C.LEAFEON_EX and not self._leafeon_ready(p),
                p.id in EEVEE_IDS and self._late_game_recovery_window(),
                p.id == C.CRUSTLE and not self._crustle_ready(p),
                p.id == C.SINISTCHA and energy_count(p) == 0,
                p.id == C.DWEBBLE,
                -energy_count(p),
                p.hp,
            ),
        )

    def _best_accel_target_from_hand(self) -> Pokemon | None:
        candidates = [pokemon for pokemon in self._my_board() if pokemon.id in GRASS_BENCH_TARGETS | {C.CHANSEY}]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda p: (
                p.id == C.LEAFEON_EX and not self._leafeon_ready(p),
                p.id == C.CRUSTLE and not self._crustle_ready(p),
                p.id in EEVEE_IDS and self._late_game_recovery_window(),
                p is self.active and p.id in (C.SHAYMIN_SEND_FLOWERS, C.CHANSEY),
                -energy_count(p),
            ),
        )

    def _valuable_legacy_target(self, pokemon: Pokemon | None, active: bool) -> bool:
        if pokemon is None:
            return False
        if pokemon.id == C.LEAFEON_EX:
            return self._late_game_recovery_window()
        return active and pokemon.id in WALL_IDS and self._opponent_ex_pressure()

    def _best_counter_gain_target(self) -> Pokemon | None:
        if not self._behind_on_prizes():
            return None
        candidates = [
            pokemon
            for pokemon in self._my_board()
            if pokemon.id in (C.LEAFEON_EX, C.CRUSTLE) and not has_any_tool(pokemon)
        ]
        if not candidates:
            candidates = [
                pokemon
                for pokemon in self._my_board()
                if pokemon.id in EEVEE_IDS and not has_any_tool(pokemon) and self._late_game_recovery_window()
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: (p.id == C.LEAFEON_EX, p.id == C.CRUSTLE, energy_count(p), p.hp))

    def _best_maximum_belt_target(self) -> Pokemon | None:
        candidates = [
            pokemon
            for pokemon in self._my_board()
            if pokemon.id == C.LEAFEON_EX and not has_any_tool(pokemon)
        ]
        if not candidates:
            candidates = [
                pokemon
                for pokemon in self._my_board()
                if pokemon.id in EEVEE_IDS and not has_any_tool(pokemon) and self._late_game_recovery_window()
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: (p.id == C.LEAFEON_EX, energy_count(p), p.hp))

    def _maximum_belt_target_score(self, pokemon: Pokemon | None) -> float:
        if pokemon is None or has_any_tool(pokemon):
            return -500
        if pokemon.id == C.LEAFEON_EX:
            return 2600 + energy_count(pokemon) * 260
        if pokemon.id in EEVEE_IDS and self._late_game_recovery_window():
            return 1300
        return -350

    def _can_take_active_ko_with_leafeon(self, pokemon: Pokemon | None) -> bool:
        if pokemon is None or pokemon.id != C.LEAFEON_EX or self.op_active is None:
            return False
        return adjusted_grass_damage(self._verdant_raw_damage(pokemon), self.op_active) >= self.op_active.hp

    def _has_evolve_source(self) -> bool:
        return any(pokemon.id in EEVEE_IDS for pokemon in self._my_board())

    def _needs_attack_energy(self) -> bool:
        return any(
            (pokemon.id == C.LEAFEON_EX and not self._leafeon_ready(pokemon))
            or (pokemon.id == C.CRUSTLE and not self._crustle_ready(pokemon))
            or pokemon.id in (C.DWEBBLE, C.SHAYMIN_SEND_FLOWERS, C.CHANSEY, C.POLTCHAGEIST, C.SINISTCHA)
            for pokemon in self._my_board()
        )

    def _needs_grass_kagura(self) -> bool:
        if self.hand_grass > 0:
            return False
        return any(pokemon.id in MAIN_LINE and energy_count(pokemon) < 2 for pokemon in self._my_board())

    def _can_pivot_to_ogerpon_for_kagura(self) -> bool:
        return False

    def _has_useful_discard(self) -> bool:
        return any(self.discard_counts[cid] > 0 for cid in RECOVERY_IDS)

    def _opponent_ex_pressure(self) -> bool:
        return any(data_for(pokemon) and (data_for(pokemon).ex or data_for(pokemon).megaEx) for _, pokemon in self._opponent_board())

    def _bench_damage_prevention_matters(self) -> bool:
        ids = {pokemon.id for _, pokemon in self._opponent_board()}
        return bool(ids & {119, 120, 121, 741, 742, 743, 235, 666})

    def _bench_damage_total(self) -> int:
        return sum(damage_on(pokemon) for _, pokemon in field_pokemon(self.me) if pokemon is not self.active)

    def _should_retreat(self) -> bool:
        if self.active is None:
            return False
        if self._leafeon_ready(self.active):
            return False
        if self._is_ex_wall(self.active):
            return False
        if self.active.id == C.DWEBBLE and self.field_counts[C.CRUSTLE] < 2:
            return False
        if self.active.id == C.SHAYMIN_SEND_FLOWERS and self._best_grass_accel_target() is not None:
            return False
        if self.active.id == C.CHANSEY and self._best_accel_target_from_hand() is not None:
            return False
        if self._can_pivot_to_ogerpon_for_kagura():
            return True
        if self._late_game_recovery_window() and any(self._leafeon_ready(pokemon) for pokemon in self.me.bench or []):
            return True
        return any(self._is_ex_wall(pokemon) for pokemon in self.me.bench or [])

    def _should_switch_item(self) -> bool:
        return self._can_pivot_to_ogerpon_for_kagura() or self._should_retreat()

    def _option_board_index(self, option) -> int:
        area = getattr(option, "area", None)
        if area == AreaType.ACTIVE or area == int(AreaType.ACTIVE):
            return 0
        return int(getattr(option, "index", 0) or 0) + 1


def fallback(obs_dict: dict) -> list[int]:
    select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if not select:
        return my_deck
    options = select.get("option") or []
    try:
        min_count = max(0, int(select.get("minCount", 0) or 0))
        max_count = max(0, int(select.get("maxCount", 0) or 0))
    except Exception:
        min_count = 0
        max_count = 0
    if max_count == 0:
        return []
    limit = min(len(options), max_count)
    if limit <= 0:
        return []
    return list(range(min(min_count, limit)))


def normalize_selection(selection: list[int], obs_dict: dict) -> list[int]:
    select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if not select:
        return my_deck
    options = select.get("option") or []
    try:
        min_count = max(0, int(select.get("minCount", 0) or 0))
        max_count = max(0, int(select.get("maxCount", 0) or 0))
    except Exception:
        min_count = 0
        max_count = 0
    if max_count == 0:
        return []
    limit = min(len(options), max_count)
    if limit <= 0:
        return []
    out: list[int] = []
    for idx in selection or []:
        if isinstance(idx, int) and 0 <= idx < len(options) and idx not in out:
            out.append(idx)
        if len(out) >= limit:
            break
    target = min(min_count, limit)
    if len(out) < target:
        for idx in range(len(options)):
            if idx not in out:
                out.append(idx)
            if len(out) >= target:
                break
    return out[:limit]


def agent(obs_dict: dict) -> list[int]:
    _DIAG["calls"] += 1
    try:
        if obs_dict.get("select") is None:
            return my_deck
        obs = to_observation_class(obs_dict)
        if obs.current is None or obs.select is None:
            return fallback(obs_dict)
        is_main = obs.select.context == SelectContext.MAIN or getattr(obs.select, "type", None) == SelectType.MAIN
        selection = LeafeonPolicy(obs).choose() if is_main or obs.select.option else fallback(obs_dict)
        normalized = normalize_selection(selection, obs_dict)
        if normalized != selection:
            _DIAG["fallback"] += 1
        return normalized
    except Exception as exc:
        _DIAG["fallback"] += 1
        _DIAG["last_error"] = f"{type(exc).__name__}: {exc}"
        return fallback(obs_dict)
