from __future__ import annotations

import os
from collections import defaultdict

from cg.api import (
    AreaType,
    Card,
    CardType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    to_observation_class,
)


class C:
    FIRE_ENERGY = 2
    PSYCHIC_ENERGY = 5
    FIGHTING_ENERGY = 6

    KORAIDON = 62
    SMOOCHUM = 183
    MEGA_KANGASKHAN_EX = 756

    UNFAIR_STAMP = 1080
    BUDDY_BUDDY_POFFIN = 1086
    NIGHT_STRETCHER = 1097
    ENERGY_SEARCH = 1119
    ULTRA_BALL = 1121
    POKEGEAR = 1122
    SWITCH = 1123
    MASTER_BALL = 1125
    MEGA_SIGNAL = 1145
    BOSS_ORDERS = 1182
    KIERAN = 1191
    XEROSICS_MACHINATIONS = 1197
    CRISPIN = 1198
    SURFER = 1203
    LILLIES_DETERMINATION = 1227
    EXCITE_STADIUM = 1251


ENERGY_IDS = {C.FIRE_ENERGY, C.PSYCHIC_ENERGY, C.FIGHTING_ENERGY}
LO_POKEMON_IDS = {58, 344, 345}
ALAKAZAM_POKEMON_IDS = {741, 742, 743}
SUPPORTERS = {
    C.BOSS_ORDERS,
    C.KIERAN,
    C.CRISPIN,
    C.XEROSICS_MACHINATIONS,
    C.SURFER,
    C.LILLIES_DETERMINATION,
}

DECK_PATH = "deck.csv"
if not os.path.exists(DECK_PATH):
    DECK_PATH = "/kaggle_simulations/agent/deck.csv"
with open(DECK_PATH, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]

card_table = {card.cardId: card for card in all_card_data()}


def get_card(
    obs: Observation,
    area,
    index: int | None,
    player_index: int | None,
) -> Pokemon | Card | None:
    if obs.current is None or area is None or index is None:
        return None
    try:
        area = AreaType(area)
        player = obs.current.players[
            obs.current.yourIndex if player_index is None else player_index
        ]
        match area:
            case AreaType.DECK:
                return obs.select.deck[index] if obs.select and obs.select.deck else None
            case AreaType.HAND:
                return player.hand[index] if player.hand else None
            case AreaType.DISCARD:
                return player.discard[index]
            case AreaType.ACTIVE:
                return player.active[index]
            case AreaType.BENCH:
                return player.bench[index]
            case AreaType.PRIZE:
                return player.prize[index]
            case AreaType.STADIUM:
                return obs.current.stadium[index]
            case AreaType.LOOKING:
                return obs.current.looking[index] if obs.current.looking else None
    except Exception:
        return None
    return None


def field_pokemon(player_state) -> list[tuple[AreaType, int, Pokemon]]:
    result: list[tuple[AreaType, int, Pokemon]] = []
    for index, pokemon in enumerate(getattr(player_state, "active", None) or []):
        if pokemon is not None:
            result.append((AreaType.ACTIVE, index, pokemon))
    for index, pokemon in enumerate(getattr(player_state, "bench", None) or []):
        if pokemon is not None:
            result.append((AreaType.BENCH, index, pokemon))
    return result


def card_id(card: Card | Pokemon | None) -> int:
    return int(getattr(card, "id", 0) or 0)


def data_for(card: Card | Pokemon | None):
    return card_table.get(card_id(card))


def energy_count(pokemon: Pokemon | None) -> int:
    if pokemon is None:
        return 0
    cards = getattr(pokemon, "energyCards", None)
    if cards is not None:
        return len(cards)
    return len(getattr(pokemon, "energies", None) or [])


def attached_energy_ids(pokemon: Pokemon | None) -> list[int]:
    if pokemon is None:
        return []
    cards = getattr(pokemon, "energyCards", None)
    if cards is not None:
        return [int(card.id) for card in cards]
    return [int(energy) for energy in (getattr(pokemon, "energies", None) or [])]


def prize_count(pokemon: Pokemon | None) -> int:
    data = data_for(pokemon)
    if not data:
        return 1
    if data.megaEx:
        return 3
    if data.ex:
        return 2
    return 1


def stadium_id(state) -> int:
    for card in getattr(state, "stadium", None) or []:
        if card is not None:
            return int(card.id)
    return 0


def effect_id(select) -> int:
    return card_id(getattr(select, "effect", None))


class KangaskhanPolicy:
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

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        for _, _, pokemon in field_pokemon(self.me):
            self.field_counts[pokemon.id] += 1
        for card in getattr(self.me, "hand", None) or []:
            self.hand_counts[card.id] += 1
        for card in getattr(self.me, "discard", None) or []:
            self.discard_counts[card.id] += 1

        self.hand_size = len(getattr(self.me, "hand", None) or [])
        self.bench_free = int(getattr(self.me, "benchMax", 5) or 5) - len(
            getattr(self.me, "bench", None) or []
        )
        self.is_lo_matchup = any(
            pokemon.id in LO_POKEMON_IDS
            for _, _, pokemon in field_pokemon(self.opponent)
        )
        visible_opponent_ids = {
            pokemon.id for _, _, pokemon in field_pokemon(self.opponent)
        }
        visible_opponent_ids.update(
            card.id for card in (getattr(self.opponent, "discard", None) or [])
        )
        self.is_alakazam_matchup = bool(
            visible_opponent_ids & ALAKAZAM_POKEMON_IDS
        )

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []
        scores = [self._score_option(option) for option in self.select.option]
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

        if self._is_smoochum_energy_search():
            target = self._best_acceleration_target(bench_only=True)
            if card_id(target) == C.KORAIDON:
                wanted = 0 if C.PSYCHIC_ENERGY in attached_energy_ids(target) else 1
            else:
                wanted = min(2, max(0, 3 - energy_count(target))) if target else 0
            count = min(wanted, self.select.maxCount, len(ranked))
            count = max(count, self.select.minCount)
            return ranked[:count]

        positive = [index for index in ranked if scores[index] > 0]
        chosen = positive[: self.select.maxCount]
        if len(chosen) < self.select.minCount:
            for index in ranked:
                if index not in chosen:
                    chosen.append(index)
                if len(chosen) >= self.select.minCount:
                    break
        return chosen[: self.select.maxCount]

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER:
            return float(option.number or 0)
        if option.type == OptionType.YES:
            if self.context == SelectContext.IS_FIRST:
                return -1000
            if self.context == SelectContext.FIRST_EFFECT:
                return 1000 if self._needs_switch() else -10
            return 1000 if self.context == SelectContext.ACTIVATE else 10
        if option.type == OptionType.NO:
            if self.context in (SelectContext.IS_FIRST, SelectContext.MULLIGAN):
                return 1000
            if self.context == SelectContext.FIRST_EFFECT:
                return 1000 if not self._needs_switch() else -10
            return 0
        if option.type == OptionType.CARD:
            return self._score_card(option)
        if option.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
            return self._score_energy_option(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            return 52000 if self._should_retreat() else -1
        if option.type == OptionType.ATTACK:
            return self._score_attack()
        if option.type == OptionType.END:
            return -100
        return 100

    def _score_card(self, option) -> float:
        owner = self.my_index if option.playerIndex is None else option.playerIndex
        card = get_card(self.obs, option.area, option.index, owner)
        if card is None:
            return 0
        if owner == self.op_index and isinstance(card, Pokemon):
            return self._opponent_target_score(card, option.area)

        cid = card.id
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._setup_active_score(cid)
        if self.context in (
            SelectContext.SETUP_BENCH_POKEMON,
            SelectContext.TO_BENCH,
            SelectContext.TO_FIELD,
        ):
            return self._bench_score(cid)
        if self.context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            return self._active_score(card) if isinstance(card, Pokemon) else 0
        if self.context in (SelectContext.TO_HAND, SelectContext.LOOK):
            return self._to_hand_score(cid)
        if self.context in (SelectContext.ATTACH_FROM, SelectContext.ATTACH_TO):
            if isinstance(card, Pokemon):
                return self._effect_attach_target_score(card, option.area)
            return self._energy_choice_score(cid)
        if self.context in (
            SelectContext.DISCARD,
            SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
        ):
            return self._discard_score(cid)
        if self.context in (SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
            return self._recycle_score(cid)
        return self._to_hand_score(cid) * 0.05

    def _setup_active_score(self, cid: int) -> float:
        if cid == C.SMOOCHUM:
            return 6000
        if cid == C.MEGA_KANGASKHAN_EX:
            return 3000
        if cid == C.KORAIDON:
            return 800
        return 10

    def _bench_score(self, cid: int) -> float:
        if cid == C.MEGA_KANGASKHAN_EX:
            return 6000 - self.field_counts[cid] * 700
        if cid == C.SMOOCHUM:
            return 4200 if self.field_counts[cid] == 0 else 300
        if cid == C.KORAIDON:
            return 5600 if self.is_lo_matchup and self.field_counts[cid] == 0 else -1
        return 10

    def _active_score(self, pokemon: Pokemon) -> float:
        if pokemon.id == C.MEGA_KANGASKHAN_EX:
            return 30000 + energy_count(pokemon) * 5000
        if pokemon.id == C.KORAIDON and self._koraidon_ready(pokemon):
            return 31000
        if pokemon.id == C.SMOOCHUM and self._best_acceleration_target(bench_only=True):
            return 24000
        return 100

    def _to_hand_score(self, cid: int) -> float:
        if cid == C.MEGA_KANGASKHAN_EX:
            return 26000 if self.field_counts[cid] == 0 else 5000
        if cid == C.SMOOCHUM:
            return 24000 if self.field_counts[cid] == 0 else 1500
        if cid == C.KORAIDON:
            return 30000 if self.is_lo_matchup and self.field_counts[cid] == 0 else 500
        if cid == C.PSYCHIC_ENERGY:
            return 21000 if self._energy_in_hand() is False else 14000
        if cid in (C.FIRE_ENERGY, C.FIGHTING_ENERGY):
            return 18500 if self.is_lo_matchup else 9000
        if cid == C.CRISPIN:
            return 20000 if self._best_kangaskhan_target() else 3000
        if cid == C.SURFER:
            return 17000 if self._needs_switch() else 4500
        if cid == C.KIERAN:
            return 15500 if self._needs_switch() else 7500
        if cid == C.LILLIES_DETERMINATION:
            return 14500 if self.hand_size <= 4 else 3500
        if cid == C.BOSS_ORDERS:
            return 13000 if self._should_boss() else 1800
        if cid == C.EXCITE_STADIUM:
            return 10500 if stadium_id(self.state) != cid else 500
        if cid == C.UNFAIR_STAMP:
            return 24000 if self.is_alakazam_matchup else 6000
        if cid == C.XEROSICS_MACHINATIONS:
            return 23000 if self.is_alakazam_matchup else 2500
        if cid in (C.MEGA_SIGNAL, C.MASTER_BALL, C.ULTRA_BALL):
            return 9500
        return 500

    def _score_energy_option(self, option) -> float:
        card = get_card(
            self.obs,
            option.area,
            option.index,
            self.my_index if option.playerIndex is None else option.playerIndex,
        )
        return self._energy_choice_score(card_id(card))

    def _energy_choice_score(self, cid: int) -> float:
        if effect_id(self.select) == C.CRISPIN:
            if self.is_lo_matchup and cid in (C.FIRE_ENERGY, C.FIGHTING_ENERGY):
                return 27000
            if cid == C.PSYCHIC_ENERGY:
                return 18000 if self.is_lo_matchup else 21000
        if self.is_lo_matchup:
            koraidon = self._field_koraidon()
            attached = attached_energy_ids(koraidon)
            if cid == C.FIRE_ENERGY and C.FIRE_ENERGY not in attached:
                return 26000
            if cid == C.FIGHTING_ENERGY and C.FIGHTING_ENERGY not in attached:
                return 25000
            if cid == C.PSYCHIC_ENERGY and C.PSYCHIC_ENERGY not in attached:
                return 19000
        if cid == C.PSYCHIC_ENERGY:
            return 20000
        if cid in (C.FIRE_ENERGY, C.FIGHTING_ENERGY):
            return 9000
        return 100

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        if card is None:
            return 0
        cid = card.id
        data = data_for(card)

        if data and data.cardType == CardType.POKEMON:
            if self.bench_free <= 0:
                return -1
            if cid == C.MEGA_KANGASKHAN_EX:
                return 42000 - self.field_counts[cid] * 1800
            if cid == C.SMOOCHUM:
                return 39000 if self.field_counts[cid] == 0 else -1
            if cid == C.KORAIDON:
                return 40500 if self.is_lo_matchup and self.field_counts[cid] == 0 else -1
            return -1

        if cid == C.BUDDY_BUDDY_POFFIN:
            return 43000 if self.bench_free and self.field_counts[C.SMOOCHUM] == 0 else -1
        if cid == C.MEGA_SIGNAL:
            return 41000 if self.field_counts[C.MEGA_KANGASKHAN_EX] == 0 else 7000
        if cid == C.ULTRA_BALL:
            missing_core = (
                self.field_counts[C.MEGA_KANGASKHAN_EX] == 0
                or self.field_counts[C.SMOOCHUM] == 0
                or (self.is_lo_matchup and self.field_counts[C.KORAIDON] == 0)
            )
            return 39000 if missing_core and self.hand_size >= 3 else 5000
        if cid == C.POKEGEAR:
            if self.state.supporterPlayed:
                return -1
            return 34000 if self._best_kangaskhan_target() else 18000
        if cid == C.ENERGY_SEARCH:
            return 33000 if not self._energy_in_hand() else 7000
        if cid == C.SWITCH:
            return 47000 if self._needs_switch() else -1
        if cid == C.NIGHT_STRETCHER:
            if self.discard_counts[C.MEGA_KANGASKHAN_EX]:
                return 30000
            if self.discard_counts[C.PSYCHIC_ENERGY] and not self._energy_in_hand():
                return 25000
            return -1
        if cid == C.UNFAIR_STAMP:
            return 65000 if self.is_alakazam_matchup else 33000
        if cid == C.CRISPIN:
            if self.state.supporterPlayed:
                return -1
            if self.is_lo_matchup and self._field_koraidon() is None:
                return -1
            target = self._best_acceleration_target()
            return 45000 if target is not None and energy_count(target) < 3 else -1
        if cid == C.XEROSICS_MACHINATIONS:
            if self.state.supporterPlayed:
                return -1
            return 43000 if self.is_alakazam_matchup else -1
        if cid == C.SURFER:
            if self.state.supporterPlayed:
                return -1
            return 44500 if self._needs_switch() else 12000 if self.hand_size <= 3 else -1
        if cid == C.KIERAN:
            if self.state.supporterPlayed:
                return -1
            if self._needs_switch():
                return 44000
            return 31000 if self._kangaskhan_can_attack() and self._opponent_is_ex() else -1
        if cid == C.BOSS_ORDERS:
            if self.state.supporterPlayed:
                return -1
            return 42000 if self._should_boss() else -1
        if cid == C.LILLIES_DETERMINATION:
            if self.state.supporterPlayed or self.me.deckCount <= 8 or self.is_lo_matchup:
                return -1
            return 37000 if self.hand_size <= 4 and not self._kangaskhan_can_attack() else 6000
        if cid == C.EXCITE_STADIUM:
            return 30000 if stadium_id(self.state) != cid else -1
        return 500

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        pokemon = get_card(
            self.obs,
            option.inPlayArea,
            option.inPlayIndex,
            self.my_index,
        )
        if card is None or pokemon is None:
            return 0
        if card.id not in ENERGY_IDS:
            return 0
        if pokemon.id == C.KORAIDON and self.is_lo_matchup:
            attached = attached_energy_ids(pokemon)
            if card.id == C.PSYCHIC_ENERGY and C.PSYCHIC_ENERGY in attached:
                return -1
            if card.id == C.FIRE_ENERGY and C.FIRE_ENERGY in attached:
                return -1
            if card.id == C.FIGHTING_ENERGY and C.FIGHTING_ENERGY in attached:
                return -1
            return 43000 + (3000 if card.id in (C.FIRE_ENERGY, C.FIGHTING_ENERGY) else 0)
        if pokemon.id != C.MEGA_KANGASKHAN_EX or energy_count(pokemon) >= 3:
            return -1
        score = 38000 + (3 - energy_count(pokemon)) * 1500
        if option.inPlayArea == AreaType.BENCH and card.id == C.PSYCHIC_ENERGY:
            score += 3000
        if option.inPlayArea == AreaType.ACTIVE:
            score += 2000
        return score

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card_id(card) == C.MEGA_KANGASKHAN_EX:
            if self.is_lo_matchup:
                return -1
            return 70000
        return 100

    def _score_attack(self) -> float:
        if card_id(self.active) == C.MEGA_KANGASKHAN_EX:
            return 20000
        if card_id(self.active) == C.SMOOCHUM:
            return 10000 if self._best_acceleration_target(bench_only=True) else 1000
        if card_id(self.active) == C.KORAIDON:
            return 22000
        return 500

    def _effect_attach_target_score(self, pokemon: Pokemon, area: AreaType) -> float:
        if pokemon.id == C.KORAIDON and self.is_lo_matchup:
            return 60000 if not self._koraidon_ready(pokemon) else -1
        if pokemon.id != C.MEGA_KANGASKHAN_EX or energy_count(pokemon) >= 3:
            return -1
        score = 30000 + (3 - energy_count(pokemon)) * 3000
        if area == AreaType.BENCH:
            score += 2000
        return score

    def _discard_score(self, cid: int) -> float:
        if cid in (C.FIRE_ENERGY, C.FIGHTING_ENERGY):
            return 1600 if self.hand_counts[cid] >= 2 else -400
        if cid in SUPPORTERS and self.hand_counts[cid] >= 2:
            return 1300
        if cid in (C.POKEGEAR, C.ENERGY_SEARCH, C.EXCITE_STADIUM) and self.hand_counts[cid] >= 2:
            return 1100
        if cid == C.SMOOCHUM and self.field_counts[cid] > 0:
            return 900
        if cid in (C.MEGA_KANGASKHAN_EX, C.PSYCHIC_ENERGY):
            return -2500
        return 300

    def _recycle_score(self, cid: int) -> float:
        if cid == C.MEGA_KANGASKHAN_EX:
            return 4000
        if cid == C.PSYCHIC_ENERGY:
            return 3200
        if cid == C.SMOOCHUM:
            return 1800
        return 100

    def _opponent_target_score(self, pokemon: Pokemon, area: AreaType) -> float:
        if area == AreaType.ACTIVE:
            return -1000
        hp = int(getattr(pokemon, "hp", 0) or 0)
        score = prize_count(pokemon) * 2500 + energy_count(pokemon) * 400
        if hp <= 200:
            score += 7000
        elif hp <= 250:
            score += 4000
        return score

    def _best_kangaskhan_target(self, bench_only: bool = False) -> Pokemon | None:
        candidates = [
            pokemon
            for area, _, pokemon in field_pokemon(self.me)
            if pokemon.id == C.MEGA_KANGASKHAN_EX
            and energy_count(pokemon) < 3
            and (not bench_only or area == AreaType.BENCH)
        ]
        return max(candidates, key=energy_count, default=None)

    def _best_acceleration_target(self, bench_only: bool = False) -> Pokemon | None:
        if self.is_lo_matchup:
            koraidon = next(
                (
                    pokemon
                    for area, _, pokemon in field_pokemon(self.me)
                    if pokemon.id == C.KORAIDON
                    and not self._koraidon_ready(pokemon)
                    and (not bench_only or area == AreaType.BENCH)
                ),
                None,
            )
            if koraidon is not None:
                return koraidon
        return self._best_kangaskhan_target(bench_only)

    def _field_koraidon(self) -> Pokemon | None:
        return next(
            (
                pokemon
                for _, _, pokemon in field_pokemon(self.me)
                if pokemon.id == C.KORAIDON
            ),
            None,
        )

    def _is_smoochum_energy_search(self) -> bool:
        if card_id(self.active) != C.SMOOCHUM or self.select.maxCount != 2:
            return False
        cards = [
            get_card(
                self.obs,
                option.area,
                option.index,
                self.my_index if option.playerIndex is None else option.playerIndex,
            )
            for option in self.select.option or []
        ]
        return bool(cards) and all(card_id(card) == C.PSYCHIC_ENERGY for card in cards)

    def _energy_in_hand(self) -> bool:
        return any(self.hand_counts[cid] for cid in ENERGY_IDS)

    def _crispin_pair_needed(self) -> bool:
        target = self._best_acceleration_target()
        return target is not None and energy_count(target) < 3

    def _ready_bench_kangaskhan(self) -> Pokemon | None:
        return next(
            (
                pokemon
                for area, _, pokemon in field_pokemon(self.me)
                if area == AreaType.BENCH
                and pokemon.id == C.MEGA_KANGASKHAN_EX
                and energy_count(pokemon) >= 3
            ),
            None,
        )

    def _ready_bench_attacker(self) -> Pokemon | None:
        if self.is_lo_matchup:
            koraidon = next(
                (
                    pokemon
                    for area, _, pokemon in field_pokemon(self.me)
                    if area == AreaType.BENCH and self._koraidon_ready(pokemon)
                ),
                None,
            )
            if koraidon is not None:
                return koraidon
        return self._ready_bench_kangaskhan()

    def _koraidon_ready(self, pokemon: Pokemon | None) -> bool:
        energies = attached_energy_ids(pokemon)
        return (
            card_id(pokemon) == C.KORAIDON
            and len(energies) >= 3
            and C.FIRE_ENERGY in energies
            and C.FIGHTING_ENERGY in energies
        )

    def _kangaskhan_can_attack(self) -> bool:
        return card_id(self.active) == C.MEGA_KANGASKHAN_EX and energy_count(self.active) >= 3

    def _needs_switch(self) -> bool:
        active_ready = self._kangaskhan_can_attack() or self._koraidon_ready(self.active)
        return not active_ready and self._ready_bench_attacker() is not None

    def _should_retreat(self) -> bool:
        return self._needs_switch()

    def _opponent_is_ex(self) -> bool:
        data = data_for(self.op_active)
        return bool(data and (data.ex or data.megaEx))

    def _should_boss(self) -> bool:
        if not self._kangaskhan_can_attack():
            return False
        return any(
            area == AreaType.BENCH and int(getattr(pokemon, "hp", 0) or 0) <= 200
            for area, _, pokemon in field_pokemon(self.opponent)
        )


def _policy_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return list(my_deck)
    return KangaskhanPolicy(obs).choose()


_DIAG = {"calls": 0, "fallback": 0, "last_error": ""}


def _select_from_raw(obs_dict):
    if isinstance(obs_dict, dict):
        return obs_dict.get("select")
    return getattr(obs_dict, "select", None)


def _field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _fallback_selection(obs_dict) -> list[int]:
    select = _select_from_raw(obs_dict)
    if not select:
        return list(my_deck)
    options = _field(select, "option", []) or []
    min_count = max(0, _as_int(_field(select, "minCount", 0), 0))
    max_count = max(0, _as_int(_field(select, "maxCount", min_count), min_count))
    limit = min(max_count, len(options))
    if limit <= 0:
        return []
    return list(range(min(min_count, limit)))


def _normalize_selection(selection, obs_dict) -> list[int]:
    select = _select_from_raw(obs_dict)
    if not select:
        return list(my_deck)
    options = _field(select, "option", []) or []
    min_count = max(0, _as_int(_field(select, "minCount", 0), 0))
    max_count = max(0, _as_int(_field(select, "maxCount", min_count), min_count))
    limit = min(max_count, len(options))
    if limit <= 0:
        return []

    cleaned: list[int] = []
    seen: set[int] = set()
    if isinstance(selection, list):
        for index in selection:
            if isinstance(index, int) and index not in seen and 0 <= index < len(options):
                cleaned.append(index)
                seen.add(index)
                if len(cleaned) >= limit:
                    break
    while len(cleaned) < min(min_count, limit):
        for index in range(len(options)):
            if index not in seen:
                cleaned.append(index)
                seen.add(index)
                break
    return cleaned[:limit]


def agent(obs_dict: dict) -> list[int]:
    _DIAG["calls"] += 1
    try:
        selection = _policy_agent(obs_dict)
    except Exception as exc:
        _DIAG["fallback"] += 1
        _DIAG["last_error"] = f"{type(exc).__name__}: {exc}"
        return _fallback_selection(obs_dict)
    normalized = _normalize_selection(selection, obs_dict)
    if normalized != selection:
        _DIAG["fallback"] += 1
    return normalized
