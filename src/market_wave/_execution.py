from __future__ import annotations

from math import exp

from ._types import (
    _EntryFlow,
    _ExecutionResult,
    _IncomingOrder,
    _MarketEvent,
    _MicrostructureState,
    _ProcessedOrder,
    _StepComputationCache,
    _TradeStats,
)
from .state import LatentState, MDFState, OrderBookState, PriceMap, TickMap


class _MarketExecutionMixin:
    def _cancel_orders(
        self,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        mdf: MDFState | None = None,
        cache: _StepComputationCache | None = None,
    ) -> PriceMap:
        mdf = mdf or self.state.mdf
        cancelled: PriceMap = {}
        for side in ("bid", "ask"):
            volumes = self._orderbook.volumes_for_side(side)
            side_weights, cancel_pressure, side_volume = self._cancel_sampling_profile(
                side,
                current_price,
                latent,
                micro,
                mdf,
                cache,
            )
            if side_volume <= 1e-12:
                continue
            if side_weights:
                regime = self._active_settings
                state_multiplier = (
                    1.0
                    + 0.50 * micro.cancel_pressure
                    + 0.28 * micro.liquidity_stress
                    + 0.17 * micro.spread_pressure
                )
                event_probability = self._cancel_event_probability(
                    side_volume,
                    cancel_pressure,
                    micro,
                )
                if self._unit_random() < event_probability:
                    expected_events = (
                        0.0058
                        * regime["cancel"]
                        * state_multiplier
                        * (0.045 + cancel_pressure)
                        * side_volume
                        / max(self._mean_child_order_size(), 1e-12)
                        / max(event_probability, 1e-12)
                    )
                    event_count = min(
                        max(1, int(6 + 4 * side_volume + 5 * self.popularity)),
                        self._sample_poisson(expected_events),
                    )
                    for _ in range(event_count):
                        price = self._sample_price(side_weights)
                        available = volumes.get(price, 0.0)
                        if available <= 1e-12:
                            continue
                        size_scale = self._clamp(
                            1.0
                            - 0.08 * micro.cancel_pressure
                            - 0.06 * micro.liquidity_stress
                            - 0.04 * micro.spread_pressure,
                            0.82,
                            1.0,
                        )
                        requested = min(available, size_scale * self._sample_order_size())
                        self._cancel_or_requote_volume(
                            cancelled,
                            side,
                            price,
                            requested,
                            current_price,
                            micro,
                            mdf,
                        )
            self._orderbook.clean()
        cancelled = self._drop_zeroes(cancelled)
        micro.last_cancelled_volume = sum(cancelled.values())
        return cancelled

    def _update_mdf_pockets(
        self,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        stats: _TradeStats,
        flow_imbalance: float,
        mdf: MDFState,
        cache: _StepComputationCache | None,
        cancelled_volume: PriceMap,
    ) -> None:
        current_tick = self.price_to_tick(current_price)
        min_tick = max(1, current_tick - self.grid_radius)
        max_tick = current_tick + self.grid_radius
        activity = self._event_cluster_signal(micro)
        stress = self._clamp(
            0.42 * micro.cancel_pressure
            + 0.36 * micro.liquidity_stress
            + 0.22 * micro.spread_pressure,
            0.0,
            2.5,
        ) / 2.5
        execution_impulse = self._clamp(
            stats.total_volume / max(1.0, 2.5 * self.popularity + 1.0),
            0.0,
            1.0,
        )
        cancel_impulse = self._clamp(
            sum(cancelled_volume.values()) / max(1.0, 2.5 * self.popularity + 1.0),
            0.0,
            1.0,
        )
        for side in ("bid", "ask"):
            pockets = self._mdf_pockets_by_side_tick.setdefault(side, {})
            side_stress = self._stress_for_book_side(side, micro)
            decay = self._clamp(
                0.88
                + 0.04 * micro.resiliency
                - 0.11 * stress
                - 0.08 * side_stress * micro.liquidity_stress
                - 0.04 * cancel_impulse,
                0.62,
                0.94,
            )
            for tick in list(pockets):
                if tick < min_tick or tick > max_tick:
                    del pockets[tick]
                    continue
                pockets[tick] *= decay
                if pockets[tick] < 0.025:
                    del pockets[tick]

            spawn_probability = self._clamp(
                0.035
                + 0.11 * activity
                + 0.08 * execution_impulse
                + 0.06 * max(0.0, micro.resiliency - 0.55)
                + 0.04 * max(0.0, -side_stress),
                0.02,
                0.28,
            )
            if self._unit_random() < spawn_probability:
                weights = self._mdf_pocket_spawn_weights(
                    side,
                    current_price,
                    latent,
                    micro,
                    mdf,
                    None,
                    flow_imbalance,
                )
                if weights:
                    price = self._sample_price(weights)
                    tick = self.price_to_tick(price)
                    strength = self._clamp(
                        0.20
                        + 0.44 * self._unit_random()
                        + 0.28 * activity
                        + 0.16 * execution_impulse
                        - 0.18 * stress,
                        0.12,
                        1.10,
                    )
                    pockets[tick] = self._clamp(pockets.get(tick, 0.0) + strength, 0.0, 1.75)
                    if self._unit_random() < 0.34 + 0.18 * activity:
                        neighbor = tick + (-1 if self._unit_random() < 0.5 else 1)
                        if min_tick <= neighbor <= max_tick:
                            pockets[neighbor] = self._clamp(
                                pockets.get(neighbor, 0.0) + 0.38 * strength,
                                0.0,
                                1.35,
                            )

    def _mdf_pocket_spawn_weights(
        self,
        side: str,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        mdf: MDFState,
        cache: _StepComputationCache | None,
        flow_imbalance: float,
    ) -> PriceMap:
        del latent, micro, flow_imbalance
        probabilities = self._entry_probabilities_for_book_side(side, mdf, cache)
        weights: PriceMap = {}
        for price, probability in probabilities.items():
            if side == "bid":
                if price >= current_price or price < self._min_price:
                    continue
            elif price <= current_price:
                continue
            weights[price] = probability
        return self._normalize_price_map(weights) if weights else {}

    def _level_cancel_probability(
        self,
        side: str,
        price: float,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        mdf: MDFState | None = None,
    ) -> float:
        regime = self._active_settings
        distance = abs(price - current_price) / self.gap
        adverse_side = (side == "bid" and latent.trend < 0) or (side == "ask" and latent.trend > 0)
        adverse = abs(latent.trend) if adverse_side else 0.0
        distance_term = 0.045 * (1.0 - exp(-distance / 5.0))
        volatility_term = 0.026 * latent.volatility / (1.0 + latent.volatility)
        side_stress = self._stress_for_book_side(side, micro)
        stress_cancel = 0.12 * micro.liquidity_stress * side_stress * exp(-distance / 2.4)
        mismatch = self._mdf_cancel_mismatch(side, price, mdf or self.state.mdf, micro=micro)
        survival_weakness = self._resting_level_survival_weakness(
            side, price, current_price, latent, micro
        )
        survival_cancel = (
            0.12 * survival_weakness + 0.20 * mismatch
        ) * (0.72 + 0.28 * exp(-distance / 3.0))
        probability = (
            0.016
            + distance_term
            + volatility_term
            + self._cancel_burst_multiplier(micro, distance, adverse)
            + stress_cancel
            + survival_cancel
        ) * regime["cancel"]
        return self._clamp(probability, 0.006, 0.32)

    def _cancel_side_pressure(
        self,
        side: str,
        current_price: float,
        micro: _MicrostructureState,
        mdf: MDFState,
        cache: _StepComputationCache | None = None,
    ) -> float:
        volumes = self._orderbook.volumes_for_side(side)
        side_volume = sum(max(0.0, volume) for volume in volumes.values())
        if side_volume <= 1e-12:
            return 0.0
        weighted_pressure = 0.0
        for price, volume in volumes.items():
            if volume <= 1e-12:
                continue
            weighted_pressure += volume * self._cancel_price_pressure(
                side,
                price,
                current_price,
                micro,
                mdf,
                cache,
                side_volume=side_volume,
            )
        return self._clamp(weighted_pressure / side_volume, 0.0, 1.25)

    def _cancel_event_probability(
        self,
        side_volume: float,
        cancel_pressure: float,
        micro: _MicrostructureState,
    ) -> float:
        if side_volume <= 1e-12:
            return 0.0
        volume_term = 1.0 - exp(-side_volume / max(7.5 + 6.0 * self.popularity, 1e-12))
        mismatch_term = 1.0 - exp(-cancel_pressure / 0.42)
        probability = (
            0.032
            + 0.35 * mismatch_term
            + 0.18 * volume_term
            + 0.12 * self._clamp(micro.cancel_pressure, 0.0, 2.0) / 2.0
            + 0.09 * self._clamp(micro.liquidity_stress, 0.0, 2.0) / 2.0
            + 0.05 * self._clamp(micro.spread_pressure, 0.0, 1.8) / 1.8
        )
        probability *= 0.82 + 0.46 * self._event_cluster_signal(micro)
        return self._clamp(probability, 0.035, 0.86)

    def _cancel_price_pressure(
        self,
        side: str,
        price: float,
        current_price: float,
        micro: _MicrostructureState,
        mdf: MDFState,
        cache: _StepComputationCache | None = None,
        side_volume: float | None = None,
        current_tick: int | None = None,
    ) -> float:
        distance = abs(price - current_price) / self.gap
        mismatch = self._mdf_cancel_mismatch(
            side,
            price,
            mdf,
            micro=micro,
            cache=cache,
            side_volume=side_volume,
        )
        side_stress = self._stress_for_book_side(side, micro)
        stress = micro.liquidity_stress * side_stress * exp(-distance / 2.5)
        survival_weakness = self._resting_level_survival_weakness(
            side,
            price,
            current_price,
            LatentState(mood=0.0, trend=0.0, volatility=0.0),
            micro,
            cache,
            current_tick,
        )
        near_touch = exp(-max(0.0, distance - 1.0) / 1.0)
        deep_stale = self._clamp((distance - 3.0) / 3.0, 0.0, 1.0)
        distance_pressure = 1.0 - exp(-max(0.0, distance - 2.0) / 2.2)
        near_alignment = (
            near_touch
            * (1.0 - mismatch)
            * (1.0 - 0.45 * self._clamp(stress, 0.0, 1.0))
            * (1.0 + 0.30 * near_touch)
        )
        stale_pressure = distance_pressure * survival_weakness * (1.0 + 0.65 * deep_stale)
        return self._clamp(
            0.015
            + 0.74 * mismatch
            + 0.20 * stress
            + 0.08 * survival_weakness
            + 0.52 * distance_pressure
            + 0.26 * stale_pressure
            + 0.10 * deep_stale
            - 0.16 * near_alignment,
            0.0,
            1.25,
        )

    def _cancel_sampling_weights(
        self,
        side: str,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        mdf: MDFState,
        cache: _StepComputationCache | None = None,
    ) -> PriceMap:
        weights, _, _ = self._cancel_sampling_profile(
            side,
            current_price,
            latent,
            micro,
            mdf,
            cache,
        )
        return weights

    def _cancel_sampling_profile(
        self,
        side: str,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        mdf: MDFState,
        cache: _StepComputationCache | None = None,
    ) -> tuple[PriceMap, float, float]:
        del latent
        volumes = self._orderbook.volumes_for_side(side)
        side_volume = sum(max(0.0, volume) for volume in volumes.values())
        if side_volume <= 1e-12:
            return {}, 0.0, 0.0

        weights: PriceMap = {}
        weighted_pressure = 0.0
        current_tick = self.price_to_tick(current_price)
        for price, volume in volumes.items():
            if volume <= 1e-12:
                continue
            pressure = self._cancel_price_pressure(
                side,
                price,
                current_price,
                micro,
                mdf,
                cache,
                side_volume=side_volume,
                current_tick=current_tick,
            )
            weighted_pressure += volume * pressure
            weights[price] = volume * pressure
        pressure = self._clamp(weighted_pressure / side_volume, 0.0, 1.25)
        return (self._normalize_price_map(weights) if weights else {}, pressure, side_volume)

    def _mdf_cancel_mismatch(
        self,
        side: str,
        price: float,
        mdf: MDFState,
        *,
        micro: _MicrostructureState | None = None,
        cache: _StepComputationCache | None = None,
        side_volume: float | None = None,
    ) -> float:
        volumes = self._orderbook.volumes_for_side(side)
        side_volume = (
            sum(max(0.0, volume) for volume in volumes.values())
            if side_volume is None
            else side_volume
        )
        if side_volume <= 1e-12:
            return 0.0
        price = self._snap_price(price)
        book_share = max(0.0, volumes.get(price, 0.0)) / side_volume
        probabilities = self._entry_probabilities_for_book_side(side, mdf, cache)
        mdf_share = probabilities.get(price, 0.0)
        excess_book = max(0.0, book_share - mdf_share)
        unsupported = 1.0 - self._clamp(mdf_share / max(book_share, 1e-12), 0.0, 1.0)
        expected = self._expected_mdf_volume_for_price(side, price, mdf, micro=micro, cache=cache)
        actual = max(0.0, volumes.get(price, 0.0))
        absolute_excess = max(0.0, actual - expected) / max(actual + expected, 1e-12)
        return self._clamp(
            0.42 * excess_book + 0.25 * unsupported + 0.33 * absolute_excess,
            0.0,
            1.0,
        )

    def _mdf_side_depth_capacity(
        self,
        side: str,
        micro: _MicrostructureState,
        cache: _StepComputationCache | None = None,
    ) -> float:
        base_depth = self._expected_mdf_side_depth(side, micro=micro, cache=cache)
        radius_multiplier = self._clamp(4.8 + 0.22 * self.grid_radius, 5.8, 8.8)
        flow_reserve = 1.0 + 0.12 * self._clamp(micro.flow_persistence, 0.0, 1.45) / 1.45
        volatility_drag = 1.0 - 0.10 * self._clamp(micro.volatility_cluster, 0.0, 2.0) / 2.0
        return max(base_depth, base_depth * radius_multiplier * flow_reserve * volatility_drag)

    def _side_depth_equilibrium(
        self,
        side: str,
        micro: _MicrostructureState,
        *,
        cache: _StepComputationCache | None = None,
        side_volume: float | None = None,
    ) -> tuple[float, float, float]:
        if side_volume is None:
            side_volume = sum(
                max(0.0, volume)
                for volume in self._orderbook.volumes_for_side(side).values()
            )
        capacity = self._mdf_side_depth_capacity(side, micro, cache)
        load = (side_volume - capacity) / max(capacity, 1e-12)
        shortage = self._clamp(-load, 0.0, 1.0)
        overfill = self._clamp(load, 0.0, 1.5)
        return shortage, overfill, capacity

    def _expected_mdf_side_depth(
        self,
        side: str,
        *,
        micro: _MicrostructureState | None = None,
        cache: _StepComputationCache | None = None,
    ) -> float:
        if cache is not None and cache.micro is micro and side in cache.expected_depth_by_side:
            return cache.expected_depth_by_side[side]
        regime = self._active_settings
        micro = micro or self._microstructure
        side_stress = self._stress_for_book_side(side, micro)
        stress_drag = self._clamp(
            1.0 - 0.28 * self._clamp(micro.liquidity_stress * side_stress, 0.0, 2.0) / 2.0,
            0.45,
            1.15,
        )
        resiliency = self._clamp(micro.resiliency, 0.20, 1.55)
        depth = max(
            0.20,
            self.popularity
            * regime["liquidity"]
            * regime["near_touch_liquidity"]
            * resiliency
            * stress_drag
            * (7.5 + 0.35 * self.grid_radius),
        )
        if cache is not None and cache.micro is micro:
            cache.expected_depth_by_side[side] = depth
        return depth

    def _expected_mdf_volume_for_price(
        self,
        side: str,
        price: float,
        mdf: MDFState,
        *,
        micro: _MicrostructureState | None = None,
        cache: _StepComputationCache | None = None,
    ) -> float:
        price = self._snap_price(price)
        cache_key = (side, price)
        if (
            cache is not None
            and cache.mdf is mdf
            and cache_key in cache.expected_volume_by_side_price
        ):
            return cache.expected_volume_by_side_price[cache_key]
        probabilities = self._entry_probabilities_for_book_side(side, mdf, cache)
        expected = self._expected_mdf_side_depth(
            side,
            micro=micro,
            cache=cache,
        ) * probabilities.get(price, 0.0)
        if cache is not None and cache.mdf is mdf:
            cache.expected_volume_by_side_price[cache_key] = expected
        return expected

    def _cancel_price_volume(
        self,
        cancelled: PriceMap,
        side: str,
        price: float,
        requested: float,
    ) -> float:
        volume_by_price = self._orderbook.volumes_for_side(side)
        removed = min(max(0.0, requested), max(0.0, volume_by_price.get(price, 0.0)))
        if removed <= 1e-12:
            return 0.0
        cancelled[price] = cancelled.get(price, 0.0) + removed
        self._orderbook.adjust_volume(side, price, -removed)
        return removed

    def _cancel_or_requote_volume(
        self,
        cancelled: PriceMap,
        side: str,
        price: float,
        requested: float,
        current_price: float,
        micro: _MicrostructureState,
        mdf: MDFState,
    ) -> float:
        should_requote = self._unit_random() < self._cancel_requote_probability(
            side,
            price,
            current_price,
            micro,
            mdf,
        )
        destination_weights = (
            self._cancel_requote_sampling_weights(
                side,
                current_price,
                price,
                mdf,
                micro,
            )
            if should_requote
            else {}
        )
        removed = self._cancel_price_volume(cancelled, side, price, requested)
        if removed <= 1e-12:
            return 0.0
        if not destination_weights:
            return removed
        destination = self._sample_price(destination_weights)
        self._orderbook.add_lot(destination, removed, side, "requote")
        return removed

    def _cancel_requote_probability(
        self,
        side: str,
        price: float,
        current_price: float,
        micro: _MicrostructureState,
        mdf: MDFState,
    ) -> float:
        probabilities = self._normalized_price_probabilities(
            mdf.buy_entry_mdf_by_price if side == "bid" else mdf.sell_entry_mdf_by_price
        )
        if not probabilities:
            return 0.0
        price = self._snap_price(price)
        max_support = max(probabilities.values())
        support = probabilities.get(price, 0.0) / max(max_support, 1e-12)
        mismatch = self._mdf_cancel_mismatch(side, price, mdf, micro=micro)
        distance = abs(price - current_price) / self.gap
        distance_drag = 1.0 - exp(-max(0.0, distance - 2.0) / 2.5)
        spread_ticks = self._current_spread_ticks()
        spread_opportunity = self._clamp((spread_ticks - 1.0) / 4.0, 0.0, 1.0)
        side_stress = self._stress_for_book_side(side, micro)
        probability = (
            0.07
            + 0.22 * support
            + 0.08 * self._clamp(micro.resiliency, 0.0, 1.4) / 1.4
            + 0.09 * spread_opportunity
            - 0.16 * mismatch
            - 0.10 * self._clamp(micro.cancel_pressure, 0.0, 2.0) / 2.0
            - 0.10 * self._clamp(micro.liquidity_stress * side_stress, 0.0, 2.0) / 2.0
            - 0.07 * distance_drag
        )
        return self._clamp(probability, 0.0, 0.30)

    def _cancel_requote_sampling_weights(
        self,
        side: str,
        current_price: float,
        old_price: float,
        mdf: MDFState,
        micro: _MicrostructureState,
    ) -> PriceMap:
        del micro
        probabilities = self._normalized_price_probabilities(
            mdf.buy_entry_mdf_by_price if side == "bid" else mdf.sell_entry_mdf_by_price
        )
        weights: PriceMap = {}
        old_price = self._snap_price(old_price)
        for price, probability in probabilities.items():
            if price == old_price:
                continue
            if side == "bid":
                if price >= current_price or price < self._min_price:
                    continue
            elif price <= current_price:
                continue
            weights[price] = probability
        return self._normalize_price_map(weights) if weights else {}

    def _resting_level_survival_weakness(
        self,
        side: str,
        price: float,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        cache: _StepComputationCache | None = None,
        current_tick: int | None = None,
    ) -> float:
        distance = abs(price - current_price) / self.gap
        if current_tick is None:
            current_tick = self.price_to_tick(current_price)
        tick = self.price_to_tick(price) - current_tick
        signals = self._cached_mdf_signals(current_price, cache)
        if side == "bid":
            shortage = signals.bid_shortage_by_tick.get(tick, 0.0)
            adverse = max(0.0, -latent.trend)
        else:
            shortage = signals.ask_shortage_by_tick.get(tick, 0.0)
            adverse = max(0.0, latent.trend)
        distance_weakness = self._clamp(
            (distance - 1.0) / max(float(self.grid_radius), 1.0),
            0.0,
            1.0,
        )
        side_stress = self._stress_for_book_side(side, micro)
        return self._clamp(
            0.35 * distance_weakness
            + 0.22 * shortage
            + 0.24 * adverse
            + 0.19 * micro.liquidity_stress * side_stress,
            0.0,
            1.0,
        )

    def _stress_for_book_side(self, side: str, micro: _MicrostructureState) -> float:
        if side == "ask":
            return self._clamp(max(0.0, micro.stress_side), 0.0, 1.0)
        return self._clamp(max(0.0, -micro.stress_side), 0.0, 1.0)

    def _cancel_burst_multiplier(
        self,
        micro: _MicrostructureState,
        distance: float,
        adverse: float,
    ) -> float:
        vulnerability = 0.65 + 0.10 * min(distance, 6.0) + 0.45 * adverse
        pressure = 1.0 - exp(-micro.cancel_pressure)
        return 0.026 * pressure * vulnerability

    def _add_post_event_quote_arrivals(
        self,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        stats: _TradeStats,
        flow_imbalance: float,
        mdf: MDFState | None = None,
        cache: _StepComputationCache | None = None,
    ) -> _EntryFlow:
        empty = _EntryFlow(orders=[], buy_intent_by_price={}, sell_intent_by_price={})
        if self.popularity <= 0:
            return empty
        mdf = mdf or self.state.mdf
        regime = self._active_settings
        stress_drag = 1.0 - 0.18 * self._clamp(micro.liquidity_stress, 0.0, 1.0)
        idle_maker_flow = (
            self.popularity
            * (0.24 + 0.15 / (1.0 + latent.volatility))
            * regime["liquidity"]
            * micro.resiliency
            * stress_drag
        )
        execution_refill = (
            stats.total_volume
            * regime["liquidity"]
            * micro.resiliency
            * (0.18 + 0.09 / (1.0 + latent.volatility))
        )
        repair_drag = self._clamp(
            1.0
            - 0.22 * self._clamp(micro.liquidity_stress, 0.0, 2.0) / 2.0
            - 0.14 * self._clamp(micro.cancel_pressure, 0.0, 2.0) / 2.0,
            0.62,
            1.0,
        )
        spread_repair = (
            self.popularity
            * regime["liquidity"]
            * micro.resiliency
            * 0.14
            * repair_drag
            * self._clamp((self._current_spread_ticks(cache) - 1.0) / 4.0, 0.0, 1.0)
        )
        base = idle_maker_flow + execution_refill + spread_repair
        if base <= 1e-12:
            return empty
        bid_tilt = self._replenishment_side_tilt("bid", flow_imbalance, latent, micro)
        ask_tilt = self._replenishment_side_tilt("ask", flow_imbalance, latent, micro)
        bid_target = base * bid_tilt
        ask_target = base * ask_tilt
        return self._merge_entry_flows(
            self._add_sampled_replenishment(
                "bid",
                "buy_entry",
                current_price,
                bid_target,
                mdf,
                latent,
                micro,
                cache,
            ),
            self._add_sampled_replenishment(
                "ask",
                "sell_entry",
                current_price,
                ask_target,
                mdf,
                latent,
                micro,
                cache,
            ),
        )

    def _add_liquidity_replenishment(
        self,
        current_price: float,
        latent: LatentState,
        imbalance: float,
        micro: _MicrostructureState,
        mdf: MDFState | None = None,
    ) -> _EntryFlow:
        return self._add_post_event_quote_arrivals(
            current_price,
            latent,
            micro,
            _TradeStats(executed_by_price={}),
            imbalance,
            mdf,
            None,
        )

    def _add_post_trade_replenishment(
        self,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        stats: _TradeStats,
        flow_imbalance: float,
        mdf: MDFState | None = None,
    ) -> _EntryFlow:
        return self._add_post_event_quote_arrivals(
            current_price,
            latent,
            micro,
            stats,
            flow_imbalance,
            mdf,
            None,
        )

    def _replenishment_side_tilt(
        self,
        side: str,
        imbalance: float,
        latent: LatentState,
        micro: _MicrostructureState,
    ) -> float:
        side_tilt = imbalance if side == "bid" else -imbalance
        trend_tilt = latent.trend if side == "bid" else -latent.trend
        pressure_drag = self._clamp(1.0 - 0.16 * micro.cancel_pressure, 0.35, 1.0)
        stress_drag = self._stress_replenishment_multiplier(side, 1, micro)
        spread_drag = self._clamp(1.0 - 0.06 * micro.spread_pressure, 0.78, 1.0)
        return pressure_drag * stress_drag * self._clamp(
            0.58 + 0.18 * side_tilt + 0.09 * trend_tilt,
            0.08,
            1.05,
        ) * spread_drag

    def _add_sampled_replenishment(
        self,
        side: str,
        kind: str,
        current_price: float,
        target_volume: float,
        mdf: MDFState,
        latent: LatentState,
        micro: _MicrostructureState,
        cache: _StepComputationCache | None = None,
    ) -> _EntryFlow:
        empty = _EntryFlow(orders=[], buy_intent_by_price={}, sell_intent_by_price={})
        if target_volume <= 1e-12:
            return empty
        weights = self._replenishment_sampling_weights(
            side,
            current_price,
            mdf,
            latent,
            micro,
            cache,
        )
        if not weights:
            return empty
        event_probability = self._replenishment_event_probability(target_volume, micro)
        if self._unit_random() >= event_probability:
            return empty
        effective_target = target_volume / max(event_probability, 1e-12)
        event_count = self._sample_order_count(
            effective_target,
            max_count=max(1, int(4 + 5 * effective_target + 2 * self.popularity)),
        )
        if event_count <= 0:
            return empty
        order_side = "buy" if side == "bid" else "sell"
        volume_by_price: PriceMap = {}
        orders: list[_IncomingOrder] = []
        for _ in range(event_count):
            price = self._sample_price(weights)
            volume = self._sample_order_size()
            if volume <= 1e-12:
                continue
            orders.append(_IncomingOrder(side=order_side, kind=kind, price=price, volume=volume))
            volume_by_price[price] = volume_by_price.get(price, 0.0) + volume
        volume_by_price = self._drop_zeroes(volume_by_price)
        if not volume_by_price:
            return empty
        self._add_lots(volume_by_price, side, kind)
        if side == "bid":
            return _EntryFlow(
                orders=orders,
                buy_intent_by_price=volume_by_price,
                sell_intent_by_price={},
            )
        return _EntryFlow(
            orders=orders,
            buy_intent_by_price={},
            sell_intent_by_price=volume_by_price,
        )

    def _merge_entry_flows(self, *flows: _EntryFlow) -> _EntryFlow:
        orders: list[_IncomingOrder] = []
        buy_intent: PriceMap = {}
        sell_intent: PriceMap = {}
        for flow in flows:
            if flow is None:
                continue
            orders.extend(flow.orders)
            buy_intent = self._merge_maps(buy_intent, flow.buy_intent_by_price)
            sell_intent = self._merge_maps(sell_intent, flow.sell_intent_by_price)
        return _EntryFlow(
            orders=orders,
            buy_intent_by_price=self._drop_zeroes(buy_intent),
            sell_intent_by_price=self._drop_zeroes(sell_intent),
        )

    def _replenishment_event_probability(
        self,
        target_volume: float,
        micro: _MicrostructureState,
    ) -> float:
        if target_volume <= 1e-12:
            return 0.0
        if self.popularity >= 10.0 and target_volume >= 0.25:
            return 1.0
        if target_volume >= 1.0:
            return 1.0
        pressure_drag = 1.0 - 0.16 * self._clamp(micro.cancel_pressure, 0.0, 2.0) / 2.0
        stress_drag = 1.0 - 0.10 * self._clamp(micro.liquidity_stress, 0.0, 2.0) / 2.0
        spread_drag = 1.0 - 0.03 * self._clamp(micro.spread_pressure, 0.0, 1.8) / 1.8
        probability = (
            0.10
            + 0.54 * (1.0 - exp(-target_volume / 0.42))
            + 0.08 * self.popularity / (1.0 + self.popularity)
        ) * pressure_drag * stress_drag * spread_drag
        probability *= 0.88 + 0.26 * self._event_cluster_signal(micro)
        return self._clamp(probability, 0.07, 0.92)

    def _replenishment_sampling_weights(
        self,
        side: str,
        current_price: float,
        mdf: MDFState,
        latent: LatentState,
        micro: _MicrostructureState,
        cache: _StepComputationCache | None = None,
    ) -> PriceMap:
        del latent, micro
        probabilities = self._entry_probabilities_for_book_side(side, mdf, cache)
        weights: PriceMap = {}
        for price, probability in probabilities.items():
            if side == "bid":
                if price >= current_price or price < self._min_price:
                    continue
            elif price <= current_price:
                continue
            weights[price] = probability
        if weights:
            return self._normalize_price_map(weights)
        return {}

    def _replenishment_volume_for_level(
        self,
        level: int,
        side: str,
        base: float,
        imbalance: float,
        latent: LatentState,
        micro: _MicrostructureState,
        mdf: MDFState | None = None,
        current_price: float | None = None,
    ) -> float:
        del imbalance, latent, micro
        current_price = self.state.price if current_price is None else current_price
        return base * self._mdf_replenishment_shape(side, level, current_price, mdf)

    def _mdf_replenishment_shape(
        self,
        side: str,
        level: int,
        current_price: float,
        mdf: MDFState | None,
    ) -> float:
        mdf = mdf or self.state.mdf
        price = (
            current_price - level * self.gap
            if side == "bid"
            else current_price + level * self.gap
        )
        probabilities = self._entry_probabilities_for_book_side(side, mdf)
        side_level_count = max(
            1,
            sum(
                1
                for tick in self.relative_tick_grid()
                if (tick < 0 if side == "bid" else tick > 0)
            ),
        )
        return probabilities.get(self._snap_price(price), 0.0) * side_level_count

    def _stress_replenishment_multiplier(
        self,
        side: str,
        level: int,
        micro: _MicrostructureState,
    ) -> float:
        side_stress = self._stress_for_book_side(side, micro)
        if side_stress <= 1e-12 or micro.liquidity_stress <= 1e-12:
            return 1.0
        near_touch = exp(-max(0, level - 1) / 2.0)
        drag = 0.62 * micro.liquidity_stress * side_stress * near_touch
        return self._clamp(1.0 - drag, 0.18, 1.0)

    def _add_lots(self, volume_by_price: PriceMap, side: str, kind: str) -> None:
        self._orderbook.add_lots(volume_by_price, side, kind)

    def _execute_market_flows(
        self,
        *,
        entry_orders: list[_IncomingOrder] | None = None,
        events: list[_MarketEvent] | None = None,
        stats: _TradeStats,
    ) -> _ExecutionResult:
        event_stream = (
            list(events)
            if events is not None
            else [
                _MarketEvent(
                    event_type="buy_limit_add" if order.side == "buy" else "sell_limit_add",
                    side=order.side,
                    price=order.price,
                    volume=order.volume,
                )
                for order in (entry_orders or [])
            ]
        )
        market_buy_volume = 0.0
        market_sell_volume = 0.0
        residual_buy_by_price: PriceMap = {}
        residual_sell_by_price: PriceMap = {}
        cancelled: PriceMap = {}
        for event in event_stream:
            if event.event_type in {"bid_cancel", "ask_cancel"}:
                book_side = "bid" if event.event_type == "bid_cancel" else "ask"
                self._cancel_price_volume(cancelled, book_side, event.price, event.volume)
                continue
            if event.side not in {"buy", "sell"}:
                continue
            order = _IncomingOrder(
                side=event.side,
                kind="buy_entry" if event.side == "buy" else "sell_entry",
                price=event.price,
                volume=event.volume,
            )
            executed_before = dict(stats.executed_by_price)
            result = self._process_incoming_order(
                order,
                stats=stats,
                rest_residual=True,
            )
            executed_delta = {
                price: stats.executed_by_price.get(price, 0.0)
                - executed_before.get(price, 0.0)
                for price in stats.executed_by_price
            }
            if order.side == "buy":
                market_buy_volume += result.executed
                for price, volume in executed_delta.items():
                    if volume <= 1e-12:
                        continue
                    residual_sell_by_price[price] = max(
                        0.0,
                        residual_sell_by_price.get(price, 0.0) - volume,
                    )
                if result.rested > 1e-12:
                    price = self._snap_price(order.price)
                    residual_buy_by_price[price] = (
                        residual_buy_by_price.get(price, 0.0) + result.rested
                    )
            else:
                market_sell_volume += result.executed
                for price, volume in executed_delta.items():
                    if volume <= 1e-12:
                        continue
                    residual_buy_by_price[price] = max(
                        0.0,
                        residual_buy_by_price.get(price, 0.0) - volume,
                    )
                if result.rested > 1e-12:
                    price = self._snap_price(order.price)
                    residual_sell_by_price[price] = (
                        residual_sell_by_price.get(price, 0.0) + result.rested
                    )
        return _ExecutionResult(
            residual_market_buy=sum(self._drop_zeroes(residual_buy_by_price).values()),
            residual_market_sell=sum(self._drop_zeroes(residual_sell_by_price).values()),
            crossed_market_volume=0.0,
            market_buy_volume=market_buy_volume,
            market_sell_volume=market_sell_volume,
            cancelled_volume_by_price=self._drop_zeroes(cancelled),
        )

    def _process_incoming_order(
        self,
        order: _IncomingOrder,
        stats: _TradeStats,
        *,
        rest_residual: bool = True,
    ) -> _ProcessedOrder:
        if order.volume <= 0:
            return _ProcessedOrder(executed=0.0, rested=0.0)
        remaining = order.volume
        executed = 0.0
        price_limit = self._snap_price(order.price)
        while remaining > 1e-12:
            price = self._best_ask() if order.side == "buy" else self._best_bid()
            if price is None:
                break
            if order.side == "buy" and price > price_limit + 1e-12:
                break
            if order.side == "sell" and price < price_limit - 1e-12:
                break

            volume_by_price = self._orderbook.volumes_for_taker_side(order.side)
            actual = min(remaining, max(0.0, volume_by_price.get(price, 0.0)))
            if actual <= 1e-12:
                self._discard_empty_head(price, "ask" if order.side == "buy" else "bid")
                continue
            resting_side = "ask" if order.side == "buy" else "bid"
            self._orderbook.adjust_volume(resting_side, price, -actual)
            remaining -= actual
            executed += actual
            stats.record(price, actual)
            self._discard_empty_head(price, "ask" if order.side == "buy" else "bid")

        restable = remaining
        if rest_residual and restable > 1e-12:
            passive_side = "bid" if order.side == "buy" else "ask"
            self._orderbook.add_lot(
                price_limit,
                restable,
                passive_side,
                order.kind,
            )
        return _ProcessedOrder(executed=executed, rested=restable)

    def _trim_orderbook_through_last_price(self, last_price: float) -> None:
        last_price = self._snap_price(last_price)
        while True:
            best_bid = self._best_bid()
            best_ask = self._best_ask()
            if best_bid is None or best_ask is None or best_bid < best_ask:
                break
            removed = False
            if best_bid >= last_price:
                self._orderbook.bid_volume_by_price.pop(best_bid, None)
                self._orderbook.invalidate("bid")
                removed = True
            if best_ask <= last_price:
                self._orderbook.ask_volume_by_price.pop(best_ask, None)
                self._orderbook.invalidate("ask")
                removed = True
            if not removed:
                if abs(best_bid - last_price) <= abs(best_ask - last_price):
                    self._orderbook.bid_volume_by_price.pop(best_bid, None)
                    self._orderbook.invalidate("bid")
                else:
                    self._orderbook.ask_volume_by_price.pop(best_ask, None)
                    self._orderbook.invalidate("ask")
        self._orderbook.clean()

    def _prune_orderbook_window(self, current_price: float) -> None:
        min_price = max(
            self._min_price,
            self._snap_price(current_price - self.grid_radius * self.gap),
        )
        max_price = self._snap_price(current_price + self.grid_radius * self.gap)
        for volume_by_price in (
            self._orderbook.bid_volume_by_price,
            self._orderbook.ask_volume_by_price,
        ):
            for price in list(volume_by_price):
                if price < min_price or price > max_price:
                    del volume_by_price[price]
                    self._orderbook.invalidate()
        self._orderbook.clean()

    def _best_bid(self) -> float | None:
        return self._orderbook.best_bid()

    def _best_ask(self) -> float | None:
        return self._orderbook.best_ask()

    def _spread(self, bid: float | None, ask: float | None) -> float | None:
        if bid is None or ask is None:
            return None
        return ask - bid

    def _current_spread_ticks(self, cache: _StepComputationCache | None = None) -> float:
        if cache is not None and cache.spread_ticks is not None:
            return cache.spread_ticks
        spread = self._spread(self._best_bid(), self._best_ask())
        if spread is None:
            spread_ticks = 2.0 + self._clamp(self._microstructure.liquidity_stress, 0.0, 2.0)
        else:
            spread_ticks = max(1.0, spread / self.gap)
        if cache is not None:
            cache.spread_ticks = spread_ticks
        return spread_ticks

    def _near_touch_imbalance(self, current_price: float) -> float:
        return self._orderbook.near_touch_imbalance(current_price, self.gap)

    def _nearby_book_depth(self, side: str, current_price: float) -> float:
        volumes = (
            self._orderbook.bid_volume_by_price
            if side == "bid"
            else self._orderbook.ask_volume_by_price
        )
        depth = 0.0
        for price, volume in volumes.items():
            distance = max(1.0, abs(current_price - price) / self.gap)
            if distance <= 5:
                depth += volume / distance
        return depth

    def _discard_empty_head(self, price: float, side: str) -> None:
        self._orderbook.discard_empty_head(price, side)

    def _clean_orderbook(self) -> None:
        self._orderbook.clean()

    def _snapshot_orderbook(self) -> OrderBookState:
        return self._orderbook.snapshot()

    def _merge_maps(self, *maps: PriceMap) -> PriceMap:
        merged: PriceMap = {}
        for values in maps:
            for price, volume in values.items():
                merged[price] = merged.get(price, 0.0) + volume
        return self._drop_zeroes(merged)

    def _price_map_to_relative_ticks(self, basis_price: float, values: PriceMap) -> TickMap:
        basis_tick = self.price_to_tick(basis_price)
        by_tick: TickMap = {}
        min_tick = -self.grid_radius
        max_tick = self.grid_radius
        for price, value in values.items():
            relative_tick = self.price_to_tick(price) - basis_tick
            if min_tick <= relative_tick <= max_tick:
                by_tick[relative_tick] = by_tick.get(relative_tick, 0.0) + value
        return by_tick

    def _single_price_volume(self, price: float, volume: float) -> PriceMap:
        if volume <= 0:
            return {}
        return {self._snap_price(price): volume}

    def _drop_zeroes(self, values: PriceMap) -> PriceMap:
        return {price: value for price, value in sorted(values.items()) if value > 1e-12}

    def _dedupe_prices(self, prices) -> list[float]:
        return sorted(set(prices))

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
