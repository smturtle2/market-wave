from __future__ import annotations

from math import exp, isfinite, log1p

from ._types import (
    _EntryFlow,
    _IncomingOrder,
    _MarketEvent,
    _MicrostructureState,
    _StepComputationCache,
)
from .distribution import MDFSignals
from .state import IntensityState, LatentState, MDFState, PriceMap, TickMap


class _MarketMDFMixin:
    def _next_mdf(
        self,
        price: float,
        price_grid: list[float],
        latent: LatentState,
        *,
        step_index: int,
        update_memory: bool,
        cache: _StepComputationCache | None = None,
    ) -> MDFState:
        del price_grid, step_index
        relative_ticks = self.relative_tick_grid()
        current_tick = self.price_to_tick(price)
        signals = self._cached_mdf_signals(price, cache)

        buy = self._entry_mdf("buy", relative_ticks, latent, signals)
        sell = self._entry_mdf("sell", relative_ticks, latent, signals)
        if update_memory:
            buy = self._mix_with_memory("buy_entry", current_tick, buy)
            sell = self._mix_with_memory("sell_entry", current_tick, sell)

        buy = self._constrain_tick_mdf(current_tick, buy)
        sell = self._constrain_tick_mdf(current_tick, sell)
        return MDFState(
            relative_ticks=relative_ticks,
            buy_entry_mdf=buy,
            sell_entry_mdf=sell,
            buy_entry_mdf_by_price=self._project_tick_mdf(current_tick, buy),
            sell_entry_mdf_by_price=self._project_tick_mdf(current_tick, sell),
        )

    def _reproject_mdf(self, price: float, mdf: MDFState) -> MDFState:
        current_tick = self.price_to_tick(price)
        buy = self._constrain_tick_mdf(current_tick, mdf.buy_entry_mdf)
        sell = self._constrain_tick_mdf(current_tick, mdf.sell_entry_mdf)
        return MDFState(
            relative_ticks=list(mdf.relative_ticks),
            buy_entry_mdf=buy,
            sell_entry_mdf=sell,
            buy_entry_mdf_by_price=self._project_tick_mdf(current_tick, buy),
            sell_entry_mdf_by_price=self._project_tick_mdf(current_tick, sell),
        )

    def _entry_mdf(
        self,
        side: str,
        relative_ticks: list[int],
        latent: LatentState,
        signals: MDFSignals,
    ) -> TickMap:
        flow = self._clamp(signals.meta_order_side, -1.0, 1.0)
        persistence = self._clamp(signals.flow_persistence, 0.0, 1.45) / 1.45
        volatility = self._clamp(latent.volatility + 0.35 * signals.volatility_cluster, 0.0, 2.5)
        stress = self._clamp(signals.liquidity_stress, 0.0, 2.0) / 2.0
        spread = self._clamp((signals.spread_ticks - 1.0) / 6.0, 0.0, 1.0)
        activity = self._clamp(signals.activity + signals.activity_event, 0.0, 3.8) / 3.8
        trend = self._clamp(0.65 * latent.trend + 0.35 * latent.mood, -1.0, 1.0)
        signed_pressure = trend + flow * (0.35 + 0.65 * persistence)
        if side == "sell":
            signed_pressure = -signed_pressure

        passive_shortage = self._side_shortage("bid" if side == "buy" else "ask", signals)
        own_load = signals.bid_depth_pressure if side == "buy" else signals.ask_depth_pressure
        passive_pull = self._clamp(passive_shortage + max(0.0, -own_load), 0.0, 2.0) / 2.0
        urgency = self._clamp(
            0.45 * activity + 0.35 * stress + 0.30 * abs(signed_pressure),
            0.0,
            1.5,
        )

        passive_center = 1.0 + 0.8 * passive_pull + 0.6 * spread - 0.5 * urgency
        marketable_center = 0.6 + 0.8 * urgency + 0.6 * max(0.0, signed_pressure)
        width = 1.0 + 1.15 * volatility + 0.9 * stress + 0.45 * spread

        raw: TickMap = {}
        for tick in relative_ticks:
            if self.price_to_tick(self.state.price) + tick < 1:
                raw[tick] = 0.0
                continue
            is_passive = tick <= 0 if side == "buy" else tick >= 0
            level = abs(float(tick))
            if is_passive:
                distance = abs(level - passive_center)
                mass = (0.80 + 0.85 * passive_pull + 0.35 * spread) * exp(-distance / width)
                mass += 0.06 / (1.0 + level)
            else:
                distance = abs(level - marketable_center)
                mass = (0.28 + 0.95 * urgency + 0.45 * max(0.0, signed_pressure)) * exp(
                    -distance / max(0.75, width * 0.8)
                )
                mass += 0.018 * activity / (1.0 + level)
            raw[tick] = max(0.0, mass)
        return self._normalize_tick_map(raw)

    def _side_shortage(self, side: str, signals: MDFSignals) -> float:
        shortages = signals.bid_shortage_by_tick if side == "bid" else signals.ask_shortage_by_tick
        gaps = signals.bid_gap_by_tick if side == "bid" else signals.ask_gap_by_tick
        levels = range(-3, 1) if side == "bid" else range(0, 4)
        total = 0.0
        weight_sum = 0.0
        for tick in levels:
            weight = 1.0 / max(1.0, abs(float(tick)))
            total += weight * (0.70 * shortages.get(tick, 0.0) + 0.30 * gaps.get(tick, 0.0))
            weight_sum += weight
        return 0.0 if weight_sum <= 1e-12 else self._clamp(total / weight_sum, 0.0, 1.5)

    def _mix_with_memory(self, key: str, current_tick: int, proposal: TickMap) -> TickMap:
        previous = self._mdf_memory.get(key)
        memory = self._clamp(self._entry_mdf_memory, 0.0, 0.35)
        if previous:
            shifted = {
                tick: previous.get(current_tick + tick, 0.0)
                for tick in proposal
                if current_tick + tick >= 1
            }
            mixed = {
                tick: (1.0 - memory) * proposal.get(tick, 0.0) + memory * shifted.get(tick, 0.0)
                for tick in proposal
            }
            proposal = self._normalize_tick_map(mixed)
        self._mdf_memory[key] = {
            current_tick + tick: value for tick, value in proposal.items() if value > 1e-12
        }
        return proposal

    def _entry_flow(self, intensity: IntensityState, mdf: MDFState) -> _EntryFlow:
        buy_orders, buy_intent = self._sample_entry_side(
            "buy", "buy_entry", intensity.buy, mdf.buy_entry_mdf_by_price
        )
        sell_orders, sell_intent = self._sample_entry_side(
            "sell", "sell_entry", intensity.sell, mdf.sell_entry_mdf_by_price
        )
        orders = buy_orders + sell_orders
        events = [
            _MarketEvent(
                event_type=self._event_type_for_incoming_order(order),
                side=order.side,
                price=order.price,
                volume=order.volume,
            )
            for order in orders
        ]
        cancel_events, bid_cancel, ask_cancel = self._sample_cancel_events()
        events.extend(cancel_events)
        return _EntryFlow(
            orders=orders,
            buy_intent_by_price=self._drop_zeroes(buy_intent),
            sell_intent_by_price=self._drop_zeroes(sell_intent),
            events=events,
            bid_cancel_intent_by_price=bid_cancel,
            ask_cancel_intent_by_price=ask_cancel,
        )

    def _sample_entry_side(
        self,
        side: str,
        kind: str,
        target_volume: float,
        mdf_by_price: PriceMap,
    ) -> tuple[list[_IncomingOrder], PriceMap]:
        probabilities = self._normalized_price_probabilities(mdf_by_price)
        if target_volume <= 1e-12 or not probabilities:
            return [], {}
        count = self._sample_market_event_count(
            target_volume,
            max_count=max(1, int(8 + 5 * target_volume + 3 * self.popularity)),
            micro=self._microstructure,
        )
        orders: list[_IncomingOrder] = []
        intent: PriceMap = {}
        for _ in range(count):
            price = self._sample_price(probabilities)
            volume = self._sample_order_size()
            if volume <= 1e-12:
                continue
            orders.append(_IncomingOrder(side=side, kind=kind, price=price, volume=volume))
            intent[price] = intent.get(price, 0.0) + volume
        return orders, intent

    def _sample_cancel_events(self) -> tuple[list[_MarketEvent], PriceMap, PriceMap]:
        micro = self._microstructure
        pressure = self._clamp(
            0.45 * micro.cancel_pressure
            + 0.30 * micro.liquidity_stress
            + 0.20 * micro.cancel_burst,
            0.0,
            2.0,
        ) / 2.0
        if pressure <= 1e-12:
            return [], {}, {}
        events: list[_MarketEvent] = []
        bid_intent: PriceMap = {}
        ask_intent: PriceMap = {}
        for book_side, volumes, intent in (
            ("bid", self._orderbook.bid_volume_by_price, bid_intent),
            ("ask", self._orderbook.ask_volume_by_price, ask_intent),
        ):
            weights = self._cancel_price_weights(book_side, volumes)
            count = self._sample_market_event_count(
                sum(volumes.values()) * (0.006 + 0.030 * pressure),
                max_count=max(1, int(3 + len(weights))),
                micro=micro,
            )
            for _ in range(count):
                if not weights:
                    continue
                price = self._sample_price(weights)
                available = max(0.0, volumes.get(price, 0.0) - intent.get(price, 0.0))
                if available <= 1e-12:
                    continue
                volume = min(available, self._sample_order_size() * (0.25 + 0.90 * pressure))
                event_type = "bid_cancel" if book_side == "bid" else "ask_cancel"
                events.append(_MarketEvent(event_type, book_side, price, volume))
                intent[price] = intent.get(price, 0.0) + volume
        return events, self._drop_zeroes(bid_intent), self._drop_zeroes(ask_intent)

    def _cancel_price_weights(self, side: str, volumes: PriceMap) -> PriceMap:
        if not volumes:
            return {}
        touch = max(volumes) if side == "bid" else min(volumes)
        weights = {
            price: max(0.0, volume) * (0.35 + 0.65 * abs(price - touch) / max(self.gap, 1e-12))
            for price, volume in volumes.items()
            if volume > 1e-12
        }
        return self._normalize_price_map(weights)

    def _event_type_for_incoming_order(self, order: _IncomingOrder) -> str:
        if order.side == "buy":
            best_ask = self._best_ask()
            return (
                "buy_marketable"
                if best_ask is not None and self._snap_price(order.price) >= best_ask - 1e-12
                else "buy_limit_add"
            )
        best_bid = self._best_bid()
        return (
            "sell_marketable"
            if best_bid is not None and self._snap_price(order.price) <= best_bid + 1e-12
            else "sell_limit_add"
        )

    def _entry_probabilities_for_book_side(
        self,
        side: str,
        mdf: MDFState,
        cache: _StepComputationCache | None = None,
    ) -> PriceMap:
        if cache is not None and cache.mdf is mdf and side in cache.entry_probabilities_by_side:
            return cache.entry_probabilities_by_side[side]
        entry_mdf = mdf.buy_entry_mdf_by_price if side == "bid" else mdf.sell_entry_mdf_by_price
        probabilities = self._normalized_price_probabilities(entry_mdf)
        if cache is not None and cache.mdf is mdf:
            cache.entry_probabilities_by_side[side] = probabilities
        return probabilities

    def _mean_child_order_size(self) -> float:
        return self._clamp(0.18 + 0.08 * self.popularity, 0.08, 0.55)

    def _sample_order_count(self, target_volume: float, *, max_count: int) -> int:
        if target_volume <= 1e-12 or max_count <= 0:
            return 0
        return min(max_count, self._sample_poisson(target_volume / self._mean_child_order_size()))

    def _sample_market_event_count(
        self,
        target_volume: float,
        *,
        max_count: int,
        micro: _MicrostructureState,
    ) -> int:
        if target_volume <= 1e-12 or max_count <= 0:
            return 0
        expected = target_volume / self._mean_child_order_size()
        activity = self._clamp(
            0.45 * micro.activity
            + 0.60 * micro.activity_event
            + 0.35 * micro.participation_burst
            + 0.25 * micro.volatility_cluster,
            0.0,
            2.5,
        ) / 2.5
        center = log1p(expected) * (0.65 + 1.25 * activity) - (0.70 - 0.45 * activity)
        noise = self._rng.gauss(0.0, 0.55 + 0.55 * activity) if hasattr(self._rng, "gauss") else 0
        raw = center + noise
        if raw <= 0.0:
            return 0
        count = int(raw)
        if self._unit_random() < raw - count:
            count += 1
        return min(max_count, count)

    def _sample_order_size(self) -> float:
        mean = self._mean_child_order_size()
        micro = self._microstructure
        activity = self._clamp(micro.activity + micro.activity_event, 0.0, 3.8) / 3.8
        stress = self._clamp(micro.liquidity_stress + micro.participation_burst, 0.0, 4.0) / 4.0
        sigma = 0.45 + 0.35 * activity + 0.25 * stress
        size = mean * self._rng.lognormvariate(-0.5 * sigma * sigma, sigma)
        if self._unit_random() < 0.025 + 0.09 * stress:
            size += mean * self._rng.paretovariate(1.8 + 0.4 * (1.0 - stress))
        return self._clamp(size, mean * 0.015, mean * 30.0)

    def _sample_poisson(self, expected: float) -> int:
        if expected <= 0:
            return 0
        if expected > 32.0 and hasattr(self._rng, "gauss"):
            return max(0, int(round(self._rng.gauss(expected, expected**0.5))))
        threshold = exp(-expected)
        product = 1.0
        count = 0
        while product > threshold:
            count += 1
            product *= self._unit_random()
        return count - 1

    def _sample_price(self, probabilities: PriceMap) -> float:
        if not probabilities:
            raise ValueError("probabilities must not be empty")
        draw = self._unit_random()
        cumulative = 0.0
        fallback = next(reversed(probabilities))
        for price, probability in probabilities.items():
            cumulative += probability
            if draw <= cumulative:
                return price
        return fallback

    def _event_cluster_signal(self, micro: _MicrostructureState) -> float:
        return self._clamp(
            0.48 * micro.activity_event / 1.8
            + 0.24 * micro.activity / 2.0
            + 0.18 * micro.cancel_pressure / 2.0
            + 0.10 * micro.spread_pressure / 1.8
            + 0.10 * micro.volatility_cluster / 2.0,
            0.0,
            1.0,
        )

    def _cached_mdf_signals(
        self,
        price: float,
        cache: _StepComputationCache | None = None,
    ) -> MDFSignals:
        if cache is None or cache.basis_price != price:
            return self._mdf_signals(price)
        if cache.signals is None:
            cache.signals = self._mdf_signals(price)
        return cache.signals

    def _mdf_signals(self, price: float) -> MDFSignals:
        orderbook = self._snapshot_orderbook()
        best_bid = self._best_bid()
        best_ask = self._best_ask()
        spread_ticks = (
            max(1.0, (best_ask - best_bid) / self.gap)
            if best_bid is not None and best_ask is not None
            else 1.0 + self._microstructure.liquidity_stress
        )
        bid_volume = sum(max(0.0, volume) for volume in orderbook.bid_volume_by_price.values())
        ask_volume = sum(max(0.0, volume) for volume in orderbook.ask_volume_by_price.values())
        target = max(1.0, self.popularity * self._active_settings["liquidity"])
        return MDFSignals(
            orderbook_imbalance=self._last_imbalance,
            last_return_ticks=self._last_return_ticks,
            last_execution_volume=self._last_execution_volume,
            executed_volume_by_tick=self._price_map_to_relative_ticks(
                price, self._last_executed_by_price
            ),
            liquidity_by_tick=self._price_map_to_relative_ticks(
                price,
                self._merge_maps(
                    orderbook.bid_volume_by_price,
                    orderbook.ask_volume_by_price,
                ),
            ),
            bid_liquidity_by_tick=self._price_map_to_relative_ticks(
                price, orderbook.bid_volume_by_price
            ),
            ask_liquidity_by_tick=self._price_map_to_relative_ticks(
                price, orderbook.ask_volume_by_price
            ),
            bid_shortage_by_tick=self._book_shortage(price, "bid", orderbook.bid_volume_by_price),
            ask_shortage_by_tick=self._book_shortage(price, "ask", orderbook.ask_volume_by_price),
            bid_gap_by_tick={},
            ask_gap_by_tick={},
            bid_front_by_tick={},
            ask_front_by_tick={},
            bid_occupancy_by_tick={},
            ask_occupancy_by_tick={},
            bid_depth_pressure=(bid_volume - target) / target,
            ask_depth_pressure=(ask_volume - target) / target,
            spread_ticks=spread_ticks,
            spread_pressure=self._microstructure.spread_pressure,
            cancel_pressure=self._microstructure.cancel_pressure,
            liquidity_stress=self._microstructure.liquidity_stress,
            stress_side=self._microstructure.stress_side,
            resiliency=self._microstructure.resiliency,
            activity=self._microstructure.activity,
            activity_event=self._microstructure.activity_event,
            flow_persistence=self._microstructure.flow_persistence,
            meta_order_side=self._microstructure.meta_order_side,
            volatility_cluster=self._microstructure.volatility_cluster,
            participation_burst=self._microstructure.participation_burst,
            liquidity_drought=self._microstructure.liquidity_drought,
            cancel_burst=self._microstructure.cancel_burst,
        )

    def _book_shortage(self, basis_price: float, side: str, volumes: PriceMap) -> TickMap:
        depth = self._price_map_to_relative_ticks(basis_price, volumes)
        ticks = [
            tick
            for tick in self.relative_tick_grid()
            if (tick < 0 if side == "bid" else tick > 0)
        ]
        expected_scale = max(0.08, self.popularity * self._active_settings["liquidity"])
        shortages: TickMap = {}
        for tick in ticks:
            expected = expected_scale / (
                max(1.0, abs(float(tick))) ** self._active_settings["depth_exponent"]
            )
            actual = max(0.0, depth.get(tick, 0.0))
            shortages[tick] = self._clamp((expected - actual) / max(expected, 1e-12), 0.0, 1.0)
        return shortages

    def _normalized_price_probabilities(self, values: PriceMap) -> PriceMap:
        return self._normalize_price_map(
            {
                self._snap_price(price): probability
                for price, probability in values.items()
                if probability > 0 and isfinite(probability) and price >= self._min_price
            }
        )

    def _project_tick_mdf(self, current_tick: int, values: TickMap) -> PriceMap:
        projected: PriceMap = {}
        for relative_tick, probability in values.items():
            absolute_tick = current_tick + relative_tick
            if absolute_tick < 1:
                continue
            projected[self._clean_number(absolute_tick * self.gap)] = probability
        return self._normalize_price_map(projected)

    def _constrain_tick_mdf(self, current_tick: int, values: TickMap) -> TickMap:
        return self._normalize_tick_map(
            {tick: probability for tick, probability in values.items() if current_tick + tick >= 1}
        )

    def _normalize_price_map(self, values: PriceMap) -> PriceMap:
        clean = {price: value for price, value in values.items() if value > 0.0 and isfinite(value)}
        total = sum(clean.values())
        if total <= 1e-12:
            return {}
        return {price: value / total for price, value in sorted(clean.items())}

    def _normalize_tick_map(self, values: TickMap) -> TickMap:
        clean = {tick: value for tick, value in values.items() if value > 0.0 and isfinite(value)}
        total = sum(clean.values())
        if total <= 1e-12:
            return {tick: 1.0 / len(values) for tick in values} if values else {}
        return {tick: clean.get(tick, 0.0) / total for tick in values}

    def _unit_random(self) -> float:
        return self._rng.random()
