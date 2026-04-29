from __future__ import annotations

from math import exp, isfinite, sin

from ._types import (
    _EntryFlow,
    _IncomingOrder,
    _MDFJudgmentSample,
    _MDFSideJudgment,
    _MicrostructureState,
    _StepComputationCache,
)
from .distribution import MDFContext, MDFSignals
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
        del price_grid
        relative_ticks = self.relative_tick_grid()
        context = MDFContext(
            current_price=price,
            current_tick=self.price_to_tick(price),
            tick_size=self.gap,
            mood=latent.mood,
            trend=latent.trend,
            volatility=latent.volatility * self._active_settings["spread"],
            regime=self._active_regime,
            augmentation_strength=self.augmentation_strength,
            step_index=step_index,
            rng=self._rng,
        )
        signals = self._cached_mdf_signals(price, cache)
        mdf_shock = self._clamp(
            0.34 * signals.liquidity_stress
            + 0.22 * signals.activity_event
            + 0.18 * min(abs(signals.last_return_ticks) / 3.0, 1.0)
            + 0.14 * signals.cancel_pressure / 2.0
            + 0.12 * latent.volatility / 1.55,
            0.0,
            1.0,
        )
        entry_memory = self._clamp(self._entry_mdf_memory * (1.0 - 0.35 * mdf_shock), 0.08, 0.20)
        floor_mix = self._clamp(self._mdf_floor_mix + 0.006 * mdf_shock, 0.010, 0.024)
        judgments = self._sample_mdf_judgments(context, signals, update_memory=update_memory)

        buy_entry_ticks = self._evolve_raw_mdf(
            "buy_entry",
            self._buy_entry_mdf_mass(relative_ticks, context, signals, judgments.buy),
            current_tick=context.current_tick,
            memory_mix=entry_memory,
            floor_mix=floor_mix,
            update_memory=update_memory,
        )
        sell_entry_ticks = self._evolve_raw_mdf(
            "sell_entry",
            self._sell_entry_mdf_mass(relative_ticks, context, signals, judgments.sell),
            current_tick=context.current_tick,
            memory_mix=entry_memory,
            floor_mix=floor_mix,
            update_memory=update_memory,
        )
        buy_entry_ticks, sell_entry_ticks = self._resolve_entry_mdf_overlap(
            relative_ticks,
            context.current_tick,
            buy_entry_ticks,
            sell_entry_ticks,
            buy_aggression=self._entry_overlap_aggression(
                "buy",
                context,
                signals,
                judgments.buy,
                mdf_shock,
            ),
            sell_aggression=self._entry_overlap_aggression(
                "sell",
                context,
                signals,
                judgments.sell,
                mdf_shock,
            ),
        )
        entry_noise_ticks = self._centered_entry_noise_mdf(
            relative_ticks,
            context.current_tick,
        )
        buy_entry_ticks = self._mix_entry_noise_mdf(
            buy_entry_ticks,
            entry_noise_ticks,
            mix=self._entry_noise_mix,
        )
        sell_entry_ticks = self._mix_entry_noise_mdf(
            sell_entry_ticks,
            entry_noise_ticks,
            mix=self._entry_noise_mix,
        )
        buy_entry = self._project_tick_mdf(context.current_tick, buy_entry_ticks)
        sell_entry = self._project_tick_mdf(context.current_tick, sell_entry_ticks)
        return MDFState(
            buy_entry_mdf_by_price=buy_entry,
            sell_entry_mdf_by_price=sell_entry,
            relative_ticks=relative_ticks,
            buy_entry_mdf=buy_entry_ticks,
            sell_entry_mdf=sell_entry_ticks,
        )

    def _reproject_mdf(
        self,
        price: float,
        mdf: MDFState,
    ) -> MDFState:
        current_tick = self.price_to_tick(price)
        buy_entry_mdf = self._constrain_tick_mdf(current_tick, mdf.buy_entry_mdf)
        sell_entry_mdf = self._constrain_tick_mdf(current_tick, mdf.sell_entry_mdf)
        return MDFState(
            buy_entry_mdf_by_price=self._project_tick_mdf(current_tick, buy_entry_mdf),
            sell_entry_mdf_by_price=self._project_tick_mdf(current_tick, sell_entry_mdf),
            relative_ticks=list(mdf.relative_ticks),
            buy_entry_mdf=buy_entry_mdf,
            sell_entry_mdf=sell_entry_mdf,
        )

    def _entry_flow(
        self,
        intensity: IntensityState,
        mdf: MDFState,
    ) -> _EntryFlow:
        buy_orders, buy_intent = self._sample_entry_side(
            "buy",
            "buy_entry",
            intensity.buy,
            mdf.buy_entry_mdf_by_price,
        )
        sell_orders, sell_intent = self._sample_entry_side(
            "sell",
            "sell_entry",
            intensity.sell,
            mdf.sell_entry_mdf_by_price,
        )
        orders = buy_orders + sell_orders

        return _EntryFlow(
            orders=orders,
            buy_intent_by_price=self._drop_zeroes(buy_intent),
            sell_intent_by_price=self._drop_zeroes(sell_intent),
        )

    def _sample_entry_side(
        self,
        side: str,
        kind: str,
        target_volume: float,
        mdf_by_price: PriceMap,
    ) -> tuple[list[_IncomingOrder], PriceMap]:
        if target_volume <= 1e-12:
            return [], {}
        probabilities = self._normalized_price_probabilities(mdf_by_price)
        if not probabilities:
            return [], {}

        event_probability = self._entry_event_probability(target_volume)
        if self._unit_random() >= event_probability:
            return [], {}

        effective_target = target_volume / max(event_probability, 1e-12)
        max_count = max(1, int(10 + 6 * effective_target + 4 * max(self.popularity, 0.0)))
        order_count = self._sample_order_count(
            effective_target,
            max_count=max_count,
        )
        if order_count <= 0:
            return [], {}

        orders: list[_IncomingOrder] = []
        intent: PriceMap = {}
        for _ in range(order_count):
            price = self._sample_price(probabilities)
            volume = self._sample_order_size()
            if volume <= 1e-12:
                continue
            orders.append(_IncomingOrder(side=side, kind=kind, price=price, volume=volume))
            intent[price] = intent.get(price, 0.0) + volume
        return orders, intent

    def _entry_event_probability(self, target_volume: float) -> float:
        if target_volume <= 1e-12:
            return 0.0
        popularity_term = self.popularity / (1.0 + self.popularity)
        cluster = self._event_cluster_signal(self._microstructure)
        probability = (
            0.075
            + 0.35 * (1.0 - exp(-target_volume / 1.2))
            + 0.43 * (1.0 - exp(-target_volume / 9.0))
            + 0.08 * popularity_term
        )
        probability *= 0.80 + 0.52 * cluster
        high_volume_floor = 0.96 * (1.0 - exp(-target_volume / 7.0))
        probability = max(probability, high_volume_floor)
        return self._clamp(probability, 0.07, 0.96)

    def _event_cluster_signal(self, micro: _MicrostructureState) -> float:
        signal = (
            0.48 * self._clamp(micro.activity_event, 0.0, 1.8) / 1.8
            + 0.24 * self._clamp(micro.activity, 0.0, 2.0) / 2.0
            + 0.18 * self._clamp(micro.cancel_pressure, 0.0, 2.0) / 2.0
            + 0.10 * self._clamp(micro.spread_pressure, 0.0, 1.8) / 1.8
        )
        return self._clamp(signal, 0.0, 1.0)

    def _normalized_price_probabilities(self, values: PriceMap) -> PriceMap:
        probabilities: PriceMap = {}
        total = 0.0
        for raw_price, raw_probability in values.items():
            if raw_probability <= 0 or not isfinite(raw_probability):
                continue
            tick = max(1, int(round(raw_price / self.gap)))
            if self._gap_is_integer:
                price = float(int(tick * self.gap))
            else:
                price = self._clean_number(tick * self.gap)
            if price < self._min_price:
                continue
            probabilities[price] = probabilities.get(price, 0.0) + raw_probability
            total += raw_probability
        if not probabilities or total <= 1e-12:
            return {}
        ordered = sorted(probabilities.items())
        if abs(total - 1.0) <= 1e-12:
            return dict(ordered)
        return {price: probability / total for price, probability in ordered}

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

    def _mean_child_order_size(self) -> float:
        return self._clamp(0.18 + 0.08 * self.popularity, 0.08, 0.55)

    def _sample_order_count(
        self,
        target_volume: float,
        *,
        max_count: int,
    ) -> int:
        if target_volume <= 1e-12 or max_count <= 0:
            return 0
        expected = target_volume / self._mean_child_order_size()
        count = self._sample_poisson(expected)
        return max(0, min(max_count, count))

    def _sample_poisson(self, expected: float) -> int:
        if expected <= 0:
            return 0
        if expected > 32.0:
            if hasattr(self._rng, "gauss"):
                return max(0, int(round(self._rng.gauss(expected, expected**0.5))))
            return max(0, int(round(expected)))
        draw = self._unit_random()
        probability = exp(-expected)
        cumulative = probability
        count = 0
        while draw > cumulative and count < 256:
            count += 1
            probability *= expected / count
            cumulative += probability
        return count

    def _sample_order_size(self) -> float:
        mean_size = self._mean_child_order_size()
        sigma = 0.62
        if hasattr(self._rng, "lognormvariate"):
            size = mean_size * self._rng.lognormvariate(-0.5 * sigma * sigma, sigma)
        else:
            size = mean_size * (0.55 + 0.90 * self._unit_random())
        return self._clamp(size, mean_size * 0.05, mean_size * 8.0)

    def _unit_random(self) -> float:
        value = self._rng.random()
        if not isfinite(value):
            return 0.0
        return self._clamp(value, 0.0, 1.0 - 1e-12)

    def _sample_mdf_judgments(
        self,
        context: MDFContext,
        signals: MDFSignals,
        *,
        update_memory: bool,
    ) -> _MDFJudgmentSample:
        volatility = self._clamp(context.volatility, 0.0, 2.2)
        stress = self._clamp(signals.liquidity_stress, 0.0, 1.8)
        event = self._clamp(signals.activity_event + 0.45 * signals.activity, 0.0, 1.8)
        spread = self._clamp(
            (signals.spread_ticks - 1.0) / max(float(self.grid_radius), 1.0),
            0.0,
            1.0,
        )
        uncertainty = self._clamp(
            0.10
            + 0.20 * volatility
            + 0.24 * stress
            + 0.18 * event
            + 0.10 * context.augmentation_strength,
            0.08,
            0.95,
        )
        common_value_noise = self._rng.gauss(0.0, 0.16 + 0.24 * uncertainty)
        buy_noise = self._rng.gauss(0.0, 0.14 + 0.26 * uncertainty)
        sell_noise = self._rng.gauss(0.0, 0.14 + 0.26 * uncertainty)

        buy_view = (
            context.trend
            + 0.42 * context.mood
            + 0.18 * signals.last_return_ticks
            + 0.12 * signals.orderbook_imbalance
            + 0.10 * max(0.0, signals.stress_side)
        )
        buy_fair_value_shift = self._clamp(
            1.10 * buy_view + common_value_noise + buy_noise,
            -2.4,
            2.4,
        )
        buy_opportunity = self._clamp(
            0.22 * max(0.0, -signals.last_return_ticks)
            + 0.20 * max(0.0, -context.trend)
            + 0.16 * max(0.0, -context.mood)
            + self._rng.gauss(0.0, 0.10 + 0.16 * uncertainty),
            0.0,
            1.5,
        )
        buy_urgency = self._clamp(
            0.10
            + 0.54 * max(0.0, buy_fair_value_shift)
            + 0.16 * event
            + 0.14 * max(0.0, signals.stress_side)
            + self._rng.gauss(0.0, 0.09 + 0.13 * uncertainty),
            0.0,
            1.8,
        )
        buy_patience = self._clamp(
            0.26
            + 0.44 * max(0.0, -buy_fair_value_shift)
            + 0.18 * volatility
            + 0.16 * buy_opportunity
            + self._rng.gauss(0.0, 0.08 + 0.12 * uncertainty),
            0.0,
            1.8,
        )
        buy_liquidity_aversion = self._clamp(
            0.08
            + 0.30 * stress
            + 0.18 * spread
            + 0.12 * max(0.0, -signals.stress_side)
            + self._rng.gauss(0.0, 0.07 + 0.12 * uncertainty),
            0.0,
            1.0,
        )
        buy_pocket_bias = self._clamp(
            0.18
            + 0.18 * uncertainty
            + 0.18 * buy_opportunity
            + self._rng.gauss(0.0, 0.08 + 0.10 * uncertainty),
            0.0,
            1.0,
        )

        sell_view = (
            0.74 * context.trend
            + 0.24 * context.mood
            + 0.10 * signals.last_return_ticks
            - 0.08 * signals.orderbook_imbalance
            - 0.10 * max(0.0, -signals.stress_side)
        )
        sell_fair_value_shift = self._clamp(
            sell_view + 0.35 * common_value_noise + sell_noise,
            -2.4,
            2.4,
        )
        profit_taking = self._clamp(
            0.26 * max(0.0, signals.last_return_ticks)
            + 0.18 * max(0.0, context.trend)
            + 0.12 * max(0.0, context.mood)
            + self._rng.gauss(0.0, 0.10 + 0.16 * uncertainty),
            0.0,
            1.5,
        )
        liquidation = self._clamp(
            0.30 * max(0.0, -context.trend)
            + 0.22 * max(0.0, -signals.last_return_ticks)
            + 0.18 * stress
            + 0.12 * max(0.0, -signals.stress_side)
            + self._rng.gauss(0.0, 0.10 + 0.16 * uncertainty),
            0.0,
            1.6,
        )
        sell_urgency = self._clamp(
            0.10
            + 0.48 * max(0.0, -sell_fair_value_shift)
            + 0.24 * liquidation
            + 0.12 * profit_taking
            + 0.16 * event
            + self._rng.gauss(0.0, 0.09 + 0.13 * uncertainty),
            0.0,
            1.9,
        )
        sell_patience = self._clamp(
            0.26
            + 0.48 * max(0.0, sell_fair_value_shift)
            + 0.16 * volatility
            + 0.10 * profit_taking
            + self._rng.gauss(0.0, 0.08 + 0.12 * uncertainty),
            0.0,
            1.8,
        )
        sell_liquidity_aversion = self._clamp(
            0.08
            + 0.30 * stress
            + 0.18 * spread
            + 0.12 * max(0.0, signals.stress_side)
            + self._rng.gauss(0.0, 0.07 + 0.12 * uncertainty),
            0.0,
            1.0,
        )
        sell_pocket_bias = self._clamp(
            0.18
            + 0.18 * uncertainty
            + 0.14 * profit_taking
            + 0.12 * liquidation
            + self._rng.gauss(0.0, 0.08 + 0.10 * uncertainty),
            0.0,
            1.0,
        )

        buy = _MDFSideJudgment(
            fair_value_shift=buy_fair_value_shift,
            urgency=buy_urgency,
            patience=buy_patience,
            opportunity=buy_opportunity,
            liquidation=0.0,
            liquidity_aversion=buy_liquidity_aversion,
            pocket_bias=buy_pocket_bias,
            uncertainty=uncertainty,
        )
        sell = _MDFSideJudgment(
            fair_value_shift=sell_fair_value_shift,
            urgency=sell_urgency,
            patience=sell_patience,
            opportunity=profit_taking,
            liquidation=liquidation,
            liquidity_aversion=sell_liquidity_aversion,
            pocket_bias=sell_pocket_bias,
            uncertainty=uncertainty,
        )
        return _MDFJudgmentSample(
            buy=self._blend_mdf_judgment("buy", buy, update_memory=update_memory),
            sell=self._blend_mdf_judgment("sell", sell, update_memory=update_memory),
        )

    def _blend_mdf_judgment(
        self,
        key: str,
        sample: _MDFSideJudgment,
        *,
        update_memory: bool,
    ) -> _MDFSideJudgment:
        previous = self._mdf_judgment_memory.get(key)
        if previous is None:
            blended = sample
        else:
            blended = _MDFSideJudgment(
                fair_value_shift=0.65 * previous.fair_value_shift + 0.35 * sample.fair_value_shift,
                urgency=0.65 * previous.urgency + 0.35 * sample.urgency,
                patience=0.65 * previous.patience + 0.35 * sample.patience,
                opportunity=0.65 * previous.opportunity + 0.35 * sample.opportunity,
                liquidation=0.65 * previous.liquidation + 0.35 * sample.liquidation,
                liquidity_aversion=(
                    0.65 * previous.liquidity_aversion + 0.35 * sample.liquidity_aversion
                ),
                pocket_bias=0.65 * previous.pocket_bias + 0.35 * sample.pocket_bias,
                uncertainty=0.65 * previous.uncertainty + 0.35 * sample.uncertainty,
            )
        if update_memory:
            self._mdf_judgment_memory[key] = blended
        return blended

    def _buy_entry_mdf_mass(
        self,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
    ) -> dict[int, float]:
        volatility = self._clamp(context.volatility, 0.0, 2.2)
        stress = self._clamp(signals.liquidity_stress, 0.0, 1.8)
        event = self._clamp(signals.activity_event + 0.45 * signals.activity, 0.0, 1.8)
        bid_shortage = signals.bid_shortage_by_tick
        bid_gap = signals.bid_gap_by_tick
        bid_front = signals.bid_front_by_tick
        bid_liquidity = self._normalize_tick_signal(dict(signals.bid_liquidity_by_tick))
        ask_liquidity = self._normalize_tick_signal(dict(signals.ask_liquidity_by_tick))
        value_center = self._clamp(
            judgment.fair_value_shift * (1.0 + 0.35 * judgment.urgency)
            - 0.55 * judgment.patience,
            -float(self.grid_radius),
            min(3.0, float(self.grid_radius)),
        )
        value_spread = 1.15 + 0.55 * volatility + 0.50 * judgment.uncertainty
        spike_evidence = self._clamp(
            0.22 * judgment.urgency
            + 0.16 * judgment.patience
            + 0.16 * event
            + 0.20 * stress
            + 0.14 * judgment.uncertainty,
            0.0,
            1.0,
        )

        raw: dict[int, float] = {}
        for tick in relative_ticks:
            level = abs(tick)
            texture = self._mdf_texture("buy", tick, context, salt=0.0)
            value_fit = self._soft_tick_band(float(tick), value_center, value_spread)
            mass = 0.007 + 0.005 * texture
            mass += (0.035 + 0.026 * judgment.uncertainty) * value_fit
            if -1 <= tick <= 0:
                mass += 0.048 + 0.018 * volatility + 0.012 * event
            elif level == 2:
                mass += 0.026 + 0.010 * volatility
            else:
                mass += 0.010 / (1.0 + 0.24 * level)

            if tick <= 0:
                passive_level = abs(tick)
                reservation_band = self._entry_reservation_band(passive_level)
                passive_center = (
                    0.95
                    + 1.70 * judgment.patience
                    + 0.55 * volatility
                    + 0.85 * judgment.opportunity
                )
                passive_band = self._soft_tick_band(
                    float(passive_level),
                    passive_center,
                    1.00 + 0.35 * volatility + 0.25 * stress,
                )
                book_signal = (
                    0.34 * bid_shortage.get(tick, 0.0)
                    + 0.24 * bid_gap.get(tick, 0.0)
                    + 0.22 * bid_front.get(tick, 0.0)
                )
                book_response = max(0.0, 1.0 - 0.70 * judgment.liquidity_aversion)
                spike_evidence = max(spike_evidence, 0.55 * self._clamp(book_signal, 0.0, 1.0))
                mass += (
                    0.026
                    + 0.042 * judgment.patience
                    + 0.038 * judgment.opportunity
                    + 0.012 * volatility
                ) * passive_band
                mass += 0.012 * reservation_band
                mass += 0.070 * passive_band * book_response * (1.0 - exp(-2.0 * book_signal))
                mass += 0.018 * bid_liquidity.get(tick, 0.0)
            else:
                market_band = self._entry_marketable_band(tick)
                opposing = ask_liquidity.get(tick, 0.0)
                crossing_depth = opposing * (0.026 + 0.120 * judgment.urgency)
                crossing_depth *= max(0.0, 1.0 - 0.35 * judgment.liquidity_aversion)
                spike_evidence = max(
                    spike_evidence,
                    0.50 * opposing * self._clamp(judgment.urgency, 0.0, 1.0),
                )
                mass += (0.026 + 0.092 * judgment.urgency + 0.018 * event) * market_band
                mass += crossing_depth

            if level >= 4:
                mass += (0.014 * volatility + 0.014 * stress) * texture
            if level >= 6:
                mass += (0.020 * volatility + 0.010 * judgment.uncertainty) * texture
            raw[tick] = mass * (0.80 + 0.38 * texture)

        self._add_buy_entry_pockets(relative_ticks, context, signals, judgment, raw)
        center_weight = 0.55 + 0.22 * spike_evidence
        adjacent_weight = 0.16 - 0.045 * spike_evidence
        outer_weight = max(0.0, (1.0 - center_weight - 2.0 * adjacent_weight) / 2.0)
        return self._smooth_tick_mass(
            raw,
            relative_ticks,
            center_weight=center_weight,
            adjacent_weight=adjacent_weight,
            outer_weight=outer_weight,
        )

    def _sell_entry_mdf_mass(
        self,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
    ) -> dict[int, float]:
        volatility = self._clamp(context.volatility, 0.0, 2.2)
        stress = self._clamp(signals.liquidity_stress, 0.0, 1.8)
        event = self._clamp(signals.activity_event + 0.45 * signals.activity, 0.0, 1.8)
        ask_shortage = signals.ask_shortage_by_tick
        ask_gap = signals.ask_gap_by_tick
        ask_front = signals.ask_front_by_tick
        ask_liquidity = self._normalize_tick_signal(dict(signals.ask_liquidity_by_tick))
        bid_liquidity = self._normalize_tick_signal(dict(signals.bid_liquidity_by_tick))
        value_center = self._clamp(
            judgment.fair_value_shift * (1.0 + 0.30 * judgment.patience)
            + 0.55 * judgment.patience
            - 0.95 * judgment.urgency
            - 0.70 * judgment.liquidation,
            -min(3.0, float(self.grid_radius)),
            float(self.grid_radius),
        )
        value_spread = 1.15 + 0.55 * volatility + 0.50 * judgment.uncertainty
        spike_evidence = self._clamp(
            0.22 * judgment.urgency
            + 0.16 * judgment.patience
            + 0.20 * judgment.liquidation
            + 0.16 * event
            + 0.18 * stress,
            0.0,
            1.0,
        )

        raw: dict[int, float] = {}
        for tick in relative_ticks:
            level = abs(tick)
            texture = self._mdf_texture("sell", tick, context, salt=0.0)
            value_fit = self._soft_tick_band(float(tick), value_center, value_spread)
            mass = 0.007 + 0.005 * texture
            mass += (0.035 + 0.026 * judgment.uncertainty) * value_fit
            if 0 <= tick <= 1:
                mass += 0.048 + 0.018 * volatility + 0.012 * event
            elif level == 2:
                mass += 0.026 + 0.010 * volatility
            else:
                mass += 0.010 / (1.0 + 0.24 * level)

            if tick >= 0:
                passive_level = tick
                reservation_band = self._entry_reservation_band(passive_level)
                passive_center = (
                    0.95
                    + 1.60 * judgment.patience
                    + 0.55 * volatility
                    - 0.50 * judgment.opportunity
                    + 0.45 * judgment.fair_value_shift
                )
                passive_band = self._soft_tick_band(
                    float(passive_level),
                    passive_center,
                    1.00 + 0.35 * volatility + 0.25 * stress,
                )
                book_signal = (
                    0.34 * ask_shortage.get(tick, 0.0)
                    + 0.24 * ask_gap.get(tick, 0.0)
                    + 0.22 * ask_front.get(tick, 0.0)
                )
                book_response = max(0.0, 1.0 - 0.70 * judgment.liquidity_aversion)
                spike_evidence = max(spike_evidence, 0.55 * self._clamp(book_signal, 0.0, 1.0))
                mass += (
                    0.026
                    + 0.040 * judgment.patience
                    + 0.030 * judgment.opportunity
                    + 0.012 * volatility
                ) * passive_band
                mass += 0.012 * reservation_band
                mass += 0.070 * passive_band * book_response * (1.0 - exp(-2.0 * book_signal))
                mass += 0.018 * ask_liquidity.get(tick, 0.0)
            else:
                market_band = self._entry_marketable_band(abs(tick))
                opposing = bid_liquidity.get(tick, 0.0)
                crossing_depth = opposing * (
                    0.026 + 0.105 * judgment.urgency + 0.190 * judgment.liquidation
                )
                crossing_depth *= max(0.0, 1.0 - 0.35 * judgment.liquidity_aversion)
                spike_evidence = max(
                    spike_evidence,
                    0.50 * opposing * self._clamp(judgment.urgency, 0.0, 1.0),
                )
                mass += (
                    0.026
                    + 0.086 * judgment.urgency
                    + 0.150 * judgment.liquidation
                    + 0.018 * event
                ) * market_band
                mass += crossing_depth

            if level >= 4:
                mass += (0.014 * volatility + 0.014 * stress) * texture
            if level >= 6:
                mass += (0.020 * volatility + 0.010 * judgment.uncertainty) * texture
            raw[tick] = mass * (0.80 + 0.38 * texture)

        self._add_sell_entry_pockets(relative_ticks, context, signals, judgment, raw)
        center_weight = 0.55 + 0.22 * spike_evidence
        adjacent_weight = 0.16 - 0.045 * spike_evidence
        outer_weight = max(0.0, (1.0 - center_weight - 2.0 * adjacent_weight) / 2.0)
        return self._smooth_tick_mass(
            raw,
            relative_ticks,
            center_weight=center_weight,
            adjacent_weight=adjacent_weight,
            outer_weight=outer_weight,
        )

    def _evolve_raw_mdf(
        self,
        key: str,
        raw_mass: dict[int, float],
        *,
        current_tick: int,
        memory_mix: float,
        floor_mix: float,
        update_memory: bool,
    ) -> dict[int, float]:
        ticks = self.relative_tick_grid()
        valid_ticks = [tick for tick in ticks if current_tick + tick >= 1]
        proposal = self._normalize_tick_map(
            {tick: max(0.0, raw_mass.get(tick, 0.0)) for tick in valid_ticks}
        )
        previous = self._mdf_memory.get(key)
        if previous is None:
            previous = proposal
        else:
            previous = self._normalize_tick_map(
                {tick: previous.get(tick, 0.0) for tick in valid_ticks}
            )
        uniform = 1.0 / len(valid_ticks)
        mixed = self._normalize_tick_map(
            {
                tick: (1.0 - memory_mix) * proposal.get(tick, 0.0)
                + memory_mix * previous.get(tick, uniform)
                for tick in valid_ticks
            }
        )
        evolved = self._normalize_tick_map(
            {
                tick: (1.0 - floor_mix) * mixed.get(tick, 0.0) + floor_mix * uniform
                for tick in valid_ticks
            }
        )
        constrained = {tick: evolved.get(tick, 0.0) for tick in ticks}
        if update_memory:
            self._mdf_memory[key] = constrained
        return constrained

    def _centered_entry_noise_mdf(
        self,
        relative_ticks: list[int],
        current_tick: int,
    ) -> TickMap:
        valid_ticks = [tick for tick in relative_ticks if current_tick + tick >= 1]
        if not valid_ticks:
            return {tick: 0.0 for tick in relative_ticks}
        sigma = max(self._entry_noise_sigma_ticks, 1e-12)
        mass = {
            tick: exp(-0.5 * (float(tick) / sigma) ** 2)
            for tick in valid_ticks
        }
        normalized = self._normalize_tick_map(mass)
        return {tick: normalized.get(tick, 0.0) for tick in relative_ticks}

    def _mix_entry_noise_mdf(
        self,
        raw_mdf: TickMap,
        noise_mdf: TickMap,
        *,
        mix: float,
    ) -> TickMap:
        mix = self._clamp(mix, 0.0, 1.0)
        ticks = list(raw_mdf)
        if not ticks:
            return {}
        if mix <= 1e-12:
            normalized = self._normalize_tick_map(raw_mdf)
            return {tick: normalized.get(tick, 0.0) for tick in ticks}
        mixed = {
            tick: (1.0 - mix) * max(0.0, raw_mdf.get(tick, 0.0))
            + mix * max(0.0, noise_mdf.get(tick, 0.0))
            for tick in ticks
        }
        normalized = self._normalize_tick_map(mixed)
        return {tick: normalized.get(tick, 0.0) for tick in ticks}

    def _resolve_entry_mdf_overlap(
        self,
        relative_ticks: list[int],
        current_tick: int,
        raw_buy: TickMap,
        raw_sell: TickMap,
        *,
        buy_aggression: float,
        sell_aggression: float,
    ) -> tuple[TickMap, TickMap]:
        valid_ticks = [tick for tick in relative_ticks if current_tick + tick >= 1]
        if not valid_ticks:
            empty = {tick: 0.0 for tick in relative_ticks}
            return empty, empty.copy()

        buy = self._normalize_tick_map(
            {tick: max(0.0, raw_buy.get(tick, 0.0)) for tick in valid_ticks}
        )
        sell = self._normalize_tick_map(
            {tick: max(0.0, raw_sell.get(tick, 0.0)) for tick in valid_ticks}
        )
        buy_aggression = self._clamp(buy_aggression, 0.0, 1.0)
        sell_aggression = self._clamp(sell_aggression, 0.0, 1.0)

        local_buy = {tick: self._local_mdf_overlap(buy, tick) for tick in valid_ticks}
        local_sell = {tick: self._local_mdf_overlap(sell, tick) for tick in valid_ticks}

        resolved_buy: TickMap = {}
        resolved_sell: TickMap = {}
        for tick in valid_ticks:
            local_total = local_buy[tick] + local_sell[tick] + 1e-12
            buy_overlap = local_sell[tick] / local_total
            sell_overlap = local_buy[tick] / local_total
            if tick < 0:
                buy_overlap *= 0.84
                sell_overlap = min(1.0, sell_overlap * 1.10)
            elif tick > 0:
                buy_overlap = min(1.0, buy_overlap * 1.10)
                sell_overlap *= 0.84
            else:
                buy_overlap = min(1.0, buy_overlap * 1.03)
                sell_overlap = min(1.0, sell_overlap * 1.03)

            resolved_buy[tick] = buy.get(tick, 0.0) * self._entry_overlap_factor(
                buy_aggression,
                buy_overlap,
            )
            resolved_sell[tick] = sell.get(tick, 0.0) * self._entry_overlap_factor(
                sell_aggression,
                sell_overlap,
            )

        resolved_buy = self._normalize_tick_map(resolved_buy)
        resolved_sell = self._normalize_tick_map(resolved_sell)
        return (
            {tick: resolved_buy.get(tick, 0.0) for tick in relative_ticks},
            {tick: resolved_sell.get(tick, 0.0) for tick in relative_ticks},
        )

    def _entry_overlap_aggression(
        self,
        side: str,
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
        mdf_shock: float,
    ) -> float:
        stress = self._clamp(signals.liquidity_stress, 0.0, 1.8) / 1.8
        event = self._clamp(signals.activity_event + 0.45 * signals.activity, 0.0, 1.8) / 1.8
        volatility = self._clamp(context.volatility, 0.0, 2.2) / 2.2
        last_return = self._clamp(signals.last_return_ticks / 4.0, -1.0, 1.0)
        if side == "buy":
            directional = self._clamp(
                0.62 * max(0.0, context.trend)
                + 0.28 * max(0.0, context.mood)
                + 0.10 * max(0.0, last_return),
                0.0,
                1.0,
            )
            liquidation = 0.0
        else:
            directional = self._clamp(
                0.62 * max(0.0, -context.trend)
                + 0.28 * max(0.0, -context.mood)
                + 0.10 * max(0.0, -last_return),
                0.0,
                1.0,
            )
            liquidation = judgment.liquidation / 1.6
        return self._clamp(
            0.20
            + 0.30 * (judgment.urgency / 1.9)
            + 0.16 * event
            + 0.14 * stress
            + 0.12 * volatility
            + 0.12 * directional
            + 0.08 * liquidation
            + 0.10 * self._clamp(mdf_shock, 0.0, 1.0)
            - 0.10 * judgment.liquidity_aversion,
            0.0,
            1.0,
        )

    def _local_mdf_overlap(self, mdf: TickMap, tick: int) -> float:
        return (
            0.70 * mdf.get(tick, 0.0)
            + 0.15 * mdf.get(tick - 1, 0.0)
            + 0.15 * mdf.get(tick + 1, 0.0)
        )

    def _entry_overlap_factor(
        self,
        aggression: float,
        overlap: float,
    ) -> float:
        overlap = self._clamp(overlap, 0.0, 1.0)
        aggression = self._clamp(aggression, 0.0, 1.0)
        retained = max(0.0, 1.0 - overlap)
        sharpness = 7.70 - 3.00 * aggression
        floor = 0.004 + 0.050 * aggression
        return floor + (1.0 - floor) * retained**sharpness

    def _constrain_tick_mdf(self, current_tick: int, mdf_by_tick: TickMap) -> TickMap:
        ticks = self.relative_tick_grid()
        valid_ticks = [tick for tick in ticks if current_tick + tick >= 1]
        valid_mass = {
            tick: max(0.0, mdf_by_tick.get(tick, 0.0))
            for tick in valid_ticks
        }
        normalized = self._normalize_tick_map(valid_mass)
        return {tick: normalized.get(tick, 0.0) for tick in ticks}

    def _smooth_tick_mass(
        self,
        raw: TickMap,
        relative_ticks: list[int],
        *,
        center_weight: float,
        adjacent_weight: float,
        outer_weight: float,
    ) -> TickMap:
        tick_set = set(relative_ticks)
        kernel = (
            (0, center_weight),
            (-1, adjacent_weight),
            (1, adjacent_weight),
            (-2, outer_weight),
            (2, outer_weight),
        )
        smoothed: TickMap = {}
        for tick, mass in raw.items():
            if mass <= 0:
                continue
            for offset, weight in kernel:
                target = tick + offset
                if target in tick_set and weight > 0:
                    smoothed[target] = smoothed.get(target, 0.0) + mass * weight
        return {tick: smoothed.get(tick, 0.0) for tick in relative_ticks}

    def _add_buy_entry_pockets(
        self,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
        raw: dict[int, float],
    ) -> None:
        centers = self._buy_entry_pocket_centers(relative_ticks, context, signals, judgment)
        self._add_entry_pocket_mass("buy", centers, context, raw)

    def _buy_entry_pocket_centers(
        self,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        volatility = self._clamp(context.volatility, 0.0, 2.2)
        stress = self._clamp(signals.liquidity_stress, 0.0, 1.8)
        bid_liquidity = self._normalize_tick_signal(dict(signals.bid_liquidity_by_tick))
        for tick in relative_ticks:
            if tick > 0:
                continue
            level = abs(tick)
            texture = self._mdf_texture("buy", tick, context, salt=5.0)
            book_signal = (
                0.34 * signals.bid_shortage_by_tick.get(tick, 0.0)
                + 0.25 * signals.bid_gap_by_tick.get(tick, 0.0)
                + 0.22 * signals.bid_front_by_tick.get(tick, 0.0)
            )
            aversion_drag = max(0.0, 1.0 - 0.55 * judgment.liquidity_aversion)
            passive_center = (
                0.95
                + 1.70 * judgment.patience
                + 0.60 * volatility
                + 0.85 * judgment.opportunity
            )
            score = (
                0.38 * aversion_drag * (1.0 - exp(-2.1 * book_signal))
                + 0.040 * bid_liquidity.get(tick, 0.0)
                + 0.16 * texture
                + 0.20
                * judgment.pocket_bias
                * self._soft_tick_band(float(level), passive_center, 1.05 + 0.25 * stress)
            )
            scores[tick] = score
        return self._ordered_entry_pocket_centers(scores)

    def _add_sell_entry_pockets(
        self,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
        raw: dict[int, float],
    ) -> None:
        centers = self._sell_entry_pocket_centers(relative_ticks, context, signals, judgment)
        self._add_entry_pocket_mass("sell", centers, context, raw)

    def _sell_entry_pocket_centers(
        self,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        volatility = self._clamp(context.volatility, 0.0, 2.2)
        stress = self._clamp(signals.liquidity_stress, 0.0, 1.8)
        ask_liquidity = self._normalize_tick_signal(dict(signals.ask_liquidity_by_tick))
        for tick in relative_ticks:
            if tick < 0:
                continue
            level = abs(tick)
            texture = self._mdf_texture("sell", tick, context, salt=5.0)
            book_signal = (
                0.34 * signals.ask_shortage_by_tick.get(tick, 0.0)
                + 0.25 * signals.ask_gap_by_tick.get(tick, 0.0)
                + 0.22 * signals.ask_front_by_tick.get(tick, 0.0)
            )
            aversion_drag = max(0.0, 1.0 - 0.55 * judgment.liquidity_aversion)
            passive_center = (
                0.95
                + 1.60 * judgment.patience
                + 0.60 * volatility
                - 0.50 * judgment.opportunity
                + 0.40 * judgment.fair_value_shift
            )
            score = (
                0.38 * aversion_drag * (1.0 - exp(-2.1 * book_signal))
                + 0.040 * ask_liquidity.get(tick, 0.0)
                + 0.16 * texture
                + 0.20
                * judgment.pocket_bias
                * self._soft_tick_band(float(level), passive_center, 1.05 + 0.25 * stress)
            )
            scores[tick] = score
        return self._ordered_entry_pocket_centers(scores)

    def _ordered_entry_pocket_centers(self, scores: dict[int, float]) -> list[tuple[int, float]]:
        ordered = sorted(scores.items(), key=lambda item: (-item[1], abs(item[0]), item[0]))
        centers: list[tuple[int, float]] = []
        for tick, score in ordered:
            if all(abs(tick - existing) > 1 for existing, _ in centers):
                centers.append((tick, score))
            if len(centers) >= 4:
                break
        return centers

    def _add_entry_pocket_mass(
        self,
        side: str,
        centers: list[tuple[int, float]],
        context: MDFContext,
        raw: dict[int, float],
    ) -> None:
        for rank, (center, score) in enumerate(centers):
            strength = self._clamp(0.024 + 0.082 * score, 0.018, 0.110)
            strength *= 1.0 / (1.0 + 0.18 * rank)
            for offset, shape in (
                (0, 0.70),
                (-1, 0.64),
                (1, 0.66),
                (-2, 0.35),
                (2, 0.34),
                (-3, 0.14),
                (3, 0.14),
            ):
                tick = center + offset
                if tick not in raw:
                    continue
                texture = self._mdf_texture(side, tick, context, salt=rank + 3.0)
                raw[tick] += strength * shape * (0.82 + 0.36 * texture)

    @staticmethod
    def _soft_tick_band(value: float, center: float, spread: float) -> float:
        return exp(-abs(value - center) / max(spread, 1e-12))

    def _entry_reservation_band(self, level: int) -> float:
        if level <= 1:
            return 0.92
        if level == 2:
            return 0.68
        if level == 3:
            return 0.50
        if level <= 5:
            return 0.32
        if level <= 8:
            return 0.20
        return 0.12

    def _entry_marketable_band(self, level: int) -> float:
        if level <= 1:
            return 1.0
        if level == 2:
            return 0.48
        if level <= 4:
            return 0.16
        return 0.035

    def _mdf_texture(
        self,
        side: str,
        tick: int,
        context: MDFContext,
        *,
        salt: float,
    ) -> float:
        side_seed = {"buy": 0.37, "sell": 0.91}[side]
        seed = 0.0 if self._seed is None else float(self._seed)
        primary = sin(
            tick * 12.9898
            + context.step_index * 0.917
            + context.current_tick * 0.071
            + seed * 0.0031
            + side_seed
            + salt * 1.618
        )
        secondary = sin(
            tick * 4.231
            + context.step_index * 0.113
            + seed * 0.017
            + side_seed * 3.0
            + salt
        )
        return 0.72 + 0.34 * (0.5 + 0.5 * primary) + 0.18 * (0.5 + 0.5 * secondary)

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
        liquidity = self._price_map_to_relative_ticks(
            price,
            self._merge_maps(orderbook.bid_volume_by_price, orderbook.ask_volume_by_price),
        )
        bid_shortage, bid_gap, bid_front = self._book_observable_maps(
            price,
            "bid",
            orderbook.bid_volume_by_price,
        )
        ask_shortage, ask_gap, ask_front = self._book_observable_maps(
            price,
            "ask",
            orderbook.ask_volume_by_price,
        )
        return MDFSignals(
            orderbook_imbalance=self._last_imbalance,
            last_return_ticks=self._last_return_ticks,
            last_execution_volume=self._last_execution_volume,
            executed_volume_by_tick=self._price_map_to_relative_ticks(
                price, self._last_executed_by_price
            ),
            liquidity_by_tick=liquidity,
            bid_liquidity_by_tick=self._price_map_to_relative_ticks(
                price, orderbook.bid_volume_by_price
            ),
            ask_liquidity_by_tick=self._price_map_to_relative_ticks(
                price, orderbook.ask_volume_by_price
            ),
            bid_shortage_by_tick=bid_shortage,
            ask_shortage_by_tick=ask_shortage,
            bid_gap_by_tick=bid_gap,
            ask_gap_by_tick=ask_gap,
            bid_front_by_tick=bid_front,
            ask_front_by_tick=ask_front,
            spread_ticks=spread_ticks,
            cancel_pressure=self._microstructure.cancel_pressure,
            liquidity_stress=self._microstructure.liquidity_stress,
            stress_side=self._microstructure.stress_side,
            resiliency=self._microstructure.resiliency,
            activity=self._microstructure.activity,
            activity_event=self._microstructure.activity_event,
        )

    def _book_observable_maps(
        self,
        basis_price: float,
        side: str,
        volume_by_price: PriceMap,
    ) -> tuple[TickMap, TickMap, TickMap]:
        """Return shortage, gap, and near-front pressure on the side's passive grid."""

        raw_depth = self._price_map_to_relative_ticks(basis_price, volume_by_price)
        regime = self._active_settings
        side_ticks = [
            tick
            for tick in self.relative_tick_grid()
            if (tick < 0 if side == "bid" else tick > 0)
        ]
        if not side_ticks:
            return {}, {}, {}
        side_ticks.sort(key=abs)

        depth_values = {tick: max(0.0, raw_depth.get(tick, 0.0)) for tick in side_ticks}
        occupancies: dict[int, float] = {}
        shortages: dict[int, float] = {}
        front_signal: dict[int, float] = {}
        expected_scale = max(0.08, self.popularity * regime["liquidity"])

        for tick in side_ticks:
            level = max(1.0, abs(float(tick)))
            expected = expected_scale * regime["near_touch_liquidity"] / (
                level ** regime["depth_exponent"]
            )
            depth = depth_values[tick]
            occupancy = depth / (depth + expected + 1e-12)
            shortage = self._clamp(1.0 - occupancy, 0.0, 1.0)
            near_weight = exp(-max(0.0, level - 1.0) / 2.2)
            occupancies[tick] = occupancy
            shortages[tick] = shortage
            front_signal[tick] = shortage * near_weight

        gap_signal: dict[int, float] = {}
        previous_occupancy = None
        for tick in side_ticks:
            occupancy = occupancies[tick]
            discontinuity = (
                0.0 if previous_occupancy is None else abs(occupancy - previous_occupancy)
            )
            gap_signal[tick] = shortages[tick] * (0.15 + discontinuity)
            previous_occupancy = occupancy

        return (
            self._normalize_tick_signal(shortages),
            self._normalize_tick_signal(gap_signal),
            self._normalize_tick_signal(front_signal),
        )

    def _normalize_tick_signal(self, values: TickMap) -> TickMap:
        peak = max(values.values(), default=0.0)
        if peak <= 1e-12:
            return {}
        return {tick: value / peak for tick, value in values.items() if value > 1e-12}

    def _project_tick_mdf(self, current_tick: int, mdf_by_tick: dict[int, float]) -> PriceMap:
        projected: PriceMap = {}
        for relative_tick, probability in mdf_by_tick.items():
            absolute_tick = current_tick + relative_tick
            if absolute_tick < 1:
                continue
            price = self.tick_to_price(absolute_tick)
            projected[price] = projected.get(price, 0.0) + probability
        total = sum(projected.values())
        if total <= 0:
            return {}
        return {price: probability / total for price, probability in sorted(projected.items())}

    def _normalize_price_map(self, values: PriceMap) -> PriceMap:
        total = sum(max(0.0, value) for value in values.values())
        if total <= 0:
            uniform = 1.0 / len(values)
            return {price: uniform for price in values}
        return {price: max(0.0, value) / total for price, value in sorted(values.items())}

    def _normalize_tick_map(self, values: dict[int, float]) -> dict[int, float]:
        total = sum(max(0.0, value) for value in values.values())
        if total <= 0:
            uniform = 1.0 / len(values)
            return {tick: uniform for tick in values}
        return {tick: max(0.0, value) / total for tick, value in values.items()}

    def _taker_share(self, latent: LatentState, imbalance: float) -> float:
        regime = self._active_settings
        base = (
            0.12
            + 0.08 * latent.volatility
            + 0.04 * abs(imbalance)
        )
        return self._clamp(base * regime["taker"], 0.08, 0.62)
