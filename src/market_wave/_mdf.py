from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite

from ._types import (
    _EntryFlow,
    _IncomingOrder,
    _MarketEvent,
    _MicrostructureState,
    _StepComputationCache,
)
from .distribution import MDFSignals
from .state import IntensityState, LatentState, MDFState, PriceMap, TickMap


@dataclass(frozen=True)
class _EntrySideView:
    push: float
    resistance: float
    opposing_resistance: float
    own_intent: float
    opposite_intent: float
    own_vacuum: float
    opposing_vacuum: float
    burst_alignment: float
    burst_resistance: float
    directional_pressure: float
    shortage: TickMap
    occupancy: TickMap
    front: TickMap
    passive_book_side: str


class _MarketMDFMixin:
    def _next_mdf(
        self,
        price: float,
        price_grid: list[float],
        latent: LatentState,
        *,
        step_index: int,
        cache: _StepComputationCache | None = None,
        current_price: float | None = None,
    ) -> MDFState:
        del price_grid
        if current_price is None:
            current_price = self.state.price
        gap_offsets = self.relative_tick_grid()
        current_tick = self.price_to_tick(price)
        signals = self._cached_mdf_signals(price, cache)

        buy = self._entry_mdf(
            "buy",
            price,
            current_price,
            gap_offsets,
            latent,
            signals,
            step_index=step_index,
        )
        sell = self._entry_mdf(
            "sell",
            price,
            current_price,
            gap_offsets,
            latent,
            signals,
            step_index=step_index,
        )

        buy = self._constrain_gap_mdf(current_tick, buy)
        sell = self._constrain_gap_mdf(current_tick, sell)
        return MDFState(
            buy_entry_mdf=buy,
            sell_entry_mdf=sell,
        )

    def _entry_mdf(
        self,
        side: str,
        basis_price: float,
        current_price: float,
        gap_offsets: list[int],
        latent: LatentState,
        signals: MDFSignals,
        *,
        step_index: int,
    ) -> TickMap:
        volatility = self._clamp(latent.volatility + 0.35 * signals.volatility_cluster, 0.0, 2.5)
        stress = self._clamp(signals.liquidity_stress, 0.0, 2.0) / 2.0
        spread = self._clamp((signals.spread_ticks - 1.0) / 6.0, 0.0, 1.0)
        activity = self._clamp(
            signals.activity + signals.activity_event + 0.70 * signals.arrival_cluster,
            0.0,
            5.3,
        ) / 5.3
        regime_trend_abs = abs(self._active_settings["trend"]) / 0.30
        condition_volatility_context = self._clamp(
            (self._active_settings["volatility"] - 0.98) / 0.77,
            0.0,
            1.0,
        )
        directional_context = self._clamp(
            0.35
            + 0.42 * regime_trend_abs
            + 0.15 * signals.flow_persistence / 1.45
            - 0.26 * condition_volatility_context * (1.0 - regime_trend_abs),
            0.25,
            1.0,
        )
        trend = self._clamp(
            directional_context * (0.65 * latent.trend + 0.35 * latent.mood),
            -1.0,
            1.0,
        )
        realized_direction = self._clamp(
            0.40 * self._clamp(signals.last_return_ticks / 4.0, -1.0, 1.0)
            + 0.34 * signals.orderbook_imbalance
            + 0.26 * self._clamp(signals.meta_order_side, -1.0, 1.0),
            -1.0,
            1.0,
        )
        participant = self._participant_pressure
        high_participation = self._clamp((self.popularity - 1.0) / 2.5, 0.0, 1.0)
        continuation = self._clamp(participant.flow_continuation, 0.0, 1.8) / 1.8
        absorption = self._clamp(participant.absorption, 0.0, 1.8) / 1.8
        exhaustion = self._clamp(participant.exhaustion, 0.0, 1.8) / 1.8
        intent_memory = self._clamp(participant.signed_intent_memory, -1.0, 1.0)
        side_view = self._entry_side_view(
            side,
            trend=trend,
            realized_direction=realized_direction,
            continuation=continuation,
            intent_memory=intent_memory,
            signals=signals,
        )
        push = side_view.push
        resistance = side_view.resistance
        opposing_resistance = side_view.opposing_resistance
        own_intent = side_view.own_intent
        opposite_intent = side_view.opposite_intent
        own_vacuum = side_view.own_vacuum
        opposing_vacuum = side_view.opposing_vacuum
        burst_alignment = side_view.burst_alignment
        burst_resistance = side_view.burst_resistance
        directional_pressure = side_view.directional_pressure
        shortage = side_view.shortage
        occupancy = side_view.occupancy
        front = side_view.front
        regime_trend = self._clamp(self._active_settings["trend"] / 0.30, -1.0, 1.0)
        trend_aligned_sweep = max(0.0, regime_trend if side == "buy" else -regime_trend)
        trend_counter_liquidity = max(0.0, -regime_trend if side == "buy" else regime_trend)

        passive_shortage = self._side_shortage(side_view.passive_book_side, signals)
        own_load = signals.bid_depth_pressure if side == "buy" else signals.ask_depth_pressure
        passive_pull = self._clamp(passive_shortage + max(0.0, -own_load), 0.0, 2.0) / 2.0
        severe_own_vacuum = self._clamp((own_vacuum - 0.42) / 0.58, 0.0, 1.0)
        severe_book_vacuum = self._clamp(
            (self._clamp(signals.book_vacuum, 0.0, 1.8) / 1.8 - 0.42) / 0.58,
            0.0,
            1.0,
        )
        low_participation = self._clamp((0.80 - self.popularity) / 0.80, 0.0, 1.0)
        repair_mode = self._clamp(
            0.62 * severe_own_vacuum + 0.26 * severe_book_vacuum,
            0.0,
            1.0,
        ) * self._clamp(1.0 - 0.32 * low_participation, 0.62, 1.0)
        urgency_shortage = passive_shortage * self._clamp(
            0.35 + 0.45 * stress + 0.35 * activity,
            0.0,
            1.0,
        )
        urgency = self._clamp(
            0.18 * activity
            + 0.30 * stress
            + 0.30 * max(0.0, directional_pressure)
            + 0.18 * push
            + 0.28 * continuation * own_intent
            + 0.18 * urgency_shortage
            + 0.18 * trend_aligned_sweep
            + 0.10 * participant.noise_pressure,
            0.0,
            1.5,
        )
        wall_pressure = self._clamp(
            0.45 * resistance
            + 0.34 * absorption * (0.55 + opposite_intent)
            + 0.35 * passive_pull
            + 0.22 * spread
            + 0.12 * participant.noise_pressure,
            0.0,
            1.8,
        )

        arrival = self._clamp(signals.arrival_cluster, 0.0, 2.2) / 2.2
        participation = self._clamp(signals.participation_burst, 0.0, 2.0) / 2.0
        drought = self._clamp(signals.liquidity_drought, 0.0, 2.0) / 2.0
        condition_volatility = condition_volatility_context
        volatile_state = self._clamp(
            (volatility + 0.35 * self._clamp(signals.volatility_cluster, 0.0, 2.0) - 0.55)
            / 1.45
            + 0.38 * condition_volatility,
            0.0,
            1.0,
        )
        dislocation = self._clamp(
            0.38 * stress
            + 0.28 * drought
            + 0.22 * spread
            + 0.26 * self._clamp(signals.displacement_pressure, 0.0, 2.0) / 2.0,
            0.0,
            1.0,
        ) * (0.20 + 0.80 * arrival) * volatile_state
        burst_state = volatile_state * self._clamp(
            0.34 * stress
            + 0.26 * arrival
            + 0.24 * participation
            + 0.20 * self._clamp(signals.displacement_pressure, 0.0, 2.0) / 2.0,
            0.0,
            1.0,
        )
        aligned_burst = burst_state * burst_alignment
        chaotic_burst = burst_state * condition_volatility * (0.35 + 0.65 * arrival)
        opposing_burst = burst_state * burst_resistance
        passive_center = (
            0.8
            + 0.7 * passive_pull
            + 0.55 * spread
            + 1.60 * low_participation
            - 0.30 * urgency
            - 0.08 * high_participation
        )
        passive_center = max(0.45, passive_center - 0.45 * repair_mode)
        displacement = self._clamp(signals.displacement_pressure, 0.0, 2.0) / 2.0
        marketable_center = (
            0.45
            + 0.95 * urgency
            + 0.60 * max(0.0, directional_pressure)
            + 0.55 * trend_aligned_sweep
            + 1.35 * dislocation
            + 0.85 * displacement
            + 1.45 * aligned_burst
            + 1.10 * chaotic_burst
        )
        taker_pressure = self._clamp(
            max(0.0, directional_pressure)
            + 0.65 * push
            + 0.55 * continuation * own_intent
            + 0.36 * abs(intent_memory) * max(0.0, directional_pressure)
            + 0.44 * trend_aligned_sweep
            + 0.40 * urgency_shortage
            + 0.28 * drought
            + 0.42 * dislocation
            + 0.64 * aligned_burst
            + 0.44 * chaotic_burst
            - 0.55 * wall_pressure
            - 0.38 * absorption,
            0.0,
            1.8,
        ) / 1.8
        taker_pressure *= self._clamp(
            1.0
            - 0.50 * repair_mode
            - 0.46 * opposing_vacuum * (1.0 - 0.70 * trend_aligned_sweep)
            - 0.28 * opposing_burst,
            0.10,
            1.0,
        )
        stress_bundle = self._clamp(stress + drought + dislocation, 0.0, 1.0)
        effective_taker_pressure = taker_pressure * (0.30 + 0.70 * stress_bundle)
        tail_state = self._clamp(
            0.10
            + 0.52 * volatile_state
            + 0.32 * stress_bundle
            + 0.24 * arrival
            + 0.28 * aligned_burst
            + 0.32 * chaotic_burst
            + 0.20 * condition_volatility,
            0.10,
            1.0,
        )
        width = (
            1.0
            + 1.20 * volatility
            + 1.05 * stress
            + 0.45 * spread
            + 0.35 * continuation
            + 0.38 * arrival
            + 0.34 * self._clamp(signals.volatility_cluster, 0.0, 2.0) / 2.0
            + 0.28 * self._clamp(signals.churn_pressure, 0.0, 2.0) / 2.0
        )
        retreat_pressure = self._clamp(
            0.34 * drought
            + 0.26 * stress
            + 0.28 * self._clamp(signals.cancel_pressure, 0.0, 2.0) / 2.0
            + 0.22 * spread
            + 0.30 * low_participation
            - 0.44 * repair_mode
            - 0.18 * self._clamp(signals.resiliency, 0.2, 1.4) / 1.4,
            0.0,
            1.0,
        )
        repair_share = self._clamp(
            0.58
            + 0.54 * repair_mode
            + 0.30 * passive_pull
            + 0.14 * (1.0 - low_participation)
            + 0.22 * high_participation
            + 0.16 * trend_counter_liquidity
            - 0.38 * retreat_pressure
            - 0.20 * effective_taker_pressure
            - 0.40 * max(0.0, -directional_pressure)
            - 2.10 * low_participation,
            0.02,
            0.97,
        )
        taker_share = self._clamp(
            0.002
            + 0.34 * effective_taker_pressure
            + 0.30 * dislocation
            + 0.24 * volatile_state * stress_bundle
            + 0.26 * aligned_burst
            + 0.16 * chaotic_burst
            + 0.06 * condition_volatility * (0.35 + 0.65 * stress_bundle)
            + 0.10 * max(0.0, directional_pressure)
            + 0.08 * trend_aligned_sweep
            + 0.24 * trend_aligned_sweep * max(0.0, directional_pressure)
            - 0.24 * repair_mode
            - 0.16 * absorption
            + 0.06 * high_participation * stress_bundle,
            0.001,
            0.48,
        )
        retreat_center = (
            passive_center
            + 0.55
            + 1.08 * retreat_pressure
            + 0.24 * spread
            + 7.20 * low_participation
            - 0.25 * high_participation
        )
        thin_inside_gap = 3.20 * low_participation * self._clamp(
            1.0 - 0.55 * repair_mode,
            0.25,
            1.0,
        )
        repair_center = max(
            0.25,
            0.55
            + 0.45 * spread
            + 0.55 * low_participation
            + thin_inside_gap
            - 0.38 * repair_mode
            - 0.08 * high_participation,
        )
        current_displacement = self._clamp(
            (current_price - basis_price) / max(self.gap, 1e-12),
            -float(self.grid_radius),
            float(self.grid_radius),
        )
        if side == "buy":
            passive_shift = min(0.0, current_displacement)
            active_shift = max(0.0, current_displacement)
        else:
            passive_shift = max(0.0, current_displacement)
            active_shift = min(0.0, current_displacement)
        passive_center += passive_shift
        repair_center += passive_shift
        retreat_center += passive_shift
        marketable_center += active_shift
        def passive_entry_mass(tick: int, level: float) -> float:
            repair_distance = abs(level - repair_center)
            retreat_distance = abs(level - retreat_center)
            level_shortage = shortage.get(tick, 0.0)
            level_occupancy = occupancy.get(tick, 0.0)
            front_support = front.get(tick, 0.0)
            far_entry_zone = self._clamp(
                (level - 4.5) / max(1.0, 0.24 * self.grid_radius),
                0.0,
                1.0,
            )
            quote_lifecycle_capacity = self._clamp(
                1.0 - (0.94 + 0.10 * high_participation) * far_entry_zone,
                0.035,
                1.0,
            ) * self._clamp(1.0 - 0.28 * high_participation * far_entry_zone, 0.60, 1.0)
            parking_lifecycle_capacity = self._clamp(
                1.0 - (0.68 + 0.10 * high_participation) * far_entry_zone,
                0.12,
                1.0,
            ) * self._clamp(1.0 - 0.34 * high_participation * far_entry_zone, 0.56, 1.0)
            passive_tail_state = self._clamp(
                tail_state + 0.58 * low_participation + 0.28 * drought + 0.22 * spread,
                0.10,
                1.35,
            )
            passive_force = (
                0.62
                + 0.72 * passive_pull
                + 0.58 * wall_pressure
                + 0.54 * level_shortage
                + 0.58 * repair_mode
                + 0.34 * trend_counter_liquidity
                + 0.26 * front_support
                + 0.08 * high_participation * (1.0 / max(1.0, level))
                + 0.24 * spread
                + 0.16 * exhaustion
                + 0.12 * participation
            )
            passive_force *= self._clamp(
                1.0 - 0.46 * max(0.0, -directional_pressure),
                0.32,
                1.0,
            )
            occupancy_excess = max(0.0, level_occupancy - 1.0)
            occupancy_gap = max(0.0, 1.0 - level_occupancy)
            wall_texture = 1.0 + 0.16 * front_support + 0.12 * occupancy_gap
            crowding_drag = wall_texture / (1.0 + 1.18 * occupancy_excess)
            inside_liquidity_capacity = self._clamp(
                (level + 0.35) / (0.35 + thin_inside_gap),
                0.24 + 0.36 * repair_mode,
                1.0,
            )
            inside_liquidity_capacity *= self._clamp(
                (level + 0.55) / (0.55 + thin_inside_gap),
                0.35 + 0.30 * repair_mode,
                1.0,
            )
            passive_width = width * (0.42 + 0.42 * passive_tail_state)
            repair_mass = (
                repair_share
                * quote_lifecycle_capacity
                * exp(-repair_distance / max(0.58, passive_width * 0.62))
            )
            retreat_mass = (
                (1.0 - repair_share)
                * parking_lifecycle_capacity
                * exp(-retreat_distance / max(0.72, passive_width))
            )
            mass = (
                passive_force
                * (repair_mass + retreat_mass)
                * crowding_drag
                * inside_liquidity_capacity
            )
            return mass + (
                0.014 + 0.050 * passive_tail_state + 0.020 * level_shortage + 0.012 * occupancy_gap
            ) * quote_lifecycle_capacity / (1.0 + level)

        def marketable_entry_mass(level: float) -> float:
            distance = abs(level - marketable_center)
            depletion_brake = self._clamp(1.0 - 0.36 * repair_mode, 0.42, 1.0)
            marketable_force = (
                0.04 * effective_taker_pressure
                + 0.48 * urgency * effective_taker_pressure
                + 0.34 * push
                + 0.30 * continuation * own_intent
                + 0.14 * urgency_shortage
                + 0.34 * dislocation
                + 0.28 * volatile_state * stress_bundle
                + 0.42 * aligned_burst
                + 0.28 * chaotic_burst
                + 0.09 * arrival * effective_taker_pressure
                + 0.08 * participation * effective_taker_pressure
                - 0.34 * opposing_resistance * (0.45 + absorption)
                - 0.24 * opposing_burst
                - 0.06 * passive_shortage * max(0.0, 1.0 - stress - continuation)
                - 0.20 * exhaustion * own_intent
            ) * depletion_brake
            mass = taker_share * max(0.0, marketable_force) * exp(
                -distance / max(0.62, width * (0.46 + 0.62 * tail_state))
            )
            return mass + (
                0.018 * activity
                + 0.012 * arrival
                + 0.010 * self._clamp(signals.volatility_cluster, 0.0, 2.0) / 2.0
            ) * taker_pressure * tail_state / (1.0 + level)

        raw: TickMap = {}
        basis_tick = self.price_to_tick(basis_price)
        for tick in gap_offsets:
            if basis_tick + tick < 1:
                raw[tick] = 0.0
                continue
            level = abs(float(tick))
            if self._is_passive_entry_tick(side, tick):
                mass = passive_entry_mass(tick, level)
            else:
                mass = marketable_entry_mass(level)
            raw[tick] = max(0.0, mass)
        return self._normalize_tick_map(raw)

    @staticmethod
    def _is_passive_entry_tick(side: str, tick: int) -> bool:
        return tick <= 0 if side == "buy" else tick >= 0

    def _entry_side_view(
        self,
        side: str,
        *,
        trend: float,
        realized_direction: float,
        continuation: float,
        intent_memory: float,
        signals: MDFSignals,
    ) -> _EntrySideView:
        participant = self._participant_pressure
        if side == "buy":
            push = participant.upward_push
            resistance = participant.downward_resistance
            opposing_resistance = participant.upward_resistance
            own_intent = max(0.0, intent_memory)
            opposite_intent = max(0.0, -intent_memory)
            own_vacuum = self._clamp(signals.bid_vacuum, 0.0, 1.8) / 1.8
            opposing_vacuum = self._clamp(signals.ask_vacuum, 0.0, 1.8) / 1.8
            burst_alignment = max(0.0, realized_direction)
            burst_resistance = max(0.0, -realized_direction)
            directional_pressure = (
                trend
                + push
                + 1.25 * continuation * own_intent
                + 0.55 * own_intent
                - opposing_resistance
            )
            return _EntrySideView(
                push=push,
                resistance=resistance,
                opposing_resistance=opposing_resistance,
                own_intent=own_intent,
                opposite_intent=opposite_intent,
                own_vacuum=own_vacuum,
                opposing_vacuum=opposing_vacuum,
                burst_alignment=burst_alignment,
                burst_resistance=burst_resistance,
                directional_pressure=directional_pressure,
                shortage=signals.bid_shortage_by_tick,
                occupancy=signals.bid_occupancy_by_tick,
                front=signals.bid_front_by_tick,
                passive_book_side="bid",
            )
        push = participant.downward_push
        resistance = participant.upward_resistance
        opposing_resistance = participant.downward_resistance
        own_intent = max(0.0, -intent_memory)
        opposite_intent = max(0.0, intent_memory)
        own_vacuum = self._clamp(signals.ask_vacuum, 0.0, 1.8) / 1.8
        opposing_vacuum = self._clamp(signals.bid_vacuum, 0.0, 1.8) / 1.8
        burst_alignment = max(0.0, -realized_direction)
        burst_resistance = max(0.0, realized_direction)
        directional_pressure = (
            -trend
            + push
            + 1.25 * continuation * own_intent
            + 0.55 * own_intent
            - opposing_resistance
        )
        return _EntrySideView(
            push=push,
            resistance=resistance,
            opposing_resistance=opposing_resistance,
            own_intent=own_intent,
            opposite_intent=opposite_intent,
            own_vacuum=own_vacuum,
            opposing_vacuum=opposing_vacuum,
            burst_alignment=burst_alignment,
            burst_resistance=burst_resistance,
            directional_pressure=directional_pressure,
            shortage=signals.ask_shortage_by_tick,
            occupancy=signals.ask_occupancy_by_tick,
            front=signals.ask_front_by_tick,
            passive_book_side="ask",
        )

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

    def _entry_flow(
        self,
        intensity: IntensityState,
        mdf: MDFState,
        *,
        mdf_price_basis: float,
        current_price: float,
    ) -> _EntryFlow:
        if self._time_bucket_has_no_arrivals():
            return _EntryFlow(
                orders=[],
                buy_intent_by_price={},
                sell_intent_by_price={},
                events=[],
                bid_cancel_intent_by_price={},
                ask_cancel_intent_by_price={},
            )
        buy_orders, buy_intent = self._sample_entry_side(
            "buy",
            "buy_entry",
            intensity.buy,
            mdf.buy_entry_mdf,
            basis_price=mdf_price_basis,
        )
        sell_orders, sell_intent = self._sample_entry_side(
            "sell",
            "sell_entry",
            intensity.sell,
            mdf.sell_entry_mdf,
            basis_price=mdf_price_basis,
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
        refresh_events, refresh_buy, refresh_sell = self._sample_quote_refresh_events(
            mdf,
            mdf_price_basis=mdf_price_basis,
            current_price=current_price,
        )
        events.extend(refresh_events)
        buy_intent = self._merge_maps(buy_intent, refresh_buy)
        sell_intent = self._merge_maps(sell_intent, refresh_sell)
        if not events:
            idle_probability = self._clamp(
                0.70 - 0.16 * self._event_cluster_signal(self._microstructure),
                0.35,
                0.78,
            )
            if self._unit_random() < idle_probability:
                return _EntryFlow(
                    orders=[],
                    buy_intent_by_price={},
                    sell_intent_by_price={},
                    events=[],
                    bid_cancel_intent_by_price={},
                    ask_cancel_intent_by_price={},
                )
        cancel_events, bid_cancel, ask_cancel = self._sample_cancel_events(
            mdf,
            mdf_price_basis=mdf_price_basis,
        )
        events.extend(cancel_events)
        self._rng.shuffle(events)
        return _EntryFlow(
            orders=orders,
            buy_intent_by_price=self._drop_zeroes(buy_intent),
            sell_intent_by_price=self._drop_zeroes(sell_intent),
            events=events,
            bid_cancel_intent_by_price=bid_cancel,
            ask_cancel_intent_by_price=ask_cancel,
        )

    def _time_bucket_has_no_arrivals(self) -> bool:
        if self.popularity <= 1e-12:
            return True
        micro = self._microstructure
        event_cluster = self._event_cluster_signal(micro)
        activity = self._clamp(
            micro.activity + micro.activity_event + 0.70 * micro.arrival_cluster,
            0.0,
            5.3,
        ) / 5.3
        drought = self._clamp(micro.liquidity_drought, 0.0, 2.0) / 2.0
        thin_protection = self._clamp((0.55 - self.popularity) / 0.55, 0.0, 1.0)
        continuity_lull = self._clamp(
            0.050
            * (1.0 - 0.45 * event_cluster)
            * (1.0 - 0.72 * thin_protection),
            0.0,
            0.050,
        )
        quiet_clock = self._clamp(
            0.060 / (self.popularity + 0.30)
            + 0.085 * (1.0 - event_cluster) * (1.0 - activity)
            + 0.044 * drought
            + continuity_lull,
            0.0,
            0.34,
        )
        seed = 0 if self._seed is None else int(self._seed)
        tick = int(self.state.step_index) + 1
        clock = (tick * 1103515245 + seed * 12345 + 0x45D9F3B) & 0x7FFFFFFF
        return clock / 0x80000000 < quiet_clock

    def _sample_quote_refresh_events(
        self,
        mdf: MDFState,
        *,
        mdf_price_basis: float,
        current_price: float,
    ) -> tuple[list[_MarketEvent], PriceMap, PriceMap]:
        if self.popularity <= 1e-12:
            return [], {}, {}
        micro = self._microstructure
        event_cluster = self._event_cluster_signal(micro)
        spread_pressure = self._clamp((self._current_spread_ticks() - 1.5) / 5.5, 0.0, 1.0)
        high_participation = self._clamp((self.popularity - 1.0) / 2.5, 0.0, 1.0)
        low_participation = self._clamp((0.80 - self.popularity) / 0.80, 0.0, 1.0)
        refresh_capacity = self._clamp(1.0 - 0.96 * low_participation, 0.12, 1.0)
        events: list[_MarketEvent] = []
        buy_intent: PriceMap = {}
        sell_intent: PriceMap = {}
        for book_side in ("bid", "ask"):
            side_depth = self._nearby_book_depth(book_side, current_price)
            expected = max(0.5 * self._expected_nearby_depth(), 1e-12)
            shortage = self._clamp(1.0 - side_depth / expected, 0.0, 1.0)
            missing_touch = (
                self._best_bid() is None if book_side == "bid" else self._best_ask() is None
            )
            side_vacuum = self._side_book_vacuum(book_side, current_price)
            regime_trend = self._clamp(self._active_settings["trend"] / 0.30, -1.0, 1.0)
            counter_trend_maker = max(
                0.0,
                -regime_trend if book_side == "bid" else regime_trend,
            )
            maintenance_pressure = (
                0.62 * shortage
                + 0.34 * spread_pressure
                + 0.34 * side_vacuum
                + 0.30 * counter_trend_maker * max(shortage, side_vacuum)
                + 0.18 * high_participation
                + 0.12 * event_cluster
            ) * refresh_capacity
            emergency_pressure = (
                0.18 * side_vacuum
                + 0.24 * counter_trend_maker * (side_vacuum + (0.50 if missing_touch else 0.0))
                + (0.42 if missing_touch else 0.0)
            ) * self._clamp(0.55 + 0.45 * refresh_capacity, 0.0, 1.0)
            refresh_pressure = self._clamp(
                maintenance_pressure + emergency_pressure,
                0.0,
                1.0,
            )
            if refresh_pressure <= 1e-12:
                continue
            quiet_book = self._clamp(
                1.0 - max(shortage, spread_pressure, side_vacuum),
                0.0,
                1.0,
            )
            probability = self._clamp(
                0.09
                + refresh_capacity
                * (
                    0.72 * refresh_pressure
                    + 0.14 * self._clamp(micro.resiliency, 0.2, 1.4)
                )
                + (0.10 if missing_touch else 0.0),
                0.0,
                0.92,
            )
            probability *= self._clamp(
                1.0 - 0.62 * quiet_book * (1.0 - event_cluster),
                0.25,
                1.0,
            )
            if self._unit_random() >= probability:
                continue
            weights = self._quote_refresh_weights(
                book_side,
                mdf,
                mdf_price_basis=mdf_price_basis,
                current_price=current_price,
            )
            if not weights and missing_touch:
                fallback_price = (
                    self._snap_price(current_price - self.gap)
                    if book_side == "bid"
                    else self._snap_price(current_price + self.gap)
                )
                if fallback_price >= self._min_price:
                    weights = {fallback_price: 1.0}
            if not weights:
                continue
            expected_events = self._clamp(
                (
                    1.05
                    + 3.60 * refresh_pressure
                    + 0.92 * high_participation
                    + 0.75 * side_vacuum
                )
                * self._clamp(self.popularity, 0.25, 3.0),
                0.0,
                12.0,
            ) * refresh_capacity
            max_events = max(1, int(2 + 5 * refresh_pressure + 3 * high_participation))
            count = min(max_events, self._sample_poisson(expected_events))
            if missing_touch or side_vacuum > 0.78:
                count = max(1, count)
            if missing_touch and counter_trend_maker > 0.5:
                count = max(2, count)
            order_side = "buy" if book_side == "bid" else "sell"
            for _ in range(count):
                price = self._sample_distribution_value(weights)
                volume = self._sample_order_size(
                    passive_liquidity=self._clamp(
                        0.45
                        + 0.55 * refresh_pressure
                        + (0.22 if missing_touch else 0.0)
                        + 0.16 * counter_trend_maker,
                        0.0,
                        1.0,
                    )
                )
                if volume <= 1e-12:
                    continue
                order = _IncomingOrder(order_side, f"{order_side}_entry", price, volume)
                events.append(
                    _MarketEvent(
                        self._event_type_for_incoming_order(order),
                        order.side,
                        order.price,
                        order.volume,
                    )
                )
                target = buy_intent if order_side == "buy" else sell_intent
                target[price] = target.get(price, 0.0) + volume
        return events, self._drop_zeroes(buy_intent), self._drop_zeroes(sell_intent)

    def _quote_refresh_weights(
        self,
        book_side: str,
        mdf: MDFState,
        *,
        mdf_price_basis: float,
        current_price: float,
    ) -> PriceMap:
        basis_tick = self.price_to_tick(mdf_price_basis)
        probabilities = (
            self._project_gap_mdf_to_prices(basis_tick, mdf.buy_entry_mdf)
            if book_side == "bid"
            else self._project_gap_mdf_to_prices(basis_tick, mdf.sell_entry_mdf)
        )
        best_bid = self._best_bid()
        best_ask = self._best_ask()
        low_participation = self._clamp((0.80 - self.popularity) / 0.80, 0.0, 1.0)
        high_participation = self._clamp((self.popularity - 1.0) / 2.5, 0.0, 1.0)
        weights: PriceMap = {}
        for price, probability in probabilities.items():
            if probability <= 0.0:
                continue
            distance = abs(price - current_price) / max(self.gap, 1e-12)
            near_weight = self._clamp((4.0 - distance) / 4.0, 0.0, 1.0)
            if near_weight <= 1e-12:
                continue
            if book_side == "bid":
                if price > current_price + 1e-12:
                    continue
                if best_ask is not None and price >= best_ask - 1e-12:
                    continue
            else:
                if price < current_price - 1e-12:
                    continue
                if best_bid is not None and price <= best_bid + 1e-12:
                    continue
            patient_quote_weight = self._clamp(distance / 4.0, 0.0, 1.0)
            touch_seeking_weight = near_weight ** 1.35
            inner_band_capacity = self._clamp(
                (distance + 0.55) / (0.55 + 1.10 * low_participation),
                0.16 + 0.18 * (1.0 - low_participation),
                1.0,
            )
            weights[price] = probability * (
                0.18
                + touch_seeking_weight * (1.12 - 0.34 * low_participation)
                + 0.72 * high_participation * touch_seeking_weight
                + 0.18 * low_participation * patient_quote_weight
            ) * inner_band_capacity
        return self._normalize_price_map(weights)

    def _sample_entry_side(
        self,
        side: str,
        kind: str,
        target_volume: float,
        entry_mdf: TickMap,
        *,
        basis_price: float,
    ) -> tuple[list[_IncomingOrder], PriceMap]:
        probabilities = self._normalize_tick_map(entry_mdf)
        if target_volume <= 1e-12 or not probabilities:
            return [], {}
        count = self._sample_market_event_count(
            target_volume,
            max_count=max(
                1,
                int(5 + 4.20 * target_volume + 2.0 * self.popularity),
            ),
            micro=self._microstructure,
        )
        orders: list[_IncomingOrder] = []
        intent: PriceMap = {}
        for _ in range(count):
            gap_offset = self._sample_distribution_value(probabilities)
            price = self._price_from_gap_offset(basis_price, gap_offset)
            aggressiveness = self._sampled_order_aggressiveness(side, price, gap_offset)
            passive_liquidity = self._sampled_passive_liquidity_pressure(side, gap_offset)
            volume = self._sample_order_size(
                aggressiveness=aggressiveness,
                passive_liquidity=passive_liquidity,
            )
            if volume <= 1e-12:
                continue
            orders.append(_IncomingOrder(side=side, kind=kind, price=price, volume=volume))
            intent[price] = intent.get(price, 0.0) + volume
        return orders, intent

    def _sample_cancel_events(
        self,
        mdf: MDFState,
        *,
        mdf_price_basis: float,
    ) -> tuple[list[_MarketEvent], PriceMap, PriceMap]:
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
        event_cluster = self._event_cluster_signal(micro)
        no_cancel_probability = self._clamp(
            0.18
            + 0.42 * (1.0 - event_cluster) * (1.0 - 0.35 * pressure)
            - 0.10 * pressure,
            0.0,
            0.68,
        )
        if self._unit_random() < no_cancel_probability:
            return [], {}, {}
        events: list[_MarketEvent] = []
        bid_intent: PriceMap = {}
        ask_intent: PriceMap = {}
        for book_side, volumes, intent in (
            ("bid", self._orderbook.bid_volume_by_price, bid_intent),
            ("ask", self._orderbook.ask_volume_by_price, ask_intent),
        ):
            side_depth = self._nearby_book_depth(book_side, self.state.price)
            side_vacuum = self._clamp(
                1.0 - side_depth / max(0.5 * self._expected_nearby_depth(), 1e-12),
                0.0,
                1.0,
            )
            support = (
                self._project_gap_mdf_to_prices(
                    self.price_to_tick(mdf_price_basis),
                    mdf.buy_entry_mdf,
                )
                if book_side == "bid"
                else self._project_gap_mdf_to_prices(
                    self.price_to_tick(mdf_price_basis),
                    mdf.sell_entry_mdf,
                )
            )
            overhang = self._book_overhang(book_side, volumes, support)
            displaced_overhang = self._displaced_book_overhang(volumes)
            displaced_event_pressure = self._clamp(
                (displaced_overhang - 0.22) / 0.78,
                0.0,
                1.0,
            )
            side_pressure = (
                pressure + 0.5 * overhang + 0.38 * displaced_event_pressure
            ) * self._clamp(
                1.0 - 0.48 * side_vacuum,
                0.30,
                1.0,
            )
            side_cancel_probability = self._clamp(
                0.10
                + 0.40 * side_pressure
                + 0.12 * event_cluster
                + 0.14 * overhang
                + 0.18 * displaced_event_pressure,
                0.03,
                0.72,
            )
            if self._unit_random() >= side_cancel_probability:
                continue
            level_weights: PriceMap = {}
            available_by_price: PriceMap = {}
            for price, book_volume in list(volumes.items()):
                available = max(0.0, book_volume - intent.get(price, 0.0))
                if available <= 1e-12:
                    continue
                hazard = self._cancel_level_hazard(
                    book_side,
                    price,
                    available,
                    support,
                    side_pressure,
                )
                if hazard <= 1e-12:
                    continue
                level_weights[price] = available * hazard * (0.35 + hazard)
                available_by_price[price] = available
            level_probabilities = self._normalize_price_map(level_weights)
            if not level_probabilities:
                continue
            max_events = max(
                1,
                min(
                    len(level_probabilities),
                    int(
                        1
                        + 4.0 * side_pressure
                        + 2.0 * event_cluster
                        + 3.0 * overhang
                        + 2.0 * displaced_event_pressure
                    ),
                ),
            )
            expected_events = self._clamp(
                (
                    0.18
                    + 1.70 * side_pressure
                    + 0.72 * event_cluster
                    + 1.20 * overhang
                    + 0.90 * displaced_event_pressure
                )
                * self._clamp(len(level_probabilities) / 8.0, 0.45, 1.45),
                0.0,
                float(max_events),
            )
            event_count = min(max_events, self._sample_poisson(expected_events))
            for _ in range(event_count):
                if not level_probabilities:
                    break
                price = self._sample_distribution_value(level_probabilities)
                available = max(
                    0.0,
                    min(
                        available_by_price.get(price, 0.0),
                        volumes.get(price, 0.0) - intent.get(price, 0.0),
                    ),
                )
                if available <= 1e-12:
                    level_probabilities = self._normalize_price_map(
                        {
                            level_price: probability
                            for level_price, probability in level_probabilities.items()
                            if level_price != price
                        }
                    )
                    continue
                stale_distance = abs(price - (max(volumes) if book_side == "bid" else min(volumes)))
                stale_distance /= max(self.gap, 1e-12)
                price_stale_distance = abs(price - self.state.price) / max(self.gap, 1e-12)
                stale_size = self._clamp(
                    max(stale_distance - 3.0, price_stale_distance - 4.0) / 8.0,
                    0.0,
                    1.0,
                )
                volume = min(
                    available,
                    self._sample_order_size()
                    * (0.25 + 0.95 * side_pressure + 1.35 * stale_size),
                )
                event_type = "bid_cancel" if book_side == "bid" else "ask_cancel"
                events.append(_MarketEvent(event_type, book_side, price, volume))
                intent[price] = intent.get(price, 0.0) + volume
                remaining = available - volume
                if remaining <= 1e-12:
                    level_probabilities = self._normalize_price_map(
                        {
                            level_price: probability
                            for level_price, probability in level_probabilities.items()
                            if level_price != price
                        }
                    )
                else:
                    available_by_price[price] = remaining
        return events, self._drop_zeroes(bid_intent), self._drop_zeroes(ask_intent)

    def _cancel_level_hazard(
        self,
        side: str,
        price: float,
        available: float,
        support: PriceMap,
        side_pressure: float,
    ) -> float:
        volumes = self._orderbook.volumes_for_side(side)
        if not volumes or available <= 1e-12:
            return 0.0
        support = self._normalized_price_probabilities(support)
        touch = max(volumes) if side == "bid" else min(volumes)
        distance = abs(price - touch) / max(self.gap, 1e-12)
        current_distance = abs(price - self.state.price) / max(self.gap, 1e-12)
        unsupported_share = max(0.0, self._book_share(side, price) - support.get(price, 0.0))
        age_pressure = self._quote_staleness(side, price)
        adverse_pressure = self._adverse_quote_pressure(side, price)
        low_participation = self._clamp((0.80 - self.popularity) / 0.80, 0.0, 1.0)
        thin_inside_churn = (
            low_participation
            * self._clamp((2.5 - distance) / 2.5, 0.0, 1.0)
            * self._clamp(
                0.35
                + 0.45 * self._microstructure.liquidity_drought / 2.0
                + 0.25 * self._microstructure.cancel_burst / 2.0,
                0.0,
                1.0,
            )
        )
        near_touch_protection = self._clamp((2.5 - distance) / 2.5, 0.0, 1.0)
        deep_stale_pressure = self._clamp(
            (distance - 2.5) / max(1.25, 0.26 * self.grid_radius),
            0.0,
            1.0,
        )
        displaced_stale_pressure = self._clamp(
            (current_distance - 3.5) / max(1.6, 0.30 * self.grid_radius),
            0.0,
            1.0,
        )
        level_pressure = self._clamp(
            0.08 * side_pressure
            + 0.70 * deep_stale_pressure
            + 0.54 * displaced_stale_pressure
            + 0.22 * unsupported_share
            + 0.22 * age_pressure
            + 0.18 * adverse_pressure
            + 0.18 * thin_inside_churn
            + 0.10
            * self._clamp(self._microstructure.liquidity_drought, 0.0, 2.0)
            / 2.0
            * (1.0 / max(1.0, distance)),
            0.0,
            0.88,
        )
        return level_pressure * self._clamp(1.0 - 0.58 * near_touch_protection, 0.28, 1.0)

    def _book_overhang(self, side: str, volumes: PriceMap, support: PriceMap) -> float:
        total = sum(max(0.0, volume) for volume in volumes.values())
        if total <= 1e-12:
            return 0.0
        support = self._normalized_price_probabilities(support)
        overhang = 0.0
        for price, volume in volumes.items():
            share = max(0.0, volume) / total
            overhang += max(0.0, share - support.get(price, 0.0))
        return self._clamp(overhang, 0.0, 1.0)

    def _displaced_book_overhang(self, volumes: PriceMap) -> float:
        total = sum(max(0.0, volume) for volume in volumes.values())
        if total <= 1e-12:
            return 0.0
        overhang = 0.0
        for price, volume in volumes.items():
            distance = abs(price - self.state.price) / max(self.gap, 1e-12)
            displacement = self._clamp((distance - 5.0) / 9.0, 0.0, 1.0)
            overhang += max(0.0, volume) / total * displacement
        return self._clamp(overhang, 0.0, 1.0)

    def _book_share(self, side: str, price: float) -> float:
        volumes = self._orderbook.volumes_for_side(side)
        total = sum(max(0.0, volume) for volume in volumes.values())
        if total <= 1e-12:
            return 0.0
        return max(0.0, volumes.get(price, 0.0)) / total

    def _event_type_for_incoming_order(self, order: _IncomingOrder) -> str:
        if order.side == "buy":
            best_ask = self._best_ask()
            if best_ask is None and self._snap_price(order.price) >= self.state.price + self.gap:
                return "buy_marketable"
            return (
                "buy_marketable"
                if best_ask is not None and self._snap_price(order.price) >= best_ask - 1e-12
                else "buy_limit_add"
            )
        best_bid = self._best_bid()
        if best_bid is None and self._snap_price(order.price) <= self.state.price - self.gap:
            return "sell_marketable"
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
        if cache is not None and cache.mdf is mdf and side in cache.entry_support_by_book_side:
            return cache.entry_support_by_book_side[side]
        basis_price = cache.basis_price if cache is not None else self.state.mdf_price_basis
        entry_mdf = (
            self._project_gap_mdf_to_prices(self.price_to_tick(basis_price), mdf.buy_entry_mdf)
            if side == "bid"
            else self._project_gap_mdf_to_prices(
                self.price_to_tick(basis_price),
                mdf.sell_entry_mdf,
            )
        )
        probabilities = self._normalized_price_probabilities(entry_mdf)
        if cache is not None and cache.mdf is mdf:
            cache.entry_support_by_book_side[side] = probabilities
        return probabilities

    def _mean_child_order_size(self) -> float:
        popularity_scale = self._clamp(self.popularity, 0.0, 4.0) ** 0.5
        return self._clamp(0.105 + 0.014 * popularity_scale, 0.085, 0.170)

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
        arrival = self._clamp(micro.arrival_cluster, 0.0, 2.2) / 2.2
        persistence = self._clamp(micro.flow_persistence, 0.0, 1.45) / 1.45
        volatility = self._clamp(micro.volatility_cluster, 0.0, 2.0) / 2.0
        drought = self._clamp(micro.liquidity_drought, 0.0, 2.0) / 2.0
        resiliency = self._clamp(micro.resiliency, 0.2, 1.4)
        raw_vacuum = self._clamp(micro.book_vacuum, 0.0, 1.8) / 1.8
        vacuum = self._clamp((raw_vacuum - 0.42) / 0.58, 0.0, 1.0)
        churn = self._clamp(micro.churn_pressure, 0.0, 2.0) / 2.0
        displacement = self._clamp(micro.displacement_pressure, 0.0, 2.0) / 2.0
        cluster = self._clamp(
            0.34 * activity + 0.34 * arrival + 0.18 * persistence + 0.14 * volatility,
            0.0,
            1.0,
        )
        stress_arrival = self._clamp(
            0.32 * volatility + 0.26 * displacement + 0.22 * arrival + 0.20 * persistence,
            0.0,
            1.0,
        )
        vacuum_probe = self._clamp(
            vacuum * (0.30 + 0.70 * (1.0 - drought)) * (0.35 + 0.65 * resiliency),
            0.0,
            1.0,
        )
        passive_capacity = self._clamp(resiliency * (1.0 - 0.45 * drought), 0.20, 1.40)
        base_quote_renewal = self._clamp(
            expected
            * (0.07 + 0.12 * resiliency + 0.08 * vacuum_probe)
            * (1.0 - 0.28 * drought)
            / passive_capacity,
            0.0,
            max_count,
        )
        clustered_arrivals = self._clamp(
            expected
            * (
                0.18 * cluster
                + 0.14 * arrival
                + 0.14 * volatility
                + 0.14 * displacement
                + 0.08 * persistence
            )
            * (1.0 + 0.45 * cluster + 0.55 * stress_arrival)
            * (1.0 - 0.22 * churn),
            0.0,
            max_count,
        )
        quiet_zero_pressure = self._clamp(
            0.38 + 0.42 * drought + 0.30 * churn - 0.28 * vacuum - 0.34 * cluster,
            0.0,
            0.90,
        )
        popularity_silence = self._clamp(0.13 / (self.popularity + 0.30), 0.025, 0.34)
        slice_silence = self._clamp(
            0.065 / (self.popularity + 0.42) * (1.0 - 0.58 * cluster),
            0.0,
            0.085,
        )
        zero_probability = (
            0.120
            + quiet_zero_pressure / (1.0 + 0.55 * expected)
            + popularity_silence * (1.0 - 0.75 * cluster)
            + slice_silence
        )
        if self._unit_random() < self._clamp(zero_probability, 0.0, 0.80):
            return 0
        probe_arrivals = self._clamp(
            expected
            * 0.18
            * vacuum_probe
            * (1.0 - 0.35 * churn)
            * self._clamp((0.70 - self.popularity) / 0.70, 0.0, 1.0),
            0.0,
            max_count,
        )
        mean = base_quote_renewal + clustered_arrivals + probe_arrivals
        if mean <= 1e-12:
            return 0
        overdispersion = self._clamp(
            0.20 + 1.45 * cluster + 0.70 * volatility + 0.85 * stress_arrival,
            0.20,
            3.20,
        )
        shape = max(0.32, mean / overdispersion)
        sampled_mean = self._rng.gammavariate(shape, mean / shape)
        count = self._sample_poisson(sampled_mean)
        return min(max_count, count)

    def _sample_order_size(
        self,
        *,
        aggressiveness: float = 0.0,
        passive_liquidity: float = 0.0,
    ) -> float:
        base_mean = self._mean_child_order_size()
        micro = self._microstructure
        aggressiveness = self._clamp(aggressiveness, 0.0, 1.0)
        passive_liquidity = self._clamp(passive_liquidity, 0.0, 1.0)
        activity = self._clamp(
            micro.activity + micro.activity_event + 0.70 * micro.arrival_cluster,
            0.0,
            5.3,
        ) / 5.3
        stress = self._clamp(
            micro.liquidity_stress
            + micro.participation_burst
            + 0.55 * micro.volatility_cluster
            + 0.36 * micro.displacement_pressure
            + 0.18 * micro.churn_pressure,
            0.0,
            6.2,
        ) / 6.2
        regime_volatility = self._clamp(
            (self._active_settings["volatility"] - 0.78) / 0.97,
            0.0,
            1.0,
        )
        mean = base_mean * (
            1.0
            + passive_liquidity
            * (
                0.70
                + 0.34 * self._clamp((self.popularity - 1.0) / 2.5, 0.0, 1.0)
                - 0.42 * self._clamp((0.80 - self.popularity) / 0.80, 0.0, 1.0)
                + 0.28 * self._clamp(micro.resiliency, 0.2, 1.4)
                - 0.20 * self._clamp(micro.liquidity_drought, 0.0, 2.0) / 2.0
            )
            + aggressiveness
            * (
                0.54
                + 0.36 * activity
                + 0.40 * stress
                + 0.54 * regime_volatility
            )
        )
        sigma = (
            0.34
            + 0.22 * activity
            + 0.24 * stress
            + 0.10 * regime_volatility
            + 0.09 * aggressiveness
            - 0.08 * passive_liquidity
        )
        size = mean * self._rng.lognormvariate(-0.5 * sigma * sigma, sigma)
        tail_probability = (
            0.010
            + 0.055 * stress
            + 0.035 * regime_volatility
            + 0.038 * aggressiveness
        )
        if self._unit_random() < tail_probability:
            size += mean * self._rng.paretovariate(
                1.95 + 0.65 * (1.0 - max(stress, aggressiveness))
            )
        return self._clamp(size, base_mean * 0.010, base_mean * (16.0 + 8.0 * aggressiveness))

    def _sampled_order_aggressiveness(self, side: str, price: float, gap_offset: float) -> float:
        if side == "buy":
            directional_offset = max(0.0, gap_offset)
            touch = self._best_ask()
            crossed_ticks = (
                0.0 if touch is None else max(0.0, (self._snap_price(price) - touch) / self.gap)
            )
            directional_flow = self._clamp(
                self._active_settings["trend"] + 0.35 * self._microstructure.meta_order_side,
                -1.0,
                1.0,
            )
        else:
            directional_offset = max(0.0, -gap_offset)
            touch = self._best_bid()
            crossed_ticks = (
                0.0 if touch is None else max(0.0, (touch - self._snap_price(price)) / self.gap)
            )
            directional_flow = self._clamp(
                -self._active_settings["trend"]
                - 0.35 * self._microstructure.meta_order_side,
                -1.0,
                1.0,
            )
        return self._clamp(
            0.34 * directional_offset / max(1.0, 0.45 * self.grid_radius)
            + 0.66 * crossed_ticks / 4.0
            + 0.24
            * max(0.0, directional_flow)
            * self._clamp(directional_offset / 3.0, 0.0, 1.0),
            0.0,
            1.0,
        )

    def _sampled_passive_liquidity_pressure(self, side: str, gap_offset: float) -> float:
        is_passive = gap_offset <= 0 if side == "buy" else gap_offset >= 0
        if not is_passive:
            return 0.0
        distance = abs(float(gap_offset))
        near_touch = self._clamp((3.0 - distance) / 3.0, 0.0, 1.0)
        if near_touch <= 1e-12:
            return 0.0
        side_depth = self._nearby_book_depth("bid" if side == "buy" else "ask", self.state.price)
        shortage = self._clamp(
            1.0 - side_depth / max(0.5 * self._expected_nearby_depth(), 1e-12),
            0.0,
            1.0,
        )
        low_participation = self._clamp((0.80 - self.popularity) / 0.80, 0.0, 1.0)
        return near_touch * self._clamp(
            0.42 + 0.48 * shortage + 0.18 * self._microstructure.resiliency,
            0.0,
            1.0,
        ) * self._clamp(1.0 - 0.86 * low_participation, 0.16, 1.18)

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

    def _sample_distribution_value(self, probabilities: TickMap) -> float:
        if not probabilities:
            raise ValueError("probabilities must not be empty")
        draw = self._unit_random()
        cumulative = 0.0
        fallback = next(reversed(probabilities))
        for value, probability in probabilities.items():
            cumulative += probability
            if draw <= cumulative:
                return value
        return fallback

    def _event_cluster_signal(self, micro: _MicrostructureState) -> float:
        return self._clamp(
            0.48 * micro.activity_event / 1.8
            + 0.24 * micro.activity / 2.0
            + 0.20 * micro.arrival_cluster / 2.2
            + 0.14 * micro.cancel_pressure / 2.0
            + 0.08 * micro.spread_pressure / 1.8
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
        if best_bid is not None and best_ask is not None:
            spread_ticks = max(1.0, (best_ask - best_bid) / self.gap)
        else:
            spread_ticks = (
                2.0
                + 1.4 * self._clamp(self._microstructure.liquidity_stress, 0.0, 2.0)
                + 1.2 * self._clamp(self._microstructure.book_vacuum, 0.0, 1.8)
                + 0.8 * self._clamp(self._microstructure.liquidity_drought, 0.0, 2.0)
            )
        bid_volume = sum(max(0.0, volume) for volume in orderbook.bid_volume_by_price.values())
        ask_volume = sum(max(0.0, volume) for volume in orderbook.ask_volume_by_price.values())
        target = max(1.0, self.popularity * self._active_settings["liquidity"])
        bid_shape = self._book_shape_signals(price, "bid", orderbook.bid_volume_by_price)
        ask_shape = self._book_shape_signals(price, "ask", orderbook.ask_volume_by_price)
        bid_vacuum = self._side_book_vacuum("bid", price)
        ask_vacuum = self._side_book_vacuum("ask", price)
        return MDFSignals(
            orderbook_imbalance=self._realized_flow.flow_imbalance,
            last_return_ticks=self._realized_flow.return_ticks,
            last_execution_volume=self._realized_flow.execution_volume,
            executed_volume_by_tick=self._price_map_to_relative_ticks(
                price, self._realized_flow.executed_by_price
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
            bid_gap_by_tick=bid_shape["gap"],
            ask_gap_by_tick=ask_shape["gap"],
            bid_front_by_tick=bid_shape["front"],
            ask_front_by_tick=ask_shape["front"],
            bid_occupancy_by_tick=bid_shape["occupancy"],
            ask_occupancy_by_tick=ask_shape["occupancy"],
            bid_depth_pressure=(bid_volume - target) / target,
            ask_depth_pressure=(ask_volume - target) / target,
            book_vacuum=self._microstructure.book_vacuum,
            bid_vacuum=bid_vacuum,
            ask_vacuum=ask_vacuum,
            churn_pressure=self._microstructure.churn_pressure,
            displacement_pressure=self._microstructure.displacement_pressure,
            spread_ticks=spread_ticks,
            spread_pressure=self._microstructure.spread_pressure,
            cancel_pressure=self._microstructure.cancel_pressure,
            liquidity_stress=self._microstructure.liquidity_stress,
            stress_side=self._microstructure.stress_side,
            resiliency=self._microstructure.resiliency,
            activity=self._microstructure.activity,
            activity_event=self._microstructure.activity_event,
            arrival_cluster=self._microstructure.arrival_cluster,
            flow_persistence=self._microstructure.flow_persistence,
            meta_order_side=self._microstructure.meta_order_side,
            volatility_cluster=self._microstructure.volatility_cluster,
            participation_burst=self._microstructure.participation_burst,
            liquidity_drought=self._microstructure.liquidity_drought,
            cancel_burst=self._microstructure.cancel_burst,
        )

    def _side_book_vacuum(self, side: str, price: float) -> float:
        depth = self._nearby_book_depth(side, price)
        expected = max(0.5 * self._expected_nearby_depth(), 1e-12)
        vacuum = self._clamp(1.0 - depth / expected, 0.0, 1.0)
        if side == "bid" and self._best_bid() is None:
            return 1.0
        if side == "ask" and self._best_ask() is None:
            return 1.0
        return vacuum

    def _book_shape_signals(
        self,
        basis_price: float,
        side: str,
        volumes: PriceMap,
    ) -> dict[str, TickMap]:
        depth = self._price_map_to_relative_ticks(basis_price, volumes)
        ticks = [
            tick
            for tick in self.relative_tick_grid()
            if (tick < 0 if side == "bid" else tick > 0)
        ]
        if not ticks:
            return {"gap": {}, "front": {}, "occupancy": {}}
        touch_tick = max(depth) if side == "bid" and depth else min(depth) if depth else None
        expected_scale = max(0.08, self.popularity * self._active_settings["liquidity"])
        gap: TickMap = {}
        front: TickMap = {}
        occupancy: TickMap = {}
        for tick in ticks:
            expected = expected_scale / (
                max(1.0, abs(float(tick))) ** self._active_settings["depth_exponent"]
            )
            actual = max(0.0, depth.get(tick, 0.0))
            gap[tick] = 1.0 if actual <= 1e-12 else 0.0
            occupancy[tick] = self._clamp(actual / max(expected, 1e-12), 0.0, 3.0)
            if touch_tick is None:
                front[tick] = 0.0
            else:
                front[tick] = exp(-abs(float(tick - touch_tick)) / 1.5)
        return {"gap": gap, "front": front, "occupancy": occupancy}

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

    def _project_gap_mdf_to_prices(self, current_tick: int, values: TickMap) -> PriceMap:
        projected: PriceMap = {}
        for gap_offset, probability in values.items():
            absolute_tick = current_tick + int(round(gap_offset))
            if absolute_tick < 1:
                continue
            price = self._clean_number(absolute_tick * self.gap)
            projected[price] = projected.get(price, 0.0) + probability
        return self._normalize_price_map(projected)

    def _constrain_gap_mdf(self, current_tick: int, values: TickMap) -> TickMap:
        return self._normalize_tick_map(
            {
                gap_offset: probability
                for gap_offset, probability in values.items()
                if current_tick + int(round(gap_offset)) >= 1
            }
        )

    def _price_from_gap_offset(self, basis_price: float, gap_offset: float) -> float:
        basis_tick = self.price_to_tick(basis_price)
        executable_tick = max(1, basis_tick + int(round(gap_offset)))
        return self.tick_to_price(executable_tick)

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
            return {tick: 0.0 for tick in values}
        return {tick: clean.get(tick, 0.0) / total for tick in values}

    def _unit_random(self) -> float:
        return self._rng.random()
