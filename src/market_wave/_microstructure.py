from __future__ import annotations

from math import exp, log1p

from ._types import _MicrostructureInputs, _MicrostructureState, _ParticipantPressureState
from .state import IntensityState, LatentState


class _MarketMicrostructureMixin:
    def _initial_latent(self) -> LatentState:
        settings = self._active_settings
        mood = self._clamp(
            self._rng.gauss(settings["mood"], 0.18),
            -0.7,
            0.7,
        )
        trend = self._clamp(
            self._rng.gauss(settings["trend"] + 0.20 * mood, 0.14),
            -0.7,
            0.7,
        )
        volatility = self._clamp(
            0.12
            + 0.18 * self._rng.random()
            + abs(self._rng.gauss(0.0, 0.11))
            + settings["volatility_bias"],
            0.05,
            0.75,
        )
        return LatentState(mood=mood, trend=trend, volatility=volatility)

    def _next_latent(self, latent: LatentState) -> LatentState:
        regime = self._active_settings
        micro = self._microstructure
        mood_noise = self._rng.gauss(0.0, 0.11)
        trend_noise = self._rng.gauss(0.0, 0.08)
        jump_probability = 0.024 * regime["volatility"] * (1.0 + self.augmentation_strength)
        jump = self._rng.gauss(0.0, 0.55) if self._rng.random() < jump_probability else 0.0
        directional_memory = micro.meta_order_side * (0.45 + 0.35 * micro.flow_persistence)
        directional_persistence = regime["directional_persistence"]
        return_momentum = self._clamp(
            self._realized_flow.return_ticks / 4.0,
            -1.0,
            1.0,
        ) * self._clamp(
            0.45 * micro.volatility_cluster + 0.35 * micro.participation_burst,
            0.0,
            1.2,
        )
        signed_flow = (
            0.075 * self._realized_flow.flow_imbalance
            + 0.115 * self._realized_flow.intent_imbalance
            + 0.060 * self._realized_flow.rested_imbalance
            + 0.030 * directional_memory
            + 0.015 * return_momentum
        ) * directional_persistence * self._clamp(0.66 + 0.22 * directional_persistence, 0.58, 1.0)
        exhaustion = self._trend_exhaustion(micro)
        squeeze_impulse = self._squeeze_impulse(micro)
        mood_target = regime["mood"] + 0.06 * latent.trend - 0.025 * exhaustion

        mood = self._clamp(
            0.74 * latent.mood
            + 0.26 * mood_target
            + signed_flow
            + mood_noise
            + 0.22 * jump,
            -1.0,
            1.0,
        )
        trend_target = regime["trend"] + 0.10 * mood
        trend_target += (
            0.05 * squeeze_impulse + 0.060 * directional_memory * directional_persistence
        )
        trend_target -= regime["trend_exhaustion"] * exhaustion * (
            1.0 if trend_target >= 0.0 else -1.0
        )
        trend = self._clamp(
            0.74 * latent.trend
            + 0.26 * trend_target
            + 0.030 * return_momentum
            + 0.045 * directional_memory * directional_persistence
            + 0.08 * jump
            + trend_noise,
            -1.0,
            1.0,
        )
        noise_shock = max(0.0, abs(mood_noise) - 0.08)
        shock = 0.32 * noise_shock + abs(jump) * 0.24
        execution_pressure = min(
            self._realized_flow.execution_volume / max(0.7, self.popularity),
            4.0,
        )
        realized = (
            0.050 * min(self._realized_flow.abs_return_ticks, 3.0)
            + 0.010 * execution_pressure
            + 0.050 * micro.volatility_cluster
        )
        volatility_target = 0.28 + 0.22 * regime["volatility"] + regime["volatility_bias"]
        volatility = self._clamp(
            0.88 * latent.volatility
            + 0.12 * volatility_target
            + shock
            + realized
            + 0.014 * abs(self._realized_flow.flow_imbalance)
            + 0.050 * abs(self._realized_flow.intent_imbalance)
            + 0.028 * abs(self._realized_flow.rested_imbalance)
            + 0.020 * micro.flow_persistence,
            0.04,
            1.55,
        )
        return LatentState(mood=mood, trend=trend, volatility=volatility)

    def _trend_exhaustion(self, micro: _MicrostructureState) -> float:
        directional_stretch = self._clamp(
            0.78 * self._clamp(micro.recent_signed_return / 5.0, -1.0, 1.0)
            + 0.22 * micro.recent_flow_imbalance,
            -1.0,
            1.0,
        )
        trend_side = 1.0 if micro.meta_order_side >= 0.0 else -1.0
        return max(0.0, trend_side * directional_stretch)

    def _squeeze_impulse(self, micro: _MicrostructureState) -> float:
        return self._clamp(micro.squeeze_pressure, -1.5, 1.5)

    def _next_intensity(
        self,
        latent: LatentState,
        micro: _MicrostructureState | None = None,
    ) -> IntensityState:
        regime = self._active_settings
        micro = micro or self._microstructure
        augmentation = 1.0 + self.augmentation_strength * self._rng.uniform(-0.18, 0.28)
        flow_reversal = self._flow_reversal_pressure(micro)
        activity = micro.activity
        arrival_cluster = self._clamp(micro.arrival_cluster, 0.0, 2.2)
        participation_burst = self._clamp(micro.participation_burst, 0.0, 2.0)
        liquidity_drought = self._clamp(micro.liquidity_drought, 0.0, 2.0)
        cancel_burst = self._clamp(micro.cancel_burst, 0.0, 2.0)
        activity_multiplier = (
            1.0
            + 0.26 * self._clamp(activity, 0.0, 2.2)
            + 0.18 * participation_burst
            + 0.18 * arrival_cluster
        )
        event_multiplier = (
            1.0
            + 0.14 * self._clamp(micro.activity_event, 0.0, 1.8)
            + 0.12 * participation_burst
            + 0.18 * arrival_cluster
        )
        dry_up = self._clamp(
            1.0 - 0.07 * micro.cancel_pressure - 0.06 * liquidity_drought - 0.035 * cancel_burst,
            0.66,
            1.0,
        )
        raw_total = (
            self.popularity
            * (1.0 + 1.35 * latent.volatility)
            * regime["intensity"]
            * augmentation
            * activity_multiplier
            * event_multiplier
            * dry_up
        )
        previous_total = self.state.intensity.total
        total = (
            0.55 * raw_total + 0.45 * previous_total
            if previous_total > 1e-12
            else raw_total
        )
        if previous_total > 1e-12:
            previous_pressure = self._clamp(
                previous_total / max(self.popularity * regime["intensity"], 1e-12) - 1.0,
                0.0,
                2.0,
            )
            total *= 1.0 + 0.08 * previous_pressure
        directional_memory = micro.meta_order_side * (
            0.42 + 0.58 * self._clamp(micro.flow_persistence, 0.0, 1.45) / 1.45
        )
        directional_persistence = regime["directional_persistence"]
        participant = self._participant_pressure
        continuation = self._clamp(participant.flow_continuation, 0.0, 1.8) / 1.8
        intent_memory = self._clamp(participant.signed_intent_memory, -1.0, 1.0)
        momentum_state = self._clamp(
            0.45 * micro.volatility_cluster + 0.35 * micro.participation_burst,
            0.0,
            1.2,
        ) / 1.2
        signed_return_momentum = self._clamp(self._realized_flow.return_ticks / 4.0, -1.0, 1.0)
        reference_pressure = self._clamp(
            (self._reference_price - self.state.price) / max(self.gap * self.grid_radius, 1e-12),
            -1.0,
            1.0,
        )
        directional_scale = self._clamp(
            0.55
            + 0.45 * abs(regime["trend"]) / 0.30
            + 0.16 * micro.flow_persistence / 1.45,
            0.55,
            1.0,
        )
        buy_ratio = self._clamp(
            0.5
            + directional_scale
            * (
                0.20 * latent.mood
                + 0.145 * latent.trend
                + 0.040 * self._realized_flow.flow_imbalance
                + 0.080 * directional_memory * directional_persistence
                + 0.055 * momentum_state * signed_return_momentum
                + 0.065 * continuation * intent_memory
            )
            + 0.025 * self._squeeze_impulse(micro)
            - 0.30 * regime["flow_reversal"] * flow_reversal
            + 0.055 * reference_pressure,
            0.08,
            0.92,
        )
        sell_ratio = 1.0 - buy_ratio
        return IntensityState(
            total=total,
            buy=total * buy_ratio,
            sell=total * sell_ratio,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
        )

    def _next_participant_pressure(
        self,
        latent: LatentState,
        micro: _MicrostructureState,
        pre_imbalance: float,
    ) -> _ParticipantPressureState:
        previous = self._participant_pressure
        realized = self._realized_flow
        bid_depth = self._nearby_book_depth("bid", self.state.price)
        ask_depth = self._nearby_book_depth("ask", self.state.price)
        total_depth = bid_depth + ask_depth
        expected_depth = self._expected_nearby_depth()
        bid_depth_ratio = bid_depth / max(0.5 * expected_depth, 1e-12)
        ask_depth_ratio = ask_depth / max(0.5 * expected_depth, 1e-12)
        depth_shortage = self._clamp(1.0 - total_depth / max(expected_depth, 1e-12), 0.0, 1.0)
        depth_surplus = self._clamp(total_depth / max(expected_depth, 1e-12) - 1.0, 0.0, 2.0)
        depth_imbalance = 0.0 if total_depth <= 1e-12 else (bid_depth - ask_depth) / total_depth
        bid_depletion = self._clamp(1.0 - bid_depth_ratio, 0.0, 1.0)
        ask_depletion = self._clamp(1.0 - ask_depth_ratio, 0.0, 1.0)
        signed_return = self._clamp(realized.return_ticks / 4.0, -1.0, 1.0)
        signed_flow = self._clamp(realized.flow_imbalance, -1.0, 1.0)
        signed_intent = self._clamp(
            0.46 * realized.intent_imbalance
            + 0.30 * realized.rested_imbalance
            + 0.24 * signed_flow,
            -1.0,
            1.0,
        )
        submitted_total = realized.submitted_buy_volume + realized.submitted_sell_volume
        rested_total = realized.rested_buy_volume + realized.rested_sell_volume
        sampled_confidence = self._clamp(
            log1p((submitted_total + 0.60 * rested_total) / max(0.35, self.popularity))
            / 1.35,
            0.0,
            1.0,
        )
        bid_cancel = sum(realized.bid_cancelled_by_price.values())
        ask_cancel = sum(realized.ask_cancelled_by_price.values())
        cancel_total = bid_cancel + ask_cancel
        cancel_imbalance = (
            0.0 if cancel_total <= 1e-12 else (ask_cancel - bid_cancel) / cancel_total
        )
        trend = self._clamp(
            0.66 * latent.trend + 0.20 * latent.mood + 0.22 * micro.meta_order_side,
            -1.0,
            1.0,
        )
        pressure_noise = self._rng.gauss(0.0, 0.16 + 0.08 * micro.activity_event)
        sampled_intent_evidence = self._clamp(
            0.58 * realized.intent_imbalance
            + 0.32 * realized.rested_imbalance
            + 0.10 * signed_flow,
            -1.0,
            1.0,
        )
        memory_decay = 0.86 - 0.14 * sampled_confidence
        intent_memory = self._clamp(
            memory_decay * previous.signed_intent_memory
            + (1.0 - memory_decay) * sampled_intent_evidence * sampled_confidence,
            -1.0,
            1.0,
        )
        execution_alignment = max(0.0, signed_intent * signed_flow)
        return_alignment = max(0.0, signed_intent * signed_return)
        meta_alignment = max(0.0, signed_intent * self._clamp(micro.meta_order_side, -1.0, 1.0))
        absorbed_intent = self._clamp(
            abs(signed_intent)
            * (1.0 - 0.55 * execution_alignment - 0.45 * return_alignment),
            0.0,
            1.0,
        )
        same_side_depletion = abs(signed_intent) * (
            bid_depletion if signed_intent < 0.0 else ask_depletion
        )
        continuation_evidence = self._clamp(
            0.36 * execution_alignment
            + 0.28 * return_alignment
            + 0.18 * meta_alignment
            + 0.16 * micro.flow_persistence / 1.45 * (0.4 + 0.6 * execution_alignment)
            + 0.24 * sampled_confidence * max(0.0, sampled_intent_evidence * intent_memory)
            + 0.18
            * sampled_confidence
            * max(0.0, realized.rested_imbalance * signed_intent)
            + 0.16 * micro.participation_burst / 2.0
            + 0.08 * max(0.0, abs(signed_return) - 0.10) * (0.5 + return_alignment),
            0.0,
            1.8,
        )
        absorption_evidence = self._clamp(
            0.20 * abs(realized.rested_imbalance) * (1.0 - 0.55 * execution_alignment)
            + 0.28 * depth_surplus
            + 0.30 * absorbed_intent
            + 0.20 * micro.spread_pressure / 1.8,
            0.0,
            1.8,
        )
        exhaustion_evidence = self._clamp(
            0.26 * micro.flow_persistence / 1.45
            + 0.34 * absorption_evidence / 1.8
            + 0.42 * max(0.0, abs(signed_return) - 0.28)
            + 0.24 * self._clamp(abs(micro.recent_signed_return) / 6.0, 0.0, 1.0)
            + 0.18 * depth_surplus
            + 0.32 * same_side_depletion
            - 0.16 * continuation_evidence / 1.8,
            0.0,
            1.8,
        )

        upward_evidence = self._clamp(
            0.38 * max(0.0, trend)
            + 0.28 * max(0.0, signed_flow)
            + 0.30 * max(0.0, intent_memory)
            + 0.26 * max(0.0, intent_memory) * continuation_evidence / 1.8
            + 0.22 * max(0.0, signed_return)
            + 0.16 * max(0.0, cancel_imbalance)
            + 0.16 * max(0.0, -depth_imbalance),
            0.0,
            1.6,
        ) * self._clamp(1.0 - 0.20 * ask_depth_ratio + 0.18 * depth_shortage, 0.55, 1.25)
        downward_evidence = self._clamp(
            0.38 * max(0.0, -trend)
            + 0.28 * max(0.0, -signed_flow)
            + 0.30 * max(0.0, -intent_memory)
            + 0.26 * max(0.0, -intent_memory) * continuation_evidence / 1.8
            + 0.22 * max(0.0, -signed_return)
            + 0.16 * max(0.0, -cancel_imbalance)
            + 0.16 * max(0.0, depth_imbalance),
            0.0,
            1.6,
        ) * self._clamp(1.0 - 0.20 * bid_depth_ratio + 0.18 * depth_shortage, 0.55, 1.25)
        upward_resistance_evidence = self._clamp(
            0.34 * max(0.0, signed_return)
            + 0.26 * max(0.0, signed_flow)
            + 0.26 * self._clamp(ask_depth_ratio - 1.0, 0.0, 1.5)
            + 0.30 * ask_depletion * max(0.0, signed_intent)
            + 0.16 * depth_surplus
            + 0.16 * micro.spread_pressure / 1.8,
            0.0,
            1.6,
        )
        downward_resistance_evidence = self._clamp(
            0.34 * max(0.0, -signed_return)
            + 0.26 * max(0.0, -signed_flow)
            + 0.26 * self._clamp(bid_depth_ratio - 1.0, 0.0, 1.5)
            + 0.30 * bid_depletion * max(0.0, -signed_intent)
            + 0.16 * depth_surplus
            + 0.16 * micro.spread_pressure / 1.8,
            0.0,
            1.6,
        )
        noise_evidence = self._clamp(
            0.32 * micro.activity
            + 0.28 * micro.volatility_cluster
            + 0.20 * abs(pre_imbalance - signed_flow)
            + abs(pressure_noise),
            0.0,
            2.0,
        )

        def evolve(old: float, evidence: float, noise_sign: float) -> float:
            noisy = evidence + 0.18 * pressure_noise * noise_sign
            return self._clamp(0.78 * old + 0.22 * noisy, 0.0, 1.8)

        return _ParticipantPressureState(
            upward_push=evolve(previous.upward_push, upward_evidence, 1.0),
            downward_push=evolve(previous.downward_push, downward_evidence, -1.0),
            upward_resistance=evolve(
                previous.upward_resistance,
                upward_resistance_evidence,
                -0.5,
            ),
            downward_resistance=evolve(
                previous.downward_resistance,
                downward_resistance_evidence,
                0.5,
            ),
            flow_continuation=evolve(
                previous.flow_continuation,
                continuation_evidence,
                0.0,
            ),
            absorption=evolve(previous.absorption, absorption_evidence, 0.0),
            exhaustion=evolve(previous.exhaustion, exhaustion_evidence, 0.0),
            signed_intent_memory=intent_memory,
            noise_pressure=evolve(previous.noise_pressure, noise_evidence, 0.0),
        )

    def _flow_reversal_pressure(self, micro: _MicrostructureState) -> float:
        persistent_flow = self._clamp(micro.flow_persistence, 0.0, 1.45) / 1.45
        participant = self._participant_pressure
        resistance_balance = (
            participant.upward_resistance
            + participant.downward_resistance
            - participant.upward_push
            - participant.downward_push
        )
        return_reversal_weight = self._clamp(
            0.16 + 0.54 * participant.exhaustion / 1.8 - 0.50 * participant.flow_continuation / 1.8,
            0.0,
            1.0,
        )
        exhaustion = self._clamp(
            0.08 + 0.24 * resistance_balance + 0.44 * participant.exhaustion / 1.8,
            0.0,
            1.0,
        )
        recent_return = self._clamp(micro.recent_signed_return / 6.0, -1.0, 1.0)
        reversal = exhaustion * (
            0.18 * persistent_flow * micro.recent_flow_imbalance
            + 0.56 * return_reversal_weight * recent_return
        )
        return self._clamp(
            reversal,
            -1.0,
            1.0,
        )

    def _next_microstructure_state(
        self,
        previous_latent: LatentState,
        latent: LatentState,
        imbalance: float,
    ) -> _MicrostructureState:
        """Update internal book pressure that links realized flow to future liquidity."""
        regime = self._active_settings
        previous = self._microstructure
        inputs = self._microstructure_inputs(previous_latent, latent, imbalance)
        bid_depth = self._nearby_book_depth("bid", self.state.price)
        ask_depth = self._nearby_book_depth("ask", self.state.price)
        expected_depth = self._expected_nearby_depth()
        bid_vacuum = self._clamp(1.0 - bid_depth / max(0.5 * expected_depth, 1e-12), 0.0, 1.0)
        ask_vacuum = self._clamp(1.0 - ask_depth / max(0.5 * expected_depth, 1e-12), 0.0, 1.0)
        if self._best_bid() is None:
            bid_vacuum = 1.0
        if self._best_ask() is None:
            ask_vacuum = 1.0
        spread_gap = self._clamp((self._current_spread_ticks() - 2.0) / 5.0, 0.0, 1.0)
        book_vacuum_target = self._clamp(
            0.42 * max(bid_vacuum, ask_vacuum)
            + 0.32 * (bid_vacuum + ask_vacuum) / 2.0
            + 0.18 * spread_gap
            + 0.08 * previous.liquidity_drought / 2.0,
            0.0,
            1.8,
        )
        book_vacuum = self._clamp(
            0.70 * previous.book_vacuum + 0.30 * book_vacuum_target,
            0.0,
            1.8,
        )
        realized_activity = (
            0.46 * inputs.execution_pressure
            + 0.34 * inputs.return_shock
            + 0.16 * inputs.volatility_shock
            + 0.10 * inputs.imbalance_shock
        )
        activity = self._clamp(
            regime["activity_decay"] * previous.activity
            + regime["activity_gain"] * realized_activity,
            0.0,
            2.0,
        )

        cancelled_volume = sum(self._realized_flow.cancelled_by_price.values())
        actual_cancel_activity = min(log1p(cancelled_volume / max(0.7, self.popularity)), 1.6)
        shock = (
            0.28 * inputs.return_shock
            + 0.22 * inputs.imbalance_shock
            + 0.28 * inputs.volatility_shock
            + 0.12 * actual_cancel_activity
            + 0.04 * max(0.0, activity - previous.activity)
        )
        # Cancellation bursts are stateful: shocks raise pressure, then pressure decays.
        cancel_pressure = self._clamp(
            (regime["cancel_decay"] - 0.04 * (1.0 - min(actual_cancel_activity, 1.0)))
            * previous.cancel_pressure
            + regime["cancel_burst"] * shock * (0.74 + 0.26 * min(actual_cancel_activity, 1.0)),
            0.0,
            2.0,
        )
        cancel_pressure *= self._clamp(1.0 - 0.24 * book_vacuum, 0.52, 1.0)

        recent_signed_return = self._clamp(
            0.86 * previous.recent_signed_return + 0.14 * inputs.signed_return,
            -12.0,
            12.0,
        )
        recent_flow_imbalance = self._clamp(
            0.82 * previous.recent_flow_imbalance + 0.18 * inputs.flow_imbalance,
            -1.0,
            1.0,
        )
        participation_target = self._clamp(
            0.34 * inputs.execution_pressure
            + 0.42 * inputs.return_shock
            + 0.22 * abs(inputs.flow_imbalance)
            + 0.24 * abs(self._realized_flow.intent_imbalance)
            + 0.10 * abs(self._realized_flow.rested_imbalance)
            + 0.20 * previous.volatility_cluster / 2.0
            + 0.18 * previous.arrival_cluster / 2.2
            + 0.14 * previous.flow_persistence / 1.45
            + 0.08 * self._clamp(activity, 0.0, 2.0) / 2.0,
            0.0,
            2.0,
        )
        participation_target *= self._clamp(1.0 - 0.24 * previous.churn_pressure / 2.0, 0.55, 1.0)
        participation_memory = self._clamp(
            0.89 + 0.040 * self._clamp(regime["volatility"] - 1.0, 0.0, 1.0),
            0.89,
            0.940,
        )
        participation_burst = self._clamp(
            participation_memory * previous.participation_burst
            + 0.23 * regime["event_gain"] * participation_target,
            0.0,
            2.0,
        )
        signed_return_unit = self._clamp(inputs.signed_return / 4.0, -1.0, 1.0)
        raw_meta_evidence = self._clamp(
            0.32 * inputs.flow_imbalance
            + 0.24 * self._realized_flow.intent_imbalance
            + 0.14 * self._realized_flow.rested_imbalance
            + 0.28 * signed_return_unit
            + 0.34 * latent.trend
            + 0.16 * previous.meta_order_side,
            -1.0,
            1.0,
        ) * regime["directional_persistence"]
        opposing_meta = raw_meta_evidence * previous.meta_order_side < -0.02
        flip_gate = self._clamp((abs(raw_meta_evidence) - 0.20) / 0.80, 0.0, 1.0)
        meta_evidence = (
            raw_meta_evidence * (0.38 + 0.62 * flip_gate)
            if opposing_meta
            else raw_meta_evidence
            * (1.0 + 0.18 * previous.flow_persistence / 1.45 + 0.10 * participation_burst / 2.0)
        )
        meta_shock = 0.0
        shock_probability = self._clamp(
            0.010
            + 0.018 * self.augmentation_strength
            + 0.012 * self._clamp(previous.activity_event, 0.0, 1.8) / 1.8,
            0.0,
            0.055,
        )
        if self._rng.random() < shock_probability:
            meta_shock = self._rng.gauss(0.0, 0.34 + 0.10 * regime["volatility"])
        meta_memory = self._clamp(
            0.90
            + 0.040 * previous.flow_persistence / 1.45
            + 0.018 * participation_burst / 2.0,
            0.90,
            0.956,
        )
        evidence_weight = 0.10 + 0.075 * (0.0 if opposing_meta else 1.0) + 0.04 * flip_gate
        meta_order_side = self._clamp(
            meta_memory * previous.meta_order_side + evidence_weight * meta_evidence + meta_shock,
            -1.0,
            1.0,
        )
        flow_alignment = max(0.0, meta_order_side * inputs.flow_imbalance)
        intent_alignment = max(0.0, meta_order_side * self._realized_flow.intent_imbalance)
        flow_persistence_target = self._clamp(
            0.46 * abs(meta_order_side)
            + 0.22 * abs(inputs.flow_imbalance)
            + 0.26 * abs(self._realized_flow.intent_imbalance)
            + 0.14 * inputs.execution_pressure
            + 0.34 * flow_alignment
            + 0.30 * intent_alignment
            + 0.18 * previous.activity_event / 1.8
            + 0.14 * participation_burst / 2.0,
            0.0,
            1.45,
        ) * self._clamp(1.0 - 0.30 * previous.churn_pressure / 2.0, 0.50, 1.0)
        flow_memory = self._clamp(
            0.88
            + 0.028 * regime["directional_persistence"]
            + 0.018 * participation_burst / 2.0,
            0.88,
            0.93,
        )
        flow_persistence = self._clamp(
            flow_memory * previous.flow_persistence + 0.15 * flow_persistence_target,
            0.0,
            1.45,
        )
        submitted_pressure = min(
            log1p(
                (
                    self._realized_flow.submitted_buy_volume
                    + self._realized_flow.submitted_sell_volume
                )
                / max(0.7, self.popularity)
            ),
            2.0,
        )
        displacement_target = self._clamp(
            0.42 * inputs.return_shock
            + 0.30 * inputs.execution_pressure
            + 0.18 * abs(inputs.flow_imbalance)
            + 0.12 * abs(self._realized_flow.intent_imbalance)
            + 0.10 * inputs.volatility_shock
            + 0.18 * self._clamp(regime["volatility"] - 1.0, 0.0, 1.0)
            + 0.12 * previous.volatility_cluster / 2.0,
            0.0,
            2.0,
        )
        displacement_pressure = self._clamp(
            0.78 * previous.displacement_pressure + 0.24 * displacement_target,
            0.0,
            2.0,
        )
        arrival_amplifier = self._clamp(
            1.0 - 0.22 * previous.churn_pressure / 2.0,
            0.58,
            1.0,
        )
        cancel_cluster = self._clamp(actual_cancel_activity + cancel_pressure, 0.0, 2.0) / 2.0
        arrival_cluster_target = self._clamp(
            0.30 * submitted_pressure * arrival_amplifier
            + 0.26 * inputs.execution_pressure
            + 0.20 * activity
            + 0.20 * previous.activity_event / 1.8
            + 0.18 * previous.volatility_cluster / 2.0
            + 0.16 * flow_persistence / 1.45
            + 0.12 * cancel_cluster,
            0.0,
            2.2,
        )
        arrival_memory = self._clamp(
            0.88
            + 0.055 * self._clamp(regime["volatility"] - 1.0, 0.0, 1.0)
            + 0.018 * flow_persistence / 1.45,
            0.88,
            0.955,
        )
        arrival_cluster = self._clamp(
            arrival_memory * previous.arrival_cluster
            + 0.27 * regime["event_gain"] * arrival_cluster_target,
            0.0,
            2.2,
        )
        volatility_cluster_target = self._clamp(
            0.52 * inputs.return_shock
            + 0.22 * inputs.execution_pressure
            + 0.24 * inputs.volatility_shock
            + 0.34 * previous.volatility_cluster / 2.0
            + 0.30 * participation_burst / 2.0
            + 0.34 * arrival_cluster / 2.2
            + 0.18 * previous.liquidity_drought / 2.0
            + 0.20 * displacement_pressure / 2.0
            + 0.10 * previous.churn_pressure / 2.0
            + 0.18 * flow_persistence / 1.45,
            0.0,
            2.0,
        )
        volatility_memory = self._clamp(
            0.89
            + 0.070 * self._clamp(regime["volatility"] - 1.0, 0.0, 1.0)
            + 0.014 * previous.liquidity_drought / 2.0,
            0.89,
            0.972,
        )
        volatility_cluster = self._clamp(
            volatility_memory * previous.volatility_cluster + 0.20 * volatility_cluster_target,
            0.0,
            2.0,
        )
        activity_event = self._next_activity_event(previous, inputs, regime)
        activity_event = self._clamp(
            activity_event
            + 0.12 * flow_persistence
            + 0.08 * volatility_cluster
            + 0.10 * participation_burst
            + 0.12 * arrival_cluster,
            0.0,
            1.8,
        )
        activity_event *= self._clamp(1.0 - 0.22 * previous.churn_pressure / 2.0, 0.55, 1.0)
        squeeze_pressure = self._next_squeeze_pressure(previous, inputs, regime)
        liquidity_stress = self._next_liquidity_stress(
            previous,
            inputs,
            cancel_pressure,
            regime,
        )
        liquidity_stress = self._clamp(
            liquidity_stress
            + 0.06 * volatility_cluster
            + 0.05 * participation_burst
            + 0.06 * previous.liquidity_drought,
            0.0,
            2.0,
        )
        spread_pressure = self._next_spread_pressure(
            previous,
            inputs,
            cancel_pressure,
            liquidity_stress,
            regime,
        )
        churn_target = self._clamp(
            0.34 * spread_pressure / 1.8
            + 0.30 * cancel_pressure / 2.0
            + 0.22 * liquidity_stress / 2.0
            + 0.18 * previous.volatility_cluster / 2.0
            - 0.34 * displacement_pressure / 2.0,
            0.0,
            2.0,
        )
        churn_pressure = self._clamp(
            0.78 * previous.churn_pressure + 0.24 * churn_target,
            0.0,
            2.0,
        )
        liquidity_drought_target = self._clamp(
            0.36 * liquidity_stress
            + 0.28 * spread_pressure / 1.8
            + 0.22 * cancel_pressure
            + 0.14 * inputs.execution_pressure
            + 0.10 * book_vacuum
            - 0.12 * previous.resiliency,
            0.0,
            2.0,
        )
        liquidity_drought = self._clamp(
            0.86 * previous.liquidity_drought + 0.18 * liquidity_drought_target,
            0.0,
            2.0,
        )
        cancel_burst_target = self._clamp(
            0.46 * actual_cancel_activity
            + 0.30 * cancel_pressure
            + 0.20 * liquidity_stress
            + 0.16 * inputs.return_shock,
            0.0,
            2.0,
        )
        cancel_burst = self._clamp(
            0.76 * previous.cancel_burst + 0.28 * cancel_burst_target,
            0.0,
            2.0,
        )
        stress_side = self._next_stress_side(previous, inputs)
        resiliency_target = regime["resiliency"] * self._clamp(
            1.0
            - 0.18 * cancel_pressure
            - 0.16 * liquidity_stress
            - 0.08 * spread_pressure
            - 0.10 * liquidity_drought
            - 0.06 * book_vacuum
            + 0.10 * previous.activity
            + 0.08 * self._clamp((self.popularity - 1.0) / 2.5, 0.0, 1.0)
            - 0.05 * activity_event,
            0.30,
            1.30,
        )
        resiliency = self._clamp(0.76 * previous.resiliency + 0.24 * resiliency_target, 0.20, 1.40)
        micro = _MicrostructureState(
            activity=activity,
            cancel_pressure=cancel_pressure,
            resiliency=resiliency,
            recent_signed_return=recent_signed_return,
            recent_flow_imbalance=recent_flow_imbalance,
            squeeze_pressure=squeeze_pressure,
            activity_event=activity_event,
            liquidity_stress=liquidity_stress,
            stress_side=stress_side,
            spread_pressure=spread_pressure,
            flow_persistence=flow_persistence,
            meta_order_side=meta_order_side,
            volatility_cluster=volatility_cluster,
            participation_burst=participation_burst,
            liquidity_drought=liquidity_drought,
            cancel_burst=cancel_burst,
            arrival_cluster=arrival_cluster,
            book_vacuum=book_vacuum,
            churn_pressure=churn_pressure,
            displacement_pressure=displacement_pressure,
        )
        self._microstructure = micro
        return micro

    def _microstructure_inputs(
        self,
        previous_latent: LatentState,
        latent: LatentState,
        imbalance: float,
    ) -> _MicrostructureInputs:
        execution_pressure = min(
            log1p(self._realized_flow.execution_volume / max(0.7, self.popularity)),
            2.0,
        )
        volatility_jump = max(0.0, latent.volatility - previous_latent.volatility)
        return _MicrostructureInputs(
            execution_pressure=execution_pressure,
            return_shock=1.0 - exp(-max(0.0, self._realized_flow.abs_return_ticks - 0.55) / 1.25),
            volatility_shock=1.0 - exp(-volatility_jump / 0.25),
            imbalance_shock=abs(imbalance - self._realized_flow.flow_imbalance),
            signed_return=self._realized_flow.return_ticks,
            flow_imbalance=self._realized_flow.flow_imbalance,
            squeeze_setup=self._squeeze_setup_pressure(),
        )

    def _next_activity_event(
        self,
        previous: _MicrostructureState,
        inputs: _MicrostructureInputs,
        regime: dict[str, float],
    ) -> float:
        burst_seed = (
            max(0.0, inputs.return_shock - 0.24)
            + 0.42 * inputs.volatility_shock
            + 0.30 * max(0.0, abs(inputs.flow_imbalance) - 0.25)
            + 0.24 * max(0.0, inputs.execution_pressure - 0.70)
            + 0.16 * previous.arrival_cluster / 2.2
            + 0.12 * previous.volatility_cluster / 2.0
        )
        return self._clamp(
            0.80 * previous.activity_event + 1.08 * regime["event_gain"] * burst_seed,
            0.0,
            1.8,
        )

    def _next_squeeze_pressure(
        self,
        previous: _MicrostructureState,
        inputs: _MicrostructureInputs,
        regime: dict[str, float],
    ) -> float:
        signed_return = self._clamp(inputs.signed_return / 3.0, -1.0, 1.0)
        release = 0.34 * signed_return if previous.squeeze_pressure * signed_return > 0.0 else 0.0
        return self._clamp(
            0.56 * previous.squeeze_pressure
            + regime["squeeze_gain"] * inputs.squeeze_setup
            - 0.34 * release,
            -1.5,
            1.5,
        )

    def _next_liquidity_stress(
        self,
        previous: _MicrostructureState,
        inputs: _MicrostructureInputs,
        cancel_pressure: float,
        regime: dict[str, float],
    ) -> float:
        raw_stress = (
            0.34 * inputs.return_shock
            + 0.26 * inputs.execution_pressure
            + 0.22 * inputs.imbalance_shock
            + 0.18 * cancel_pressure
        ) * regime["stress"]
        return self._clamp(0.70 * previous.liquidity_stress + 0.30 * raw_stress, 0.0, 2.0)

    def _next_spread_pressure(
        self,
        previous: _MicrostructureState,
        inputs: _MicrostructureInputs,
        cancel_pressure: float,
        liquidity_stress: float,
        regime: dict[str, float],
    ) -> float:
        spread_ticks = self._current_spread_ticks()
        spread_gap = self._clamp((spread_ticks - 2.0) / 5.0, 0.0, 1.0)
        repair_pull = self._clamp((spread_ticks - 3.0) / 5.0, 0.0, 1.0)
        participation_repair = self._clamp((self.popularity - 1.0) / 2.5, 0.0, 1.0)
        raw_pressure = (
            0.26 * inputs.return_shock
            + 0.20 * inputs.execution_pressure
            + 0.18 * abs(inputs.flow_imbalance)
            + 0.22 * cancel_pressure
            + 0.20 * liquidity_stress
            + 0.10 * spread_gap
            - 0.16 * repair_pull
            - 0.12 * participation_repair
        ) * regime["stress"]
        smoothed = self._clamp(
            0.78 * previous.spread_pressure + 0.22 * raw_pressure,
            0.0,
            1.72,
        )
        if smoothed <= 1.45:
            return smoothed
        return self._clamp(1.45 + 0.27 * (1.0 - exp(-(smoothed - 1.45) / 0.27)), 0.0, 1.72)

    def _next_stress_side(
        self,
        previous: _MicrostructureState,
        inputs: _MicrostructureInputs,
    ) -> float:
        directional_pressure = self._clamp(
            0.62 * inputs.flow_imbalance
            + 0.38 * self._clamp(inputs.signed_return / 5.0, -1.0, 1.0),
            -1.0,
            1.0,
        )
        return self._clamp(0.74 * previous.stress_side + 0.26 * directional_pressure, -1.0, 1.0)

    def _squeeze_setup_pressure(self) -> float:
        price = self.state.price
        bid_depth = self._nearby_book_depth("bid", price)
        ask_depth = self._nearby_book_depth("ask", price)
        total_depth = bid_depth + ask_depth
        if total_depth <= 1e-12:
            return 0.0
        bid_crowding = bid_depth / total_depth
        ask_crowding = ask_depth / total_depth
        ask_thinness = 1.0 / (1.0 + ask_depth / max(self.popularity, 1e-12))
        bid_thinness = 1.0 / (1.0 + bid_depth / max(self.popularity, 1e-12))
        signed_return = self._clamp(self._realized_flow.return_ticks / 3.0, -1.0, 1.0)
        signed_flow = self._clamp(self._realized_flow.flow_imbalance, -1.0, 1.0)
        trigger = self._clamp(
            0.55 * signed_return + 0.45 * signed_flow,
            -1.0,
            1.0,
        )
        upward_setup = 0.58 * bid_crowding + 0.42 * ask_thinness
        downward_setup = 0.58 * ask_crowding + 0.42 * bid_thinness
        return self._clamp(
            max(0.0, trigger) * upward_setup - max(0.0, -trigger) * downward_setup,
            -1.0,
            1.0,
        )
