from __future__ import annotations

from math import exp, log1p

from ._types import _MicrostructureInputs, _MicrostructureState
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
        return_momentum = self._clamp(self._last_return_ticks / 4.0, -1.0, 1.0) * self._clamp(
            0.45 * micro.volatility_cluster + 0.35 * micro.participation_burst,
            0.0,
            1.2,
        )
        signed_flow = (
            0.09 * self._last_imbalance
            + 0.045 * directional_memory
            + 0.020 * return_momentum
        )
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
        trend_target += 0.05 * squeeze_impulse + 0.060 * directional_memory
        trend_target -= regime["trend_exhaustion"] * exhaustion
        trend = self._clamp(
            0.74 * latent.trend
            + 0.26 * trend_target
            + 0.030 * return_momentum
            + 0.045 * directional_memory
            + 0.08 * jump
            + trend_noise,
            -1.0,
            1.0,
        )
        noise_shock = max(0.0, abs(mood_noise) - 0.08)
        shock = 0.32 * noise_shock + abs(jump) * 0.24
        execution_pressure = min(
            self._last_execution_volume / max(0.7, self.popularity),
            4.0,
        )
        realized = (
            0.050 * min(self._last_abs_return_ticks, 3.0)
            + 0.010 * execution_pressure
            + 0.050 * micro.volatility_cluster
        )
        volatility_target = 0.28 + 0.22 * regime["volatility"] + regime["volatility_bias"]
        volatility = self._clamp(
            0.88 * latent.volatility
            + 0.12 * volatility_target
            + shock
            + realized
            + 0.014 * abs(self._last_imbalance)
            + 0.020 * micro.flow_persistence,
            0.04,
            1.55,
        )
        return LatentState(mood=mood, trend=trend, volatility=volatility)

    def _trend_exhaustion(self, micro: _MicrostructureState) -> float:
        return self._clamp(
            0.70 * self._clamp(micro.recent_signed_return / 8.0, -1.0, 1.0)
            + 0.30 * micro.recent_flow_imbalance,
            -1.0,
            1.0,
        )

    def _squeeze_impulse(self, micro: _MicrostructureState) -> float:
        return self._clamp(micro.squeeze_pressure, 0.0, 1.5)

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
        participation_burst = self._clamp(micro.participation_burst, 0.0, 2.0)
        liquidity_drought = self._clamp(micro.liquidity_drought, 0.0, 2.0)
        cancel_burst = self._clamp(micro.cancel_burst, 0.0, 2.0)
        activity_multiplier = (
            1.0
            + 0.26 * self._clamp(activity, 0.0, 2.2)
            + 0.18 * participation_burst
        )
        event_multiplier = (
            1.0
            + 0.14 * self._clamp(micro.activity_event, 0.0, 1.8)
            + 0.12 * participation_burst
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
        momentum_state = self._clamp(
            0.45 * micro.volatility_cluster + 0.35 * micro.participation_burst,
            0.0,
            1.2,
        ) / 1.2
        signed_return_momentum = self._clamp(self._last_return_ticks / 4.0, -1.0, 1.0)
        buy_ratio = self._clamp(
            0.5
            + 0.24 * latent.mood
            + 0.18 * latent.trend
            + 0.055 * self._last_imbalance
            + 0.025 * self._squeeze_impulse(micro)
            + 0.105 * directional_memory
            + 0.070 * momentum_state * signed_return_momentum
            - 0.35 * regime["flow_reversal"] * flow_reversal,
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

    def _flow_reversal_pressure(self, micro: _MicrostructureState) -> float:
        persistent_flow = self._clamp(micro.flow_persistence, 0.0, 1.45) / 1.45
        return_reversal_weight = 1.0 - 0.75 * persistent_flow
        return self._clamp(
            0.65 * micro.recent_flow_imbalance
            + 0.35
            * return_reversal_weight
            * self._clamp(micro.recent_signed_return / 6.0, -1.0, 1.0),
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

        actual_cancel_activity = min(
            log1p(previous.last_cancelled_volume / max(0.7, self.popularity)),
            1.6,
        )
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
            + 0.20 * previous.volatility_cluster / 2.0
            + 0.14 * previous.flow_persistence / 1.45
            + 0.08 * self._clamp(activity, 0.0, 2.0) / 2.0,
            0.0,
            2.0,
        )
        participation_burst = self._clamp(
            0.86 * previous.participation_burst
            + 0.20 * regime["event_gain"] * participation_target,
            0.0,
            2.0,
        )
        signed_return_unit = self._clamp(inputs.signed_return / 4.0, -1.0, 1.0)
        raw_meta_evidence = self._clamp(
            0.50 * inputs.flow_imbalance
            + 0.34 * signed_return_unit
            + 0.16 * previous.meta_order_side,
            -1.0,
            1.0,
        )
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
        meta_memory = 0.90 + 0.035 * previous.flow_persistence / 1.45
        evidence_weight = 0.11 + 0.08 * (0.0 if opposing_meta else 1.0) + 0.04 * flip_gate
        meta_order_side = self._clamp(
            meta_memory * previous.meta_order_side + evidence_weight * meta_evidence + meta_shock,
            -1.0,
            1.0,
        )
        flow_alignment = max(0.0, meta_order_side * inputs.flow_imbalance)
        flow_persistence_target = self._clamp(
            0.42 * abs(meta_order_side)
            + 0.28 * abs(inputs.flow_imbalance)
            + 0.14 * inputs.execution_pressure
            + 0.34 * flow_alignment
            + 0.18 * previous.activity_event / 1.8
            + 0.14 * participation_burst / 2.0,
            0.0,
            1.45,
        )
        flow_persistence = self._clamp(
            0.86 * previous.flow_persistence + 0.14 * flow_persistence_target,
            0.0,
            1.45,
        )
        volatility_cluster_target = self._clamp(
            0.52 * inputs.return_shock
            + 0.18 * self._clamp(inputs.execution_pressure - 0.55, 0.0, 2.0)
            + 0.18 * inputs.volatility_shock
            + 0.34 * participation_burst / 2.0
            + 0.20 * previous.liquidity_drought / 2.0
            + 0.16 * flow_persistence / 1.45,
            0.0,
            2.0,
        )
        volatility_cluster = self._clamp(
            0.83 * previous.volatility_cluster + 0.17 * volatility_cluster_target,
            0.0,
            2.0,
        )
        activity_event = self._next_activity_event(previous, inputs, regime)
        activity_event = self._clamp(
            activity_event
            + 0.12 * flow_persistence
            + 0.08 * volatility_cluster
            + 0.10 * participation_burst,
            0.0,
            1.8,
        )
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
        liquidity_drought_target = self._clamp(
            0.36 * liquidity_stress
            + 0.28 * spread_pressure / 1.8
            + 0.22 * cancel_pressure
            + 0.14 * inputs.execution_pressure
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
            - 0.08 * liquidity_drought
            + 0.08 * previous.activity
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
            last_cancelled_volume=previous.last_cancelled_volume,
            flow_persistence=flow_persistence,
            meta_order_side=meta_order_side,
            volatility_cluster=volatility_cluster,
            participation_burst=participation_burst,
            liquidity_drought=liquidity_drought,
            cancel_burst=cancel_burst,
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
            log1p(self._last_execution_volume / max(0.7, self.popularity)),
            2.0,
        )
        volatility_jump = max(0.0, latent.volatility - previous_latent.volatility)
        return _MicrostructureInputs(
            execution_pressure=execution_pressure,
            return_shock=1.0 - exp(-max(0.0, self._last_abs_return_ticks - 0.55) / 1.25),
            volatility_shock=1.0 - exp(-volatility_jump / 0.25),
            imbalance_shock=abs(imbalance - self._last_imbalance),
            signed_return=self._last_return_ticks,
            flow_imbalance=self._last_imbalance,
            squeeze_setup=self._squeeze_setup_pressure(),
        )

    def _next_activity_event(
        self,
        previous: _MicrostructureState,
        inputs: _MicrostructureInputs,
        regime: dict[str, float],
    ) -> float:
        burst_seed = (
            max(0.0, inputs.return_shock - 0.32)
            + 0.42 * inputs.volatility_shock
            + 0.30 * max(0.0, abs(inputs.flow_imbalance) - 0.25)
            + 0.24 * max(0.0, inputs.execution_pressure - 0.70)
        )
        return self._clamp(
            0.72 * previous.activity_event + 1.18 * regime["event_gain"] * burst_seed,
            0.0,
            1.8,
        )

    def _next_squeeze_pressure(
        self,
        previous: _MicrostructureState,
        inputs: _MicrostructureInputs,
        regime: dict[str, float],
    ) -> float:
        release = max(0.0, -inputs.signed_return) / 3.0
        return self._clamp(
            0.56 * previous.squeeze_pressure
            + regime["squeeze_gain"] * inputs.squeeze_setup
            - 0.34 * release,
            0.0,
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
        raw_pressure = (
            0.26 * inputs.return_shock
            + 0.20 * inputs.execution_pressure
            + 0.18 * abs(inputs.flow_imbalance)
            + 0.22 * cancel_pressure
            + 0.20 * liquidity_stress
            + 0.10 * spread_gap
            - 0.16 * repair_pull
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
        ask_thinness = 1.0 / (1.0 + ask_depth / max(self.popularity, 1e-12))
        upward_trigger = self._clamp(
            0.55 * max(0.0, self._last_return_ticks) / 3.0
            + 0.45 * max(0.0, self._last_imbalance)
            - 0.10,
            0.0,
            1.0,
        )
        setup = 0.58 * bid_crowding + 0.42 * ask_thinness
        return self._clamp(setup * upward_trigger, 0.0, 1.0)
