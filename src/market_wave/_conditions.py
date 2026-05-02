from __future__ import annotations

from math import exp, log1p

from ._types import _MarketConditionInputs, _MarketConditionState
from .state import LatentState


class _MarketConditionsMixin:
    @staticmethod
    def _regime_names() -> set[str]:
        return {"normal", "trend_up", "trend_down", "high_vol", "thin_liquidity", "squeeze"}

    def _condition_preset(self, regime: str) -> _MarketConditionState:
        presets = {
            "normal": _MarketConditionState(),
            "trend_up": _MarketConditionState(
                trend_bias=0.21,
                volatility_pressure=0.08,
                liquidity_tightness=0.04,
                stress_pressure=0.06,
                participation_bias=0.16,
            ),
            "trend_down": _MarketConditionState(
                trend_bias=-0.16,
                volatility_pressure=0.08,
                liquidity_tightness=0.04,
                stress_pressure=0.06,
                participation_bias=0.16,
            ),
            "high_vol": _MarketConditionState(
                volatility_pressure=0.78,
                liquidity_tightness=0.24,
                stress_pressure=0.68,
                participation_bias=0.36,
            ),
            "thin_liquidity": _MarketConditionState(
                volatility_pressure=0.30,
                liquidity_tightness=1.08,
                stress_pressure=0.58,
                participation_bias=-0.30,
            ),
            "squeeze": _MarketConditionState(
                volatility_pressure=0.62,
                liquidity_tightness=0.56,
                stress_pressure=0.72,
                participation_bias=0.22,
                squeeze_pressure=0.78,
            ),
        }
        return presets.get("normal" if regime == "auto" else regime, presets["normal"])

    def _condition_settings(self, condition: _MarketConditionState) -> dict[str, float]:
        trend = self._clamp(condition.trend_bias, -1.0, 1.0)
        volatility = self._clamp(condition.volatility_pressure, 0.0, 1.8)
        tightness = self._clamp(condition.liquidity_tightness, 0.0, 1.8)
        stress = self._clamp(condition.stress_pressure, 0.0, 1.8)
        participation = self._clamp(condition.participation_bias, -0.8, 1.2)
        squeeze = self._clamp(condition.squeeze_pressure, 0.0, 1.5)
        trend_abs = abs(trend)

        return {
            "mood": 0.12 * trend,
            "trend": 0.55 * trend,
            "volatility": self._clamp(
                1.0 + 0.55 * volatility + 0.08 * tightness + 0.04 * trend_abs,
                0.78,
                1.75,
            ),
            "volatility_bias": self._clamp(
                0.065 * volatility + 0.018 * stress + 0.010 * tightness,
                0.0,
                0.12,
            ),
            "intensity": self._clamp(
                1.0
                + 0.28 * participation
                + 0.14 * volatility
                - 0.22 * tightness
                + 0.08 * trend_abs
                + 0.10 * squeeze,
                0.52,
                1.52,
            ),
            "taker": self._clamp(
                1.0 + 0.16 * volatility + 0.10 * tightness + 0.06 * squeeze,
                0.88,
                1.42,
            ),
            "cancel": self._clamp(
                1.0 + 0.20 * volatility + 0.26 * tightness + 0.20 * stress,
                0.86,
                1.62,
            ),
            "liquidity": self._clamp(
                1.0 - 0.48 * tightness - 0.12 * stress + 0.05 * participation,
                0.32,
                1.12,
            ),
            "spread": self._clamp(
                1.0 + 0.22 * volatility + 0.17 * tightness + 0.08 * stress,
                0.92,
                1.58,
            ),
            "activity_gain": self._clamp(
                0.20 + 0.045 * volatility + 0.025 * max(participation, 0.0),
                0.15,
                0.32,
            ),
            "activity_decay": self._clamp(
                0.76 + 0.020 * trend_abs - 0.018 * volatility,
                0.70,
                0.82,
            ),
            "cancel_burst": self._clamp(
                0.68 + 0.16 * volatility + 0.14 * tightness + 0.10 * stress,
                0.55,
                1.08,
            ),
            "cancel_decay": self._clamp(0.70 + 0.018 * tightness - 0.012 * volatility, 0.64, 0.76),
            "depth_exponent": self._clamp(
                1.35 + 0.20 * tightness - 0.18 * volatility - 0.04 * participation,
                1.02,
                1.62,
            ),
            "near_touch_liquidity": self._clamp(
                1.0 - 0.33 * tightness - 0.13 * volatility + 0.06 * participation,
                0.52,
                1.10,
            ),
            "resiliency": self._clamp(
                0.92 - 0.26 * volatility - 0.36 * tightness - 0.18 * stress
                + 0.08 * participation,
                0.39,
                1.08,
            ),
            "stress": self._clamp(
                0.78 + 0.32 * volatility + 0.28 * tightness + 0.30 * stress,
                0.62,
                1.58,
            ),
            "event_gain": self._clamp(
                0.36 + 0.20 * volatility + 0.18 * stress + 0.12 * squeeze,
                0.28,
                0.86,
            ),
            "book_noise": self._clamp(
                0.18 + 0.08 * volatility + 0.08 * tightness + 0.04 * stress,
                0.12,
                0.38,
            ),
            "trend_exhaustion": self._clamp(
                0.055 + 0.012 * trend_abs + 0.008 * volatility,
                0.04,
                0.09,
            ),
            "flow_reversal": self._clamp(
                0.045 + 0.015 * tightness + 0.010 * volatility,
                0.025,
                0.075,
            ),
            "squeeze_gain": self._clamp(0.34 * squeeze, 0.0, 0.44),
            "directional_persistence": self._clamp(
                1.0
                + 0.28 * trend_abs
                - (1.0 - trend_abs) * (0.34 * volatility + 0.20 * stress),
                0.38,
                1.16,
            ),
        }

    def _condition_label(self, condition: _MarketConditionState) -> str:
        if condition.squeeze_pressure >= 0.42:
            return "squeeze"
        if condition.liquidity_tightness >= 0.62:
            return "thin_liquidity"
        if condition.volatility_pressure >= 0.58 or condition.stress_pressure >= 0.72:
            return "high_vol"
        if condition.trend_bias >= 0.15:
            return "trend_up"
        if condition.trend_bias <= -0.15:
            return "trend_down"
        return "normal"

    def _market_condition_inputs(
        self,
        latent: LatentState,
        imbalance: float,
    ) -> _MarketConditionInputs:
        execution_pressure = min(
            log1p(self._realized_flow.execution_volume / max(0.7, self.popularity)),
            2.0,
        )
        spread_gap = self._clamp((self._current_spread_ticks() - 2.0) / 5.0, 0.0, 1.0)
        bid_depth = self._nearby_book_depth("bid", self.state.price)
        ask_depth = self._nearby_book_depth("ask", self.state.price)
        expected_depth = self._expected_nearby_depth()
        depth_shortage = self._clamp(
            1.0 - (bid_depth + ask_depth) / max(expected_depth, 1e-12),
            0.0,
            1.0,
        )
        volatility_jump = max(0.0, latent.volatility - self.state.latent.volatility)
        return _MarketConditionInputs(
            execution_pressure=execution_pressure,
            return_shock=1.0 - exp(-self._realized_flow.abs_return_ticks / 1.4),
            volatility_shock=1.0 - exp(-volatility_jump / 0.25),
            imbalance_shock=abs(imbalance - self._realized_flow.flow_imbalance),
            signed_return=self._realized_flow.return_ticks,
            flow_imbalance=self._realized_flow.flow_imbalance,
            spread_gap=spread_gap,
            depth_shortage=depth_shortage,
            cancel_pressure=self._microstructure.cancel_pressure,
            squeeze_setup=self._squeeze_setup_pressure(),
        )

    def _next_market_condition(
        self,
        previous: _MarketConditionState,
        inputs: _MarketConditionInputs,
    ) -> _MarketConditionState:
        noise_scale = 0.010 + 0.018 * self.augmentation_strength

        def signed_next(
            previous_value: float,
            evidence: float,
            *,
            memory: float,
            low: float,
            high: float,
        ) -> float:
            noise = self._rng.gauss(0.0, noise_scale)
            return self._clamp(
                memory * previous_value + (1.0 - memory) * evidence + noise,
                low,
                high,
            )

        def pressure_next(
            previous_value: float,
            evidence: float,
            *,
            memory: float,
            high: float,
        ) -> float:
            noise = self._rng.gauss(0.0, noise_scale)
            return self._clamp(
                memory * previous_value + (1.0 - memory) * evidence + noise,
                0.0,
                high,
            )

        signed_return = self._clamp(inputs.signed_return / 5.0, -1.0, 1.0)
        trend_evidence = self._clamp(
            0.58 * inputs.flow_imbalance + 0.42 * signed_return,
            -1.0,
            1.0,
        )
        volatility_evidence = self._clamp(
            0.62 * inputs.return_shock
            + 0.24 * inputs.execution_pressure
            + 0.24 * inputs.volatility_shock
            + 0.12 * abs(inputs.flow_imbalance),
            0.0,
            1.8,
        )
        liquidity_evidence = self._clamp(
            0.48 * inputs.depth_shortage
            + 0.34 * inputs.spread_gap
            + 0.18 * self._clamp(inputs.cancel_pressure, 0.0, 2.0) / 2.0,
            0.0,
            1.8,
        )
        stress_evidence = self._clamp(
            0.34 * inputs.return_shock
            + 0.24 * inputs.execution_pressure
            + 0.24 * inputs.imbalance_shock
            + 0.18 * self._clamp(inputs.cancel_pressure, 0.0, 2.0) / 2.0,
            0.0,
            1.8,
        )
        participation_evidence = self._clamp(
            0.44 * inputs.execution_pressure
            + 0.26 * inputs.return_shock
            + 0.20 * abs(inputs.flow_imbalance)
            - 0.18 * inputs.depth_shortage,
            -0.8,
            1.2,
        )
        squeeze_evidence = self._clamp(
            0.82 * abs(inputs.squeeze_setup)
            + 0.18 * abs(signed_return),
            0.0,
            1.5,
        )

        return _MarketConditionState(
            trend_bias=signed_next(
                previous.trend_bias,
                trend_evidence,
                memory=0.985,
                low=-1.0,
                high=1.0,
            ),
            volatility_pressure=pressure_next(
                previous.volatility_pressure,
                volatility_evidence,
                memory=0.960,
                high=1.8,
            ),
            liquidity_tightness=pressure_next(
                previous.liquidity_tightness,
                liquidity_evidence,
                memory=0.960,
                high=1.8,
            ),
            stress_pressure=pressure_next(
                previous.stress_pressure,
                stress_evidence,
                memory=0.955,
                high=1.8,
            ),
            participation_bias=signed_next(
                previous.participation_bias,
                participation_evidence,
                memory=0.950,
                low=-0.8,
                high=1.2,
            ),
            squeeze_pressure=pressure_next(
                previous.squeeze_pressure,
                squeeze_evidence,
                memory=0.950,
                high=1.5,
            ),
        )
