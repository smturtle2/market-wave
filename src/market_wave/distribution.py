from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import exp, isfinite, log
from random import Random
from typing import Protocol

TickMap = dict[int, float]


@dataclass(frozen=True)
class MDFContext:
    current_price: float
    current_tick: int
    tick_size: float
    mood: float
    trend: float
    volatility: float
    regime: str
    augmentation_strength: float
    step_index: int
    rng: Random


@dataclass(frozen=True)
class MDFSignals:
    orderbook_imbalance: float = 0.0
    last_return_ticks: float = 0.0
    last_execution_volume: float = 0.0
    executed_volume_by_tick: Mapping[int, float] = field(default_factory=dict)
    liquidity_by_tick: Mapping[int, float] = field(default_factory=dict)
    long_position_mass_by_tick: Mapping[int, float] = field(default_factory=dict)
    short_position_mass_by_tick: Mapping[int, float] = field(default_factory=dict)


class MDFModel(Protocol):
    def scores(
        self,
        side: str,
        intent: str,
        relative_ticks: Sequence[int],
        context: MDFContext,
        signals: MDFSignals | None = None,
    ) -> list[float]: ...


@dataclass(frozen=True)
class RelativeMDFComponent:
    weight: float
    center_tick: float
    spread_ticks: float


@dataclass(frozen=True)
class DynamicMDFModel:
    components: tuple[RelativeMDFComponent, ...] | None = None
    value_weight: float = 1.0
    trend_weight: float = 0.48
    liquidity_weight: float = 0.16
    memory_weight: float = 0.18
    risk_weight: float = 0.30
    orderbook_weight: float = 0.20

    def scores(
        self,
        side: str,
        intent: str,
        relative_ticks: Sequence[int],
        context: MDFContext,
        signals: MDFSignals | None = None,
    ) -> list[float]:
        signals = signals or MDFSignals()
        uses_default_components = self.components is None
        components = self.components or self._default_components(side, intent, context)
        liquidity_scale = _scale_map(signals.liquidity_by_tick)
        memory_scale = _scale_map(signals.executed_volume_by_tick)
        long_risk = _scale_map(signals.long_position_mass_by_tick)
        short_risk = _scale_map(signals.short_position_mass_by_tick)
        directional_side = 1.0 if side in {"buy", "short"} else -1.0
        trend_push = context.trend + 0.35 * context.mood + 0.20 * signals.last_return_ticks
        imbalance_push = signals.orderbook_imbalance
        noise_strength = 0.50 * context.augmentation_strength

        scores: list[float] = []
        for tick in relative_ticks:
            value_score = _component_score(tick, components)
            if uses_default_components and intent == "entry":
                value_score -= _entry_crossing_penalty(side, tick)
            trend_score = directional_side * trend_push * tick / max(1, len(relative_ticks))
            orderbook_score = directional_side * imbalance_push * tick / max(1, len(relative_ticks))
            liquidity_score = liquidity_scale.get(tick, 0.0)
            memory_score = memory_scale.get(tick, 0.0)
            if intent == "exit" and side == "long":
                risk_score = long_risk.get(tick, 0.0) + max(0.0, -tick) * 0.03
            elif intent == "exit" and side == "short":
                risk_score = short_risk.get(tick, 0.0) + max(0.0, tick) * 0.03
            else:
                risk_score = 0.0
            noise = context.rng.uniform(-noise_strength, noise_strength) if noise_strength else 0.0
            scores.append(
                self.value_weight * value_score
                + self.trend_weight * trend_score
                + self.orderbook_weight * orderbook_score
                + self.liquidity_weight * liquidity_score
                + self.memory_weight * memory_score
                + self.risk_weight * risk_score
                + noise
            )
        return scores

    def _default_components(
        self, side: str, intent: str, context: MDFContext
    ) -> tuple[RelativeMDFComponent, ...]:
        if intent == "entry" and side in {"buy", "sell"}:
            return self._entry_components(side, context)
        spread = 1.0 + 5.5 * context.volatility
        if intent == "exit" and side == "long":
            return (
                RelativeMDFComponent(0.58, 3.0, spread * 1.2),
                RelativeMDFComponent(0.42, -2.0, spread * 1.1),
            )
        return (
            RelativeMDFComponent(0.58, -3.0, spread * 1.2),
            RelativeMDFComponent(0.42, 2.0, spread * 1.1),
        )

    def _entry_components(
        self, side: str, context: MDFContext
    ) -> tuple[RelativeMDFComponent, ...]:
        side_direction = -1.0 if side == "buy" else 1.0
        side_urgency = context.trend + 0.35 * context.mood
        if side == "sell":
            side_urgency *= -1.0
        side_urgency = _clamp(side_urgency, -1.0, 1.0)

        passive_bias = max(0.0, -side_urgency)
        chase_bias = max(0.0, side_urgency)
        weights = _normalize_weights(
            (
                0.18 + 0.05 * passive_bias,
                0.36 + 0.04 * passive_bias - 0.04 * chase_bias,
                0.30 + 0.02 * chase_bias,
                0.16 + 0.07 * chase_bias - 0.03 * passive_bias,
            )
        )

        volatility = max(0.0, context.volatility)
        spread = 0.85 + 3.6 * volatility
        volatility_extension = min(2.0, 1.2 * volatility)
        weak_shift = 0.7 * (context.trend + 0.40 * context.mood)
        fair_shift = context.trend * (1.6 + 1.2 * volatility) + context.mood * 0.70
        chase_shift = context.trend * (1.2 + 0.8 * volatility) + context.mood * 0.50

        # These are reservation-price zones. They create passive interest through
        # the MDF itself instead of pinning synthetic walls to absolute prices.
        return (
            RelativeMDFComponent(
                weights[0],
                side_direction * (7.0 + volatility_extension) + weak_shift,
                spread * 1.45,
            ),
            RelativeMDFComponent(
                weights[1],
                side_direction * (3.5 + 0.6 * volatility_extension) + weak_shift,
                spread * 0.90,
            ),
            RelativeMDFComponent(
                weights[2],
                side_direction * 0.8 + fair_shift,
                spread * 0.70,
            ),
            RelativeMDFComponent(
                weights[3],
                -side_direction * (2.0 + 0.4 * volatility_extension) + chase_shift,
                spread * 0.60,
            ),
        )


def _component_score(tick: int, components: Sequence[RelativeMDFComponent]) -> float:
    mass = 0.0
    for component in components:
        if component.weight <= 0:
            continue
        if not (
            isfinite(component.weight)
            and isfinite(component.center_tick)
            and isfinite(component.spread_ticks)
        ):
            continue
        spread = max(component.spread_ticks, 1e-12)
        mass += component.weight * exp(-abs(tick - component.center_tick) / spread)
    return log(max(mass, 1e-12))


def _entry_crossing_penalty(side: str, tick: int) -> float:
    aggressive_ticks = tick if side == "buy" else -tick
    if aggressive_ticks <= 0:
        return 0.0
    return 0.12 * aggressive_ticks**1.35


def _normalize_weights(values: Sequence[float]) -> tuple[float, ...]:
    cleaned = tuple(max(1e-12, value) for value in values)
    total = sum(cleaned)
    return tuple(value / total for value in cleaned)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _scale_map(values: Mapping[int, float]) -> TickMap:
    cleaned = {tick: max(0.0, value) for tick, value in values.items() if isfinite(value)}
    peak = max(cleaned.values(), default=0.0)
    if peak <= 0:
        return {}
    return {tick: value / peak for tick, value in cleaned.items()}
