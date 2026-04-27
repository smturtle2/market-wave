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
        spread = 1.0 + 5.5 * context.volatility
        trend_shift = context.trend * (2.5 + 3.0 * context.volatility)
        mood_shift = context.mood * 2.0
        if intent == "entry" and side == "buy":
            return (
                RelativeMDFComponent(
                    0.72 + max(context.mood, 0.0) * 0.18, -1 + trend_shift, spread
                ),
                RelativeMDFComponent(0.28, -4 + mood_shift, spread * 1.8),
            )
        if intent == "entry" and side == "sell":
            return (
                RelativeMDFComponent(
                    0.72 + max(-context.mood, 0.0) * 0.18, 1 + trend_shift, spread
                ),
                RelativeMDFComponent(0.28, 4 + mood_shift, spread * 1.8),
            )
        if intent == "exit" and side == "long":
            return (
                RelativeMDFComponent(0.58, 3.0, spread * 1.2),
                RelativeMDFComponent(0.42, -2.0, spread * 1.1),
            )
        return (
            RelativeMDFComponent(0.58, -3.0, spread * 1.2),
            RelativeMDFComponent(0.42, 2.0, spread * 1.1),
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


def _scale_map(values: Mapping[int, float]) -> TickMap:
    cleaned = {tick: max(0.0, value) for tick, value in values.items() if isfinite(value)}
    peak = max(cleaned.values(), default=0.0)
    if peak <= 0:
        return {}
    return {tick: value / peak for tick, value in cleaned.items()}
