from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite
from random import Random
from typing import Protocol


@dataclass(frozen=True)
class MixtureComponent:
    weight: float
    center_price: float
    spread: float


@dataclass(frozen=True)
class RelativeMixtureComponent:
    weight: float
    center_tick: float
    spread_ticks: float


@dataclass(frozen=True)
class DistributionContext:
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


class DistributionModel(Protocol):
    def pmf(
        self,
        side: str,
        intent: str,
        relative_ticks: Sequence[int],
        context: DistributionContext,
    ) -> list[float]: ...


@dataclass(frozen=True)
class DiscreteMixtureDistribution:
    components: tuple[MixtureComponent, ...]

    def pmf(self, price_grid: list[float]) -> dict[float, float]:
        if any(not isfinite(price) for price in price_grid):
            raise ValueError("price_grid must contain only finite prices")
        for component in self.components:
            if not (
                isfinite(component.weight)
                and isfinite(component.center_price)
                and isfinite(component.spread)
            ):
                raise ValueError("mixture components must contain only finite values")
            if component.spread <= 0:
                raise ValueError("mixture component spread must be positive")

        masses = {}
        for price in price_grid:
            mass = 0.0
            for component in self.components:
                if component.weight <= 0:
                    continue
                spread = max(component.spread, 1e-12)
                kernel = exp(-abs(price - component.center_price) / spread)
                mass += component.weight * kernel
            masses[price] = mass

        total = sum(masses.values())
        if total <= 0:
            if not price_grid:
                return {}
            uniform = 1.0 / len(price_grid)
            return {price: uniform for price in price_grid}

        return {price: mass / total for price, mass in masses.items()}


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    cleaned = [max(0.0, float(value)) if isfinite(float(value)) else 0.0 for value in values]
    total = sum(cleaned)
    if total <= 0:
        return [1.0 / len(cleaned) for _ in cleaned]
    return [value / total for value in cleaned]


@dataclass(frozen=True)
class LaplaceMixturePMF:
    components: tuple[RelativeMixtureComponent, ...] | None = None

    def pmf(
        self,
        side: str,
        intent: str,
        relative_ticks: Sequence[int],
        context: DistributionContext,
    ) -> list[float]:
        components = self.components or self._default_components(side, intent, context)
        masses = []
        spread_scale = 1.0 + 0.35 * context.augmentation_strength
        for tick in relative_ticks:
            mass = 0.0
            for component in components:
                if component.weight <= 0:
                    continue
                spread = max(component.spread_ticks * spread_scale, 1e-12)
                mass += component.weight * exp(-abs(tick - component.center_tick) / spread)
            masses.append(mass)
        return _normalize(masses)

    def _default_components(
        self, side: str, intent: str, context: DistributionContext
    ) -> tuple[RelativeMixtureComponent, ...]:
        spread = 1.0 + 5.5 * context.volatility
        trend_shift = context.trend * (2.5 + 3.0 * context.volatility)
        mood_shift = context.mood * 2.0
        if intent == "entry" and side == "buy":
            return (
                RelativeMixtureComponent(
                    0.72 + max(context.mood, 0.0) * 0.18, -1 + trend_shift, spread
                ),
                RelativeMixtureComponent(0.28, -4 + mood_shift, spread * 1.8),
            )
        if intent == "entry" and side == "sell":
            return (
                RelativeMixtureComponent(
                    0.72 + max(-context.mood, 0.0) * 0.18, 1 + trend_shift, spread
                ),
                RelativeMixtureComponent(0.28, 4 + mood_shift, spread * 1.8),
            )
        if intent == "exit" and side == "long":
            return (
                RelativeMixtureComponent(0.58, 3.0, spread * 1.2),
                RelativeMixtureComponent(0.42, -2.0, spread * 1.1),
            )
        return (
            RelativeMixtureComponent(0.58, -3.0, spread * 1.2),
            RelativeMixtureComponent(0.42, 2.0, spread * 1.1),
        )


@dataclass(frozen=True)
class SkewedPMF:
    base: DistributionModel | None = None
    skew: float = 0.8

    def pmf(
        self,
        side: str,
        intent: str,
        relative_ticks: Sequence[int],
        context: DistributionContext,
    ) -> list[float]:
        base = (self.base or LaplaceMixturePMF()).pmf(side, intent, relative_ticks, context)
        direction = 1.0 if side in {"buy", "short"} else -1.0
        regime_skew = context.trend + 0.5 * context.mood
        weights = [
            probability
            * max(
                0.05, 1.0 + direction * self.skew * regime_skew * tick / max(1, len(relative_ticks))
            )
            for probability, tick in zip(base, relative_ticks, strict=False)
        ]
        return _normalize(weights)


@dataclass(frozen=True)
class FatTailPMF:
    tail_weight: float = 0.18

    def pmf(
        self,
        side: str,
        intent: str,
        relative_ticks: Sequence[int],
        context: DistributionContext,
    ) -> list[float]:
        base = LaplaceMixturePMF().pmf(side, intent, relative_ticks, context)
        tail = _normalize([1.0 / (1.0 + abs(tick)) ** 1.25 for tick in relative_ticks])
        weight = min(0.55, max(0.0, self.tail_weight + 0.10 * context.augmentation_strength))
        return _normalize(
            [
                (1.0 - weight) * left + weight * right
                for left, right in zip(base, tail, strict=False)
            ]
        )


@dataclass(frozen=True)
class NoisyPMF:
    base: DistributionModel | None = None
    strength: float | None = None

    def pmf(
        self,
        side: str,
        intent: str,
        relative_ticks: Sequence[int],
        context: DistributionContext,
    ) -> list[float]:
        base = (self.base or LaplaceMixturePMF()).pmf(side, intent, relative_ticks, context)
        strength = context.augmentation_strength if self.strength is None else self.strength
        if strength <= 0:
            return base
        perturbed = [
            probability * context.rng.lognormvariate(0.0, 0.18 * strength) for probability in base
        ]
        return _normalize(perturbed)
