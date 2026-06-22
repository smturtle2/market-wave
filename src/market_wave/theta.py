from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import GenerationConfig


@dataclass(frozen=True)
class WorldTheta:
    id: str
    currency: str
    tick_size_decimal: str
    quantity_unit_decimal: str
    min_quantity_unit: int
    max_quantity_unit: int
    initial_mid_tick: int
    initial_book_levels: int
    initial_orders_per_level: int
    base_order_quantity: int
    depth_slope: int
    top_levels: int
    steps: int
    start_timestamp: str
    snapshot_every: int
    minimum_spread_ticks: int
    base_activity: float
    base_limit_weight: float
    base_take_weight: float
    base_cancel_weight: float
    base_replace_weight: float
    feedback_decay: float
    latent_bound: float
    max_clock_ms: int

    @property
    def matrix_summary(self) -> dict[str, Any]:
        return {
            "theta_id": self.id,
            "market_world": {
                "initial_book_levels": self.initial_book_levels,
                "initial_orders_per_level": self.initial_orders_per_level,
                "minimum_spread_ticks": self.minimum_spread_ticks,
                "feedback_decay": self.feedback_decay,
                "latent_bound": self.latent_bound,
            },
        }


@dataclass(frozen=True)
class LatentState:
    liquidity: float
    volatility: float
    spread_pressure: float
    order_flow: float
    activity: float
    trend_pressure: float
    sweep_intensity: float
    cancel_pressure: float
    replenishment_pressure: float
    shock: float

    def clipped(self, bound: float) -> "LatentState":
        return LatentState(
            *[float(np.clip(value, -bound, bound)) for value in self.as_tuple()]
        )

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.liquidity,
            self.volatility,
            self.spread_pressure,
            self.order_flow,
            self.activity,
            self.trend_pressure,
            self.sweep_intensity,
            self.cancel_pressure,
            self.replenishment_pressure,
            self.shock,
        )


def sample_world_theta(rng: np.random.Generator, config: GenerationConfig) -> WorldTheta:
    config.validate()
    theta = WorldTheta(
        id=f"theta_{config.seed:016x}",
        currency=config.currency,
        tick_size_decimal=config.tick_size_decimal,
        quantity_unit_decimal=config.quantity_unit_decimal,
        min_quantity_unit=config.min_quantity_unit,
        max_quantity_unit=config.max_quantity_unit,
        initial_mid_tick=config.initial_mid_tick,
        initial_book_levels=max(config.top_levels + 2, config.initial_book_levels),
        initial_orders_per_level=config.initial_orders_per_level,
        base_order_quantity=int(rng.integers(80, 150)),
        depth_slope=int(rng.integers(1, 4)),
        top_levels=config.top_levels,
        steps=config.steps,
        start_timestamp=config.start_timestamp,
        snapshot_every=config.snapshot_every,
        minimum_spread_ticks=1,
        base_activity=float(rng.uniform(0.9, 1.15)),
        base_limit_weight=float(rng.uniform(1.5, 2.0)),
        base_take_weight=float(rng.uniform(0.35, 0.55)),
        base_cancel_weight=float(rng.uniform(0.45, 0.70)),
        base_replace_weight=float(rng.uniform(0.12, 0.20)),
        feedback_decay=0.94,
        latent_bound=4.0,
        max_clock_ms=450,
    )
    validate_theta_or_fail(theta)
    return theta


def init_latent_field(rng: np.random.Generator, theta: WorldTheta) -> LatentState:
    return LatentState(
        liquidity=float(rng.normal(0.15, 0.12)),
        volatility=float(rng.normal(0.05, 0.10)),
        spread_pressure=float(rng.normal(0.0, 0.10)),
        order_flow=float(rng.normal(0.0, 0.14)),
        activity=float(rng.normal(0.25, 0.12)),
        trend_pressure=float(rng.normal(0.0, 0.18)),
        sweep_intensity=float(rng.normal(0.10, 0.12)),
        cancel_pressure=float(rng.normal(0.0, 0.10)),
        replenishment_pressure=float(rng.normal(0.0, 0.10)),
        shock=0.0,
    ).clipped(theta.latent_bound)


def validate_theta_or_fail(theta: WorldTheta) -> None:
    if theta.min_quantity_unit < 1:
        raise ValueError("min quantity must be positive")
    if theta.max_quantity_unit < theta.min_quantity_unit:
        raise ValueError("max quantity must be >= min quantity")
    if theta.initial_book_levels < theta.top_levels + 2:
        raise ValueError("initial book levels must cover public top levels")
    if theta.initial_orders_per_level < 1:
        raise ValueError("initial orders per level must be positive")
    if theta.minimum_spread_ticks < 1:
        raise ValueError("minimum spread must be at least one tick")
    for value in (
        theta.base_activity,
        theta.base_limit_weight,
        theta.base_take_weight,
        theta.base_cancel_weight,
        theta.base_replace_weight,
        theta.feedback_decay,
        theta.latent_bound,
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("theta bounds must be finite positive values")


def bounded_smooth_projection(value: float, bound: float) -> float:
    return float(bound * np.tanh(float(value) / bound))


def deterministic_clip_and_normalize(value: float, bound: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -bound, bound))
