from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from random import Random

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
    bid_liquidity_by_tick: Mapping[int, float] = field(default_factory=dict)
    ask_liquidity_by_tick: Mapping[int, float] = field(default_factory=dict)
    bid_shortage_by_tick: Mapping[int, float] = field(default_factory=dict)
    ask_shortage_by_tick: Mapping[int, float] = field(default_factory=dict)
    bid_gap_by_tick: Mapping[int, float] = field(default_factory=dict)
    ask_gap_by_tick: Mapping[int, float] = field(default_factory=dict)
    bid_front_by_tick: Mapping[int, float] = field(default_factory=dict)
    ask_front_by_tick: Mapping[int, float] = field(default_factory=dict)
    bid_occupancy_by_tick: Mapping[int, float] = field(default_factory=dict)
    ask_occupancy_by_tick: Mapping[int, float] = field(default_factory=dict)
    bid_depth_pressure: float = 0.0
    ask_depth_pressure: float = 0.0
    spread_ticks: float = 1.0
    spread_pressure: float = 0.0
    cancel_pressure: float = 0.0
    liquidity_stress: float = 0.0
    stress_side: float = 0.0
    resiliency: float = 1.0
    activity: float = 0.0
    activity_event: float = 0.0
    flow_persistence: float = 0.0
    meta_order_side: float = 0.0
    volatility_cluster: float = 0.0
    participation_burst: float = 0.0
    liquidity_drought: float = 0.0
    cancel_burst: float = 0.0
