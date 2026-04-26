from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

PriceMap = dict[float, float]


@dataclass(frozen=True)
class IntensityState:
    total: float
    buy: float
    sell: float
    buy_ratio: float
    sell_ratio: float


@dataclass(frozen=True)
class LatentState:
    mood: float
    trend: float
    volatility: float


@dataclass(frozen=True)
class DistributionState:
    buy_entry_pmf: PriceMap
    sell_entry_pmf: PriceMap
    long_exit_pmf: PriceMap
    short_exit_pmf: PriceMap


@dataclass(frozen=True)
class OrderBookState:
    bid_volume_by_price: PriceMap = field(default_factory=dict)
    ask_volume_by_price: PriceMap = field(default_factory=dict)


@dataclass(frozen=True)
class PositionMassState:
    long_exit_mass_by_price: PriceMap = field(default_factory=dict)
    short_exit_mass_by_price: PriceMap = field(default_factory=dict)


@dataclass(frozen=True)
class MarketState:
    price: float
    step_index: int
    intensity: IntensityState
    latent: LatentState
    price_grid: list[float]
    distributions: DistributionState
    orderbook: OrderBookState
    position_mass: PositionMassState


@dataclass(frozen=True)
class StepInfo:
    step_index: int
    price_before: float
    price_after: float
    price_change: float
    intensity: IntensityState
    mood: float
    trend: float
    volatility: float
    price_grid: list[float]
    buy_entry_pmf: PriceMap
    sell_entry_pmf: PriceMap
    long_exit_pmf: PriceMap
    short_exit_pmf: PriceMap
    buy_volume_by_price: PriceMap
    sell_volume_by_price: PriceMap
    entry_volume_by_price: PriceMap
    exit_volume_by_price: PriceMap
    cancelled_volume_by_price: PriceMap
    executed_volume_by_price: PriceMap
    total_executed_volume: float
    market_buy_volume: float
    market_sell_volume: float
    crossed_market_volume: float
    residual_market_buy_volume: float
    residual_market_sell_volume: float
    trade_count: int
    vwap_price: float | None
    best_bid_before: float | None
    best_ask_before: float | None
    best_bid_after: float | None
    best_ask_after: float | None
    spread_before: float | None
    spread_after: float | None
    order_flow_imbalance: float
    orderbook_before: OrderBookState
    orderbook_after: OrderBookState
    position_mass_before: PositionMassState
    position_mass_after: PositionMassState

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)
