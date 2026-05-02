from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

PriceMap = dict[float, float]
TickMap = dict[float, float]

MUTABLE_SNAPSHOT_NOTE = (
    "State dataclasses are frozen at the attribute level, but their list and dict "
    "fields are plain mutable containers for JSON-friendly exports. Treat them as "
    "read-only observations unless you explicitly copy them."
)


@dataclass(frozen=True)
class IntensityState:
    """Submitted-flow intensity split for one market state."""

    total: float
    buy: float
    sell: float
    buy_ratio: float
    sell_ratio: float


@dataclass(frozen=True)
class LatentState:
    """Latent mood, trend, and volatility values used by the simulator."""

    mood: float
    trend: float
    volatility: float


@dataclass(frozen=True)
class MDFState:
    """Market Distribution Function snapshot.

    Keys are gap-unit offsets sampled by the engine before conversion to prices.
    The dataclass is frozen, but the contained maps remain mutable plain
    containers for compatibility with ``to_dict()``/JSON export workflows.
    """

    buy_entry_mdf: TickMap = field(default_factory=dict)
    sell_entry_mdf: TickMap = field(default_factory=dict)


@dataclass(frozen=True)
class OrderBookState:
    """Aggregated order-book snapshot with volume keyed by price."""

    bid_volume_by_price: PriceMap = field(default_factory=dict)
    ask_volume_by_price: PriceMap = field(default_factory=dict)


@dataclass(frozen=True)
class MarketState:
    """Current market observation.

    See ``MUTABLE_SNAPSHOT_NOTE``: nested containers are not deeply immutable.
    Use ``Market.snapshot()`` when you need a mutation-safe copy of current state.
    """

    price: float
    step_index: int
    current_tick: int
    tick_grid: list[int]
    intensity: IntensityState
    latent: LatentState
    price_grid: list[float]
    mdf_price_basis: float
    mdf: MDFState
    orderbook: OrderBookState


@dataclass(frozen=True)
class StepInfo:
    """Per-step observation and diagnostics.

    ``StepInfo`` is the main observation record returned by ``Market.step()``.
    It includes pre/post prices, active condition labels, MDF snapshots,
    submitted volume, cancelled volume, executions, and order-book snapshots.
    Nested dict/list fields are plain mutable containers; treat the object as a
    read-only observation unless you explicitly copy it.
    """

    step_index: int
    price_before: float
    price_after: float
    price_change: float
    tick_before: int
    tick_after: int
    tick_change: int
    intensity: IntensityState
    mood: float
    trend: float
    volatility: float
    regime: str
    augmentation_strength: float
    price_grid: list[float]
    mdf_price_basis: float
    buy_entry_mdf: TickMap
    sell_entry_mdf: TickMap
    buy_volume_by_price: PriceMap
    sell_volume_by_price: PriceMap
    entry_volume_by_price: PriceMap
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
    mean_quote_age: float
    order_flow_imbalance: float
    orderbook_before: OrderBookState
    orderbook_after: OrderBookState

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dataclass dictionary."""

        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        """Serialize the step observation to JSON."""

        return json.dumps(self.to_dict(), **kwargs)
