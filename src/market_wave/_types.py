from __future__ import annotations

from dataclasses import dataclass, field

from .distribution import MDFSignals
from .state import MDFState, PriceMap


@dataclass
class _IncomingOrder:
    side: str
    kind: str
    price: float
    volume: float


@dataclass
class _MarketEvent:
    event_type: str
    side: str
    price: float
    volume: float


@dataclass(frozen=True)
class _EventSizeState:
    maker_scale: float
    buy_taker_scale: float
    sell_taker_scale: float
    cancel_scale: float
    taker_tail_probability: float
    taker_medium_probability: float


@dataclass
class _EntryFlow:
    orders: list[_IncomingOrder]
    buy_intent_by_price: PriceMap
    sell_intent_by_price: PriceMap
    events: list[_MarketEvent] = field(default_factory=list)
    bid_cancel_intent_by_price: PriceMap = field(default_factory=dict)
    ask_cancel_intent_by_price: PriceMap = field(default_factory=dict)


@dataclass
class _TradeStats:
    executed_by_price: PriceMap
    buy_executed_by_price: PriceMap = field(default_factory=dict)
    sell_executed_by_price: PriceMap = field(default_factory=dict)
    total_volume: float = 0.0
    notional: float = 0.0
    trade_count: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_levels_walked: int = 0
    sell_levels_walked: int = 0
    first_price: float | None = None
    last_price: float | None = None
    min_price: float | None = None
    max_price: float | None = None

    def record(self, price: float, volume: float, side: str | None = None) -> None:
        if volume <= 0:
            return
        self.executed_by_price[price] = self.executed_by_price.get(price, 0.0) + volume
        self.total_volume += volume
        self.notional += price * volume
        self.trade_count += 1
        if side == "buy":
            self.buy_executed_by_price[price] = (
                self.buy_executed_by_price.get(price, 0.0) + volume
            )
            self.buy_volume += volume
            self.buy_levels_walked += 1
        elif side == "sell":
            self.sell_executed_by_price[price] = (
                self.sell_executed_by_price.get(price, 0.0) + volume
            )
            self.sell_volume += volume
            self.sell_levels_walked += 1
        if self.first_price is None:
            self.first_price = price
        self.last_price = price
        self.min_price = price if self.min_price is None else min(self.min_price, price)
        self.max_price = price if self.max_price is None else max(self.max_price, price)


@dataclass
class _ExecutionResult:
    residual_market_buy: float
    residual_market_sell: float
    crossed_market_volume: float
    market_buy_volume: float = 0.0
    market_sell_volume: float = 0.0
    cancelled_volume_by_price: PriceMap = field(default_factory=dict)
    bid_cancelled_volume_by_price: PriceMap = field(default_factory=dict)
    ask_cancelled_volume_by_price: PriceMap = field(default_factory=dict)


@dataclass(frozen=True)
class _RealizedFlow:
    return_ticks: float = 0.0
    abs_return_ticks: float = 0.0
    execution_volume: float = 0.0
    submitted_buy_volume: float = 0.0
    submitted_sell_volume: float = 0.0
    rested_buy_volume: float = 0.0
    rested_sell_volume: float = 0.0
    executed_by_price: PriceMap = field(default_factory=dict)
    cancelled_by_price: PriceMap = field(default_factory=dict)
    bid_cancelled_by_price: PriceMap = field(default_factory=dict)
    ask_cancelled_by_price: PriceMap = field(default_factory=dict)
    flow_imbalance: float = 0.0
    intent_imbalance: float = 0.0
    rested_imbalance: float = 0.0
    residual_imbalance: float = 0.0
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None


@dataclass(frozen=True)
class _ParticipantPressureState:
    upward_push: float = 0.0
    downward_push: float = 0.0
    upward_resistance: float = 0.0
    downward_resistance: float = 0.0
    flow_continuation: float = 0.0
    absorption: float = 0.0
    exhaustion: float = 0.0
    signed_intent_memory: float = 0.0
    noise_pressure: float = 0.0


@dataclass(frozen=True)
class _ProcessedOrder:
    executed: float
    rested: float


@dataclass
class _MicrostructureState:
    activity: float = 0.0
    cancel_pressure: float = 0.0
    resiliency: float = 1.0
    recent_signed_return: float = 0.0
    recent_flow_imbalance: float = 0.0
    squeeze_pressure: float = 0.0
    activity_event: float = 0.0
    liquidity_stress: float = 0.0
    stress_side: float = 0.0
    spread_pressure: float = 0.0
    flow_persistence: float = 0.0
    meta_order_side: float = 0.0
    volatility_cluster: float = 0.0
    participation_burst: float = 0.0
    liquidity_drought: float = 0.0
    cancel_burst: float = 0.0
    arrival_cluster: float = 0.0
    book_vacuum: float = 0.0
    churn_pressure: float = 0.0
    displacement_pressure: float = 0.0


@dataclass(frozen=True)
class _MicrostructureInputs:
    execution_pressure: float
    return_shock: float
    volatility_shock: float
    imbalance_shock: float
    signed_return: float
    flow_imbalance: float
    squeeze_setup: float


@dataclass(frozen=True)
class _MarketConditionInputs:
    execution_pressure: float
    return_shock: float
    volatility_shock: float
    imbalance_shock: float
    signed_return: float
    flow_imbalance: float
    spread_gap: float
    depth_shortage: float
    cancel_pressure: float
    squeeze_setup: float


@dataclass(frozen=True)
class _MarketConditionState:
    trend_bias: float = 0.0
    volatility_pressure: float = 0.0
    liquidity_tightness: float = 0.0
    stress_pressure: float = 0.0
    participation_bias: float = 0.0
    squeeze_pressure: float = 0.0


@dataclass
class _StepComputationCache:
    basis_price: float
    mdf: MDFState | None = None
    micro: _MicrostructureState | None = None
    signals: MDFSignals | None = None
    spread_ticks: float | None = None
    entry_support_by_book_side: dict[str, PriceMap] = field(default_factory=dict)
