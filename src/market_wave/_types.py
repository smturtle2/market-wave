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
class _EntryFlow:
    orders: list[_IncomingOrder]
    buy_intent_by_price: PriceMap
    sell_intent_by_price: PriceMap


@dataclass
class _TradeStats:
    executed_by_price: PriceMap
    total_volume: float = 0.0
    notional: float = 0.0
    trade_count: int = 0
    last_price: float | None = None

    def record(self, price: float, volume: float) -> None:
        if volume <= 0:
            return
        self.executed_by_price[price] = self.executed_by_price.get(price, 0.0) + volume
        self.total_volume += volume
        self.notional += price * volume
        self.trade_count += 1
        self.last_price = price


@dataclass
class _ExecutionResult:
    residual_market_buy: float
    residual_market_sell: float
    crossed_market_volume: float
    market_buy_volume: float = 0.0
    market_sell_volume: float = 0.0


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
    last_cancelled_volume: float = 0.0


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


@dataclass(frozen=True)
class _MDFSideJudgment:
    fair_value_shift: float
    urgency: float
    patience: float
    opportunity: float
    liquidation: float
    liquidity_aversion: float
    pocket_bias: float
    uncertainty: float


@dataclass(frozen=True)
class _MDFJudgmentSample:
    buy: _MDFSideJudgment
    sell: _MDFSideJudgment


@dataclass
class _StepComputationCache:
    basis_price: float
    mdf: MDFState | None = None
    micro: _MicrostructureState | None = None
    signals: MDFSignals | None = None
    spread_ticks: float | None = None
    entry_probabilities_by_side: dict[str, PriceMap] = field(default_factory=dict)
    expected_depth_by_side: dict[str, float] = field(default_factory=dict)
    expected_volume_by_side_price: dict[tuple[str, float], float] = field(default_factory=dict)
