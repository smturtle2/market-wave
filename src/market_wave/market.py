from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import exp, isfinite, log1p, sin
from random import Random

from .distribution import (
    MDFContext,
    MDFSignals,
)
from .state import (
    IntensityState,
    LatentState,
    MarketState,
    MDFState,
    OrderBookState,
    PriceMap,
    StepInfo,
    TickMap,
)


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


@dataclass
class _OrderBook:
    bid_volume_by_price: PriceMap = field(default_factory=dict)
    ask_volume_by_price: PriceMap = field(default_factory=dict)
    _best_bid_cache: float | None = field(default=None, init=False, repr=False)
    _best_ask_cache: float | None = field(default=None, init=False, repr=False)
    _bid_best_dirty: bool = field(default=True, init=False, repr=False)
    _ask_best_dirty: bool = field(default=True, init=False, repr=False)

    def invalidate(self, side: str | None = None) -> None:
        if side in (None, "bid"):
            self._bid_best_dirty = True
        if side in (None, "ask"):
            self._ask_best_dirty = True

    def add_lot(
        self,
        price: float,
        volume: float,
        side: str,
        kind: str,
    ) -> None:
        del kind
        if volume <= 0:
            return
        volume_totals = self.volumes_for_side(side)
        volume_totals[price] = volume_totals.get(price, 0.0) + volume
        self.invalidate(side)

    def add_lots(self, volume_by_price: PriceMap, side: str, kind: str) -> None:
        for price, volume in volume_by_price.items():
            self.add_lot(price, volume, side, kind)

    def volumes_for_side(self, side: str) -> PriceMap:
        return self.bid_volume_by_price if side == "bid" else self.ask_volume_by_price

    def volumes_for_taker_side(self, side: str) -> PriceMap:
        return self.ask_volume_by_price if side == "buy" else self.bid_volume_by_price

    def adjust_volume(self, side: str, price: float, delta: float) -> None:
        if abs(delta) <= 1e-12:
            return
        volume_totals = self.volumes_for_side(side)
        next_volume = volume_totals.get(price, 0.0) + delta
        if next_volume > 1e-12:
            volume_totals[price] = next_volume
        else:
            volume_totals.pop(price, None)
        self.invalidate(side)

    def best_bid(self) -> float | None:
        if self._bid_best_dirty:
            self._best_bid_cache = max(
                (price for price, volume in self.bid_volume_by_price.items() if volume > 1e-12),
                default=None,
            )
            self._bid_best_dirty = False
        return self._best_bid_cache

    def best_ask(self) -> float | None:
        if self._ask_best_dirty:
            self._best_ask_cache = min(
                (price for price, volume in self.ask_volume_by_price.items() if volume > 1e-12),
                default=None,
            )
            self._ask_best_dirty = False
        return self._best_ask_cache

    def discard_empty_head(self, price: float, side: str) -> None:
        volume_totals = self.volumes_for_side(side)
        if volume_totals.get(price, 0.0) <= 1e-12:
            volume_totals.pop(price, None)
            self.invalidate(side)

    def clean(self) -> None:
        for side, volume_by_price in (
            ("bid", self.bid_volume_by_price),
            ("ask", self.ask_volume_by_price),
        ):
            for price in list(volume_by_price):
                if volume_by_price[price] <= 1e-12:
                    del volume_by_price[price]
                    self.invalidate(side)

    def snapshot(self) -> OrderBookState:
        return OrderBookState(
            bid_volume_by_price=self._drop_zeroes(self.bid_volume_by_price),
            ask_volume_by_price=self._drop_zeroes(self.ask_volume_by_price),
        )

    def near_touch_imbalance(self, current_price: float, gap: float) -> float:
        bid_depth = 0.0
        ask_depth = 0.0
        for price, volume in self.bid_volume_by_price.items():
            distance = max(1.0, abs(current_price - price) / gap)
            if distance <= 5:
                bid_depth += volume / distance
        for price, volume in self.ask_volume_by_price.items():
            distance = max(1.0, abs(price - current_price) / gap)
            if distance <= 5:
                ask_depth += volume / distance
        total = bid_depth + ask_depth
        if total <= 1e-12:
            return 0.0
        return max(-1.0, min(1.0, (bid_depth - ask_depth) / total))

    @staticmethod
    def _drop_zeroes(values: PriceMap) -> PriceMap:
        return {price: value for price, value in sorted(values.items()) if value > 1e-12}


class Market:
    @staticmethod
    def _regime_names() -> set[str]:
        return {"normal", "trend_up", "trend_down", "high_vol", "thin_liquidity", "squeeze"}

    def _condition_preset(self, regime: str) -> _MarketConditionState:
        presets = {
            "normal": _MarketConditionState(),
            "trend_up": _MarketConditionState(
                trend_bias=0.36,
                volatility_pressure=0.06,
                participation_bias=0.12,
            ),
            "trend_down": _MarketConditionState(
                trend_bias=-0.36,
                volatility_pressure=0.10,
                liquidity_tightness=0.05,
                stress_pressure=0.08,
                participation_bias=0.14,
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
                trend_bias=0.10,
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
            "mood": 0.12 * trend + 0.015 * squeeze,
            "trend": 0.22 * trend + 0.050 * squeeze,
            "volatility": self._clamp(
                1.0 + 0.36 * volatility + 0.08 * tightness + 0.04 * trend_abs,
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
        }

    def _condition_label(self, condition: _MarketConditionState) -> str:
        if condition.squeeze_pressure >= 0.42:
            return "squeeze"
        if condition.liquidity_tightness >= 0.62:
            return "thin_liquidity"
        if condition.volatility_pressure >= 0.58 or condition.stress_pressure >= 0.72:
            return "high_vol"
        if condition.trend_bias >= 0.22:
            return "trend_up"
        if condition.trend_bias <= -0.22:
            return "trend_down"
        return "normal"

    def _market_condition_inputs(
        self,
        latent: LatentState,
        imbalance: float,
    ) -> _MarketConditionInputs:
        execution_pressure = min(
            log1p(self._last_execution_volume / max(0.7, self.popularity)),
            2.0,
        )
        spread_gap = self._clamp((self._current_spread_ticks() - 2.0) / 5.0, 0.0, 1.0)
        bid_depth = self._nearby_book_depth("bid", self.state.price)
        ask_depth = self._nearby_book_depth("ask", self.state.price)
        expected_depth = max(0.8, self.popularity * (3.6 + 0.30 * self.grid_radius))
        depth_shortage = self._clamp(
            1.0 - (bid_depth + ask_depth) / max(expected_depth, 1e-12),
            0.0,
            1.0,
        )
        volatility_jump = max(0.0, latent.volatility - self.state.latent.volatility)
        return _MarketConditionInputs(
            execution_pressure=execution_pressure,
            return_shock=1.0 - exp(-self._last_abs_return_ticks / 1.4),
            volatility_shock=1.0 - exp(-volatility_jump / 0.25),
            imbalance_shock=abs(imbalance - self._last_imbalance),
            signed_return=self._last_return_ticks,
            flow_imbalance=self._last_imbalance,
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
        preset = self._condition_preset(self.regime)
        preset_weight = 0.22 if self.regime == "auto" else 0.62
        adaptive_weight = 1.0 - preset_weight
        noise_scale = 0.010 + 0.018 * self.augmentation_strength

        def signed_next(
            previous_value: float,
            preset_value: float,
            evidence: float,
            *,
            memory: float,
            low: float,
            high: float,
        ) -> float:
            target = preset_weight * preset_value + adaptive_weight * evidence
            noise = self._rng.gauss(0.0, noise_scale)
            return self._clamp(memory * previous_value + (1.0 - memory) * target + noise, low, high)

        def pressure_next(
            previous_value: float,
            preset_value: float,
            evidence: float,
            *,
            memory: float,
            high: float,
        ) -> float:
            mixed = preset_weight * preset_value + adaptive_weight * evidence
            target = max(0.72 * preset_value, mixed)
            noise = self._rng.gauss(0.0, noise_scale)
            return self._clamp(memory * previous_value + (1.0 - memory) * target + noise, 0.0, high)

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
            0.82 * inputs.squeeze_setup
            + 0.18 * max(0.0, signed_return),
            0.0,
            1.5,
        )

        return _MarketConditionState(
            trend_bias=signed_next(
                previous.trend_bias,
                preset.trend_bias,
                trend_evidence,
                memory=0.86,
                low=-1.0,
                high=1.0,
            ),
            volatility_pressure=pressure_next(
                previous.volatility_pressure,
                preset.volatility_pressure,
                volatility_evidence,
                memory=0.82,
                high=1.8,
            ),
            liquidity_tightness=pressure_next(
                previous.liquidity_tightness,
                preset.liquidity_tightness,
                liquidity_evidence,
                memory=0.84,
                high=1.8,
            ),
            stress_pressure=pressure_next(
                previous.stress_pressure,
                preset.stress_pressure,
                stress_evidence,
                memory=0.80,
                high=1.8,
            ),
            participation_bias=signed_next(
                previous.participation_bias,
                preset.participation_bias,
                participation_evidence,
                memory=0.82,
                low=-0.8,
                high=1.2,
            ),
            squeeze_pressure=pressure_next(
                previous.squeeze_pressure,
                preset.squeeze_pressure,
                squeeze_evidence,
                memory=0.78,
                high=1.5,
            ),
        )

    def __init__(
        self,
        initial_price: float,
        gap: float,
        popularity: float = 1.0,
        seed: int | None = None,
        grid_radius: int = 20,
        augmentation_strength: float = 0.0,
        regime: str = "normal",
    ) -> None:
        if not isfinite(initial_price) or initial_price <= 0:
            raise ValueError("initial_price must be a positive finite number")
        if not isfinite(gap) or gap <= 0:
            raise ValueError("gap must be a positive finite number")
        if grid_radius < 1:
            raise ValueError("grid_radius must be at least 1")
        if not isfinite(popularity) or popularity < 0:
            raise ValueError("popularity must be a non-negative finite number")
        if not isfinite(augmentation_strength) or augmentation_strength < 0:
            raise ValueError("augmentation_strength must be a non-negative finite number")
        if regime not in self._regime_names() | {"auto"}:
            raise ValueError("regime must be one of the supported regimes or 'auto'")

        self.gap = float(gap)
        self._gap_is_integer = self.gap.is_integer()
        self.popularity = float(popularity)
        self._min_price = self.gap
        self.grid_radius = int(grid_radius)
        self.augmentation_strength = float(augmentation_strength)
        self._entry_mdf_memory = 0.18
        self._mdf_floor_mix = 0.012
        self._entry_noise_mix = 0.023
        self._entry_noise_sigma_ticks = self._clamp(0.14 * self.grid_radius, 1.25, 2.50)
        self.regime = regime
        self._market_condition = self._condition_preset(regime)
        self._active_regime = self._condition_label(self._market_condition)
        self._active_settings = self._condition_settings(self._market_condition)
        self._rng = Random(seed)
        self._seed = seed
        self.history: list[StepInfo] = []
        self._orderbook = _OrderBook()
        self._last_return_ticks = 0.0
        self._last_abs_return_ticks = 0.0
        self._last_imbalance = 0.0
        self._last_execution_volume = 0.0
        self._last_executed_by_price: PriceMap = {}
        self._price_residual_ticks = 0.0
        self._microstructure = _MicrostructureState()
        self._mdf_memory: dict[str, dict[int, float]] = {}
        self._mdf_judgment_memory: dict[str, _MDFSideJudgment] = {}

        price = self._snap_price(float(initial_price))
        grid = self._price_grid(price)
        tick_grid = self.relative_tick_grid()
        current_tick = self.price_to_tick(price)
        intensity = IntensityState(total=0.0, buy=0.0, sell=0.0, buy_ratio=0.5, sell_ratio=0.5)
        latent = self._initial_latent()
        uniform = 1.0 / len(grid)
        initial_mdf_by_price = {price_level: uniform for price_level in grid}
        initial_mdf = {tick: 1.0 / len(tick_grid) for tick in tick_grid}
        mdf = MDFState(
            relative_ticks=tick_grid,
            buy_entry_mdf=initial_mdf,
            sell_entry_mdf=initial_mdf.copy(),
            buy_entry_mdf_by_price=initial_mdf_by_price,
            sell_entry_mdf_by_price=initial_mdf_by_price.copy(),
        )
        self.state = MarketState(
            price=price,
            step_index=0,
            current_tick=current_tick,
            tick_grid=tick_grid,
            intensity=intensity,
            latent=latent,
            price_grid=grid,
            mdf=mdf,
            orderbook=OrderBookState(),
        )

    def _initial_latent(self) -> LatentState:
        settings = self._active_settings
        mood = self._clamp(
            self._rng.gauss(settings["mood"], 0.18),
            -0.7,
            0.7,
        )
        trend = self._clamp(
            self._rng.gauss(settings["trend"] + 0.20 * mood, 0.14),
            -0.7,
            0.7,
        )
        volatility = self._clamp(
            0.12
            + 0.18 * self._rng.random()
            + abs(self._rng.gauss(0.0, 0.11))
            + settings["volatility_bias"],
            0.05,
            0.75,
        )
        return LatentState(mood=mood, trend=trend, volatility=volatility)

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def tick_size(self) -> float:
        return self.gap

    def snapshot(self) -> MarketState:
        """Return a deep copy of the current public market state.

        ``Market.state`` is retained as the current alpha-compatible state object
        and contains mutable plain containers. Use this method when downstream
        code needs to inspect state without risking accidental mutation of the
        live market object.
        """

        return deepcopy(self.state)

    def step(self, n: int, *, keep_history: bool = True) -> list[StepInfo]:
        if n < 0:
            raise ValueError("n must be non-negative")

        steps = [self._step_once() for _ in range(n)]
        if keep_history:
            self.history.extend(steps)
        return steps

    def stream(self, n: int, *, keep_history: bool = False):
        if n < 0:
            raise ValueError("n must be non-negative")
        for _ in range(n):
            step = self._step_once()
            if keep_history:
                self.history.append(step)
            yield step

    def history_records(self) -> list[dict]:
        return [step.to_dict() for step in self.history]

    def plot_history(
        self,
        *,
        ax=None,
        style: str = "market_wave",
        last: int | None = None,
        layout: str = "panel",
        orderbook: bool | None = None,
        orderbook_snapshot: str = "after",
        orderbook_depth: int | None = None,
    ):
        if not self.history:
            raise ValueError("history is empty; call step(n) before plotting")
        if last is not None and last <= 0:
            raise ValueError("last must be positive")
        if layout not in {"panel", "overlay"}:
            raise ValueError("layout must be 'panel' or 'overlay'")
        if orderbook_snapshot not in {"before", "after"}:
            raise ValueError("orderbook_snapshot must be 'before' or 'after'")
        if orderbook_depth is not None and orderbook_depth <= 0:
            raise ValueError("orderbook_depth must be positive")
        include_orderbook = layout == "panel" if orderbook is None else orderbook
        if layout == "overlay" and include_orderbook:
            raise ValueError("orderbook heatmap is only supported with layout='panel'")
        if ax is not None and layout == "panel":
            raise ValueError("ax is only supported with layout='overlay'")

        import matplotlib.pyplot as plt

        steps = self.history[-last:] if last is not None else self.history
        context = self._plot_style_context(style)
        with plt.style.context(context):
            if layout == "panel" and ax is None:
                return self._plot_history_panel(
                    plt,
                    steps,
                    style,
                    orderbook=include_orderbook,
                    orderbook_snapshot=orderbook_snapshot,
                    orderbook_depth=orderbook_depth,
                )
            return self._plot_history_overlay(plt, steps, ax, style)

    def plot(
        self,
        *,
        ax=None,
        style: str = "market_wave",
        last: int | None = None,
        layout: str = "panel",
        orderbook: bool | None = None,
        orderbook_snapshot: str = "after",
        orderbook_depth: int | None = None,
    ):
        return self.plot_history(
            ax=ax,
            style=style,
            last=last,
            layout=layout,
            orderbook=orderbook,
            orderbook_snapshot=orderbook_snapshot,
            orderbook_depth=orderbook_depth,
        )

    def _plot_style_context(self, style: str) -> str:
        if style == "market_wave":
            return "default"
        if style == "market_wave_dark":
            return "dark_background"
        return style

    def _plot_history_panel(
        self,
        plt,
        steps: list[StepInfo],
        style: str,
        *,
        orderbook: bool,
        orderbook_snapshot: str,
        orderbook_depth: int | None,
    ):
        if orderbook:
            fig, axes = plt.subplots(
                5,
                1,
                figsize=(12, 10.4),
                sharex=True,
                constrained_layout=True,
                gridspec_kw={"height_ratios": [3.2, 1.25, 1.25, 1.1, 1.1]},
            )
            price_ax, ask_ax, bid_ax, volume_ax, imbalance_ax = axes
        else:
            fig, axes = plt.subplots(
                3,
                1,
                figsize=(12, 7.8),
                sharex=True,
                constrained_layout=True,
                gridspec_kw={"height_ratios": [3.4, 1.25, 1.25]},
            )
            price_ax, volume_ax, imbalance_ax = axes
        x_values, prices, vwap_x, vwap_y, volumes, imbalances = self._plot_series(steps)
        colors = self._plot_colors(style)

        price_ax.plot(x_values, prices, color=colors["price"], linewidth=2.0, label="price")
        if vwap_x:
            price_ax.plot(
                vwap_x,
                vwap_y,
                color=colors["vwap"],
                linewidth=1.25,
                linestyle="--",
                alpha=0.9,
                label="vwap",
            )
        if orderbook:
            self._plot_orderbook_heatmap_axes(
                fig,
                bid_ax,
                ask_ax,
                steps,
                colors,
                snapshot=orderbook_snapshot,
                depth=orderbook_depth,
            )
        volume_ax.bar(x_values, volumes, width=0.86, color=colors["volume"], alpha=0.72)
        imbalance_colors = [
            colors["buy_imbalance"] if imbalance >= 0 else colors["sell_imbalance"]
            for imbalance in imbalances
        ]
        imbalance_ax.bar(x_values, imbalances, width=0.86, color=imbalance_colors, alpha=0.75)
        imbalance_ax.axhline(0, color=colors["zero"], linewidth=0.9)

        price_ax.set_title(
            f"market-wave simulation ({len(steps)} steps)",
            loc="left",
            pad=12,
            fontsize=13,
            fontweight="bold",
        )
        price_ax.set_ylabel("price")
        volume_ax.set_ylabel("volume")
        imbalance_ax.set_ylabel("imbalance")
        imbalance_ax.set_xlabel("step")
        price_ax.legend(loc="upper left", frameon=False, ncols=2)
        self._style_market_wave_axes(fig, axes, colors)
        if orderbook:
            bid_ax.grid(False)
            ask_ax.grid(False)
        return fig, price_ax

    def _plot_history_overlay(self, plt, steps: list[StepInfo], ax, style: str):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6.4), constrained_layout=True)
        else:
            fig = ax.figure
        volume_ax = ax.twinx()
        x_values, prices, vwap_x, vwap_y, volumes, imbalances = self._plot_series(steps)
        colors = self._plot_colors(style)
        baseline = min(prices)
        price_range = max(prices) - baseline
        imbalance_scale = price_range * 0.10 if price_range > 0 else max(self.gap, 1.0)
        positive = [
            imbalance * imbalance_scale if imbalance > 0 else 0.0 for imbalance in imbalances
        ]
        negative = [
            imbalance * imbalance_scale if imbalance < 0 else 0.0 for imbalance in imbalances
        ]

        volume_ax.bar(
            x_values,
            volumes,
            width=0.86,
            color=colors["volume"],
            alpha=0.24,
            label="executed volume",
            zorder=1,
        )
        ax.plot(x_values, prices, color=colors["price"], linewidth=2.0, label="price", zorder=4)
        if vwap_x:
            ax.plot(
                vwap_x,
                vwap_y,
                color=colors["vwap"],
                linewidth=1.2,
                linestyle="--",
                alpha=0.85,
                label="vwap",
                zorder=3,
            )
        ax.bar(
            x_values,
            positive,
            bottom=baseline,
            width=0.82,
            color=colors["buy_imbalance"],
            alpha=0.22,
            label="buy imbalance",
            zorder=2,
        )
        ax.bar(
            x_values,
            negative,
            bottom=baseline,
            width=0.82,
            color=colors["sell_imbalance"],
            alpha=0.22,
            label="sell imbalance",
            zorder=2,
        )

        ax.set_title(
            f"market-wave simulation ({len(steps)} steps)",
            loc="left",
            pad=12,
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("step")
        ax.set_ylabel("price")
        volume_ax.set_ylabel("executed volume")
        self._style_market_wave_axes(fig, [ax, volume_ax], colors)
        volume_ax.grid(False)
        handles, labels = ax.get_legend_handles_labels()
        volume_handles, volume_labels = volume_ax.get_legend_handles_labels()
        ax.legend(
            handles + volume_handles,
            labels + volume_labels,
            loc="upper left",
            frameon=False,
            ncols=2,
        )
        return fig, ax

    def _plot_series(self, steps: list[StepInfo]):
        x_values = [step.step_index for step in steps]
        prices = [step.price_after for step in steps]
        vwap_x = [step.step_index for step in steps if step.vwap_price is not None]
        vwap_y = [step.vwap_price for step in steps if step.vwap_price is not None]
        volumes = [step.total_executed_volume for step in steps]
        imbalances = [step.order_flow_imbalance for step in steps]
        return x_values, prices, vwap_x, vwap_y, volumes, imbalances

    def _plot_orderbook_heatmap_axes(
        self,
        fig,
        bid_ax,
        ask_ax,
        steps: list[StepInfo],
        colors: dict[str, str],
        *,
        snapshot: str,
        depth: int | None,
    ) -> None:
        import matplotlib.colors as mcolors

        x_values, levels, bid_matrix, ask_matrix = self._orderbook_heatmap_matrices(
            steps, snapshot=snapshot, depth=depth
        )
        extent = self._heatmap_extent(x_values, levels)
        bid_map = mcolors.LinearSegmentedColormap.from_list(
            "market_wave_bid_depth",
            [colors["panel"], colors["bid_depth_mid"], colors["bid_depth"]],
        )
        ask_map = mcolors.LinearSegmentedColormap.from_list(
            "market_wave_ask_depth",
            [colors["panel"], colors["ask_depth_mid"], colors["ask_depth"]],
        )
        bid_ax.imshow(
            bid_matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=extent,
            cmap=bid_map,
            vmin=0.0,
            vmax=self._heatmap_vmax(bid_matrix),
        )
        ask_ax.imshow(
            ask_matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=extent,
            cmap=ask_map,
            vmin=0.0,
            vmax=self._heatmap_vmax(ask_matrix),
        )
        for axis, label in ((bid_ax, "bid level"), (ask_ax, "ask level")):
            axis.set_ylabel(label)
            axis.set_yticks(self._level_ticks(levels))
            axis.set_ylim(0.5, max(levels) + 0.5)
        bid_ax.invert_yaxis()

    def _orderbook_heatmap_matrices(
        self,
        steps: list[StepInfo],
        *,
        snapshot: str,
        depth: int | None,
    ):
        depth = depth or min(self.grid_radius, 20)
        x_values = [step.step_index for step in steps]
        levels = list(range(1, depth + 1))
        bid_matrix = [[0.0 for _ in steps] for _ in levels]
        ask_matrix = [[0.0 for _ in steps] for _ in levels]
        for column, step in enumerate(steps):
            orderbook = step.orderbook_after if snapshot == "after" else step.orderbook_before
            basis_tick = step.tick_after if snapshot == "after" else step.tick_before
            for price, volume in orderbook.bid_volume_by_price.items():
                level = basis_tick - self.price_to_tick(price)
                if 1 <= level <= depth:
                    bid_matrix[level - 1][column] += volume
            for price, volume in orderbook.ask_volume_by_price.items():
                level = self.price_to_tick(price) - basis_tick
                if 1 <= level <= depth:
                    ask_matrix[level - 1][column] += volume
        return x_values, levels, bid_matrix, ask_matrix

    def _heatmap_extent(self, x_values: list[int], levels: list[int]) -> list[float]:
        if not x_values:
            return [0.5, 1.5, 0.5, len(levels) + 0.5]
        if len(x_values) == 1:
            left = x_values[0] - 0.5
            right = x_values[0] + 0.5
        else:
            step_width = min(
                abs(right - left) for left, right in zip(x_values, x_values[1:], strict=False)
            )
            left = x_values[0] - step_width / 2
            right = x_values[-1] + step_width / 2
        return [left, right, 0.5, len(levels) + 0.5]

    def _heatmap_vmax(self, matrix: list[list[float]]) -> float:
        values = sorted(value for row in matrix for value in row if value > 0)
        if not values:
            return 1.0
        index = min(len(values) - 1, int(0.95 * (len(values) - 1)))
        return max(values[index], 1e-12)

    def _level_ticks(self, levels: list[int]) -> list[int]:
        if len(levels) <= 6:
            return levels
        stride = max(1, len(levels) // 4)
        ticks = [levels[0], *levels[stride - 1 :: stride]]
        if ticks[-1] != levels[-1]:
            ticks.append(levels[-1])
        return sorted(set(ticks))

    def _plot_colors(self, style: str) -> dict[str, str]:
        if style == "market_wave_dark":
            return {
                "figure": "#09090b",
                "panel": "#111827",
                "grid": "#334155",
                "text": "#e5e7eb",
                "muted": "#94a3b8",
                "price": "#22d3ee",
                "vwap": "#f472b6",
                "volume": "#64748b",
                "buy_imbalance": "#fb7185",
                "sell_imbalance": "#38bdf8",
                "bid_depth_mid": "#7f1d1d",
                "bid_depth": "#fb7185",
                "ask_depth_mid": "#075985",
                "ask_depth": "#38bdf8",
                "zero": "#94a3b8",
            }
        return {
            "figure": "#f8fafc",
            "panel": "#ffffff",
            "grid": "#dbe3ef",
            "text": "#0f172a",
            "muted": "#64748b",
            "price": "#2563eb",
            "vwap": "#9333ea",
            "volume": "#94a3b8",
            "buy_imbalance": "#dc2626",
            "sell_imbalance": "#2563eb",
            "bid_depth_mid": "#fecaca",
            "bid_depth": "#dc2626",
            "ask_depth_mid": "#bfdbfe",
            "ask_depth": "#2563eb",
            "zero": "#94a3b8",
        }

    def _style_market_wave_axes(self, fig, axes, colors: dict[str, str]) -> None:
        fig.set_facecolor(colors["figure"])
        for axis in axes:
            axis.set_facecolor(colors["panel"])
            axis.grid(True, which="major", color=colors["grid"], alpha=0.75, linewidth=0.8)
            axis.margins(x=0.01)
            axis.tick_params(colors=colors["muted"])
            axis.xaxis.label.set_color(colors["muted"])
            axis.yaxis.label.set_color(colors["muted"])
            axis.title.set_color(colors["text"])
            for spine in axis.spines.values():
                spine.set_color(colors["grid"])

    def _step_once(self) -> StepInfo:
        state = self.state
        step_index = state.step_index + 1
        price_before = state.price
        self._clean_orderbook()

        price_grid = self._price_grid(price_before)
        orderbook_before = self._snapshot_orderbook()
        best_bid_before = self._best_bid()
        best_ask_before = self._best_ask()
        spread_before = self._spread(best_bid_before, best_ask_before)
        pre_imbalance = self._near_touch_imbalance(price_before)

        condition_inputs = self._market_condition_inputs(state.latent, pre_imbalance)
        self._market_condition = self._next_market_condition(
            self._market_condition,
            condition_inputs,
        )
        self._active_settings = self._condition_settings(self._market_condition)
        self._active_regime = self._condition_label(self._market_condition)
        latent = self._next_latent(state.latent)
        micro = self._next_microstructure_state(state.latent, latent, pre_imbalance)
        intensity = self._next_intensity(latent, micro)
        pre_trade_cache = _StepComputationCache(price_before, micro=micro)
        mdf = self._next_mdf(
            price_before,
            price_grid,
            latent,
            step_index=step_index,
            update_memory=True,
            cache=pre_trade_cache,
        )
        pre_trade_cache.mdf = mdf

        cancelled_volume = self._cancel_orders(price_before, latent, micro, mdf, pre_trade_cache)
        entry_flow = self._entry_flow(intensity, mdf)

        stats = _TradeStats(executed_by_price={})
        execution = self._execute_market_flows(
            entry_orders=entry_flow.orders,
            stats=stats,
        )
        self._clean_orderbook()

        price_after = self._next_price_after_trading(price_before, stats, execution, latent)
        price_after = self._snap_price(price_after)
        self._last_return_ticks = (price_after - price_before) / self.gap
        self._last_abs_return_ticks = abs(self._last_return_ticks)
        self._last_execution_volume = stats.total_volume
        self._last_executed_by_price = self._drop_zeroes(stats.executed_by_price)
        total_market_buy = execution.market_buy_volume
        total_market_sell = execution.market_sell_volume
        residual_market_buy = execution.residual_market_buy
        residual_market_sell = execution.residual_market_sell
        order_flow_imbalance = self._step_order_flow_imbalance(
            price_after,
            execution,
            cancelled_volume,
        )
        self._last_imbalance = order_flow_imbalance
        self._trim_orderbook_through_last_price(price_after)
        self._prune_orderbook_window(price_after)
        post_trade_mdf = self._reproject_mdf(price_after, mdf)
        post_trade_cache = _StepComputationCache(price_after, mdf=post_trade_mdf, micro=micro)
        self._add_post_event_quote_arrivals(
            price_after,
            latent,
            micro,
            stats,
            order_flow_imbalance,
            post_trade_mdf,
            post_trade_cache,
        )
        self._refresh_post_event_deep_quotes(
            cancelled_volume,
            price_after,
            micro,
            post_trade_mdf,
            post_trade_cache,
        )
        cancelled_volume = self._drop_zeroes(cancelled_volume)
        micro.last_cancelled_volume = sum(cancelled_volume.values())
        self._trim_orderbook_through_last_price(price_after)
        self._clean_orderbook()

        state_grid = self._price_grid(price_after)
        state_mdf = post_trade_mdf
        orderbook_after = self._snapshot_orderbook()
        best_bid_after = self._best_bid()
        best_ask_after = self._best_ask()
        spread_after = self._spread(best_bid_after, best_ask_after)

        entry_volume = self._merge_maps(
            entry_flow.buy_intent_by_price,
            entry_flow.sell_intent_by_price,
        )
        buy_volume = entry_flow.buy_intent_by_price
        sell_volume = entry_flow.sell_intent_by_price
        vwap = stats.notional / stats.total_volume if stats.total_volume > 0 else None
        step_info = StepInfo(
            step_index=step_index,
            price_before=price_before,
            price_after=price_after,
            price_change=price_after - price_before,
            tick_before=self.price_to_tick(price_before),
            tick_after=self.price_to_tick(price_after),
            tick_change=self.price_to_tick(price_after) - self.price_to_tick(price_before),
            intensity=intensity,
            mood=latent.mood,
            trend=latent.trend,
            volatility=latent.volatility,
            regime=self._active_regime,
            augmentation_strength=self.augmentation_strength,
            price_grid=price_grid,
            mdf_price_basis=price_before,
            relative_ticks=mdf.relative_ticks,
            buy_entry_mdf=mdf.buy_entry_mdf,
            sell_entry_mdf=mdf.sell_entry_mdf,
            buy_entry_mdf_by_price=mdf.buy_entry_mdf_by_price,
            sell_entry_mdf_by_price=mdf.sell_entry_mdf_by_price,
            buy_volume_by_price=buy_volume,
            sell_volume_by_price=sell_volume,
            entry_volume_by_price=entry_volume,
            cancelled_volume_by_price=cancelled_volume,
            executed_volume_by_price=self._drop_zeroes(stats.executed_by_price),
            total_executed_volume=stats.total_volume,
            market_buy_volume=total_market_buy,
            market_sell_volume=total_market_sell,
            crossed_market_volume=execution.crossed_market_volume,
            residual_market_buy_volume=residual_market_buy,
            residual_market_sell_volume=residual_market_sell,
            trade_count=stats.trade_count,
            vwap_price=vwap,
            best_bid_before=best_bid_before,
            best_ask_before=best_ask_before,
            best_bid_after=best_bid_after,
            best_ask_after=best_ask_after,
            spread_before=spread_before,
            spread_after=spread_after,
            order_flow_imbalance=order_flow_imbalance,
            orderbook_before=orderbook_before,
            orderbook_after=orderbook_after,
        )

        self.state = MarketState(
            price=price_after,
            step_index=step_index,
            current_tick=self.price_to_tick(price_after),
            tick_grid=self.relative_tick_grid(),
            intensity=intensity,
            latent=latent,
            price_grid=state_grid,
            mdf=state_mdf,
            orderbook=orderbook_after,
        )
        return step_info

    def _age_state(self) -> None:
        return None

    def _step_order_flow_imbalance(
        self,
        current_price: float,
        execution: _ExecutionResult,
        cancelled_volume: PriceMap,
    ) -> float:
        trade_delta = execution.market_buy_volume - execution.market_sell_volume
        trade_total = execution.market_buy_volume + execution.market_sell_volume
        residual_delta = execution.residual_market_buy - execution.residual_market_sell
        residual_total = execution.residual_market_buy + execution.residual_market_sell
        cancel_delta = 0.0
        cancel_total = 0.0
        for price, volume in cancelled_volume.items():
            if volume <= 1e-12:
                continue
            cancel_total += volume
            if price < current_price:
                cancel_delta -= volume
            elif price > current_price:
                cancel_delta += volume
        near_depth = self._nearby_book_depth("bid", current_price) + self._nearby_book_depth(
            "ask",
            current_price,
        )
        near_imbalance = self._near_touch_imbalance(current_price)
        context_numerator = (
            0.24 * residual_delta
            + 0.12 * cancel_delta
            + 0.10 * near_imbalance * near_depth
        )
        context_denominator = (
            0.50 * residual_total
            + 0.45 * cancel_total
            + 0.32 * near_depth
            + 2.0 * self._mean_child_order_size()
        )
        if context_denominator <= 1e-12 and trade_total <= 1e-12:
            return 0.0
        context_imbalance = (
            context_numerator / context_denominator
            if context_denominator > 1e-12
            else 0.0
        )
        if trade_total <= 1e-12:
            return self._clamp(context_imbalance, -1.0, 1.0)
        trade_imbalance = self._clamp(trade_delta / trade_total, -1.0, 1.0)
        trade_confidence = self._clamp(
            trade_total
            / (
                trade_total
                + 0.50 * residual_total
                + 0.35 * cancel_total
                + 0.22 * near_depth
                + self._mean_child_order_size()
            ),
            0.62,
            0.92,
        )
        return self._clamp(
            trade_confidence * trade_imbalance
            + (1.0 - trade_confidence) * context_imbalance,
            -1.0,
            1.0,
        )

    def _next_price_after_trading(
        self,
        price_before: float,
        stats: _TradeStats,
        execution: _ExecutionResult,
        latent: LatentState | None = None,
    ) -> float:
        if stats.total_volume <= 0 or stats.last_price is None:
            return price_before
        latent = latent or self.state.latent
        regime = self._active_settings
        vwap = stats.notional / stats.total_volume
        execution_price = 0.68 * stats.last_price + 0.32 * vwap
        execution_move_ticks = (execution_price - price_before) / self.gap
        flow_total = execution.market_buy_volume + execution.market_sell_volume
        flow_imbalance = (
            self._clamp(
                (execution.market_buy_volume - execution.market_sell_volume) / flow_total,
                -1.0,
                1.0,
            )
            if flow_total > 1e-12
            else 0.0
        )
        if abs(execution_move_ticks) <= 1e-12 and abs(flow_imbalance) <= 1e-12:
            return price_before
        volume_confidence = self._clamp(
            stats.total_volume / max(0.7, self.popularity),
            0.35,
            1.65,
        )
        volatility_response = self._clamp(
            0.86 + 0.18 * regime["volatility"] + 0.14 * latent.volatility,
            0.90,
            1.50,
        )
        # Last price shows where trades occurred; flow only nudges revealed pressure.
        price_discovery = self._clamp(abs(execution_move_ticks) / 1.35, 0.25, 1.0)
        flow_move_ticks = (
            flow_imbalance
            * (0.06 + 0.045 * volume_confidence)
            * price_discovery
            * volatility_response
        )
        proposed_ticks = (
            execution_move_ticks * (0.52 + 0.12 * volume_confidence) * volatility_response
        )
        proposed_ticks += flow_move_ticks
        max_move_ticks = self._clamp(1.0 + 0.45 * latent.volatility, 1.0, 2.0)
        proposed_ticks = self._clamp(proposed_ticks, -max_move_ticks, max_move_ticks)
        self._price_residual_ticks = self._clamp(
            self._price_residual_ticks + proposed_ticks,
            -max_move_ticks,
            max_move_ticks,
        )
        emitted_ticks = int(round(self._price_residual_ticks))
        if emitted_ticks == 0:
            return price_before
        self._price_residual_ticks -= emitted_ticks
        return max(self._min_price, price_before + emitted_ticks * self.gap)

    def _price_grid(self, center_price: float) -> list[float]:
        center = self._snap_price(center_price)
        return self._dedupe_prices(
            self._snap_price(center + offset * self.gap)
            for offset in range(-self.grid_radius, self.grid_radius + 1)
        )

    def price_to_tick(self, price: float) -> int:
        return max(1, int(round(price / self.gap)))

    def tick_to_price(self, tick: int) -> float:
        snapped = max(1, int(tick)) * self.gap
        if self._gap_is_integer:
            return float(int(snapped))
        return self._clean_number(snapped)

    def relative_tick_grid(self) -> list[int]:
        return list(range(-self.grid_radius, self.grid_radius + 1))

    def _snap_price(self, price: float) -> float:
        tick = max(1, int(round(price / self.gap)))
        snapped = tick * self.gap
        if self._gap_is_integer:
            return float(int(snapped))
        return self._clean_number(snapped)

    def _clean_number(self, value: float) -> float:
        rounded = round(value, 10)
        if rounded.is_integer():
            return float(int(rounded))
        return rounded

    def _next_latent(self, latent: LatentState) -> LatentState:
        regime = self._active_settings
        micro = self._microstructure
        mood_noise = self._rng.gauss(0.0, 0.11)
        trend_noise = self._rng.gauss(0.0, 0.08)
        jump_probability = 0.024 * regime["volatility"] * (1.0 + self.augmentation_strength)
        jump = self._rng.gauss(0.0, 0.55) if self._rng.random() < jump_probability else 0.0
        signed_flow = 0.08 * self._last_imbalance - 0.03 * self._last_return_ticks
        exhaustion = self._trend_exhaustion(micro)
        squeeze_impulse = self._squeeze_impulse(micro)
        mood_target = regime["mood"] + 0.06 * latent.trend - 0.025 * exhaustion

        mood = self._clamp(
            0.74 * latent.mood
            + 0.26 * mood_target
            + signed_flow
            + mood_noise
            + 0.22 * jump,
            -1.0,
            1.0,
        )
        trend_target = regime["trend"] + 0.10 * mood
        trend_target += 0.05 * squeeze_impulse
        trend_target -= regime["trend_exhaustion"] * exhaustion
        trend = self._clamp(
            0.74 * latent.trend
            + 0.26 * trend_target
            - 0.06 * self._last_return_ticks
            + 0.08 * jump
            + trend_noise,
            -1.0,
            1.0,
        )
        noise_shock = max(0.0, abs(mood_noise) - 0.08)
        shock = 0.32 * noise_shock + abs(jump) * 0.24
        execution_pressure = min(
            self._last_execution_volume / max(0.7, self.popularity),
            4.0,
        )
        # Volatility is a regime target plus fresh shocks, not an accumulated level.
        realized = 0.030 * min(self._last_abs_return_ticks, 3.0) + 0.004 * execution_pressure
        volatility_target = 0.28 + 0.22 * regime["volatility"] + regime["volatility_bias"]
        volatility = self._clamp(
            0.84 * latent.volatility
            + 0.16 * volatility_target
            + shock
            + realized
            + 0.008 * abs(self._last_imbalance),
            0.04,
            1.55,
        )
        return LatentState(mood=mood, trend=trend, volatility=volatility)

    def _trend_exhaustion(self, micro: _MicrostructureState) -> float:
        return self._clamp(
            0.70 * self._clamp(micro.recent_signed_return / 8.0, -1.0, 1.0)
            + 0.30 * micro.recent_flow_imbalance,
            -1.0,
            1.0,
        )

    def _squeeze_impulse(self, micro: _MicrostructureState) -> float:
        return self._clamp(micro.squeeze_pressure, 0.0, 1.5)

    def _next_intensity(
        self,
        latent: LatentState,
        micro: _MicrostructureState | None = None,
    ) -> IntensityState:
        regime = self._active_settings
        micro = micro or self._microstructure
        augmentation = 1.0 + self.augmentation_strength * self._rng.uniform(-0.18, 0.28)
        flow_reversal = self._flow_reversal_pressure(micro)
        activity = micro.activity
        activity_multiplier = 1.0 + 0.28 * self._clamp(activity, 0.0, 2.2)
        event_multiplier = 1.0 + 0.18 * self._clamp(micro.activity_event, 0.0, 1.8)
        dry_up = self._clamp(1.0 - 0.08 * micro.cancel_pressure, 0.74, 1.0)
        raw_total = (
            self.popularity
            * (1.0 + 1.35 * latent.volatility)
            * regime["intensity"]
            * augmentation
            * activity_multiplier
            * event_multiplier
            * dry_up
        )
        previous_total = self.state.intensity.total
        total = (
            0.55 * raw_total + 0.45 * previous_total
            if previous_total > 1e-12
            else raw_total
        )
        if previous_total > 1e-12:
            previous_pressure = self._clamp(
                previous_total / max(self.popularity * regime["intensity"], 1e-12) - 1.0,
                0.0,
                2.0,
            )
            total *= 1.0 + 0.08 * previous_pressure
        buy_ratio = self._clamp(
            0.5
            + 0.24 * latent.mood
            + 0.18 * latent.trend
            + 0.04 * self._last_imbalance
            + 0.025 * self._squeeze_impulse(micro)
            - 0.06 * self._last_return_ticks
            - regime["flow_reversal"] * flow_reversal,
            0.08,
            0.92,
        )
        sell_ratio = 1.0 - buy_ratio
        return IntensityState(
            total=total,
            buy=total * buy_ratio,
            sell=total * sell_ratio,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
        )

    def _flow_reversal_pressure(self, micro: _MicrostructureState) -> float:
        return self._clamp(
            0.65 * micro.recent_flow_imbalance
            + 0.35 * self._clamp(micro.recent_signed_return / 6.0, -1.0, 1.0),
            -1.0,
            1.0,
        )

    def _next_microstructure_state(
        self,
        previous_latent: LatentState,
        latent: LatentState,
        imbalance: float,
    ) -> _MicrostructureState:
        """Update internal book pressure that links realized flow to future liquidity."""
        regime = self._active_settings
        previous = self._microstructure
        inputs = self._microstructure_inputs(previous_latent, latent, imbalance)
        realized_activity = (
            0.46 * inputs.execution_pressure
            + 0.34 * inputs.return_shock
            + 0.16 * inputs.volatility_shock
            + 0.10 * inputs.imbalance_shock
        )
        activity = self._clamp(
            regime["activity_decay"] * previous.activity
            + regime["activity_gain"] * realized_activity,
            0.0,
            2.0,
        )

        shock = (
            0.40 * inputs.return_shock
            + 0.22 * inputs.imbalance_shock
            + 0.28 * inputs.volatility_shock
            + 0.04 * max(0.0, activity - previous.activity)
        )
        # Cancellation bursts are stateful: shocks raise pressure, then pressure decays.
        cancel_pressure = self._clamp(
            regime["cancel_decay"] * previous.cancel_pressure
            + regime["cancel_burst"] * shock,
            0.0,
            2.0,
        )

        recent_signed_return = self._clamp(
            0.86 * previous.recent_signed_return + 0.14 * inputs.signed_return,
            -12.0,
            12.0,
        )
        recent_flow_imbalance = self._clamp(
            0.78 * previous.recent_flow_imbalance + 0.22 * inputs.flow_imbalance,
            -1.0,
            1.0,
        )
        activity_event = self._next_activity_event(previous, inputs, regime)
        squeeze_pressure = self._next_squeeze_pressure(previous, inputs, regime)
        liquidity_stress = self._next_liquidity_stress(
            previous,
            inputs,
            cancel_pressure,
            regime,
        )
        spread_pressure = self._next_spread_pressure(
            previous,
            inputs,
            cancel_pressure,
            liquidity_stress,
            regime,
        )
        stress_side = self._next_stress_side(previous, inputs)
        resiliency_target = regime["resiliency"] * self._clamp(
            1.0
            - 0.18 * cancel_pressure
            - 0.16 * liquidity_stress
            - 0.08 * spread_pressure
            + 0.08 * previous.activity
            - 0.05 * activity_event,
            0.30,
            1.30,
        )
        resiliency = self._clamp(0.76 * previous.resiliency + 0.24 * resiliency_target, 0.20, 1.40)
        micro = _MicrostructureState(
            activity=activity,
            cancel_pressure=cancel_pressure,
            resiliency=resiliency,
            recent_signed_return=recent_signed_return,
            recent_flow_imbalance=recent_flow_imbalance,
            squeeze_pressure=squeeze_pressure,
            activity_event=activity_event,
            liquidity_stress=liquidity_stress,
            stress_side=stress_side,
            spread_pressure=spread_pressure,
            last_cancelled_volume=previous.last_cancelled_volume,
        )
        self._microstructure = micro
        return micro

    def _microstructure_inputs(
        self,
        previous_latent: LatentState,
        latent: LatentState,
        imbalance: float,
    ) -> _MicrostructureInputs:
        execution_pressure = min(
            log1p(self._last_execution_volume / max(0.7, self.popularity)),
            2.0,
        )
        volatility_jump = max(0.0, latent.volatility - previous_latent.volatility)
        return _MicrostructureInputs(
            execution_pressure=execution_pressure,
            return_shock=1.0 - exp(-self._last_abs_return_ticks / 1.4),
            volatility_shock=1.0 - exp(-volatility_jump / 0.25),
            imbalance_shock=abs(imbalance - self._last_imbalance),
            signed_return=self._last_return_ticks,
            flow_imbalance=self._last_imbalance,
            squeeze_setup=self._squeeze_setup_pressure(),
        )

    def _next_activity_event(
        self,
        previous: _MicrostructureState,
        inputs: _MicrostructureInputs,
        regime: dict[str, float],
    ) -> float:
        burst_seed = (
            max(0.0, inputs.return_shock - 0.32)
            + 0.42 * inputs.volatility_shock
            + 0.30 * max(0.0, abs(inputs.flow_imbalance) - 0.25)
            + 0.24 * max(0.0, inputs.execution_pressure - 0.70)
        )
        return self._clamp(
            0.72 * previous.activity_event + 1.18 * regime["event_gain"] * burst_seed,
            0.0,
            1.8,
        )

    def _next_squeeze_pressure(
        self,
        previous: _MicrostructureState,
        inputs: _MicrostructureInputs,
        regime: dict[str, float],
    ) -> float:
        release = max(0.0, -inputs.signed_return) / 3.0
        return self._clamp(
            0.56 * previous.squeeze_pressure
            + regime["squeeze_gain"] * inputs.squeeze_setup
            - 0.34 * release,
            0.0,
            1.5,
        )

    def _next_liquidity_stress(
        self,
        previous: _MicrostructureState,
        inputs: _MicrostructureInputs,
        cancel_pressure: float,
        regime: dict[str, float],
    ) -> float:
        raw_stress = (
            0.34 * inputs.return_shock
            + 0.26 * inputs.execution_pressure
            + 0.22 * inputs.imbalance_shock
            + 0.18 * cancel_pressure
        ) * regime["stress"]
        return self._clamp(0.70 * previous.liquidity_stress + 0.30 * raw_stress, 0.0, 2.0)

    def _next_spread_pressure(
        self,
        previous: _MicrostructureState,
        inputs: _MicrostructureInputs,
        cancel_pressure: float,
        liquidity_stress: float,
        regime: dict[str, float],
    ) -> float:
        spread_ticks = self._current_spread_ticks()
        spread_gap = self._clamp((spread_ticks - 2.0) / 5.0, 0.0, 1.0)
        raw_pressure = (
            0.26 * inputs.return_shock
            + 0.20 * inputs.execution_pressure
            + 0.18 * abs(inputs.flow_imbalance)
            + 0.22 * cancel_pressure
            + 0.20 * liquidity_stress
            + 0.18 * spread_gap
        ) * regime["stress"]
        return self._clamp(
            0.78 * previous.spread_pressure + 0.22 * raw_pressure,
            0.0,
            1.8,
        )

    def _next_stress_side(
        self,
        previous: _MicrostructureState,
        inputs: _MicrostructureInputs,
    ) -> float:
        directional_pressure = self._clamp(
            0.62 * inputs.flow_imbalance
            + 0.38 * self._clamp(inputs.signed_return / 5.0, -1.0, 1.0),
            -1.0,
            1.0,
        )
        return self._clamp(0.74 * previous.stress_side + 0.26 * directional_pressure, -1.0, 1.0)

    def _squeeze_setup_pressure(self) -> float:
        price = self.state.price
        bid_depth = self._nearby_book_depth("bid", price)
        ask_depth = self._nearby_book_depth("ask", price)
        total_depth = bid_depth + ask_depth
        if total_depth <= 1e-12:
            return 0.0
        bid_crowding = bid_depth / total_depth
        ask_thinness = 1.0 / (1.0 + ask_depth / max(self.popularity, 1e-12))
        upward_trigger = self._clamp(
            0.55 * max(0.0, self._last_return_ticks) / 3.0
            + 0.45 * max(0.0, self._last_imbalance)
            - 0.10,
            0.0,
            1.0,
        )
        setup = 0.58 * bid_crowding + 0.42 * ask_thinness
        return self._clamp(setup * upward_trigger, 0.0, 1.0)

    def _next_mdf(
        self,
        price: float,
        price_grid: list[float],
        latent: LatentState,
        *,
        step_index: int,
        update_memory: bool,
        cache: _StepComputationCache | None = None,
    ) -> MDFState:
        del price_grid
        relative_ticks = self.relative_tick_grid()
        context = MDFContext(
            current_price=price,
            current_tick=self.price_to_tick(price),
            tick_size=self.gap,
            mood=latent.mood,
            trend=latent.trend,
            volatility=latent.volatility * self._active_settings["spread"],
            regime=self._active_regime,
            augmentation_strength=self.augmentation_strength,
            step_index=step_index,
            rng=self._rng,
        )
        signals = self._cached_mdf_signals(price, cache)
        mdf_shock = self._clamp(
            0.34 * signals.liquidity_stress
            + 0.22 * signals.activity_event
            + 0.18 * min(abs(signals.last_return_ticks) / 3.0, 1.0)
            + 0.14 * signals.cancel_pressure / 2.0
            + 0.12 * latent.volatility / 1.55,
            0.0,
            1.0,
        )
        entry_memory = self._clamp(self._entry_mdf_memory * (1.0 - 0.35 * mdf_shock), 0.08, 0.20)
        floor_mix = self._clamp(self._mdf_floor_mix + 0.006 * mdf_shock, 0.010, 0.024)
        judgments = self._sample_mdf_judgments(context, signals, update_memory=update_memory)

        buy_entry_ticks = self._evolve_raw_mdf(
            "buy_entry",
            self._buy_entry_mdf_mass(relative_ticks, context, signals, judgments.buy),
            current_tick=context.current_tick,
            memory_mix=entry_memory,
            floor_mix=floor_mix,
            update_memory=update_memory,
        )
        sell_entry_ticks = self._evolve_raw_mdf(
            "sell_entry",
            self._sell_entry_mdf_mass(relative_ticks, context, signals, judgments.sell),
            current_tick=context.current_tick,
            memory_mix=entry_memory,
            floor_mix=floor_mix,
            update_memory=update_memory,
        )
        buy_entry_ticks, sell_entry_ticks = self._resolve_entry_mdf_overlap(
            relative_ticks,
            context.current_tick,
            buy_entry_ticks,
            sell_entry_ticks,
            buy_aggression=self._entry_overlap_aggression(
                "buy",
                context,
                signals,
                judgments.buy,
                mdf_shock,
            ),
            sell_aggression=self._entry_overlap_aggression(
                "sell",
                context,
                signals,
                judgments.sell,
                mdf_shock,
            ),
        )
        entry_noise_ticks = self._centered_entry_noise_mdf(
            relative_ticks,
            context.current_tick,
        )
        buy_entry_ticks = self._mix_entry_noise_mdf(
            buy_entry_ticks,
            entry_noise_ticks,
            mix=self._entry_noise_mix,
        )
        sell_entry_ticks = self._mix_entry_noise_mdf(
            sell_entry_ticks,
            entry_noise_ticks,
            mix=self._entry_noise_mix,
        )
        buy_entry = self._project_tick_mdf(context.current_tick, buy_entry_ticks)
        sell_entry = self._project_tick_mdf(context.current_tick, sell_entry_ticks)
        return MDFState(
            buy_entry_mdf_by_price=buy_entry,
            sell_entry_mdf_by_price=sell_entry,
            relative_ticks=relative_ticks,
            buy_entry_mdf=buy_entry_ticks,
            sell_entry_mdf=sell_entry_ticks,
        )

    def _reproject_mdf(
        self,
        price: float,
        mdf: MDFState,
    ) -> MDFState:
        current_tick = self.price_to_tick(price)
        buy_entry_mdf = self._constrain_tick_mdf(current_tick, mdf.buy_entry_mdf)
        sell_entry_mdf = self._constrain_tick_mdf(current_tick, mdf.sell_entry_mdf)
        return MDFState(
            buy_entry_mdf_by_price=self._project_tick_mdf(current_tick, buy_entry_mdf),
            sell_entry_mdf_by_price=self._project_tick_mdf(current_tick, sell_entry_mdf),
            relative_ticks=list(mdf.relative_ticks),
            buy_entry_mdf=buy_entry_mdf,
            sell_entry_mdf=sell_entry_mdf,
        )

    def _entry_flow(
        self,
        intensity: IntensityState,
        mdf: MDFState,
    ) -> _EntryFlow:
        buy_orders, buy_intent = self._sample_entry_side(
            "buy",
            "buy_entry",
            intensity.buy,
            mdf.buy_entry_mdf_by_price,
        )
        sell_orders, sell_intent = self._sample_entry_side(
            "sell",
            "sell_entry",
            intensity.sell,
            mdf.sell_entry_mdf_by_price,
        )
        orders = buy_orders + sell_orders

        return _EntryFlow(
            orders=orders,
            buy_intent_by_price=self._drop_zeroes(buy_intent),
            sell_intent_by_price=self._drop_zeroes(sell_intent),
        )

    def _sample_entry_side(
        self,
        side: str,
        kind: str,
        target_volume: float,
        mdf_by_price: PriceMap,
    ) -> tuple[list[_IncomingOrder], PriceMap]:
        if target_volume <= 1e-12:
            return [], {}
        probabilities = self._normalized_price_probabilities(mdf_by_price)
        if not probabilities:
            return [], {}

        event_probability = self._entry_event_probability(target_volume)
        if self._unit_random() >= event_probability:
            return [], {}

        effective_target = target_volume / max(event_probability, 1e-12)
        max_count = max(1, int(10 + 6 * effective_target + 4 * max(self.popularity, 0.0)))
        order_count = self._sample_order_count(
            effective_target,
            max_count=max_count,
        )
        if order_count <= 0:
            return [], {}

        orders: list[_IncomingOrder] = []
        intent: PriceMap = {}
        for _ in range(order_count):
            price = self._sample_price(probabilities)
            volume = self._sample_order_size()
            if volume <= 1e-12:
                continue
            orders.append(_IncomingOrder(side=side, kind=kind, price=price, volume=volume))
            intent[price] = intent.get(price, 0.0) + volume
        return orders, intent

    def _entry_event_probability(self, target_volume: float) -> float:
        if target_volume <= 1e-12:
            return 0.0
        popularity_term = self.popularity / (1.0 + self.popularity)
        cluster = self._event_cluster_signal(self._microstructure)
        probability = (
            0.075
            + 0.35 * (1.0 - exp(-target_volume / 1.2))
            + 0.43 * (1.0 - exp(-target_volume / 9.0))
            + 0.08 * popularity_term
        )
        probability *= 0.80 + 0.52 * cluster
        high_volume_floor = 0.96 * (1.0 - exp(-target_volume / 7.0))
        probability = max(probability, high_volume_floor)
        return self._clamp(probability, 0.07, 0.96)

    def _event_cluster_signal(self, micro: _MicrostructureState) -> float:
        signal = (
            0.48 * self._clamp(micro.activity_event, 0.0, 1.8) / 1.8
            + 0.24 * self._clamp(micro.activity, 0.0, 2.0) / 2.0
            + 0.18 * self._clamp(micro.cancel_pressure, 0.0, 2.0) / 2.0
            + 0.10 * self._clamp(micro.spread_pressure, 0.0, 1.8) / 1.8
        )
        return self._clamp(signal, 0.0, 1.0)

    def _normalized_price_probabilities(self, values: PriceMap) -> PriceMap:
        probabilities: PriceMap = {}
        total = 0.0
        for raw_price, raw_probability in values.items():
            if raw_probability <= 0 or not isfinite(raw_probability):
                continue
            tick = max(1, int(round(raw_price / self.gap)))
            if self._gap_is_integer:
                price = float(int(tick * self.gap))
            else:
                price = self._clean_number(tick * self.gap)
            if price < self._min_price:
                continue
            probabilities[price] = probabilities.get(price, 0.0) + raw_probability
            total += raw_probability
        if not probabilities or total <= 1e-12:
            return {}
        ordered = sorted(probabilities.items())
        if abs(total - 1.0) <= 1e-12:
            return dict(ordered)
        return {price: probability / total for price, probability in ordered}

    def _entry_probabilities_for_book_side(
        self,
        side: str,
        mdf: MDFState,
        cache: _StepComputationCache | None = None,
    ) -> PriceMap:
        if cache is not None and cache.mdf is mdf and side in cache.entry_probabilities_by_side:
            return cache.entry_probabilities_by_side[side]
        entry_mdf = mdf.buy_entry_mdf_by_price if side == "bid" else mdf.sell_entry_mdf_by_price
        probabilities = self._normalized_price_probabilities(entry_mdf)
        if cache is not None and cache.mdf is mdf:
            cache.entry_probabilities_by_side[side] = probabilities
        return probabilities

    def _sample_price(self, probabilities: PriceMap) -> float:
        if not probabilities:
            raise ValueError("probabilities must not be empty")
        draw = self._unit_random()
        cumulative = 0.0
        fallback = next(reversed(probabilities))
        for price, probability in probabilities.items():
            cumulative += probability
            if draw <= cumulative:
                return price
        return fallback

    def _mean_child_order_size(self) -> float:
        return self._clamp(0.18 + 0.08 * self.popularity, 0.08, 0.55)

    def _sample_order_count(
        self,
        target_volume: float,
        *,
        max_count: int,
    ) -> int:
        if target_volume <= 1e-12 or max_count <= 0:
            return 0
        expected = target_volume / self._mean_child_order_size()
        count = self._sample_poisson(expected)
        return max(0, min(max_count, count))

    def _sample_poisson(self, expected: float) -> int:
        if expected <= 0:
            return 0
        if expected > 32.0:
            if hasattr(self._rng, "gauss"):
                return max(0, int(round(self._rng.gauss(expected, expected**0.5))))
            return max(0, int(round(expected)))
        draw = self._unit_random()
        probability = exp(-expected)
        cumulative = probability
        count = 0
        while draw > cumulative and count < 256:
            count += 1
            probability *= expected / count
            cumulative += probability
        return count

    def _sample_order_size(self) -> float:
        mean_size = self._mean_child_order_size()
        sigma = 0.62
        if hasattr(self._rng, "lognormvariate"):
            size = mean_size * self._rng.lognormvariate(-0.5 * sigma * sigma, sigma)
        else:
            size = mean_size * (0.55 + 0.90 * self._unit_random())
        return self._clamp(size, mean_size * 0.05, mean_size * 8.0)

    def _unit_random(self) -> float:
        value = self._rng.random()
        if not isfinite(value):
            return 0.0
        return self._clamp(value, 0.0, 1.0 - 1e-12)

    def _sample_mdf_judgments(
        self,
        context: MDFContext,
        signals: MDFSignals,
        *,
        update_memory: bool,
    ) -> _MDFJudgmentSample:
        volatility = self._clamp(context.volatility, 0.0, 2.2)
        stress = self._clamp(signals.liquidity_stress, 0.0, 1.8)
        event = self._clamp(signals.activity_event + 0.45 * signals.activity, 0.0, 1.8)
        spread = self._clamp(
            (signals.spread_ticks - 1.0) / max(float(self.grid_radius), 1.0),
            0.0,
            1.0,
        )
        uncertainty = self._clamp(
            0.10
            + 0.20 * volatility
            + 0.24 * stress
            + 0.18 * event
            + 0.10 * context.augmentation_strength,
            0.08,
            0.95,
        )
        common_value_noise = self._rng.gauss(0.0, 0.16 + 0.24 * uncertainty)
        buy_noise = self._rng.gauss(0.0, 0.14 + 0.26 * uncertainty)
        sell_noise = self._rng.gauss(0.0, 0.14 + 0.26 * uncertainty)

        buy_view = (
            context.trend
            + 0.42 * context.mood
            + 0.18 * signals.last_return_ticks
            + 0.12 * signals.orderbook_imbalance
            + 0.10 * max(0.0, signals.stress_side)
        )
        buy_fair_value_shift = self._clamp(
            1.10 * buy_view + common_value_noise + buy_noise,
            -2.4,
            2.4,
        )
        buy_opportunity = self._clamp(
            0.22 * max(0.0, -signals.last_return_ticks)
            + 0.20 * max(0.0, -context.trend)
            + 0.16 * max(0.0, -context.mood)
            + self._rng.gauss(0.0, 0.10 + 0.16 * uncertainty),
            0.0,
            1.5,
        )
        buy_urgency = self._clamp(
            0.10
            + 0.54 * max(0.0, buy_fair_value_shift)
            + 0.16 * event
            + 0.14 * max(0.0, signals.stress_side)
            + self._rng.gauss(0.0, 0.09 + 0.13 * uncertainty),
            0.0,
            1.8,
        )
        buy_patience = self._clamp(
            0.26
            + 0.44 * max(0.0, -buy_fair_value_shift)
            + 0.18 * volatility
            + 0.16 * buy_opportunity
            + self._rng.gauss(0.0, 0.08 + 0.12 * uncertainty),
            0.0,
            1.8,
        )
        buy_liquidity_aversion = self._clamp(
            0.08
            + 0.30 * stress
            + 0.18 * spread
            + 0.12 * max(0.0, -signals.stress_side)
            + self._rng.gauss(0.0, 0.07 + 0.12 * uncertainty),
            0.0,
            1.0,
        )
        buy_pocket_bias = self._clamp(
            0.18
            + 0.18 * uncertainty
            + 0.18 * buy_opportunity
            + self._rng.gauss(0.0, 0.08 + 0.10 * uncertainty),
            0.0,
            1.0,
        )

        sell_view = (
            0.74 * context.trend
            + 0.24 * context.mood
            + 0.10 * signals.last_return_ticks
            - 0.08 * signals.orderbook_imbalance
            - 0.10 * max(0.0, -signals.stress_side)
        )
        sell_fair_value_shift = self._clamp(
            sell_view + 0.35 * common_value_noise + sell_noise,
            -2.4,
            2.4,
        )
        profit_taking = self._clamp(
            0.26 * max(0.0, signals.last_return_ticks)
            + 0.18 * max(0.0, context.trend)
            + 0.12 * max(0.0, context.mood)
            + self._rng.gauss(0.0, 0.10 + 0.16 * uncertainty),
            0.0,
            1.5,
        )
        liquidation = self._clamp(
            0.30 * max(0.0, -context.trend)
            + 0.22 * max(0.0, -signals.last_return_ticks)
            + 0.18 * stress
            + 0.12 * max(0.0, -signals.stress_side)
            + self._rng.gauss(0.0, 0.10 + 0.16 * uncertainty),
            0.0,
            1.6,
        )
        sell_urgency = self._clamp(
            0.10
            + 0.48 * max(0.0, -sell_fair_value_shift)
            + 0.24 * liquidation
            + 0.12 * profit_taking
            + 0.16 * event
            + self._rng.gauss(0.0, 0.09 + 0.13 * uncertainty),
            0.0,
            1.9,
        )
        sell_patience = self._clamp(
            0.26
            + 0.48 * max(0.0, sell_fair_value_shift)
            + 0.16 * volatility
            + 0.10 * profit_taking
            + self._rng.gauss(0.0, 0.08 + 0.12 * uncertainty),
            0.0,
            1.8,
        )
        sell_liquidity_aversion = self._clamp(
            0.08
            + 0.30 * stress
            + 0.18 * spread
            + 0.12 * max(0.0, signals.stress_side)
            + self._rng.gauss(0.0, 0.07 + 0.12 * uncertainty),
            0.0,
            1.0,
        )
        sell_pocket_bias = self._clamp(
            0.18
            + 0.18 * uncertainty
            + 0.14 * profit_taking
            + 0.12 * liquidation
            + self._rng.gauss(0.0, 0.08 + 0.10 * uncertainty),
            0.0,
            1.0,
        )

        buy = _MDFSideJudgment(
            fair_value_shift=buy_fair_value_shift,
            urgency=buy_urgency,
            patience=buy_patience,
            opportunity=buy_opportunity,
            liquidation=0.0,
            liquidity_aversion=buy_liquidity_aversion,
            pocket_bias=buy_pocket_bias,
            uncertainty=uncertainty,
        )
        sell = _MDFSideJudgment(
            fair_value_shift=sell_fair_value_shift,
            urgency=sell_urgency,
            patience=sell_patience,
            opportunity=profit_taking,
            liquidation=liquidation,
            liquidity_aversion=sell_liquidity_aversion,
            pocket_bias=sell_pocket_bias,
            uncertainty=uncertainty,
        )
        return _MDFJudgmentSample(
            buy=self._blend_mdf_judgment("buy", buy, update_memory=update_memory),
            sell=self._blend_mdf_judgment("sell", sell, update_memory=update_memory),
        )

    def _blend_mdf_judgment(
        self,
        key: str,
        sample: _MDFSideJudgment,
        *,
        update_memory: bool,
    ) -> _MDFSideJudgment:
        previous = self._mdf_judgment_memory.get(key)
        if previous is None:
            blended = sample
        else:
            blended = _MDFSideJudgment(
                fair_value_shift=0.65 * previous.fair_value_shift + 0.35 * sample.fair_value_shift,
                urgency=0.65 * previous.urgency + 0.35 * sample.urgency,
                patience=0.65 * previous.patience + 0.35 * sample.patience,
                opportunity=0.65 * previous.opportunity + 0.35 * sample.opportunity,
                liquidation=0.65 * previous.liquidation + 0.35 * sample.liquidation,
                liquidity_aversion=(
                    0.65 * previous.liquidity_aversion + 0.35 * sample.liquidity_aversion
                ),
                pocket_bias=0.65 * previous.pocket_bias + 0.35 * sample.pocket_bias,
                uncertainty=0.65 * previous.uncertainty + 0.35 * sample.uncertainty,
            )
        if update_memory:
            self._mdf_judgment_memory[key] = blended
        return blended

    def _buy_entry_mdf_mass(
        self,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
    ) -> dict[int, float]:
        volatility = self._clamp(context.volatility, 0.0, 2.2)
        stress = self._clamp(signals.liquidity_stress, 0.0, 1.8)
        event = self._clamp(signals.activity_event + 0.45 * signals.activity, 0.0, 1.8)
        bid_shortage = signals.bid_shortage_by_tick
        bid_gap = signals.bid_gap_by_tick
        bid_front = signals.bid_front_by_tick
        bid_liquidity = self._normalize_tick_signal(dict(signals.bid_liquidity_by_tick))
        ask_liquidity = self._normalize_tick_signal(dict(signals.ask_liquidity_by_tick))
        value_center = self._clamp(
            judgment.fair_value_shift * (1.0 + 0.35 * judgment.urgency)
            - 0.55 * judgment.patience,
            -float(self.grid_radius),
            min(3.0, float(self.grid_radius)),
        )
        value_spread = 1.15 + 0.55 * volatility + 0.50 * judgment.uncertainty
        spike_evidence = self._clamp(
            0.22 * judgment.urgency
            + 0.16 * judgment.patience
            + 0.16 * event
            + 0.20 * stress
            + 0.14 * judgment.uncertainty,
            0.0,
            1.0,
        )

        raw: dict[int, float] = {}
        for tick in relative_ticks:
            level = abs(tick)
            texture = self._mdf_texture("buy", tick, context, salt=0.0)
            value_fit = self._soft_tick_band(float(tick), value_center, value_spread)
            mass = 0.007 + 0.005 * texture
            mass += (0.035 + 0.026 * judgment.uncertainty) * value_fit
            if -1 <= tick <= 0:
                mass += 0.048 + 0.018 * volatility + 0.012 * event
            elif level == 2:
                mass += 0.026 + 0.010 * volatility
            else:
                mass += 0.010 / (1.0 + 0.24 * level)

            if tick <= 0:
                passive_level = abs(tick)
                reservation_band = self._entry_reservation_band(passive_level)
                passive_center = (
                    0.95
                    + 1.70 * judgment.patience
                    + 0.55 * volatility
                    + 0.85 * judgment.opportunity
                )
                passive_band = self._soft_tick_band(
                    float(passive_level),
                    passive_center,
                    1.00 + 0.35 * volatility + 0.25 * stress,
                )
                book_signal = (
                    0.34 * bid_shortage.get(tick, 0.0)
                    + 0.24 * bid_gap.get(tick, 0.0)
                    + 0.22 * bid_front.get(tick, 0.0)
                )
                book_response = max(0.0, 1.0 - 0.70 * judgment.liquidity_aversion)
                spike_evidence = max(spike_evidence, 0.55 * self._clamp(book_signal, 0.0, 1.0))
                mass += (
                    0.026
                    + 0.042 * judgment.patience
                    + 0.038 * judgment.opportunity
                    + 0.012 * volatility
                ) * passive_band
                mass += 0.012 * reservation_band
                mass += 0.070 * passive_band * book_response * (1.0 - exp(-2.0 * book_signal))
                mass += 0.018 * bid_liquidity.get(tick, 0.0)
            else:
                market_band = self._entry_marketable_band(tick)
                opposing = ask_liquidity.get(tick, 0.0)
                crossing_depth = opposing * (0.026 + 0.120 * judgment.urgency)
                crossing_depth *= max(0.0, 1.0 - 0.35 * judgment.liquidity_aversion)
                spike_evidence = max(
                    spike_evidence,
                    0.50 * opposing * self._clamp(judgment.urgency, 0.0, 1.0),
                )
                mass += (0.026 + 0.092 * judgment.urgency + 0.018 * event) * market_band
                mass += crossing_depth

            if level >= 4:
                mass += (0.014 * volatility + 0.014 * stress) * texture
            if level >= 6:
                mass += (0.020 * volatility + 0.010 * judgment.uncertainty) * texture
            raw[tick] = mass * (0.80 + 0.38 * texture)

        self._add_buy_entry_pockets(relative_ticks, context, signals, judgment, raw)
        center_weight = 0.55 + 0.22 * spike_evidence
        adjacent_weight = 0.16 - 0.045 * spike_evidence
        outer_weight = max(0.0, (1.0 - center_weight - 2.0 * adjacent_weight) / 2.0)
        return self._smooth_tick_mass(
            raw,
            relative_ticks,
            center_weight=center_weight,
            adjacent_weight=adjacent_weight,
            outer_weight=outer_weight,
        )

    def _sell_entry_mdf_mass(
        self,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
    ) -> dict[int, float]:
        volatility = self._clamp(context.volatility, 0.0, 2.2)
        stress = self._clamp(signals.liquidity_stress, 0.0, 1.8)
        event = self._clamp(signals.activity_event + 0.45 * signals.activity, 0.0, 1.8)
        ask_shortage = signals.ask_shortage_by_tick
        ask_gap = signals.ask_gap_by_tick
        ask_front = signals.ask_front_by_tick
        ask_liquidity = self._normalize_tick_signal(dict(signals.ask_liquidity_by_tick))
        bid_liquidity = self._normalize_tick_signal(dict(signals.bid_liquidity_by_tick))
        value_center = self._clamp(
            judgment.fair_value_shift * (1.0 + 0.30 * judgment.patience)
            + 0.55 * judgment.patience
            - 0.95 * judgment.urgency
            - 0.70 * judgment.liquidation,
            -min(3.0, float(self.grid_radius)),
            float(self.grid_radius),
        )
        value_spread = 1.15 + 0.55 * volatility + 0.50 * judgment.uncertainty
        spike_evidence = self._clamp(
            0.22 * judgment.urgency
            + 0.16 * judgment.patience
            + 0.20 * judgment.liquidation
            + 0.16 * event
            + 0.18 * stress,
            0.0,
            1.0,
        )

        raw: dict[int, float] = {}
        for tick in relative_ticks:
            level = abs(tick)
            texture = self._mdf_texture("sell", tick, context, salt=0.0)
            value_fit = self._soft_tick_band(float(tick), value_center, value_spread)
            mass = 0.007 + 0.005 * texture
            mass += (0.035 + 0.026 * judgment.uncertainty) * value_fit
            if 0 <= tick <= 1:
                mass += 0.048 + 0.018 * volatility + 0.012 * event
            elif level == 2:
                mass += 0.026 + 0.010 * volatility
            else:
                mass += 0.010 / (1.0 + 0.24 * level)

            if tick >= 0:
                passive_level = tick
                reservation_band = self._entry_reservation_band(passive_level)
                passive_center = (
                    0.95
                    + 1.60 * judgment.patience
                    + 0.55 * volatility
                    - 0.50 * judgment.opportunity
                    + 0.45 * judgment.fair_value_shift
                )
                passive_band = self._soft_tick_band(
                    float(passive_level),
                    passive_center,
                    1.00 + 0.35 * volatility + 0.25 * stress,
                )
                book_signal = (
                    0.34 * ask_shortage.get(tick, 0.0)
                    + 0.24 * ask_gap.get(tick, 0.0)
                    + 0.22 * ask_front.get(tick, 0.0)
                )
                book_response = max(0.0, 1.0 - 0.70 * judgment.liquidity_aversion)
                spike_evidence = max(spike_evidence, 0.55 * self._clamp(book_signal, 0.0, 1.0))
                mass += (
                    0.026
                    + 0.040 * judgment.patience
                    + 0.030 * judgment.opportunity
                    + 0.012 * volatility
                ) * passive_band
                mass += 0.012 * reservation_band
                mass += 0.070 * passive_band * book_response * (1.0 - exp(-2.0 * book_signal))
                mass += 0.018 * ask_liquidity.get(tick, 0.0)
            else:
                market_band = self._entry_marketable_band(abs(tick))
                opposing = bid_liquidity.get(tick, 0.0)
                crossing_depth = opposing * (
                    0.026 + 0.105 * judgment.urgency + 0.190 * judgment.liquidation
                )
                crossing_depth *= max(0.0, 1.0 - 0.35 * judgment.liquidity_aversion)
                spike_evidence = max(
                    spike_evidence,
                    0.50 * opposing * self._clamp(judgment.urgency, 0.0, 1.0),
                )
                mass += (
                    0.026
                    + 0.086 * judgment.urgency
                    + 0.150 * judgment.liquidation
                    + 0.018 * event
                ) * market_band
                mass += crossing_depth

            if level >= 4:
                mass += (0.014 * volatility + 0.014 * stress) * texture
            if level >= 6:
                mass += (0.020 * volatility + 0.010 * judgment.uncertainty) * texture
            raw[tick] = mass * (0.80 + 0.38 * texture)

        self._add_sell_entry_pockets(relative_ticks, context, signals, judgment, raw)
        center_weight = 0.55 + 0.22 * spike_evidence
        adjacent_weight = 0.16 - 0.045 * spike_evidence
        outer_weight = max(0.0, (1.0 - center_weight - 2.0 * adjacent_weight) / 2.0)
        return self._smooth_tick_mass(
            raw,
            relative_ticks,
            center_weight=center_weight,
            adjacent_weight=adjacent_weight,
            outer_weight=outer_weight,
        )

    def _evolve_raw_mdf(
        self,
        key: str,
        raw_mass: dict[int, float],
        *,
        current_tick: int,
        memory_mix: float,
        floor_mix: float,
        update_memory: bool,
    ) -> dict[int, float]:
        ticks = self.relative_tick_grid()
        valid_ticks = [tick for tick in ticks if current_tick + tick >= 1]
        proposal = self._normalize_tick_map(
            {tick: max(0.0, raw_mass.get(tick, 0.0)) for tick in valid_ticks}
        )
        previous = self._mdf_memory.get(key)
        if previous is None:
            previous = proposal
        else:
            previous = self._normalize_tick_map(
                {tick: previous.get(tick, 0.0) for tick in valid_ticks}
            )
        uniform = 1.0 / len(valid_ticks)
        mixed = self._normalize_tick_map(
            {
                tick: (1.0 - memory_mix) * proposal.get(tick, 0.0)
                + memory_mix * previous.get(tick, uniform)
                for tick in valid_ticks
            }
        )
        evolved = self._normalize_tick_map(
            {
                tick: (1.0 - floor_mix) * mixed.get(tick, 0.0) + floor_mix * uniform
                for tick in valid_ticks
            }
        )
        constrained = {tick: evolved.get(tick, 0.0) for tick in ticks}
        if update_memory:
            self._mdf_memory[key] = constrained
        return constrained

    def _centered_entry_noise_mdf(
        self,
        relative_ticks: list[int],
        current_tick: int,
    ) -> TickMap:
        valid_ticks = [tick for tick in relative_ticks if current_tick + tick >= 1]
        if not valid_ticks:
            return {tick: 0.0 for tick in relative_ticks}
        sigma = max(self._entry_noise_sigma_ticks, 1e-12)
        mass = {
            tick: exp(-0.5 * (float(tick) / sigma) ** 2)
            for tick in valid_ticks
        }
        normalized = self._normalize_tick_map(mass)
        return {tick: normalized.get(tick, 0.0) for tick in relative_ticks}

    def _mix_entry_noise_mdf(
        self,
        raw_mdf: TickMap,
        noise_mdf: TickMap,
        *,
        mix: float,
    ) -> TickMap:
        mix = self._clamp(mix, 0.0, 1.0)
        ticks = list(raw_mdf)
        if not ticks:
            return {}
        if mix <= 1e-12:
            normalized = self._normalize_tick_map(raw_mdf)
            return {tick: normalized.get(tick, 0.0) for tick in ticks}
        mixed = {
            tick: (1.0 - mix) * max(0.0, raw_mdf.get(tick, 0.0))
            + mix * max(0.0, noise_mdf.get(tick, 0.0))
            for tick in ticks
        }
        normalized = self._normalize_tick_map(mixed)
        return {tick: normalized.get(tick, 0.0) for tick in ticks}

    def _resolve_entry_mdf_overlap(
        self,
        relative_ticks: list[int],
        current_tick: int,
        raw_buy: TickMap,
        raw_sell: TickMap,
        *,
        buy_aggression: float,
        sell_aggression: float,
    ) -> tuple[TickMap, TickMap]:
        valid_ticks = [tick for tick in relative_ticks if current_tick + tick >= 1]
        if not valid_ticks:
            empty = {tick: 0.0 for tick in relative_ticks}
            return empty, empty.copy()

        buy = self._normalize_tick_map(
            {tick: max(0.0, raw_buy.get(tick, 0.0)) for tick in valid_ticks}
        )
        sell = self._normalize_tick_map(
            {tick: max(0.0, raw_sell.get(tick, 0.0)) for tick in valid_ticks}
        )
        buy_aggression = self._clamp(buy_aggression, 0.0, 1.0)
        sell_aggression = self._clamp(sell_aggression, 0.0, 1.0)

        local_buy = {tick: self._local_mdf_overlap(buy, tick) for tick in valid_ticks}
        local_sell = {tick: self._local_mdf_overlap(sell, tick) for tick in valid_ticks}

        resolved_buy: TickMap = {}
        resolved_sell: TickMap = {}
        for tick in valid_ticks:
            local_total = local_buy[tick] + local_sell[tick] + 1e-12
            buy_overlap = local_sell[tick] / local_total
            sell_overlap = local_buy[tick] / local_total
            if tick < 0:
                buy_overlap *= 0.84
                sell_overlap = min(1.0, sell_overlap * 1.10)
            elif tick > 0:
                buy_overlap = min(1.0, buy_overlap * 1.10)
                sell_overlap *= 0.84
            else:
                buy_overlap = min(1.0, buy_overlap * 1.03)
                sell_overlap = min(1.0, sell_overlap * 1.03)

            resolved_buy[tick] = buy.get(tick, 0.0) * self._entry_overlap_factor(
                buy_aggression,
                buy_overlap,
            )
            resolved_sell[tick] = sell.get(tick, 0.0) * self._entry_overlap_factor(
                sell_aggression,
                sell_overlap,
            )

        resolved_buy = self._normalize_tick_map(resolved_buy)
        resolved_sell = self._normalize_tick_map(resolved_sell)
        return (
            {tick: resolved_buy.get(tick, 0.0) for tick in relative_ticks},
            {tick: resolved_sell.get(tick, 0.0) for tick in relative_ticks},
        )

    def _entry_overlap_aggression(
        self,
        side: str,
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
        mdf_shock: float,
    ) -> float:
        stress = self._clamp(signals.liquidity_stress, 0.0, 1.8) / 1.8
        event = self._clamp(signals.activity_event + 0.45 * signals.activity, 0.0, 1.8) / 1.8
        volatility = self._clamp(context.volatility, 0.0, 2.2) / 2.2
        last_return = self._clamp(signals.last_return_ticks / 4.0, -1.0, 1.0)
        if side == "buy":
            directional = self._clamp(
                0.62 * max(0.0, context.trend)
                + 0.28 * max(0.0, context.mood)
                + 0.10 * max(0.0, last_return),
                0.0,
                1.0,
            )
            liquidation = 0.0
        else:
            directional = self._clamp(
                0.62 * max(0.0, -context.trend)
                + 0.28 * max(0.0, -context.mood)
                + 0.10 * max(0.0, -last_return),
                0.0,
                1.0,
            )
            liquidation = judgment.liquidation / 1.6
        return self._clamp(
            0.20
            + 0.30 * (judgment.urgency / 1.9)
            + 0.16 * event
            + 0.14 * stress
            + 0.12 * volatility
            + 0.12 * directional
            + 0.08 * liquidation
            + 0.10 * self._clamp(mdf_shock, 0.0, 1.0)
            - 0.10 * judgment.liquidity_aversion,
            0.0,
            1.0,
        )

    def _local_mdf_overlap(self, mdf: TickMap, tick: int) -> float:
        return (
            0.70 * mdf.get(tick, 0.0)
            + 0.15 * mdf.get(tick - 1, 0.0)
            + 0.15 * mdf.get(tick + 1, 0.0)
        )

    def _entry_overlap_factor(
        self,
        aggression: float,
        overlap: float,
    ) -> float:
        overlap = self._clamp(overlap, 0.0, 1.0)
        aggression = self._clamp(aggression, 0.0, 1.0)
        retained = max(0.0, 1.0 - overlap)
        sharpness = 7.70 - 3.00 * aggression
        floor = 0.004 + 0.050 * aggression
        return floor + (1.0 - floor) * retained**sharpness

    def _constrain_tick_mdf(self, current_tick: int, mdf_by_tick: TickMap) -> TickMap:
        ticks = self.relative_tick_grid()
        valid_ticks = [tick for tick in ticks if current_tick + tick >= 1]
        valid_mass = {
            tick: max(0.0, mdf_by_tick.get(tick, 0.0))
            for tick in valid_ticks
        }
        normalized = self._normalize_tick_map(valid_mass)
        return {tick: normalized.get(tick, 0.0) for tick in ticks}

    def _smooth_tick_mass(
        self,
        raw: TickMap,
        relative_ticks: list[int],
        *,
        center_weight: float,
        adjacent_weight: float,
        outer_weight: float,
    ) -> TickMap:
        tick_set = set(relative_ticks)
        kernel = (
            (0, center_weight),
            (-1, adjacent_weight),
            (1, adjacent_weight),
            (-2, outer_weight),
            (2, outer_weight),
        )
        smoothed: TickMap = {}
        for tick, mass in raw.items():
            if mass <= 0:
                continue
            for offset, weight in kernel:
                target = tick + offset
                if target in tick_set and weight > 0:
                    smoothed[target] = smoothed.get(target, 0.0) + mass * weight
        return {tick: smoothed.get(tick, 0.0) for tick in relative_ticks}

    def _add_buy_entry_pockets(
        self,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
        raw: dict[int, float],
    ) -> None:
        centers = self._buy_entry_pocket_centers(relative_ticks, context, signals, judgment)
        self._add_entry_pocket_mass("buy", centers, context, raw)

    def _buy_entry_pocket_centers(
        self,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        volatility = self._clamp(context.volatility, 0.0, 2.2)
        stress = self._clamp(signals.liquidity_stress, 0.0, 1.8)
        bid_liquidity = self._normalize_tick_signal(dict(signals.bid_liquidity_by_tick))
        for tick in relative_ticks:
            if tick > 0:
                continue
            level = abs(tick)
            texture = self._mdf_texture("buy", tick, context, salt=5.0)
            book_signal = (
                0.34 * signals.bid_shortage_by_tick.get(tick, 0.0)
                + 0.25 * signals.bid_gap_by_tick.get(tick, 0.0)
                + 0.22 * signals.bid_front_by_tick.get(tick, 0.0)
            )
            aversion_drag = max(0.0, 1.0 - 0.55 * judgment.liquidity_aversion)
            passive_center = (
                0.95
                + 1.70 * judgment.patience
                + 0.60 * volatility
                + 0.85 * judgment.opportunity
            )
            score = (
                0.38 * aversion_drag * (1.0 - exp(-2.1 * book_signal))
                + 0.040 * bid_liquidity.get(tick, 0.0)
                + 0.16 * texture
                + 0.20
                * judgment.pocket_bias
                * self._soft_tick_band(float(level), passive_center, 1.05 + 0.25 * stress)
            )
            scores[tick] = score
        return self._ordered_entry_pocket_centers(scores)

    def _add_sell_entry_pockets(
        self,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
        raw: dict[int, float],
    ) -> None:
        centers = self._sell_entry_pocket_centers(relative_ticks, context, signals, judgment)
        self._add_entry_pocket_mass("sell", centers, context, raw)

    def _sell_entry_pocket_centers(
        self,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
        judgment: _MDFSideJudgment,
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        volatility = self._clamp(context.volatility, 0.0, 2.2)
        stress = self._clamp(signals.liquidity_stress, 0.0, 1.8)
        ask_liquidity = self._normalize_tick_signal(dict(signals.ask_liquidity_by_tick))
        for tick in relative_ticks:
            if tick < 0:
                continue
            level = abs(tick)
            texture = self._mdf_texture("sell", tick, context, salt=5.0)
            book_signal = (
                0.34 * signals.ask_shortage_by_tick.get(tick, 0.0)
                + 0.25 * signals.ask_gap_by_tick.get(tick, 0.0)
                + 0.22 * signals.ask_front_by_tick.get(tick, 0.0)
            )
            aversion_drag = max(0.0, 1.0 - 0.55 * judgment.liquidity_aversion)
            passive_center = (
                0.95
                + 1.60 * judgment.patience
                + 0.60 * volatility
                - 0.50 * judgment.opportunity
                + 0.40 * judgment.fair_value_shift
            )
            score = (
                0.38 * aversion_drag * (1.0 - exp(-2.1 * book_signal))
                + 0.040 * ask_liquidity.get(tick, 0.0)
                + 0.16 * texture
                + 0.20
                * judgment.pocket_bias
                * self._soft_tick_band(float(level), passive_center, 1.05 + 0.25 * stress)
            )
            scores[tick] = score
        return self._ordered_entry_pocket_centers(scores)

    def _ordered_entry_pocket_centers(self, scores: dict[int, float]) -> list[tuple[int, float]]:
        ordered = sorted(scores.items(), key=lambda item: (-item[1], abs(item[0]), item[0]))
        centers: list[tuple[int, float]] = []
        for tick, score in ordered:
            if all(abs(tick - existing) > 1 for existing, _ in centers):
                centers.append((tick, score))
            if len(centers) >= 4:
                break
        return centers

    def _add_entry_pocket_mass(
        self,
        side: str,
        centers: list[tuple[int, float]],
        context: MDFContext,
        raw: dict[int, float],
    ) -> None:
        for rank, (center, score) in enumerate(centers):
            strength = self._clamp(0.024 + 0.082 * score, 0.018, 0.110)
            strength *= 1.0 / (1.0 + 0.18 * rank)
            for offset, shape in (
                (0, 0.70),
                (-1, 0.64),
                (1, 0.66),
                (-2, 0.35),
                (2, 0.34),
                (-3, 0.14),
                (3, 0.14),
            ):
                tick = center + offset
                if tick not in raw:
                    continue
                texture = self._mdf_texture(side, tick, context, salt=rank + 3.0)
                raw[tick] += strength * shape * (0.82 + 0.36 * texture)

    @staticmethod
    def _soft_tick_band(value: float, center: float, spread: float) -> float:
        return exp(-abs(value - center) / max(spread, 1e-12))

    def _entry_reservation_band(self, level: int) -> float:
        if level <= 1:
            return 0.92
        if level == 2:
            return 0.68
        if level == 3:
            return 0.50
        if level <= 5:
            return 0.32
        if level <= 8:
            return 0.20
        return 0.12

    def _entry_marketable_band(self, level: int) -> float:
        if level <= 1:
            return 1.0
        if level == 2:
            return 0.48
        if level <= 4:
            return 0.16
        return 0.035

    def _mdf_texture(
        self,
        side: str,
        tick: int,
        context: MDFContext,
        *,
        salt: float,
    ) -> float:
        side_seed = {"buy": 0.37, "sell": 0.91}[side]
        seed = 0.0 if self._seed is None else float(self._seed)
        primary = sin(
            tick * 12.9898
            + context.step_index * 0.917
            + context.current_tick * 0.071
            + seed * 0.0031
            + side_seed
            + salt * 1.618
        )
        secondary = sin(
            tick * 4.231
            + context.step_index * 0.113
            + seed * 0.017
            + side_seed * 3.0
            + salt
        )
        return 0.72 + 0.34 * (0.5 + 0.5 * primary) + 0.18 * (0.5 + 0.5 * secondary)

    def _cached_mdf_signals(
        self,
        price: float,
        cache: _StepComputationCache | None = None,
    ) -> MDFSignals:
        if cache is None or cache.basis_price != price:
            return self._mdf_signals(price)
        if cache.signals is None:
            cache.signals = self._mdf_signals(price)
        return cache.signals

    def _mdf_signals(self, price: float) -> MDFSignals:
        orderbook = self._snapshot_orderbook()
        best_bid = self._best_bid()
        best_ask = self._best_ask()
        spread_ticks = (
            max(1.0, (best_ask - best_bid) / self.gap)
            if best_bid is not None and best_ask is not None
            else 1.0 + self._microstructure.liquidity_stress
        )
        liquidity = self._price_map_to_relative_ticks(
            price,
            self._merge_maps(orderbook.bid_volume_by_price, orderbook.ask_volume_by_price),
        )
        bid_shortage, bid_gap, bid_front = self._book_observable_maps(
            price,
            "bid",
            orderbook.bid_volume_by_price,
        )
        ask_shortage, ask_gap, ask_front = self._book_observable_maps(
            price,
            "ask",
            orderbook.ask_volume_by_price,
        )
        return MDFSignals(
            orderbook_imbalance=self._last_imbalance,
            last_return_ticks=self._last_return_ticks,
            last_execution_volume=self._last_execution_volume,
            executed_volume_by_tick=self._price_map_to_relative_ticks(
                price, self._last_executed_by_price
            ),
            liquidity_by_tick=liquidity,
            bid_liquidity_by_tick=self._price_map_to_relative_ticks(
                price, orderbook.bid_volume_by_price
            ),
            ask_liquidity_by_tick=self._price_map_to_relative_ticks(
                price, orderbook.ask_volume_by_price
            ),
            bid_shortage_by_tick=bid_shortage,
            ask_shortage_by_tick=ask_shortage,
            bid_gap_by_tick=bid_gap,
            ask_gap_by_tick=ask_gap,
            bid_front_by_tick=bid_front,
            ask_front_by_tick=ask_front,
            spread_ticks=spread_ticks,
            cancel_pressure=self._microstructure.cancel_pressure,
            liquidity_stress=self._microstructure.liquidity_stress,
            stress_side=self._microstructure.stress_side,
            resiliency=self._microstructure.resiliency,
            activity=self._microstructure.activity,
            activity_event=self._microstructure.activity_event,
        )

    def _book_observable_maps(
        self,
        basis_price: float,
        side: str,
        volume_by_price: PriceMap,
    ) -> tuple[TickMap, TickMap, TickMap]:
        """Return shortage, gap, and near-front pressure on the side's passive grid."""

        raw_depth = self._price_map_to_relative_ticks(basis_price, volume_by_price)
        regime = self._active_settings
        side_ticks = [
            tick
            for tick in self.relative_tick_grid()
            if (tick < 0 if side == "bid" else tick > 0)
        ]
        if not side_ticks:
            return {}, {}, {}
        side_ticks.sort(key=abs)

        depth_values = {tick: max(0.0, raw_depth.get(tick, 0.0)) for tick in side_ticks}
        occupancies: dict[int, float] = {}
        shortages: dict[int, float] = {}
        front_signal: dict[int, float] = {}
        expected_scale = max(0.08, self.popularity * regime["liquidity"])

        for tick in side_ticks:
            level = max(1.0, abs(float(tick)))
            expected = expected_scale * regime["near_touch_liquidity"] / (
                level ** regime["depth_exponent"]
            )
            depth = depth_values[tick]
            occupancy = depth / (depth + expected + 1e-12)
            shortage = self._clamp(1.0 - occupancy, 0.0, 1.0)
            near_weight = exp(-max(0.0, level - 1.0) / 2.2)
            occupancies[tick] = occupancy
            shortages[tick] = shortage
            front_signal[tick] = shortage * near_weight

        gap_signal: dict[int, float] = {}
        previous_occupancy = None
        for tick in side_ticks:
            occupancy = occupancies[tick]
            discontinuity = (
                0.0 if previous_occupancy is None else abs(occupancy - previous_occupancy)
            )
            gap_signal[tick] = shortages[tick] * (0.15 + discontinuity)
            previous_occupancy = occupancy

        return (
            self._normalize_tick_signal(shortages),
            self._normalize_tick_signal(gap_signal),
            self._normalize_tick_signal(front_signal),
        )

    def _normalize_tick_signal(self, values: TickMap) -> TickMap:
        peak = max(values.values(), default=0.0)
        if peak <= 1e-12:
            return {}
        return {tick: value / peak for tick, value in values.items() if value > 1e-12}

    def _project_tick_mdf(self, current_tick: int, mdf_by_tick: dict[int, float]) -> PriceMap:
        projected: PriceMap = {}
        for relative_tick, probability in mdf_by_tick.items():
            absolute_tick = current_tick + relative_tick
            if absolute_tick < 1:
                continue
            price = self.tick_to_price(absolute_tick)
            projected[price] = projected.get(price, 0.0) + probability
        total = sum(projected.values())
        if total <= 0:
            return {}
        return {price: probability / total for price, probability in sorted(projected.items())}

    def _normalize_price_map(self, values: PriceMap) -> PriceMap:
        total = sum(max(0.0, value) for value in values.values())
        if total <= 0:
            uniform = 1.0 / len(values)
            return {price: uniform for price in values}
        return {price: max(0.0, value) / total for price, value in sorted(values.items())}

    def _normalize_tick_map(self, values: dict[int, float]) -> dict[int, float]:
        total = sum(max(0.0, value) for value in values.values())
        if total <= 0:
            uniform = 1.0 / len(values)
            return {tick: uniform for tick in values}
        return {tick: max(0.0, value) / total for tick, value in values.items()}

    def _taker_share(self, latent: LatentState, imbalance: float) -> float:
        regime = self._active_settings
        base = (
            0.12
            + 0.08 * latent.volatility
            + 0.04 * abs(imbalance)
        )
        return self._clamp(base * regime["taker"], 0.08, 0.62)

    def _cancel_orders(
        self,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        mdf: MDFState | None = None,
        cache: _StepComputationCache | None = None,
    ) -> PriceMap:
        mdf = mdf or self.state.mdf
        cancelled: PriceMap = {}
        for side in ("bid", "ask"):
            volumes = self._orderbook.volumes_for_side(side)
            side_weights, cancel_pressure, side_volume = self._cancel_sampling_profile(
                side,
                current_price,
                latent,
                micro,
                mdf,
                cache,
            )
            if side_volume <= 1e-12:
                continue
            if side_weights:
                regime = self._active_settings
                state_multiplier = (
                    1.0
                    + 0.50 * micro.cancel_pressure
                    + 0.28 * micro.liquidity_stress
                    + 0.17 * micro.spread_pressure
                )
                event_probability = self._cancel_event_probability(
                    side_volume,
                    cancel_pressure,
                    micro,
                )
                if self._unit_random() < event_probability:
                    expected_events = (
                        0.0061
                        * regime["cancel"]
                        * state_multiplier
                        * (0.045 + cancel_pressure)
                        * side_volume
                        / max(self._mean_child_order_size(), 1e-12)
                        / max(event_probability, 1e-12)
                    )
                    event_count = min(
                        max(1, int(6 + 4 * side_volume + 5 * self.popularity)),
                        self._sample_poisson(expected_events),
                    )
                    for _ in range(event_count):
                        price = self._sample_price(side_weights)
                        available = volumes.get(price, 0.0)
                        if available <= 1e-12:
                            continue
                        size_scale = self._clamp(
                            1.0
                            - 0.08 * micro.cancel_pressure
                            - 0.06 * micro.liquidity_stress
                            - 0.04 * micro.spread_pressure,
                            0.82,
                            1.0,
                        )
                        requested = min(available, size_scale * self._sample_order_size())
                        self._cancel_or_requote_volume(
                            cancelled,
                            side,
                            price,
                            requested,
                            current_price,
                            micro,
                            mdf,
                        )
            self._refresh_deep_passive_quotes(
                cancelled,
                side,
                current_price,
                micro,
                mdf,
                cache,
                side_volume,
            )
            self._orderbook.clean()
        cancelled = self._drop_zeroes(cancelled)
        micro.last_cancelled_volume = sum(cancelled.values())
        return cancelled

    def _refresh_deep_passive_quotes(
        self,
        cancelled: PriceMap,
        side: str,
        current_price: float,
        micro: _MicrostructureState,
        mdf: MDFState,
        cache: _StepComputationCache | None,
        side_volume: float,
    ) -> None:
        if side_volume <= 1e-12:
            return
        event_probability = self._deep_refresh_event_probability(side_volume, micro)
        if self._unit_random() >= event_probability:
            return
        volumes = self._orderbook.volumes_for_side(side)
        deep_prices = sorted(
            (
                (self._snap_price(price), max(0.0, volume))
                for price, volume in volumes.items()
                if volume > 1e-12
            ),
            key=lambda item: abs(item[0] - current_price),
            reverse=True,
        )
        if not deep_prices:
            return

        activity = self._event_cluster_signal(micro)
        stress = self._clamp(
            0.55 * micro.cancel_pressure
            + 0.35 * micro.liquidity_stress
            + 0.25 * micro.spread_pressure,
            0.0,
            2.5,
        ) / 2.5
        mean_child = max(self._mean_child_order_size(), 1e-12)
        side_cap = side_volume * self._clamp(
            0.010 + 0.024 * activity + 0.018 * stress,
            0.010,
            0.060,
        )
        refreshed = 0.0
        for price, _ in deep_prices:
            if refreshed >= side_cap:
                break
            distance = abs(price - current_price) / self.gap
            if distance < 4.0:
                continue
            actual = max(0.0, volumes.get(price, 0.0))
            if actual <= 1e-12:
                continue
            texture = self._quote_texture(side, price, current_price, distance)
            texture_fade = 1.0 - self._clamp(texture, 0.0, 1.0)
            expected = self._expected_mdf_volume_for_price(
                side,
                price,
                mdf,
                micro=micro,
                cache=cache,
            )
            allowance = self._deep_refresh_allowance(distance) * (0.72 + 0.28 * texture)
            excess = actual - allowance * expected - (0.14 + 0.08 * texture) * mean_child
            fade_excess = (
                actual
                * texture_fade
                * texture_fade
                * self._clamp((distance - 3.0) / 5.0, 0.0, 1.0)
                - 0.10 * mean_child
            )
            excess = max(excess, fade_excess)
            if excess <= 1e-12:
                continue
            deepness = self._clamp((distance - 4.0) / 5.0, 0.0, 1.0)
            refresh_rate = self._clamp(
                0.045
                + 0.055 * deepness
                + 0.035 * activity
                + 0.030 * stress
                + 0.060 * texture_fade,
                0.035,
                0.190,
            )
            if self._unit_random() >= refresh_rate:
                continue
            requested = min(
                excess * refresh_rate,
                actual * (0.08 + 0.05 * deepness + 0.10 * texture_fade),
                mean_child * (0.36 + 0.36 * activity + 0.20 * texture_fade),
                side_cap - refreshed,
            )
            removed = self._cancel_or_requote_volume(
                cancelled,
                side,
                price,
                requested,
                current_price,
                micro,
                mdf,
            )
            refreshed += removed

    def _refresh_post_event_deep_quotes(
        self,
        cancelled: PriceMap,
        current_price: float,
        micro: _MicrostructureState,
        mdf: MDFState,
        cache: _StepComputationCache | None,
    ) -> None:
        for side in ("bid", "ask"):
            volumes = self._orderbook.volumes_for_side(side)
            side_volume = sum(max(0.0, volume) for volume in volumes.values())
            self._refresh_deep_passive_quotes(
                cancelled,
                side,
                current_price,
                micro,
                mdf,
                cache,
                side_volume,
            )
            self._orderbook.clean()

    def _deep_refresh_event_probability(
        self,
        side_volume: float,
        micro: _MicrostructureState,
    ) -> float:
        if side_volume <= 1e-12:
            return 0.0
        activity = self._event_cluster_signal(micro)
        stress = self._clamp(
            0.55 * micro.cancel_pressure
            + 0.35 * micro.liquidity_stress
            + 0.25 * micro.spread_pressure,
            0.0,
            2.5,
        ) / 2.5
        volume_term = 1.0 - exp(-side_volume / max(10.0 + 7.0 * self.popularity, 1e-12))
        probability = 0.016 + 0.090 * activity + 0.070 * stress + 0.065 * volume_term
        return self._clamp(probability, 0.0, 0.30)

    def _deep_refresh_allowance(self, distance: float) -> float:
        return self._clamp(1.65 - 0.12 * max(0.0, distance - 4.0), 1.08, 1.65)

    def _quote_texture(
        self,
        side: str,
        price: float,
        current_price: float,
        distance: float,
    ) -> float:
        tick = self.price_to_tick(price)
        current_tick = self.price_to_tick(current_price)
        seed = 0.0 if self._seed is None else float(self._seed)
        step_index = float(self.state.step_index + 1)
        side_seed = 0.43 if side == "bid" else 1.11
        primary = 0.5 + 0.5 * sin(
            tick * 12.9898
            + current_tick * 0.071
            + step_index * 0.521
            + seed * 0.0031
            + side_seed
        )
        secondary = 0.5 + 0.5 * sin(
            tick * 4.231
            + current_tick * 0.173
            + step_index * 0.197
            + seed * 0.017
            + side_seed * 3.0
        )
        texture = 0.64 * primary + 0.36 * secondary
        if distance <= 2.0:
            return self._clamp(0.92 + 0.14 * texture, 0.92, 1.06)
        if distance <= 4.0:
            return self._clamp(0.58 + 0.58 * texture, 0.58, 1.12)
        return self._clamp(0.34 + 0.80 * texture, 0.34, 1.12)

    def _level_cancel_probability(
        self,
        side: str,
        price: float,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        mdf: MDFState | None = None,
    ) -> float:
        regime = self._active_settings
        distance = abs(price - current_price) / self.gap
        adverse_side = (side == "bid" and latent.trend < 0) or (side == "ask" and latent.trend > 0)
        adverse = abs(latent.trend) if adverse_side else 0.0
        distance_term = 0.045 * (1.0 - exp(-distance / 5.0))
        volatility_term = 0.026 * latent.volatility / (1.0 + latent.volatility)
        side_stress = self._stress_for_book_side(side, micro)
        stress_cancel = 0.12 * micro.liquidity_stress * side_stress * exp(-distance / 2.4)
        mismatch = self._mdf_cancel_mismatch(side, price, mdf or self.state.mdf, micro=micro)
        survival_weakness = self._resting_level_survival_weakness(
            side, price, current_price, latent, micro
        )
        survival_cancel = (
            0.12 * survival_weakness + 0.20 * mismatch
        ) * (0.72 + 0.28 * exp(-distance / 3.0))
        probability = (
            0.016
            + distance_term
            + volatility_term
            + self._cancel_burst_multiplier(micro, distance, adverse)
            + stress_cancel
            + survival_cancel
        ) * regime["cancel"]
        return self._clamp(probability, 0.006, 0.32)

    def _cancel_side_pressure(
        self,
        side: str,
        current_price: float,
        micro: _MicrostructureState,
        mdf: MDFState,
        cache: _StepComputationCache | None = None,
    ) -> float:
        volumes = self._orderbook.volumes_for_side(side)
        side_volume = sum(max(0.0, volume) for volume in volumes.values())
        if side_volume <= 1e-12:
            return 0.0
        weighted_pressure = 0.0
        for price, volume in volumes.items():
            if volume <= 1e-12:
                continue
            weighted_pressure += volume * self._cancel_price_pressure(
                side,
                price,
                current_price,
                micro,
                mdf,
                cache,
                side_volume=side_volume,
            )
        return self._clamp(weighted_pressure / side_volume, 0.0, 1.25)

    def _cancel_event_probability(
        self,
        side_volume: float,
        cancel_pressure: float,
        micro: _MicrostructureState,
    ) -> float:
        if side_volume <= 1e-12:
            return 0.0
        volume_term = 1.0 - exp(-side_volume / max(7.5 + 6.0 * self.popularity, 1e-12))
        mismatch_term = 1.0 - exp(-cancel_pressure / 0.42)
        probability = (
            0.032
            + 0.35 * mismatch_term
            + 0.18 * volume_term
            + 0.12 * self._clamp(micro.cancel_pressure, 0.0, 2.0) / 2.0
            + 0.09 * self._clamp(micro.liquidity_stress, 0.0, 2.0) / 2.0
            + 0.05 * self._clamp(micro.spread_pressure, 0.0, 1.8) / 1.8
        )
        probability *= 0.82 + 0.46 * self._event_cluster_signal(micro)
        return self._clamp(probability, 0.035, 0.86)

    def _cancel_price_pressure(
        self,
        side: str,
        price: float,
        current_price: float,
        micro: _MicrostructureState,
        mdf: MDFState,
        cache: _StepComputationCache | None = None,
        side_volume: float | None = None,
        current_tick: int | None = None,
    ) -> float:
        distance = abs(price - current_price) / self.gap
        mismatch = self._mdf_cancel_mismatch(
            side,
            price,
            mdf,
            micro=micro,
            cache=cache,
            side_volume=side_volume,
        )
        side_stress = self._stress_for_book_side(side, micro)
        stress = micro.liquidity_stress * side_stress * exp(-distance / 2.5)
        survival_weakness = self._resting_level_survival_weakness(
            side,
            price,
            current_price,
            LatentState(mood=0.0, trend=0.0, volatility=0.0),
            micro,
            cache,
            current_tick,
        )
        near_touch = exp(-max(0.0, distance - 1.0) / 1.0)
        deep_stale = self._clamp((distance - 3.0) / 3.0, 0.0, 1.0)
        distance_pressure = 1.0 - exp(-max(0.0, distance - 2.0) / 2.2)
        near_alignment = (
            near_touch
            * (1.0 - mismatch)
            * (1.0 - 0.45 * self._clamp(stress, 0.0, 1.0))
            * (1.0 + 0.30 * near_touch)
        )
        stale_pressure = distance_pressure * survival_weakness * (1.0 + 0.65 * deep_stale)
        return self._clamp(
            0.015
            + 0.74 * mismatch
            + 0.20 * stress
            + 0.08 * survival_weakness
            + 0.52 * distance_pressure
            + 0.26 * stale_pressure
            + 0.10 * deep_stale
            - 0.16 * near_alignment,
            0.0,
            1.25,
        )

    def _cancel_sampling_weights(
        self,
        side: str,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        mdf: MDFState,
        cache: _StepComputationCache | None = None,
    ) -> PriceMap:
        weights, _, _ = self._cancel_sampling_profile(
            side,
            current_price,
            latent,
            micro,
            mdf,
            cache,
        )
        return weights

    def _cancel_sampling_profile(
        self,
        side: str,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        mdf: MDFState,
        cache: _StepComputationCache | None = None,
    ) -> tuple[PriceMap, float, float]:
        del latent
        volumes = self._orderbook.volumes_for_side(side)
        side_volume = sum(max(0.0, volume) for volume in volumes.values())
        if side_volume <= 1e-12:
            return {}, 0.0, 0.0

        weights: PriceMap = {}
        weighted_pressure = 0.0
        current_tick = self.price_to_tick(current_price)
        for price, volume in volumes.items():
            if volume <= 1e-12:
                continue
            pressure = self._cancel_price_pressure(
                side,
                price,
                current_price,
                micro,
                mdf,
                cache,
                side_volume=side_volume,
                current_tick=current_tick,
            )
            weighted_pressure += volume * pressure
            weights[price] = volume * pressure
        pressure = self._clamp(weighted_pressure / side_volume, 0.0, 1.25)
        return (self._normalize_price_map(weights) if weights else {}, pressure, side_volume)

    def _mdf_cancel_mismatch(
        self,
        side: str,
        price: float,
        mdf: MDFState,
        *,
        micro: _MicrostructureState | None = None,
        cache: _StepComputationCache | None = None,
        side_volume: float | None = None,
    ) -> float:
        volumes = self._orderbook.volumes_for_side(side)
        side_volume = (
            sum(max(0.0, volume) for volume in volumes.values())
            if side_volume is None
            else side_volume
        )
        if side_volume <= 1e-12:
            return 0.0
        price = self._snap_price(price)
        book_share = max(0.0, volumes.get(price, 0.0)) / side_volume
        probabilities = self._entry_probabilities_for_book_side(side, mdf, cache)
        mdf_share = probabilities.get(price, 0.0)
        excess_book = max(0.0, book_share - mdf_share)
        unsupported = 1.0 - self._clamp(mdf_share / max(book_share, 1e-12), 0.0, 1.0)
        expected = self._expected_mdf_volume_for_price(side, price, mdf, micro=micro, cache=cache)
        actual = max(0.0, volumes.get(price, 0.0))
        absolute_excess = max(0.0, actual - expected) / max(actual + expected, 1e-12)
        return self._clamp(
            0.48 * excess_book + 0.24 * unsupported + 0.28 * absolute_excess,
            0.0,
            1.0,
        )

    def _expected_mdf_side_depth(
        self,
        side: str,
        *,
        micro: _MicrostructureState | None = None,
        cache: _StepComputationCache | None = None,
    ) -> float:
        if cache is not None and cache.micro is micro and side in cache.expected_depth_by_side:
            return cache.expected_depth_by_side[side]
        regime = self._active_settings
        micro = micro or self._microstructure
        side_stress = self._stress_for_book_side(side, micro)
        stress_drag = self._clamp(
            1.0 - 0.28 * self._clamp(micro.liquidity_stress * side_stress, 0.0, 2.0) / 2.0,
            0.45,
            1.15,
        )
        resiliency = self._clamp(micro.resiliency, 0.20, 1.55)
        depth = max(
            0.20,
            self.popularity
            * regime["liquidity"]
            * regime["near_touch_liquidity"]
            * resiliency
            * stress_drag
            * (7.5 + 0.35 * self.grid_radius),
        )
        if cache is not None and cache.micro is micro:
            cache.expected_depth_by_side[side] = depth
        return depth

    def _expected_mdf_volume_for_price(
        self,
        side: str,
        price: float,
        mdf: MDFState,
        *,
        micro: _MicrostructureState | None = None,
        cache: _StepComputationCache | None = None,
    ) -> float:
        price = self._snap_price(price)
        cache_key = (side, price)
        if (
            cache is not None
            and cache.mdf is mdf
            and cache_key in cache.expected_volume_by_side_price
        ):
            return cache.expected_volume_by_side_price[cache_key]
        probabilities = self._entry_probabilities_for_book_side(side, mdf, cache)
        expected = self._expected_mdf_side_depth(
            side,
            micro=micro,
            cache=cache,
        ) * probabilities.get(price, 0.0)
        if cache is not None and cache.mdf is mdf:
            cache.expected_volume_by_side_price[cache_key] = expected
        return expected

    def _cancel_price_volume(
        self,
        cancelled: PriceMap,
        side: str,
        price: float,
        requested: float,
    ) -> float:
        volume_by_price = self._orderbook.volumes_for_side(side)
        removed = min(max(0.0, requested), max(0.0, volume_by_price.get(price, 0.0)))
        if removed <= 1e-12:
            return 0.0
        cancelled[price] = cancelled.get(price, 0.0) + removed
        self._orderbook.adjust_volume(side, price, -removed)
        return removed

    def _cancel_or_requote_volume(
        self,
        cancelled: PriceMap,
        side: str,
        price: float,
        requested: float,
        current_price: float,
        micro: _MicrostructureState,
        mdf: MDFState,
    ) -> float:
        should_requote = self._unit_random() < self._cancel_requote_probability(
            side,
            price,
            current_price,
            micro,
            mdf,
        )
        destination_weights = (
            self._cancel_requote_sampling_weights(
                side,
                current_price,
                price,
                mdf,
                micro,
            )
            if should_requote
            else {}
        )
        removed = self._cancel_price_volume(cancelled, side, price, requested)
        if removed <= 1e-12:
            return 0.0
        if not destination_weights:
            return removed
        destination = self._sample_price(destination_weights)
        self._orderbook.add_lot(destination, removed, side, "requote")
        return removed

    def _cancel_requote_probability(
        self,
        side: str,
        price: float,
        current_price: float,
        micro: _MicrostructureState,
        mdf: MDFState,
    ) -> float:
        probabilities = self._normalized_price_probabilities(
            mdf.buy_entry_mdf_by_price if side == "bid" else mdf.sell_entry_mdf_by_price
        )
        if not probabilities:
            return 0.0
        price = self._snap_price(price)
        max_support = max(probabilities.values())
        support = probabilities.get(price, 0.0) / max(max_support, 1e-12)
        mismatch = self._mdf_cancel_mismatch(side, price, mdf, micro=micro)
        distance = abs(price - current_price) / self.gap
        distance_drag = 1.0 - exp(-max(0.0, distance - 2.0) / 2.5)
        spread_ticks = self._current_spread_ticks()
        spread_opportunity = self._clamp((spread_ticks - 1.0) / 4.0, 0.0, 1.0)
        side_stress = self._stress_for_book_side(side, micro)
        probability = (
            0.07
            + 0.22 * support
            + 0.08 * self._clamp(micro.resiliency, 0.0, 1.4) / 1.4
            + 0.09 * spread_opportunity
            - 0.16 * mismatch
            - 0.10 * self._clamp(micro.cancel_pressure, 0.0, 2.0) / 2.0
            - 0.10 * self._clamp(micro.liquidity_stress * side_stress, 0.0, 2.0) / 2.0
            - 0.07 * distance_drag
        )
        return self._clamp(probability, 0.0, 0.30)

    def _cancel_requote_sampling_weights(
        self,
        side: str,
        current_price: float,
        old_price: float,
        mdf: MDFState,
        micro: _MicrostructureState,
    ) -> PriceMap:
        probabilities = self._normalized_price_probabilities(
            mdf.buy_entry_mdf_by_price if side == "bid" else mdf.sell_entry_mdf_by_price
        )
        weights: PriceMap = {}
        old_price = self._snap_price(old_price)
        spread_ticks = self._current_spread_ticks()
        wide_spread_bonus = self._clamp((spread_ticks - 2.0) / 4.0, 0.0, 0.80)
        spread_pressure = self._clamp(micro.spread_pressure, 0.0, 1.8) / 1.8
        stress = self._clamp(micro.liquidity_stress, 0.0, 2.0) / 2.0
        stress_drag = self._clamp(
            1.0
            - 0.18 * self._clamp(micro.cancel_pressure, 0.0, 2.0) / 2.0
            - 0.12 * stress,
            0.55,
            1.0,
        )
        for price, probability in probabilities.items():
            if price == old_price:
                continue
            if side == "bid":
                if price >= current_price or price < self._min_price:
                    continue
                distance = (current_price - price) / self.gap
            elif price <= current_price:
                continue
            else:
                distance = (price - current_price) / self.gap
            near_touch_bias = 0.55 + 1.45 * exp(-max(0.0, distance - 1.0) / 1.25)
            if distance <= 2.0:
                near_touch_bias *= 1.0 + wide_spread_bonus * (
                    0.25
                    + 0.18 * self._clamp(micro.resiliency, 0.0, 1.4) / 1.4
                    + 0.10 * spread_pressure
                )
            deep_drag = 1.0 / (
                1.0 + (0.10 + 0.18 * stress) * max(0.0, distance - 3.0)
            )
            weights[price] = probability * near_touch_bias * stress_drag * deep_drag
        return self._normalize_price_map(weights) if weights else {}

    def _resting_level_survival_weakness(
        self,
        side: str,
        price: float,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        cache: _StepComputationCache | None = None,
        current_tick: int | None = None,
    ) -> float:
        distance = abs(price - current_price) / self.gap
        if current_tick is None:
            current_tick = self.price_to_tick(current_price)
        tick = self.price_to_tick(price) - current_tick
        signals = self._cached_mdf_signals(current_price, cache)
        if side == "bid":
            shortage = signals.bid_shortage_by_tick.get(tick, 0.0)
            adverse = max(0.0, -latent.trend)
        else:
            shortage = signals.ask_shortage_by_tick.get(tick, 0.0)
            adverse = max(0.0, latent.trend)
        distance_weakness = self._clamp(
            (distance - 1.0) / max(float(self.grid_radius), 1.0),
            0.0,
            1.0,
        )
        side_stress = self._stress_for_book_side(side, micro)
        return self._clamp(
            0.35 * distance_weakness
            + 0.22 * shortage
            + 0.24 * adverse
            + 0.19 * micro.liquidity_stress * side_stress,
            0.0,
            1.0,
        )

    def _stress_for_book_side(self, side: str, micro: _MicrostructureState) -> float:
        if side == "ask":
            return self._clamp(max(0.0, micro.stress_side), 0.0, 1.0)
        return self._clamp(max(0.0, -micro.stress_side), 0.0, 1.0)

    def _cancel_burst_multiplier(
        self,
        micro: _MicrostructureState,
        distance: float,
        adverse: float,
    ) -> float:
        vulnerability = 0.65 + 0.10 * min(distance, 6.0) + 0.45 * adverse
        pressure = 1.0 - exp(-micro.cancel_pressure)
        return 0.026 * pressure * vulnerability

    def _add_post_event_quote_arrivals(
        self,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        stats: _TradeStats,
        flow_imbalance: float,
        mdf: MDFState | None = None,
        cache: _StepComputationCache | None = None,
    ) -> None:
        if self.popularity <= 0:
            return
        mdf = mdf or self.state.mdf
        regime = self._active_settings
        stress_drag = 1.0 - 0.18 * self._clamp(micro.liquidity_stress, 0.0, 1.0)
        idle_maker_flow = (
            self.popularity
            * (0.12 + 0.13 / (1.0 + latent.volatility))
            * regime["liquidity"]
            * micro.resiliency
            * stress_drag
        )
        execution_refill = (
            stats.total_volume
            * regime["liquidity"]
            * micro.resiliency
            * (0.13 + 0.07 / (1.0 + latent.volatility))
        )
        repair_drag = self._clamp(
            1.0
            - 0.22 * self._clamp(micro.liquidity_stress, 0.0, 2.0) / 2.0
            - 0.14 * self._clamp(micro.cancel_pressure, 0.0, 2.0) / 2.0,
            0.62,
            1.0,
        )
        spread_repair = (
            self.popularity
            * regime["liquidity"]
            * micro.resiliency
            * 0.10
            * repair_drag
            * self._clamp((self._current_spread_ticks(cache) - 1.0) / 4.0, 0.0, 1.0)
        )
        base = idle_maker_flow + execution_refill + spread_repair
        if base <= 1e-12:
            return
        generic_base = 0.76 * base
        spread_gap = self._clamp((self._current_spread_ticks(cache) - 1.0) / 4.0, 0.0, 1.0)
        front_base = base * self._clamp(
            0.18
            + 0.28 * spread_gap
            + 0.08 * self._clamp(micro.liquidity_stress, 0.0, 2.0) / 2.0,
            0.14,
            0.48,
        )
        bid_tilt = self._replenishment_side_tilt("bid", flow_imbalance, latent, micro)
        ask_tilt = self._replenishment_side_tilt("ask", flow_imbalance, latent, micro)
        bid_target = generic_base * bid_tilt
        ask_target = generic_base * ask_tilt
        self._add_sampled_replenishment(
            "bid",
            "buy_entry",
            current_price,
            bid_target,
            mdf,
            latent,
            micro,
            cache,
        )
        self._add_sampled_replenishment(
            "ask",
            "sell_entry",
            current_price,
            ask_target,
            mdf,
            latent,
            micro,
            cache,
        )
        self._add_sampled_replenishment(
            "bid",
            "buy_entry",
            current_price,
            front_base * bid_tilt,
            mdf,
            latent,
            micro,
            cache,
            max_passive_distance=2,
        )
        self._add_sampled_replenishment(
            "ask",
            "sell_entry",
            current_price,
            front_base * ask_tilt,
            mdf,
            latent,
            micro,
            cache,
            max_passive_distance=2,
        )

    def _add_liquidity_replenishment(
        self,
        current_price: float,
        latent: LatentState,
        imbalance: float,
        micro: _MicrostructureState,
        mdf: MDFState | None = None,
    ) -> None:
        self._add_post_event_quote_arrivals(
            current_price,
            latent,
            micro,
            _TradeStats(executed_by_price={}),
            imbalance,
            mdf,
            None,
        )

    def _add_post_trade_replenishment(
        self,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
        stats: _TradeStats,
        flow_imbalance: float,
        mdf: MDFState | None = None,
    ) -> None:
        self._add_post_event_quote_arrivals(
            current_price,
            latent,
            micro,
            stats,
            flow_imbalance,
            mdf,
            None,
        )

    def _replenishment_side_tilt(
        self,
        side: str,
        imbalance: float,
        latent: LatentState,
        micro: _MicrostructureState,
    ) -> float:
        side_tilt = imbalance if side == "bid" else -imbalance
        trend_tilt = latent.trend if side == "bid" else -latent.trend
        pressure_drag = self._clamp(1.0 - 0.16 * micro.cancel_pressure, 0.35, 1.0)
        stress_drag = self._stress_replenishment_multiplier(side, 1, micro)
        spread_drag = self._clamp(1.0 - 0.06 * micro.spread_pressure, 0.78, 1.0)
        return pressure_drag * stress_drag * self._clamp(
            0.58 + 0.18 * side_tilt + 0.09 * trend_tilt,
            0.08,
            1.05,
        ) * spread_drag

    def _add_sampled_replenishment(
        self,
        side: str,
        kind: str,
        current_price: float,
        target_volume: float,
        mdf: MDFState,
        latent: LatentState,
        micro: _MicrostructureState,
        cache: _StepComputationCache | None = None,
        max_passive_distance: int | None = None,
    ) -> None:
        if target_volume <= 1e-12:
            return
        weights = self._replenishment_sampling_weights(
            side,
            current_price,
            mdf,
            latent,
            micro,
            cache,
            max_passive_distance=max_passive_distance,
        )
        if not weights:
            return
        event_probability = self._replenishment_event_probability(target_volume, micro)
        if self._unit_random() >= event_probability:
            return
        effective_target = target_volume / max(event_probability, 1e-12)
        event_count = self._sample_order_count(
            effective_target,
            max_count=max(1, int(4 + 5 * effective_target + 2 * self.popularity)),
        )
        if event_count <= 0:
            return
        volume_by_price: PriceMap = {}
        for _ in range(event_count):
            price = self._sample_price(weights)
            volume = self._sample_order_size()
            volume_by_price[price] = volume_by_price.get(price, 0.0) + volume
        if volume_by_price:
            self._add_lots(volume_by_price, side, kind)

    def _replenishment_event_probability(
        self,
        target_volume: float,
        micro: _MicrostructureState,
    ) -> float:
        if target_volume <= 1e-12:
            return 0.0
        if self.popularity >= 10.0 and target_volume >= 0.25:
            return 1.0
        if target_volume >= 1.0:
            return 1.0
        pressure_drag = 1.0 - 0.16 * self._clamp(micro.cancel_pressure, 0.0, 2.0) / 2.0
        stress_drag = 1.0 - 0.10 * self._clamp(micro.liquidity_stress, 0.0, 2.0) / 2.0
        spread_drag = 1.0 - 0.03 * self._clamp(micro.spread_pressure, 0.0, 1.8) / 1.8
        probability = (
            0.10
            + 0.54 * (1.0 - exp(-target_volume / 0.42))
            + 0.08 * self.popularity / (1.0 + self.popularity)
        ) * pressure_drag * stress_drag * spread_drag
        probability *= 0.88 + 0.26 * self._event_cluster_signal(micro)
        return self._clamp(probability, 0.07, 0.92)

    def _replenishment_sampling_weights(
        self,
        side: str,
        current_price: float,
        mdf: MDFState,
        latent: LatentState,
        micro: _MicrostructureState,
        cache: _StepComputationCache | None = None,
        max_passive_distance: int | None = None,
    ) -> PriceMap:
        del latent
        probabilities = self._entry_probabilities_for_book_side(side, mdf, cache)
        weights: PriceMap = {}
        spread_ticks = self._current_spread_ticks(cache)
        spread_pressure = self._clamp(micro.spread_pressure, 0.0, 1.8) / 1.8
        wide_spread_bonus = self._clamp((spread_ticks - 2.0) / 4.0, 0.0, 0.80)
        side_stress = self._stress_for_book_side(side, micro)
        near_stress_drag = self._clamp(
            1.0 - 0.10 * spread_pressure - 0.04 * micro.liquidity_stress * side_stress,
            0.80,
            1.30,
        )
        volumes = self._orderbook.volumes_for_side(side)
        for price, probability in probabilities.items():
            if side == "bid":
                if price >= current_price or price < self._min_price:
                    continue
                distance = (current_price - price) / self.gap
            elif price <= current_price:
                continue
            else:
                distance = (price - current_price) / self.gap
            if max_passive_distance is not None and distance > max_passive_distance:
                continue
            near_touch_bias = 0.78 + 2.10 * exp(-max(0.0, distance - 1.0) / 1.35)
            if distance <= 2.0:
                quote_improvement = 1.0 + (
                    0.46
                    * wide_spread_bonus
                    * micro.resiliency
                    * self._clamp(1.0 - 0.35 * spread_pressure, 0.45, 1.0)
                )
                near_touch_bias *= quote_improvement * near_stress_drag
            expected = self._expected_mdf_volume_for_price(
                side,
                price,
                mdf,
                micro=micro,
                cache=cache,
            )
            actual = max(0.0, volumes.get(price, 0.0))
            scale = max(expected, self._mean_child_order_size(), 1e-12)
            shortage = max(0.0, expected - actual) / scale
            overdepth = max(0.0, actual - expected) / scale
            shortage_gain = 0.82 + 0.16 * wide_spread_bonus if distance <= 2.0 else 0.58
            overdepth_penalty = 0.98 + 0.26 * self._clamp(distance - 3.0, 0.0, 4.0)
            depth_tilt = (1.0 + shortage_gain * self._clamp(shortage, 0.0, 2.0)) / (
                1.0 + overdepth_penalty * self._clamp(overdepth, 0.0, 3.2)
            )
            stress_tail = self._clamp(
                micro.liquidity_stress + 0.70 * micro.spread_pressure,
                0.0,
                2.4,
            ) / 2.4
            deep_tail_drag = 1.0 / (1.0 + 0.34 * stress_tail * max(0.0, distance - 5.0))
            texture = self._quote_texture(side, price, current_price, distance)
            weights[price] = (
                probability
                * near_touch_bias
                * depth_tilt
                * deep_tail_drag
                * texture
            )
        if weights:
            return self._normalize_price_map(weights)
        return {}

    def _replenishment_volume_for_level(
        self,
        level: int,
        side: str,
        base: float,
        imbalance: float,
        latent: LatentState,
        micro: _MicrostructureState,
        mdf: MDFState | None = None,
        current_price: float | None = None,
    ) -> float:
        regime = self._active_settings
        current_price = self.state.price if current_price is None else current_price
        shape = base * self._mdf_replenishment_shape(side, level, current_price, mdf)
        side_tilt = imbalance if side == "bid" else -imbalance
        trend_tilt = latent.trend if side == "bid" else -latent.trend
        pressure_drag = self._clamp(1.0 - 0.16 * micro.cancel_pressure, 0.35, 1.0)
        level_noise = self._book_level_noise(regime["book_noise"])
        stress_drag = self._stress_replenishment_multiplier(side, level, micro)
        volume = shape * pressure_drag * stress_drag * self._clamp(
            1.0 + 0.20 * side_tilt + 0.10 * trend_tilt,
            0.45,
            1.65,
        ) * level_noise
        cap = max(0.025, self.popularity * 0.45 / (1.0 + 0.18 * level))
        return min(volume, cap)

    def _mdf_replenishment_shape(
        self,
        side: str,
        level: int,
        current_price: float,
        mdf: MDFState | None,
    ) -> float:
        del mdf
        regime = self._active_settings
        baseline = 0.25 * regime["near_touch_liquidity"] / (
            max(1.0, float(level)) ** (regime["depth_exponent"] + 0.20)
        )
        signals = self._mdf_signals(current_price)
        tick = -level if side == "bid" else level
        if side == "bid":
            shortage = signals.bid_shortage_by_tick.get(tick, 0.0)
            front = signals.bid_front_by_tick.get(tick, 0.0)
            gap = signals.bid_gap_by_tick.get(tick, 0.0)
        else:
            shortage = signals.ask_shortage_by_tick.get(tick, 0.0)
            front = signals.ask_front_by_tick.get(tick, 0.0)
            gap = signals.ask_gap_by_tick.get(tick, 0.0)
        repair_signal = self._clamp(0.40 * shortage + 0.28 * front + 0.18 * gap, 0.0, 1.0)
        side_stress = self._stress_for_book_side(side, self._microstructure)
        stress_avoidance = self._clamp(signals.liquidity_stress * side_stress, 0.0, 1.0)
        maker_willingness = self._clamp(
            0.72 + 0.36 * signals.resiliency + 0.22 * repair_signal - 0.58 * stress_avoidance,
            0.10,
            1.45,
        )
        return baseline * maker_willingness

    def _stress_replenishment_multiplier(
        self,
        side: str,
        level: int,
        micro: _MicrostructureState,
    ) -> float:
        side_stress = self._stress_for_book_side(side, micro)
        if side_stress <= 1e-12 or micro.liquidity_stress <= 1e-12:
            return 1.0
        near_touch = exp(-max(0, level - 1) / 2.0)
        drag = 0.62 * micro.liquidity_stress * side_stress * near_touch
        return self._clamp(1.0 - drag, 0.18, 1.0)

    def _book_level_noise(self, sigma: float) -> float:
        if sigma <= 0:
            return 1.0
        return self._clamp(
            self._rng.lognormvariate(-0.5 * sigma * sigma, sigma),
            0.45,
            2.10,
        )

    def _add_lots(self, volume_by_price: PriceMap, side: str, kind: str) -> None:
        self._orderbook.add_lots(volume_by_price, side, kind)

    def _execute_market_flows(
        self,
        *,
        entry_orders: list[_IncomingOrder],
        stats: _TradeStats,
    ) -> _ExecutionResult:
        orders = list(entry_orders)
        self._rng.shuffle(orders)
        market_buy_volume = 0.0
        market_sell_volume = 0.0
        residual_market_buy = 0.0
        residual_market_sell = 0.0
        residual_buy_by_price: PriceMap = {}
        residual_sell_by_price: PriceMap = {}
        for order in orders:
            result = self._process_incoming_order(
                order,
                stats=stats,
                rest_residual=False,
            )
            if order.side == "buy":
                market_buy_volume += result.executed
                residual_market_buy += result.rested
                if result.rested > 1e-12:
                    price = self._snap_price(order.price)
                    residual_buy_by_price[price] = (
                        residual_buy_by_price.get(price, 0.0) + result.rested
                    )
            else:
                market_sell_volume += result.executed
                residual_market_sell += result.rested
                if result.rested > 1e-12:
                    price = self._snap_price(order.price)
                    residual_sell_by_price[price] = (
                        residual_sell_by_price.get(price, 0.0) + result.rested
                    )
        self._orderbook.add_lots(residual_buy_by_price, "bid", "buy_entry")
        self._orderbook.add_lots(residual_sell_by_price, "ask", "sell_entry")
        return _ExecutionResult(
            residual_market_buy=residual_market_buy,
            residual_market_sell=residual_market_sell,
            crossed_market_volume=0.0,
            market_buy_volume=market_buy_volume,
            market_sell_volume=market_sell_volume,
        )

    def _process_incoming_order(
        self,
        order: _IncomingOrder,
        stats: _TradeStats,
        *,
        rest_residual: bool = True,
    ) -> _ProcessedOrder:
        if order.volume <= 0:
            return _ProcessedOrder(executed=0.0, rested=0.0)
        remaining = order.volume
        executed = 0.0
        price_limit = self._snap_price(order.price)
        while remaining > 1e-12:
            price = self._best_ask() if order.side == "buy" else self._best_bid()
            if price is None:
                break
            if order.side == "buy" and price > price_limit + 1e-12:
                break
            if order.side == "sell" and price < price_limit - 1e-12:
                break

            volume_by_price = self._orderbook.volumes_for_taker_side(order.side)
            actual = min(remaining, max(0.0, volume_by_price.get(price, 0.0)))
            if actual <= 1e-12:
                self._discard_empty_head(price, "ask" if order.side == "buy" else "bid")
                continue
            resting_side = "ask" if order.side == "buy" else "bid"
            self._orderbook.adjust_volume(resting_side, price, -actual)
            remaining -= actual
            executed += actual
            stats.record(price, actual)
            self._discard_empty_head(price, "ask" if order.side == "buy" else "bid")

        restable = remaining
        if rest_residual and restable > 1e-12:
            passive_side = "bid" if order.side == "buy" else "ask"
            self._orderbook.add_lot(
                price_limit,
                restable,
                passive_side,
                order.kind,
            )
        return _ProcessedOrder(executed=executed, rested=restable)

    def _trim_orderbook_through_last_price(self, last_price: float) -> None:
        last_price = self._snap_price(last_price)
        while True:
            best_bid = self._best_bid()
            best_ask = self._best_ask()
            if best_bid is None or best_ask is None or best_bid < best_ask:
                break
            removed = False
            if best_bid >= last_price:
                self._orderbook.bid_volume_by_price.pop(best_bid, None)
                self._orderbook.invalidate("bid")
                removed = True
            if best_ask <= last_price:
                self._orderbook.ask_volume_by_price.pop(best_ask, None)
                self._orderbook.invalidate("ask")
                removed = True
            if not removed:
                if abs(best_bid - last_price) <= abs(best_ask - last_price):
                    self._orderbook.bid_volume_by_price.pop(best_bid, None)
                    self._orderbook.invalidate("bid")
                else:
                    self._orderbook.ask_volume_by_price.pop(best_ask, None)
                    self._orderbook.invalidate("ask")
        self._orderbook.clean()

    def _prune_orderbook_window(self, current_price: float) -> None:
        min_price = max(
            self._min_price,
            self._snap_price(current_price - self.grid_radius * self.gap),
        )
        max_price = self._snap_price(current_price + self.grid_radius * self.gap)
        for volume_by_price in (
            self._orderbook.bid_volume_by_price,
            self._orderbook.ask_volume_by_price,
        ):
            for price in list(volume_by_price):
                if price < min_price or price > max_price:
                    del volume_by_price[price]
                    self._orderbook.invalidate()
        self._orderbook.clean()

    def _best_bid(self) -> float | None:
        return self._orderbook.best_bid()

    def _best_ask(self) -> float | None:
        return self._orderbook.best_ask()

    def _spread(self, bid: float | None, ask: float | None) -> float | None:
        if bid is None or ask is None:
            return None
        return ask - bid

    def _current_spread_ticks(self, cache: _StepComputationCache | None = None) -> float:
        if cache is not None and cache.spread_ticks is not None:
            return cache.spread_ticks
        spread = self._spread(self._best_bid(), self._best_ask())
        if spread is None:
            spread_ticks = 2.0 + self._clamp(self._microstructure.liquidity_stress, 0.0, 2.0)
        else:
            spread_ticks = max(1.0, spread / self.gap)
        if cache is not None:
            cache.spread_ticks = spread_ticks
        return spread_ticks

    def _near_touch_imbalance(self, current_price: float) -> float:
        return self._orderbook.near_touch_imbalance(current_price, self.gap)

    def _nearby_book_depth(self, side: str, current_price: float) -> float:
        volumes = (
            self._orderbook.bid_volume_by_price
            if side == "bid"
            else self._orderbook.ask_volume_by_price
        )
        depth = 0.0
        for price, volume in volumes.items():
            distance = max(1.0, abs(current_price - price) / self.gap)
            if distance <= 5:
                depth += volume / distance
        return depth

    def _discard_empty_head(self, price: float, side: str) -> None:
        self._orderbook.discard_empty_head(price, side)

    def _clean_orderbook(self) -> None:
        self._orderbook.clean()

    def _snapshot_orderbook(self) -> OrderBookState:
        return self._orderbook.snapshot()

    def _merge_maps(self, *maps: PriceMap) -> PriceMap:
        merged: PriceMap = {}
        for values in maps:
            for price, volume in values.items():
                merged[price] = merged.get(price, 0.0) + volume
        return self._drop_zeroes(merged)

    def _price_map_to_relative_ticks(self, basis_price: float, values: PriceMap) -> TickMap:
        basis_tick = self.price_to_tick(basis_price)
        by_tick: TickMap = {}
        min_tick = -self.grid_radius
        max_tick = self.grid_radius
        for price, value in values.items():
            relative_tick = self.price_to_tick(price) - basis_tick
            if min_tick <= relative_tick <= max_tick:
                by_tick[relative_tick] = by_tick.get(relative_tick, 0.0) + value
        return by_tick

    def _single_price_volume(self, price: float, volume: float) -> PriceMap:
        if volume <= 0:
            return {}
        return {self._snap_price(price): volume}

    def _drop_zeroes(self, values: PriceMap) -> PriceMap:
        return {price: value for price, value in sorted(values.items()) if value > 1e-12}

    def _dedupe_prices(self, prices) -> list[float]:
        return sorted(set(prices))

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
