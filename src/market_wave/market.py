from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from math import exp, isfinite, log, log1p
from random import Random

from .distribution import (
    DynamicMDFModel,
    MDFContext,
    MDFModel,
    MDFSignals,
)
from .state import (
    IntensityState,
    LatentState,
    MarketState,
    MDFState,
    OrderBookState,
    PositionMassState,
    PriceMap,
    StepInfo,
    TickMap,
)


@dataclass
class _OrderLot:
    volume: float
    kind: str
    age: int = 0
    cohort_id: int | None = None


@dataclass
class _PositionCohort:
    side: str
    entry_price: float
    mass: float
    age: int = 0
    id: int = 0


@dataclass
class _ExitOrder:
    price: float
    volume: float
    cohort_id: int


@dataclass
class _IncomingOrder:
    side: str
    kind: str
    price: float
    volume: float
    cohort_id: int | None = None


@dataclass
class _EntryFlow:
    orders: list[_IncomingOrder]
    buy_intent_by_price: PriceMap
    sell_intent_by_price: PriceMap


@dataclass
class _ExitFlow:
    market_orders: list[_IncomingOrder]
    limit_orders: list[_ExitOrder]
    intent_volume_by_price: PriceMap

    @property
    def market_volume(self) -> float:
        return sum(order.volume for order in self.market_orders)


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
    residual_short_exit: float
    residual_market_sell: float
    residual_long_exit: float
    crossed_market_volume: float
    market_buy_volume: float = 0.0
    market_sell_volume: float = 0.0


@dataclass
class _MicrostructureState:
    activity: float = 0.0
    cancel_pressure: float = 0.0
    resiliency: float = 1.0
    recent_signed_return: float = 0.0
    recent_flow_imbalance: float = 0.0
    squeeze_pressure: float = 0.0
    activity_event: float = 0.0
    wall_pressure_by_absolute_tick: TickMap = field(default_factory=dict)
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


@dataclass
class _OrderBook:
    bid_lots: dict[float, list[_OrderLot]]
    ask_lots: dict[float, list[_OrderLot]]
    bid_volume_by_price: PriceMap = field(default_factory=dict)
    ask_volume_by_price: PriceMap = field(default_factory=dict)

    def age(self) -> None:
        for lots_by_price in (self.bid_lots, self.ask_lots):
            for lots in lots_by_price.values():
                for lot in lots:
                    lot.age += 1

    def add_lot(
        self,
        price: float,
        volume: float,
        side: str,
        kind: str,
        cohort_id: int | None = None,
    ) -> None:
        if volume <= 0:
            return
        lots_by_price = self.lots_for_side(side)
        volume_totals = self.volumes_for_side(side)
        lots = lots_by_price.setdefault(price, [])
        for lot in reversed(lots):
            if lot.kind == kind and lot.cohort_id == cohort_id:
                total = lot.volume + volume
                if total > 1e-12:
                    lot.age = int(round((lot.age * lot.volume) / total))
                    lot.volume = total
                break
        else:
            lots.append(_OrderLot(volume=volume, kind=kind, cohort_id=cohort_id))
        volume_totals[price] = volume_totals.get(price, 0.0) + volume

    def add_lots(self, volume_by_price: PriceMap, side: str, kind: str) -> None:
        for price, volume in volume_by_price.items():
            self.add_lot(price, volume, side, kind)

    def lots_for_side(self, side: str) -> dict[float, list[_OrderLot]]:
        return self.bid_lots if side == "bid" else self.ask_lots

    def lots_for_taker_side(self, side: str) -> dict[float, list[_OrderLot]]:
        return self.ask_lots if side == "buy" else self.bid_lots

    def volumes_for_side(self, side: str) -> PriceMap:
        return self.bid_volume_by_price if side == "bid" else self.ask_volume_by_price

    def adjust_volume(self, side: str, price: float, delta: float) -> None:
        if abs(delta) <= 1e-12:
            return
        volume_totals = self.volumes_for_side(side)
        next_volume = volume_totals.get(price, 0.0) + delta
        if next_volume > 1e-12:
            volume_totals[price] = next_volume
        else:
            volume_totals.pop(price, None)

    def rebuild_totals(self) -> None:
        self.bid_volume_by_price = self._aggregate_lots(self.bid_lots)
        self.ask_volume_by_price = self._aggregate_lots(self.ask_lots)

    def best_bid(self) -> float | None:
        prices = [price for price, volume in self.bid_volume_by_price.items() if volume > 1e-12]
        return max(prices) if prices else None

    def best_ask(self) -> float | None:
        prices = [price for price, volume in self.ask_volume_by_price.items() if volume > 1e-12]
        return min(prices) if prices else None

    def discard_empty_head(self, price: float, side: str) -> None:
        lots_by_price = self.lots_for_side(side)
        volume_totals = self.volumes_for_side(side)
        lots = lots_by_price.get(price, [])
        while lots and lots[0].volume <= 1e-12:
            lots.pop(0)
        if not lots and price in lots_by_price:
            del lots_by_price[price]
            volume_totals.pop(price, None)
        elif not lots:
            volume_totals.pop(price, None)

    def clean(self) -> None:
        for lots_by_price in (self.bid_lots, self.ask_lots):
            for price in list(lots_by_price):
                lots_by_price[price] = [lot for lot in lots_by_price[price] if lot.volume > 1e-12]
                if not lots_by_price[price]:
                    del lots_by_price[price]
        self.rebuild_totals()

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
    def _aggregate_lots(lots_by_price: dict[float, list[_OrderLot]]) -> PriceMap:
        return {
            price: volume
            for price, volume in sorted(
                (price, sum(lot.volume for lot in lots)) for price, lots in lots_by_price.items()
            )
            if volume > 1e-12
        }

    @staticmethod
    def _drop_zeroes(values: PriceMap) -> PriceMap:
        return {price: value for price, value in sorted(values.items()) if value > 1e-12}


class Market:
    @staticmethod
    def _regime_names() -> set[str]:
        return {"normal", "trend_up", "trend_down", "high_vol", "thin_liquidity", "squeeze"}

    def _next_regime(self) -> str:
        if self.regime != "auto":
            return self.regime
        switch_probability = self._clamp(0.015 + 0.08 * self.augmentation_strength, 0.0, 0.18)
        if self._rng.random() >= switch_probability:
            return self._active_regime
        regimes = sorted(self._regime_names())
        weights = {
            "normal": 0.42,
            "trend_up": 0.14,
            "trend_down": 0.14,
            "high_vol": 0.12,
            "thin_liquidity": 0.10,
            "squeeze": 0.08,
        }
        draw = self._rng.random() * sum(weights.values())
        cumulative = 0.0
        for regime in regimes:
            cumulative += weights[regime]
            if draw <= cumulative:
                return regime
        return "normal"

    def _regime_settings(self, regime: str) -> dict[str, float]:
        settings = {
            "normal": {
                "mood": 0.0,
                "trend": 0.0,
                "volatility": 1.0,
                "volatility_bias": 0.0,
                "intensity": 1.0,
                "taker": 1.0,
                "cancel": 1.0,
                "liquidity": 1.0,
                "spread": 1.0,
                "activity_gain": 0.20,
                "activity_decay": 0.76,
                "cancel_burst": 0.68,
                "cancel_decay": 0.70,
                "depth_exponent": 1.35,
                "near_touch_liquidity": 1.0,
                "wall_persistence": 0.82,
                "wall_strength": 0.45,
                "resiliency": 0.92,
                "event_gain": 0.36,
                "book_noise": 0.18,
                "trend_exhaustion": 0.055,
                "flow_reversal": 0.045,
                "squeeze_gain": 0.0,
            },
            "trend_up": {
                "mood": 0.04,
                "trend": 0.08,
                "volatility": 1.05,
                "volatility_bias": 0.01,
                "intensity": 1.12,
                "taker": 1.08,
                "cancel": 1.02,
                "liquidity": 1.0,
                "spread": 1.05,
                "activity_gain": 0.22,
                "activity_decay": 0.78,
                "cancel_burst": 0.72,
                "cancel_decay": 0.70,
                "depth_exponent": 1.28,
                "near_touch_liquidity": 1.04,
                "wall_persistence": 0.84,
                "wall_strength": 0.50,
                "resiliency": 0.98,
                "event_gain": 0.40,
                "book_noise": 0.20,
                "trend_exhaustion": 0.050,
                "flow_reversal": 0.040,
                "squeeze_gain": 0.0,
            },
            "trend_down": {
                "mood": -0.04,
                "trend": -0.08,
                "volatility": 1.08,
                "volatility_bias": 0.015,
                "intensity": 1.14,
                "taker": 1.10,
                "cancel": 1.04,
                "liquidity": 0.98,
                "spread": 1.08,
                "activity_gain": 0.23,
                "activity_decay": 0.78,
                "cancel_burst": 0.76,
                "cancel_decay": 0.70,
                "depth_exponent": 1.26,
                "near_touch_liquidity": 0.98,
                "wall_persistence": 0.82,
                "wall_strength": 0.46,
                "resiliency": 0.90,
                "event_gain": 0.42,
                "book_noise": 0.20,
                "trend_exhaustion": 0.055,
                "flow_reversal": 0.045,
                "squeeze_gain": 0.0,
            },
            "high_vol": {
                "mood": 0.0,
                "trend": 0.0,
                "volatility": 1.35,
                "volatility_bias": 0.06,
                "intensity": 1.22,
                "taker": 1.18,
                "cancel": 1.25,
                "liquidity": 0.9,
                "spread": 1.35,
                "activity_gain": 0.26,
                "activity_decay": 0.76,
                "cancel_burst": 0.92,
                "cancel_decay": 0.70,
                "depth_exponent": 1.05,
                "near_touch_liquidity": 0.76,
                "wall_persistence": 0.66,
                "wall_strength": 0.30,
                "resiliency": 0.58,
                "event_gain": 0.62,
                "book_noise": 0.28,
                "trend_exhaustion": 0.060,
                "flow_reversal": 0.055,
                "squeeze_gain": 0.0,
            },
            "thin_liquidity": {
                "mood": 0.0,
                "trend": 0.0,
                "volatility": 1.18,
                "volatility_bias": 0.025,
                "intensity": 0.74,
                "taker": 1.16,
                "cancel": 1.28,
                "liquidity": 0.42,
                "spread": 1.22,
                "activity_gain": 0.19,
                "activity_decay": 0.74,
                "cancel_burst": 0.84,
                "cancel_decay": 0.70,
                "depth_exponent": 1.55,
                "near_touch_liquidity": 0.58,
                "wall_persistence": 0.58,
                "wall_strength": 0.24,
                "resiliency": 0.42,
                "event_gain": 0.58,
                "book_noise": 0.30,
                "trend_exhaustion": 0.065,
                "flow_reversal": 0.060,
                "squeeze_gain": 0.0,
            },
            "squeeze": {
                "mood": 0.015,
                "trend": 0.015,
                "volatility": 1.45,
                "volatility_bias": 0.035,
                "intensity": 1.14,
                "taker": 1.10,
                "cancel": 1.18,
                "liquidity": 0.7,
                "spread": 1.4,
                "activity_gain": 0.25,
                "activity_decay": 0.76,
                "cancel_burst": 0.82,
                "cancel_decay": 0.70,
                "depth_exponent": 1.16,
                "near_touch_liquidity": 0.70,
                "wall_persistence": 0.74,
                "wall_strength": 0.40,
                "resiliency": 0.66,
                "event_gain": 0.70,
                "book_noise": 0.28,
                "trend_exhaustion": 0.070,
                "flow_reversal": 0.045,
                "squeeze_gain": 0.34,
            },
        }
        return settings.get(regime, settings["normal"])

    def __init__(
        self,
        initial_price: float,
        gap: float,
        popularity: float = 1.0,
        seed: int | None = None,
        grid_radius: int = 20,
        mdf_model: MDFModel | None = None,
        augmentation_strength: float = 0.0,
        mdf_temperature: float = 1.0,
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
        if not isfinite(mdf_temperature) or mdf_temperature <= 0:
            raise ValueError("mdf_temperature must be a positive finite number")
        if regime not in self._regime_names() | {"auto"}:
            raise ValueError("regime must be one of the supported regimes or 'auto'")

        self.gap = float(gap)
        self.popularity = float(popularity)
        self._min_price = self.gap
        self.grid_radius = int(grid_radius)
        self.mdf_model = mdf_model or DynamicMDFModel()
        self.augmentation_strength = float(augmentation_strength)
        self.mdf_temperature = float(mdf_temperature)
        self._mdf_persistence = 0.88
        self._mdf_diffusion = 0.04
        self._mdf_floor_mix = 0.04
        self._mdf_eps = 1e-12
        self.regime = regime
        self._active_regime = "normal" if regime == "auto" else regime
        self._rng = Random(seed)
        self._seed = seed
        self.history: list[StepInfo] = []
        self._orderbook = _OrderBook(bid_lots={}, ask_lots={})
        self._long_cohorts: list[_PositionCohort] = []
        self._short_cohorts: list[_PositionCohort] = []
        self._long_cohorts_by_bucket: dict[int, _PositionCohort] = {}
        self._short_cohorts_by_bucket: dict[int, _PositionCohort] = {}
        self._cohorts_by_id: dict[int, _PositionCohort] = {}
        self._next_cohort_id = 1
        self._long_mass_total = 0.0
        self._short_mass_total = 0.0
        self._last_return_ticks = 0.0
        self._last_abs_return_ticks = 0.0
        self._last_imbalance = 0.0
        self._last_execution_volume = 0.0
        self._last_executed_by_price: PriceMap = {}
        self._microstructure = _MicrostructureState()
        self._mdf_model_accepts_signals = self._scores_accepts_signals(self.mdf_model.scores)
        self._mdf_memory: dict[str, dict[int, float]] = {}

        price = self._snap_price(float(initial_price))
        grid = self._price_grid(price)
        tick_grid = self.relative_tick_grid()
        current_tick = self.price_to_tick(price)
        intensity = IntensityState(total=0.0, buy=0.0, sell=0.0, buy_ratio=0.5, sell_ratio=0.5)
        latent = self._initial_latent()
        uniform = 1.0 / len(grid)
        initial_mdf_by_price = {price_level: uniform for price_level in grid}
        initial_mdf = {tick: 1.0 / len(tick_grid) for tick in tick_grid}
        zero_mass = {price_level: 0.0 for price_level in grid}
        mdf = MDFState(
            relative_ticks=tick_grid,
            buy_entry_mdf=initial_mdf,
            sell_entry_mdf=initial_mdf.copy(),
            long_exit_mdf=initial_mdf.copy(),
            short_exit_mdf=initial_mdf.copy(),
            buy_entry_mdf_by_price=initial_mdf_by_price,
            sell_entry_mdf_by_price=initial_mdf_by_price.copy(),
            long_exit_mdf_by_price=initial_mdf_by_price.copy(),
            short_exit_mdf_by_price=initial_mdf_by_price.copy(),
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
            position_mass=PositionMassState(
                long_exit_mass_by_price=zero_mass,
                short_exit_mass_by_price=zero_mass.copy(),
            ),
        )

    def _initial_latent(self) -> LatentState:
        regime = self._regime_settings(self._active_regime)
        mood = self._clamp(
            self._rng.gauss(regime["mood"], 0.18),
            -0.7,
            0.7,
        )
        trend = self._clamp(
            self._rng.gauss(regime["trend"] + 0.20 * mood, 0.14),
            -0.7,
            0.7,
        )
        volatility = self._clamp(
            0.12
            + 0.18 * self._rng.random()
            + abs(self._rng.gauss(0.0, 0.11))
            + regime["volatility_bias"],
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
            bid_levels = sorted(orderbook.bid_volume_by_price.items(), reverse=True)[:depth]
            ask_levels = sorted(orderbook.ask_volume_by_price.items())[:depth]
            for row, (_, volume) in enumerate(bid_levels):
                bid_matrix[row][column] = volume
            for row, (_, volume) in enumerate(ask_levels):
                ask_matrix[row][column] = volume
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
        self._age_state()
        self._clear_stale_exit_lots()
        self._clean_orderbook()
        self._clean_cohorts()

        price_grid = self._price_grid(price_before)
        orderbook_before = self._snapshot_orderbook()
        position_before = self._snapshot_position_mass()
        best_bid_before = self._best_bid()
        best_ask_before = self._best_ask()
        spread_before = self._spread(best_bid_before, best_ask_before)
        pre_imbalance = self._near_touch_imbalance(price_before)

        self._active_regime = self._next_regime()
        latent = self._next_latent(state.latent)
        micro = self._next_microstructure_state(state.latent, latent, pre_imbalance)
        intensity = self._next_intensity(latent, micro)
        mdf = self._next_mdf(
            price_before,
            price_grid,
            latent,
            step_index=step_index,
            update_memory=True,
        )

        cancelled_volume = self._cancel_orders(price_before, latent, micro)
        long_exit_flow = self._exit_flow(
            "long",
            price_before,
            latent,
            mdf.long_exit_mdf_by_price,
        )
        short_exit_flow = self._exit_flow(
            "short",
            price_before,
            latent,
            mdf.short_exit_mdf_by_price,
        )
        mdf = self._with_cohort_exit_mdfs(
            price_before,
            mdf,
            long_exit_flow.intent_volume_by_price,
            short_exit_flow.intent_volume_by_price,
        )

        self._add_liquidity_replenishment(price_before, latent, pre_imbalance, micro)
        self._add_exit_orders(short_exit_flow.limit_orders, side="bid", kind="short_exit")
        self._add_exit_orders(long_exit_flow.limit_orders, side="ask", kind="long_exit")
        entry_flow = self._entry_flow(intensity, mdf)

        stats = _TradeStats(executed_by_price={})
        execution = self._execute_market_flows(
            entry_orders=entry_flow.orders,
            short_exit_market_orders=short_exit_flow.market_orders,
            long_exit_market_orders=long_exit_flow.market_orders,
            mdf=mdf,
            stats=stats,
        )
        self._clean_orderbook()
        self._clean_cohorts()

        price_after = self._next_price_after_trading(price_before, stats, execution)
        price_after = self._snap_price(price_after)
        self._last_return_ticks = (price_after - price_before) / self.gap
        self._last_abs_return_ticks = abs(self._last_return_ticks)
        self._last_execution_volume = stats.total_volume
        self._last_executed_by_price = self._drop_zeroes(stats.executed_by_price)
        self._trim_orderbook_through_last_price(price_after)
        self._prune_orderbook_window(price_after)
        self._update_wall_memory_from_book(price_after, micro)

        state_grid = self._price_grid(price_after)
        state_mdf = self._reproject_mdf(price_after, mdf)
        position_after = self._position_mass_from_cohorts(
            state_mdf.long_exit_mdf_by_price,
            state_mdf.short_exit_mdf_by_price,
        )
        orderbook_after = self._snapshot_orderbook()
        best_bid_after = self._best_bid()
        best_ask_after = self._best_ask()
        spread_after = self._spread(best_bid_after, best_ask_after)
        total_market_buy = execution.market_buy_volume
        total_market_sell = execution.market_sell_volume
        # Incoming order leftovers now rest in the book at their sampled price.
        # These compatibility fields remain zero in the order-book-first engine.
        residual_market_buy = 0.0
        residual_market_sell = 0.0
        market_flow_delta = total_market_buy - total_market_sell
        market_flow_total = total_market_buy + total_market_sell
        order_flow_imbalance = (
            self._clamp(market_flow_delta / market_flow_total, -1.0, 1.0)
            if market_flow_total > 1e-12
            else self._near_touch_imbalance(price_after)
        )
        self._last_imbalance = order_flow_imbalance

        entry_volume = self._merge_maps(
            entry_flow.buy_intent_by_price,
            entry_flow.sell_intent_by_price,
        )
        exit_volume = self._merge_maps(
            long_exit_flow.intent_volume_by_price,
            short_exit_flow.intent_volume_by_price,
        )
        buy_volume = self._merge_maps(
            entry_flow.buy_intent_by_price,
            short_exit_flow.intent_volume_by_price,
        )
        sell_volume = self._merge_maps(
            entry_flow.sell_intent_by_price,
            long_exit_flow.intent_volume_by_price,
        )
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
            long_exit_mdf=mdf.long_exit_mdf,
            short_exit_mdf=mdf.short_exit_mdf,
            buy_entry_mdf_by_price=mdf.buy_entry_mdf_by_price,
            sell_entry_mdf_by_price=mdf.sell_entry_mdf_by_price,
            long_exit_mdf_by_price=mdf.long_exit_mdf_by_price,
            short_exit_mdf_by_price=mdf.short_exit_mdf_by_price,
            buy_volume_by_price=buy_volume,
            sell_volume_by_price=sell_volume,
            entry_volume_by_price=entry_volume,
            exit_volume_by_price=exit_volume,
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
            position_mass_before=position_before,
            position_mass_after=position_after,
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
            position_mass=position_after,
        )
        return step_info

    def _age_state(self) -> None:
        self._orderbook.age()
        for cohort in (*self._long_cohorts, *self._short_cohorts):
            cohort.age += 1

    def _next_price_after_trading(
        self,
        price_before: float,
        stats: _TradeStats,
        execution: _ExecutionResult,
    ) -> float:
        if stats.total_volume <= 0 or stats.last_price is None:
            return price_before
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
            1.40,
        )
        # Last price shows where trades occurred; flow only nudges revealed pressure.
        price_discovery = self._clamp(abs(execution_move_ticks) / 1.5, 0.25, 1.0)
        flow_move_ticks = flow_imbalance * (0.18 + 0.12 * volume_confidence) * price_discovery
        proposed_ticks = execution_move_ticks * (0.72 + 0.18 * volume_confidence)
        proposed_ticks += flow_move_ticks
        proposed_ticks = self._clamp(proposed_ticks, -3.0, 3.0)
        return max(self._min_price, price_before + proposed_ticks * self.gap)

    def _price_grid(self, center_price: float) -> list[float]:
        center = self._snap_price(center_price)
        return self._dedupe_prices(
            self._snap_price(center + offset * self.gap)
            for offset in range(-self.grid_radius, self.grid_radius + 1)
        )

    def price_to_tick(self, price: float) -> int:
        return max(1, int(round(self._snap_price(price) / self.gap)))

    def tick_to_price(self, tick: int) -> float:
        return self._snap_price(max(1, int(tick)) * self.gap)

    def relative_tick_grid(self) -> list[int]:
        return list(range(-self.grid_radius, self.grid_radius + 1))

    def _snap_price(self, price: float) -> float:
        snapped = max(self._min_price, round(price / self.gap) * self.gap)
        return self._clean_number(snapped)

    def _clean_number(self, value: float) -> float:
        rounded = round(value, 10)
        if rounded.is_integer():
            return float(int(rounded))
        return rounded

    def _next_latent(self, latent: LatentState) -> LatentState:
        regime = self._regime_settings(self._active_regime)
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
        if self._active_regime != "squeeze":
            return 0.0
        return self._clamp(micro.squeeze_pressure, 0.0, 1.5)

    def _next_intensity(
        self,
        latent: LatentState,
        micro: _MicrostructureState | None = None,
    ) -> IntensityState:
        regime = self._regime_settings(self._active_regime)
        micro = micro or self._microstructure
        augmentation = 1.0 + self.augmentation_strength * self._rng.uniform(-0.18, 0.28)
        flow_reversal = self._flow_reversal_pressure(micro)
        activity = micro.activity
        activity_multiplier = 1.0 + 0.38 * self._clamp(activity, 0.0, 2.2)
        event_multiplier = 1.0 + 0.32 * self._clamp(micro.activity_event, 0.0, 1.8)
        dry_up = self._clamp(1.0 - 0.10 * micro.cancel_pressure, 0.68, 1.0)
        total = (
            self.popularity
            * (1.0 + 2.2 * latent.volatility)
            * regime["intensity"]
            * augmentation
            * activity_multiplier
            * event_multiplier
            * dry_up
        )
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
        regime = self._regime_settings(self._active_regime)
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

        wall_pressure = self._decay_wall_pressure(previous.wall_pressure_by_absolute_tick, regime)
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
        resiliency_target = regime["resiliency"] * self._clamp(
            1.0 - 0.18 * cancel_pressure + 0.08 * previous.activity - 0.05 * activity_event,
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
            wall_pressure_by_absolute_tick=wall_pressure,
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
            max(0.0, inputs.return_shock - 0.42)
            + 0.36 * inputs.volatility_shock
            + 0.22 * max(0.0, abs(inputs.flow_imbalance) - 0.35)
            + 0.18 * max(0.0, inputs.execution_pressure - 0.90)
        )
        return self._clamp(
            0.72 * previous.activity_event + regime["event_gain"] * burst_seed,
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

    def _squeeze_setup_pressure(self) -> float:
        if self._active_regime != "squeeze" or self._short_mass_total <= 1e-12:
            return 0.0
        total_mass = self._long_mass_total + self._short_mass_total
        short_share = self._short_mass_total / max(total_mass, 1e-12)
        relative_crowding = self._clamp((short_share - 0.45) * 2.0, 0.0, 1.0)
        absolute_crowding = self._clamp(
            self._short_mass_total / max(8.0 * self.popularity, 1.0),
            0.0,
            1.0,
        )
        short_pressure = max(relative_crowding, 0.45 * absolute_crowding)
        adverse_move = self._clamp(max(0.0, self._last_return_ticks) / 3.0, 0.0, 1.0)
        buy_pressure = self._clamp(max(0.0, self._last_imbalance), 0.0, 1.0)
        trigger = self._clamp(0.65 * adverse_move + 0.35 * buy_pressure - 0.15, 0.0, 1.0)
        return self._clamp(short_pressure * trigger, 0.0, 1.0)

    def _decay_wall_pressure(self, values: TickMap, regime: dict[str, float]) -> TickMap:
        persistence = regime["wall_persistence"]
        return {
            tick: pressure * persistence
            for tick, pressure in values.items()
            if pressure * persistence > 1e-4
        }

    def _next_mdf(
        self,
        price: float,
        price_grid: list[float],
        latent: LatentState,
        *,
        step_index: int,
        update_memory: bool,
    ) -> MDFState:
        del price_grid
        relative_ticks = self.relative_tick_grid()
        context = MDFContext(
            current_price=price,
            current_tick=self.price_to_tick(price),
            tick_size=self.gap,
            mood=latent.mood,
            trend=latent.trend,
            volatility=latent.volatility * self._regime_settings(self._active_regime)["spread"],
            regime=self._active_regime,
            augmentation_strength=self.augmentation_strength,
            step_index=step_index,
            rng=self._rng,
        )
        signals = self._mdf_signals(price)
        temperature = max(0.05, self.mdf_temperature * (0.65 + context.volatility))
        buy_entry_ticks = self._evolve_mdf(
            "buy_entry",
            self._model_scores("buy", "entry", relative_ticks, context, signals),
            temperature=temperature,
            update_memory=update_memory,
        )
        sell_entry_ticks = self._evolve_mdf(
            "sell_entry",
            self._model_scores("sell", "entry", relative_ticks, context, signals),
            temperature=temperature,
            update_memory=update_memory,
        )
        long_exit_ticks = self._evolve_mdf(
            "long_exit",
            self._model_scores("long", "exit", relative_ticks, context, signals),
            temperature=temperature,
            update_memory=update_memory,
        )
        short_exit_ticks = self._evolve_mdf(
            "short_exit",
            self._model_scores("short", "exit", relative_ticks, context, signals),
            temperature=temperature,
            update_memory=update_memory,
        )
        buy_entry = self._project_tick_mdf(context.current_tick, buy_entry_ticks)
        sell_entry = self._project_tick_mdf(context.current_tick, sell_entry_ticks)
        long_exit = self._project_tick_mdf(context.current_tick, long_exit_ticks)
        short_exit = self._project_tick_mdf(context.current_tick, short_exit_ticks)
        return MDFState(
            buy_entry_mdf_by_price=buy_entry,
            sell_entry_mdf_by_price=sell_entry,
            long_exit_mdf_by_price=long_exit,
            short_exit_mdf_by_price=short_exit,
            relative_ticks=relative_ticks,
            buy_entry_mdf=buy_entry_ticks,
            sell_entry_mdf=sell_entry_ticks,
            long_exit_mdf=long_exit_ticks,
            short_exit_mdf=short_exit_ticks,
        )

    def _reproject_mdf(
        self,
        price: float,
        mdf: MDFState,
    ) -> MDFState:
        current_tick = self.price_to_tick(price)
        return MDFState(
            buy_entry_mdf_by_price=self._project_tick_mdf(current_tick, mdf.buy_entry_mdf),
            sell_entry_mdf_by_price=self._project_tick_mdf(current_tick, mdf.sell_entry_mdf),
            long_exit_mdf_by_price=self._project_tick_mdf(current_tick, mdf.long_exit_mdf),
            short_exit_mdf_by_price=self._project_tick_mdf(current_tick, mdf.short_exit_mdf),
            relative_ticks=list(mdf.relative_ticks),
            buy_entry_mdf=dict(mdf.buy_entry_mdf),
            sell_entry_mdf=dict(mdf.sell_entry_mdf),
            long_exit_mdf=dict(mdf.long_exit_mdf),
            short_exit_mdf=dict(mdf.short_exit_mdf),
        )

    def _entry_flow(
        self,
        intensity: IntensityState,
        mdf: MDFState,
    ) -> _EntryFlow:
        orders: list[_IncomingOrder] = []
        buy_intent: PriceMap = {}
        sell_intent: PriceMap = {}

        for price, probability in mdf.buy_entry_mdf_by_price.items():
            volume = intensity.buy * probability
            if volume <= 1e-12:
                continue
            price = self._snap_price(price)
            buy_intent[price] = buy_intent.get(price, 0.0) + volume
            orders.append(
                _IncomingOrder(
                    side="buy",
                    kind="buy_entry",
                    price=price,
                    volume=volume,
                )
            )

        for price, probability in mdf.sell_entry_mdf_by_price.items():
            volume = intensity.sell * probability
            if volume <= 1e-12:
                continue
            price = self._snap_price(price)
            sell_intent[price] = sell_intent.get(price, 0.0) + volume
            orders.append(
                _IncomingOrder(
                    side="sell",
                    kind="sell_entry",
                    price=price,
                    volume=volume,
                )
            )

        return _EntryFlow(
            orders=orders,
            buy_intent_by_price=self._drop_zeroes(buy_intent),
            sell_intent_by_price=self._drop_zeroes(sell_intent),
        )

    def _model_scores(
        self,
        side: str,
        intent: str,
        relative_ticks: list[int],
        context: MDFContext,
        signals: MDFSignals,
    ) -> list[float]:
        if self._mdf_model_accepts_signals:
            return self.mdf_model.scores(side, intent, relative_ticks, context, signals)
        return self.mdf_model.scores(side, intent, relative_ticks, context)

    @staticmethod
    def _scores_accepts_signals(scores) -> bool:
        try:
            parameters = list(inspect.signature(scores).parameters.values())
        except (TypeError, ValueError):
            return True
        if any(
            parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            for parameter in parameters
        ):
            return True
        return len(parameters) >= 5

    def _mdf_signals(self, price: float) -> MDFSignals:
        orderbook = self._snapshot_orderbook()
        liquidity = self._price_map_to_relative_ticks(
            price,
            self._merge_maps(orderbook.bid_volume_by_price, orderbook.ask_volume_by_price),
        )
        position = self._snapshot_position_mass()
        return MDFSignals(
            orderbook_imbalance=self._last_imbalance,
            last_return_ticks=self._last_return_ticks,
            last_execution_volume=self._last_execution_volume,
            executed_volume_by_tick=self._price_map_to_relative_ticks(
                price, self._last_executed_by_price
            ),
            liquidity_by_tick=liquidity,
            long_position_mass_by_tick=self._price_map_to_relative_ticks(
                price, position.long_exit_mass_by_price
            ),
            short_position_mass_by_tick=self._price_map_to_relative_ticks(
                price, position.short_exit_mass_by_price
            ),
        )

    def _evolve_mdf(
        self, key: str, scores: list[float], *, temperature: float, update_memory: bool
    ) -> dict[int, float]:
        ticks = self.relative_tick_grid()
        if len(scores) != len(ticks):
            raise ValueError("mdf_model.scores must return one score per relative tick")
        previous = self._mdf_memory.get(key)
        if previous is None:
            uniform = 1.0 / len(ticks)
            previous = {tick: uniform for tick in ticks}
        cleaned = [score if isfinite(score) else 0.0 for score in scores]
        logits = [
            self._mdf_persistence * log(max(previous.get(tick, 0.0), self._mdf_eps))
            + score / temperature
            for tick, score in zip(ticks, cleaned, strict=False)
        ]
        logit_peak = max(logits, default=0.0)
        weights = [exp(self._clamp(logit - logit_peak, -50.0, 0.0)) for logit in logits]
        proposal = self._normalize_tick_map(
            {tick: weight for tick, weight in zip(ticks, weights, strict=False)}
        )
        diffused = self._diffuse_tick_mdf(proposal, self._mdf_diffusion)
        uniform = 1.0 / len(ticks)
        evolved = self._normalize_tick_map(
            {
                tick: (1.0 - self._mdf_floor_mix) * diffused.get(tick, 0.0)
                + self._mdf_floor_mix * uniform
                for tick in ticks
            }
        )
        if update_memory:
            self._mdf_memory[key] = evolved
        return evolved

    def _diffuse_tick_mdf(self, values: dict[int, float], diffusion: float) -> dict[int, float]:
        if diffusion <= 0:
            return dict(values)
        ticks = self.relative_tick_grid()
        diffused = {}
        for index, tick in enumerate(ticks):
            neighbors = []
            if index > 0:
                neighbors.append(ticks[index - 1])
            if index < len(ticks) - 1:
                neighbors.append(ticks[index + 1])
            neighbor_mass = (
                sum(values.get(neighbor, 0.0) for neighbor in neighbors) / len(neighbors)
                if neighbors
                else values.get(tick, 0.0)
            )
            diffused[tick] = (1.0 - diffusion) * values.get(tick, 0.0) + diffusion * neighbor_mass
        return self._normalize_tick_map(diffused)

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

    def _with_cohort_exit_mdfs(
        self,
        basis_price: float,
        mdf: MDFState,
        long_exit_intent: PriceMap,
        short_exit_intent: PriceMap,
    ) -> MDFState:
        long_by_price = (
            self._cohort_intent_price_mdf(basis_price, long_exit_intent)
            or mdf.long_exit_mdf_by_price
        )
        short_by_price = (
            self._cohort_intent_price_mdf(basis_price, short_exit_intent)
            or mdf.short_exit_mdf_by_price
        )
        return MDFState(
            buy_entry_mdf_by_price=mdf.buy_entry_mdf_by_price,
            sell_entry_mdf_by_price=mdf.sell_entry_mdf_by_price,
            long_exit_mdf_by_price=long_by_price,
            short_exit_mdf_by_price=short_by_price,
            relative_ticks=list(mdf.relative_ticks),
            buy_entry_mdf=dict(mdf.buy_entry_mdf),
            sell_entry_mdf=dict(mdf.sell_entry_mdf),
            long_exit_mdf=self._price_map_to_normalized_relative_mdf(
                basis_price, long_by_price
            )
            or dict(mdf.long_exit_mdf),
            short_exit_mdf=self._price_map_to_normalized_relative_mdf(
                basis_price, short_by_price
            )
            or dict(mdf.short_exit_mdf),
        )

    def _cohort_intent_price_mdf(self, basis_price: float, values: PriceMap) -> PriceMap:
        grid = self._price_grid(basis_price)
        grid_set = set(grid)
        cleaned = {
            price: max(0.0, value)
            for price, value in values.items()
            if price in grid_set and value > 1e-12
        }
        total = sum(cleaned.values())
        if total <= 1e-12:
            return {}
        uniform = 1.0 / len(grid)
        floor_mix = min(0.44, max(self._mdf_floor_mix, 0.42))
        return self._normalize_price_map(
            {
                price: (1.0 - floor_mix) * cleaned.get(price, 0.0) / total
                + floor_mix * uniform
                for price in grid
            }
        )

    def _price_map_to_normalized_relative_mdf(
        self, basis_price: float, values: PriceMap
    ) -> TickMap:
        raw_by_tick = self._price_map_to_relative_ticks(basis_price, values)
        return self._normalize_tick_map(
            {tick: raw_by_tick.get(tick, 0.0) for tick in self.relative_tick_grid()}
        )

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
        regime = self._regime_settings(self._active_regime)
        base = (
            0.12
            + 0.08 * latent.volatility
            + 0.04 * abs(imbalance)
        )
        return self._clamp(base * regime["taker"], 0.08, 0.62)

    def _exit_flow(
        self,
        side: str,
        current_price: float,
        latent: LatentState,
        exit_mdf: PriceMap,
    ) -> _ExitFlow:
        del exit_mdf
        cohorts = self._long_cohorts if side == "long" else self._short_cohorts
        market_orders: list[_IncomingOrder] = []
        limit_orders: list[_ExitOrder] = []
        intent_volume_by_price: PriceMap = {}
        for cohort in cohorts:
            if cohort.mass <= 0:
                continue
            pnl_ticks = self._cohort_pnl_ticks(side, cohort, current_price)
            adverse_trend = max(0.0, -latent.trend) if side == "long" else max(0.0, latent.trend)
            stop_ticks = 2.0 + 1.4 * latent.volatility
            target_ticks = 3.0 + 1.8 * latent.volatility
            stop_pressure = self._threshold_pressure(-pnl_ticks - stop_ticks, 2.0)
            take_profit_pressure = self._threshold_pressure(pnl_ticks - target_ticks, 3.0)
            age_pressure = min(0.10, cohort.age * 0.002)
            volatility_pressure = 0.025 * latent.volatility / (1.0 + latent.volatility)
            urgency = self._clamp(
                0.006
                + age_pressure
                + 0.18 * stop_pressure
                + 0.10 * take_profit_pressure
                + 0.06 * adverse_trend
                + volatility_pressure,
                0.0,
                0.42,
            )
            desired = min(cohort.mass, cohort.mass * urgency)
            if desired <= 1e-12:
                continue

            market_share = self._clamp(
                0.14
                + 0.48 * stop_pressure
                + 0.15 * adverse_trend
                + 0.20 * latent.volatility / (1.0 + latent.volatility),
                0.12,
                0.88,
            )
            market_volume = desired * market_share
            passive_volume = desired - market_volume

            if market_volume > 1e-12:
                stop_price = self._exit_stop_price(side, current_price)
                market_orders.append(
                    _IncomingOrder(
                        side="sell" if side == "long" else "buy",
                        kind="long_exit" if side == "long" else "short_exit",
                        price=stop_price,
                        volume=market_volume,
                        cohort_id=cohort.id,
                    )
                )
                intent_volume_by_price[stop_price] = (
                    intent_volume_by_price.get(stop_price, 0.0) + market_volume
                )

            if passive_volume <= 1e-12:
                continue
            for price, volume in self._passive_exit_kernel(
                side,
                cohort,
                current_price,
                latent,
                pnl_ticks,
                target_ticks,
                passive_volume,
            ):
                limit_orders.append(_ExitOrder(price=price, volume=volume, cohort_id=cohort.id))
                intent_volume_by_price[price] = intent_volume_by_price.get(price, 0.0) + volume

        return _ExitFlow(
            market_orders=market_orders,
            limit_orders=limit_orders,
            intent_volume_by_price=self._drop_zeroes(intent_volume_by_price),
        )

    def _cohort_pnl_ticks(
        self, side: str, cohort: _PositionCohort, current_price: float
    ) -> float:
        if side == "long":
            return (current_price - cohort.entry_price) / self.gap
        return (cohort.entry_price - current_price) / self.gap

    def _threshold_pressure(self, value: float, scale: float) -> float:
        if value <= 0:
            return 0.0
        scaled = self._clamp(value / max(scale, 1e-12), -50.0, 50.0)
        return 2.0 * (1.0 / (1.0 + exp(-scaled)) - 0.5)

    def _exit_stop_price(self, side: str, current_price: float) -> float:
        if side == "long":
            return max(self._min_price, self._snap_price(current_price - self.gap))
        return self._snap_price(current_price + self.gap)

    def _passive_exit_kernel(
        self,
        side: str,
        cohort: _PositionCohort,
        current_price: float,
        latent: LatentState,
        pnl_ticks: float,
        target_ticks: float,
        volume: float,
    ) -> list[tuple[float, float]]:
        if volume <= 1e-12:
            return []
        if side == "long":
            raw_target = cohort.entry_price + target_ticks * self.gap
            target = max(current_price + self.gap, raw_target)
            side_floor = current_price + self.gap
        else:
            raw_target = cohort.entry_price - target_ticks * self.gap
            target = min(current_price - self.gap, raw_target)
            side_floor = max(self._min_price, current_price - self.gap)

        target = self._snap_price(target)
        min_price = self._snap_price(current_price - self.grid_radius * self.gap)
        max_price = self._snap_price(current_price + self.grid_radius * self.gap)
        target = self._clamp(target, max(self._min_price, min_price), max_price)
        spread_ticks = max(1.1, 1.4 + 1.4 * latent.volatility + 0.08 * abs(pnl_ticks))
        center_tick = self.price_to_tick(target)
        candidates: list[tuple[float, float]] = []
        for offset in range(-2, 3):
            price = self.tick_to_price(center_tick + offset)
            if side == "long" and price < side_floor:
                continue
            if side == "short" and price > side_floor:
                continue
            if price < self._min_price or price < min_price or price > max_price:
                continue
            weight = exp(-abs(offset) / spread_ticks)
            candidates.append((price, weight))
        total_weight = sum(weight for _, weight in candidates)
        if total_weight <= 1e-12:
            fallback = self._snap_price(side_floor)
            return [(fallback, volume)]
        return [(price, volume * weight / total_weight) for price, weight in candidates]

    def _cancel_orders(
        self,
        current_price: float,
        latent: LatentState,
        micro: _MicrostructureState,
    ) -> PriceMap:
        cancelled: PriceMap = {}
        for side, lots_by_price in (
            ("bid", self._orderbook.bid_lots),
            ("ask", self._orderbook.ask_lots),
        ):
            for price, lots in list(lots_by_price.items()):
                survivors = []
                for lot in lots:
                    probability = self._lot_cancel_probability(
                        side,
                        price,
                        current_price,
                        lot,
                        latent,
                        micro,
                    )
                    cancel_fraction = probability * self._rng.random()
                    removed = lot.volume * cancel_fraction
                    lot.volume -= removed
                    if removed > 1e-12:
                        cancelled[price] = cancelled.get(price, 0.0) + removed
                        self._orderbook.adjust_volume(side, price, -removed)
                    if lot.volume > 1e-12:
                        survivors.append(lot)
                    elif lot.volume > 0:
                        self._orderbook.adjust_volume(side, price, -lot.volume)
                if survivors:
                    lots_by_price[price] = survivors
                else:
                    del lots_by_price[price]
        cancelled = self._drop_zeroes(cancelled)
        micro.last_cancelled_volume = sum(cancelled.values())
        return cancelled

    def _lot_cancel_probability(
        self,
        side: str,
        price: float,
        current_price: float,
        lot: _OrderLot,
        latent: LatentState,
        micro: _MicrostructureState,
    ) -> float:
        regime = self._regime_settings(self._active_regime)
        distance = abs(price - current_price) / self.gap
        adverse_side = (side == "bid" and latent.trend < 0) or (side == "ask" and latent.trend > 0)
        adverse = abs(latent.trend) if adverse_side else 0.0
        age_term = 0.075 * (1.0 - exp(-lot.age / 9.0))
        distance_term = 0.045 * (1.0 - exp(-distance / 5.0))
        volatility_term = 0.026 * latent.volatility / (1.0 + latent.volatility)
        probability = (
            0.016
            + age_term
            + distance_term
            + volatility_term
            + self._cancel_burst_multiplier(micro, distance, adverse)
        ) * regime["cancel"]
        return self._clamp(probability, 0.01, 0.55)

    def _cancel_burst_multiplier(
        self,
        micro: _MicrostructureState,
        distance: float,
        adverse: float,
    ) -> float:
        vulnerability = 0.65 + 0.10 * min(distance, 6.0) + 0.45 * adverse
        pressure = 1.0 - exp(-micro.cancel_pressure)
        return 0.050 * pressure * vulnerability

    def _add_liquidity_replenishment(
        self,
        current_price: float,
        latent: LatentState,
        imbalance: float,
        micro: _MicrostructureState,
    ) -> None:
        regime = self._regime_settings(self._active_regime)
        base = (
            self.popularity
            * (0.18 + 0.20 / (1.0 + latent.volatility))
            * regime["liquidity"]
            * micro.resiliency
        )
        depth = min(13, self.grid_radius + 1)
        current_tick = self.price_to_tick(current_price)
        for level in range(1, depth):
            bid_volume = self._replenishment_volume_for_level(
                level,
                "bid",
                current_tick,
                base,
                imbalance,
                latent,
                micro,
            )
            ask_volume = self._replenishment_volume_for_level(
                level,
                "ask",
                current_tick,
                base,
                imbalance,
                latent,
                micro,
            )
            self._add_lots(
                {self._snap_price(current_price - level * self.gap): bid_volume}, "bid", "buy_entry"
            )
            self._add_lots(
                {self._snap_price(current_price + level * self.gap): ask_volume},
                "ask",
                "sell_entry",
            )

    def _replenishment_volume_for_level(
        self,
        level: int,
        side: str,
        current_tick: int,
        base: float,
        imbalance: float,
        latent: LatentState,
        micro: _MicrostructureState,
    ) -> float:
        regime = self._regime_settings(self._active_regime)
        near_touch = regime["near_touch_liquidity"] if level <= 2 else 1.0
        shape = base * near_touch / (level ** regime["depth_exponent"])
        side_tilt = imbalance if side == "bid" else -imbalance
        trend_tilt = latent.trend if side == "bid" else -latent.trend
        tick = -level if side == "bid" else level
        wall = 1.0 + regime["wall_strength"] * self._wall_pressure_at_relative_tick(
            micro,
            current_tick,
            tick,
        )
        pressure_drag = self._clamp(1.0 - 0.16 * micro.cancel_pressure, 0.35, 1.0)
        level_noise = self._book_level_noise(regime["book_noise"])
        return shape * wall * pressure_drag * self._clamp(
            1.0 + 0.20 * side_tilt + 0.10 * trend_tilt,
            0.45,
            1.65,
        ) * level_noise

    def _book_level_noise(self, sigma: float) -> float:
        if sigma <= 0:
            return 1.0
        return self._clamp(
            self._rng.lognormvariate(-0.5 * sigma * sigma, sigma),
            0.45,
            2.10,
        )

    def _update_wall_memory_from_book(
        self,
        current_price: float,
        micro: _MicrostructureState,
    ) -> None:
        orderbook = self._snapshot_orderbook()
        current_tick = self.price_to_tick(current_price)
        relative = self._price_map_to_relative_ticks(
            current_price,
            self._merge_maps(
                orderbook.bid_volume_by_price,
                orderbook.ask_volume_by_price,
            ),
        )
        positive = [volume for volume in relative.values() if volume > 1e-12]
        if not positive:
            micro.wall_pressure_by_absolute_tick = self._decay_wall_pressure(
                micro.wall_pressure_by_absolute_tick,
                self._regime_settings(self._active_regime),
            )
            return
        baseline = sorted(positive)[len(positive) // 2]
        updated = dict(micro.wall_pressure_by_absolute_tick)
        # Wall pressure is anchored by absolute tick so walls can be consumed or left behind.
        for relative_tick, volume in relative.items():
            if relative_tick == 0 or abs(relative_tick) > self.grid_radius:
                continue
            prominence = self._clamp(volume / max(baseline, 1e-12) - 1.0, 0.0, 4.0)
            absolute_tick = current_tick + relative_tick
            if absolute_tick < 1:
                continue
            old = updated.get(absolute_tick, 0.0)
            updated[absolute_tick] = self._clamp(0.86 * old + 0.14 * prominence, 0.0, 3.0)
        micro.wall_pressure_by_absolute_tick = {
            tick: pressure
            for tick, pressure in updated.items()
            if pressure > 1e-4 and abs(tick - current_tick) <= 2 * self.grid_radius
        }

    def _wall_pressure_at_relative_tick(
        self,
        micro: _MicrostructureState,
        current_tick: int,
        relative_tick: int,
    ) -> float:
        return micro.wall_pressure_by_absolute_tick.get(current_tick + relative_tick, 0.0)

    def _add_lots(self, volume_by_price: PriceMap, side: str, kind: str) -> None:
        self._orderbook.add_lots(volume_by_price, side, kind)

    def _add_exit_orders(self, orders: list[_ExitOrder], side: str, kind: str) -> None:
        for order in orders:
            self._orderbook.add_lot(
                order.price,
                order.volume,
                side,
                kind,
                cohort_id=order.cohort_id,
            )

    def _execute_market_flows(
        self,
        *,
        entry_orders: list[_IncomingOrder],
        short_exit_market_orders: list[_IncomingOrder],
        long_exit_market_orders: list[_IncomingOrder],
        mdf: MDFState,
        stats: _TradeStats,
    ) -> _ExecutionResult:
        orders = [
            *entry_orders,
            *short_exit_market_orders,
            *long_exit_market_orders,
        ]
        self._rng.shuffle(orders)
        market_buy_volume = 0.0
        market_sell_volume = 0.0
        for order in orders:
            executed = self._process_incoming_order(
                order,
                mdf=mdf,
                stats=stats,
            )
            if order.side == "buy":
                market_buy_volume += executed
            else:
                market_sell_volume += executed
        return _ExecutionResult(
            residual_market_buy=0.0,
            residual_short_exit=0.0,
            residual_market_sell=0.0,
            residual_long_exit=0.0,
            crossed_market_volume=0.0,
            market_buy_volume=market_buy_volume,
            market_sell_volume=market_sell_volume,
        )

    def _process_incoming_order(
        self,
        order: _IncomingOrder,
        mdf: MDFState,
        stats: _TradeStats,
    ) -> float:
        if order.volume <= 0:
            return 0.0
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

            lots_by_price = self._orderbook.lots_for_taker_side(order.side)
            resting_lot = lots_by_price[price][0]
            fill_volume = min(remaining, resting_lot.volume)
            taker_actual = self._available_fill_volume(
                order.kind,
                fill_volume,
                order.cohort_id,
            )
            resting_actual = self._available_fill_volume(
                resting_lot.kind, fill_volume, resting_lot.cohort_id
            )
            actual = min(taker_actual, resting_actual)
            if actual <= 1e-12:
                if self._is_exit_kind(resting_lot.kind) and resting_actual <= 1e-12:
                    resting_side = "ask" if order.side == "buy" else "bid"
                    self._orderbook.adjust_volume(resting_side, price, -resting_lot.volume)
                    resting_lot.volume = 0.0
                    self._discard_empty_head(price, resting_side)
                    continue
                break
            resting_lot.volume -= actual
            resting_side = "ask" if order.side == "buy" else "bid"
            self._orderbook.adjust_volume(resting_side, price, -actual)
            remaining -= actual
            executed += actual
            self._apply_fill(order.kind, actual, price, mdf, cohort_id=order.cohort_id)
            self._apply_fill(
                resting_lot.kind,
                actual,
                price,
                mdf,
                cohort_id=resting_lot.cohort_id,
            )
            stats.record(price, actual)
            self._discard_empty_head(price, "ask" if order.side == "buy" else "bid")

        restable = self._available_fill_volume(order.kind, remaining, order.cohort_id)
        if restable > 1e-12:
            passive_side = "bid" if order.side == "buy" else "ask"
            self._orderbook.add_lot(
                price_limit,
                restable,
                passive_side,
                order.kind,
                cohort_id=order.cohort_id,
            )
        return executed

    def _apply_fill(
        self,
        kind: str,
        volume: float,
        execution_price: float,
        mdf: MDFState,
        cohort_id: int | None = None,
    ) -> float:
        del mdf
        if volume <= 0:
            return 0.0
        if kind == "buy_entry":
            self._add_position_cohort("long", execution_price, volume)
            self._long_mass_total += volume
            return volume
        elif kind == "sell_entry":
            self._add_position_cohort("short", execution_price, volume)
            self._short_mass_total += volume
            return volume
        elif kind == "long_exit":
            removed = self._remove_cohort_mass(
                self._long_cohorts,
                execution_price,
                volume,
                cohort_id=cohort_id,
            )
            self._long_mass_total = max(0.0, self._long_mass_total - removed)
            return removed
        elif kind == "short_exit":
            removed = self._remove_cohort_mass(
                self._short_cohorts,
                execution_price,
                volume,
                cohort_id=cohort_id,
            )
            self._short_mass_total = max(0.0, self._short_mass_total - removed)
            return removed
        return 0.0

    def _add_position_cohort(self, side: str, execution_price: float, volume: float) -> None:
        bucket = self._cohort_bucket_tick(execution_price)
        cohorts_by_bucket = (
            self._long_cohorts_by_bucket if side == "long" else self._short_cohorts_by_bucket
        )
        cohort = cohorts_by_bucket.get(bucket)
        if cohort is not None:
            total = cohort.mass + volume
            if total > 1e-12:
                cohort.age = int(round((cohort.age * cohort.mass) / total))
                cohort.mass = total
            return
        cohort = _PositionCohort(
            side=side,
            entry_price=self.tick_to_price(bucket),
            mass=volume,
            id=self._next_cohort_id,
        )
        self._next_cohort_id += 1
        cohorts_by_bucket[bucket] = cohort
        self._cohorts_by_id[cohort.id] = cohort
        if side == "long":
            self._long_cohorts.append(cohort)
        else:
            self._short_cohorts.append(cohort)

    def _remove_cohort_mass(
        self,
        cohorts: list[_PositionCohort],
        execution_price: float,
        volume: float,
        cohort_id: int | None = None,
    ) -> float:
        remaining = volume
        removed_total = 0.0
        if cohort_id is not None:
            cohort = self._cohorts_by_id.get(cohort_id)
            if cohort is not None and cohort.mass > 0:
                removed = min(cohort.mass, remaining)
                cohort.mass -= removed
                return removed
        ordered = sorted(
            cohorts,
            key=lambda cohort: (
                -abs(cohort.entry_price - execution_price) / self.gap,
                -cohort.age,
            ),
        )
        for cohort in ordered:
            if remaining <= 1e-12:
                break
            removed = min(cohort.mass, remaining)
            cohort.mass -= removed
            remaining -= removed
            removed_total += removed
        return removed_total

    def _available_fill_volume(
        self, kind: str, requested: float, cohort_id: int | None = None
    ) -> float:
        if requested <= 0:
            return 0.0
        if cohort_id is not None and kind in {"long_exit", "short_exit"}:
            cohort = self._cohorts_by_id.get(cohort_id)
            return min(requested, cohort.mass) if cohort is not None else 0.0
        if kind == "long_exit":
            return min(requested, self._long_mass_total)
        if kind == "short_exit":
            return min(requested, self._short_mass_total)
        return requested

    def _is_exit_kind(self, kind: str) -> bool:
        return kind in {"long_exit", "short_exit"}

    def _position_mass_from_cohorts(
        self, long_exit_mdf_by_price: PriceMap, short_exit_mdf_by_price: PriceMap
    ) -> PositionMassState:
        return PositionMassState(
            long_exit_mass_by_price=self._drop_zeroes(
                {
                    price: self._long_mass_total * probability
                    for price, probability in long_exit_mdf_by_price.items()
                }
            ),
            short_exit_mass_by_price=self._drop_zeroes(
                {
                    price: self._short_mass_total * probability
                    for price, probability in short_exit_mdf_by_price.items()
                }
            ),
        )

    def _clear_stale_exit_lots(self) -> None:
        for lots_by_price in (self._orderbook.bid_lots, self._orderbook.ask_lots):
            for price in list(lots_by_price):
                lots_by_price[price] = [
                    lot for lot in lots_by_price[price] if not self._is_exit_kind(lot.kind)
                ]
                if not lots_by_price[price]:
                    del lots_by_price[price]
        self._orderbook.rebuild_totals()

    def _trim_orderbook_through_last_price(self, last_price: float) -> None:
        for price in list(self._orderbook.bid_lots):
            if price >= last_price:
                del self._orderbook.bid_lots[price]
        for price in list(self._orderbook.ask_lots):
            if price <= last_price:
                del self._orderbook.ask_lots[price]
        self._orderbook.rebuild_totals()

    def _prune_orderbook_window(self, current_price: float) -> None:
        min_price = max(
            self._min_price,
            self._snap_price(current_price - self.grid_radius * self.gap),
        )
        max_price = self._snap_price(current_price + self.grid_radius * self.gap)
        for lots_by_price in (self._orderbook.bid_lots, self._orderbook.ask_lots):
            for price in list(lots_by_price):
                if price < min_price or price > max_price:
                    del lots_by_price[price]
        self._orderbook.rebuild_totals()

    def _clean_cohorts(self) -> None:
        self._prune_cohort_buckets(self._long_cohorts_by_bucket)
        self._prune_cohort_buckets(self._short_cohorts_by_bucket)
        self._long_cohorts = sorted(
            self._long_cohorts_by_bucket.values(), key=lambda cohort: cohort.entry_price
        )
        self._short_cohorts = sorted(
            self._short_cohorts_by_bucket.values(), key=lambda cohort: cohort.entry_price
        )
        self._cohorts_by_id = {
            cohort.id: cohort for cohort in (*self._long_cohorts, *self._short_cohorts)
        }
        self._long_mass_total = sum(cohort.mass for cohort in self._long_cohorts)
        self._short_mass_total = sum(cohort.mass for cohort in self._short_cohorts)

    def _prune_cohort_buckets(self, cohorts_by_bucket: dict[int, _PositionCohort]) -> None:
        for bucket, cohort in list(cohorts_by_bucket.items()):
            if cohort.mass <= 1e-8:
                del cohorts_by_bucket[bucket]

    def _cohort_bucket_tick(self, entry_price: float) -> int:
        tick = self.price_to_tick(entry_price)
        bucket_width = max(1, self.grid_radius // 4)
        return int(round(tick / bucket_width)) * bucket_width

    def _best_bid(self) -> float | None:
        return self._orderbook.best_bid()

    def _best_ask(self) -> float | None:
        return self._orderbook.best_ask()

    def _spread(self, bid: float | None, ask: float | None) -> float | None:
        if bid is None or ask is None:
            return None
        return ask - bid

    def _near_touch_imbalance(self, current_price: float) -> float:
        return self._orderbook.near_touch_imbalance(current_price, self.gap)

    def _discard_empty_head(self, price: float, side: str) -> None:
        self._orderbook.discard_empty_head(price, side)

    def _clean_orderbook(self) -> None:
        self._orderbook.clean()

    def _snapshot_orderbook(self) -> OrderBookState:
        return self._orderbook.snapshot()

    def _snapshot_position_mass(self) -> PositionMassState:
        return PositionMassState(
            long_exit_mass_by_price=dict(self.state.position_mass.long_exit_mass_by_price),
            short_exit_mass_by_price=dict(self.state.position_mass.short_exit_mass_by_price),
        )

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
