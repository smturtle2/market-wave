from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass
from math import ceil, exp, isfinite, log
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


@dataclass
class _PositionCohort:
    side: str
    entry_price: float
    mass: float
    age: int = 0


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


@dataclass
class _OrderBook:
    bid_lots: dict[float, list[_OrderLot]]
    ask_lots: dict[float, list[_OrderLot]]

    def age(self) -> None:
        for lots_by_price in (self.bid_lots, self.ask_lots):
            for lots in lots_by_price.values():
                for lot in lots:
                    lot.age += 1

    def add_lots(self, volume_by_price: PriceMap, side: str, kind: str) -> None:
        lots_by_price = self.lots_for_side(side)
        for price, volume in volume_by_price.items():
            if volume <= 0:
                continue
            lots_by_price.setdefault(price, []).append(_OrderLot(volume=volume, kind=kind))

    def lots_for_side(self, side: str) -> dict[float, list[_OrderLot]]:
        return self.bid_lots if side == "bid" else self.ask_lots

    def lots_for_taker_side(self, side: str) -> dict[float, list[_OrderLot]]:
        return self.ask_lots if side == "buy" else self.bid_lots

    def best_bid(self) -> float | None:
        prices = [
            price
            for price, lots in self.bid_lots.items()
            if sum(lot.volume for lot in lots) > 1e-12
        ]
        return max(prices) if prices else None

    def best_ask(self) -> float | None:
        prices = [
            price
            for price, lots in self.ask_lots.items()
            if sum(lot.volume for lot in lots) > 1e-12
        ]
        return min(prices) if prices else None

    def discard_empty_head(self, price: float, side: str) -> None:
        lots_by_price = self.lots_for_side(side)
        lots = lots_by_price.get(price, [])
        while lots and lots[0].volume <= 1e-12:
            lots.pop(0)
        if not lots and price in lots_by_price:
            del lots_by_price[price]

    def clean(self) -> None:
        for lots_by_price in (self.bid_lots, self.ask_lots):
            for price in list(lots_by_price):
                lots_by_price[price] = [lot for lot in lots_by_price[price] if lot.volume > 1e-12]
                if not lots_by_price[price]:
                    del lots_by_price[price]

    def snapshot(self) -> OrderBookState:
        return OrderBookState(
            bid_volume_by_price=self._aggregate_lots(self.bid_lots),
            ask_volume_by_price=self._aggregate_lots(self.ask_lots),
        )

    def near_touch_imbalance(self, current_price: float, gap: float) -> float:
        bid_depth = 0.0
        ask_depth = 0.0
        for price, lots in self.bid_lots.items():
            distance = max(1.0, abs(current_price - price) / gap)
            if distance <= 5:
                bid_depth += sum(lot.volume for lot in lots) / distance
        for price, lots in self.ask_lots.items():
            distance = max(1.0, abs(price - current_price) / gap)
            if distance <= 5:
                ask_depth += sum(lot.volume for lot in lots) / distance
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
            },
            "high_vol": {
                "mood": 0.0,
                "trend": 0.0,
                "volatility": 1.35,
                "volatility_bias": 0.06,
                "intensity": 1.35,
                "taker": 1.24,
                "cancel": 1.25,
                "liquidity": 0.9,
                "spread": 1.35,
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
            },
            "squeeze": {
                "mood": 0.06,
                "trend": 0.12,
                "volatility": 1.45,
                "volatility_bias": 0.08,
                "intensity": 1.55,
                "taker": 1.34,
                "cancel": 1.18,
                "liquidity": 0.7,
                "spread": 1.4,
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
        self._last_return_ticks = 0.0
        self._last_abs_return_ticks = 0.0
        self._last_imbalance = 0.0
        self._last_execution_volume = 0.0
        self._last_executed_by_price: PriceMap = {}
        self._mdf_model_accepts_signals = self._scores_accepts_signals(self.mdf_model.scores)
        self._mdf_memory: dict[str, dict[int, float]] = {}
        self._anchor_price: float

        price = self._snap_price(float(initial_price))
        self._anchor_price = price
        grid = self._price_grid(price)
        tick_grid = self.relative_tick_grid()
        current_tick = self.price_to_tick(price)
        intensity = IntensityState(total=0.0, buy=0.0, sell=0.0, buy_ratio=0.5, sell_ratio=0.5)
        latent = LatentState(mood=0.0, trend=0.0, volatility=0.25)
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

        price_grid = self._price_grid(price_before)
        orderbook_before = self._snapshot_orderbook()
        position_before = self._snapshot_position_mass()
        best_bid_before = self._best_bid()
        best_ask_before = self._best_ask()
        spread_before = self._spread(best_bid_before, best_ask_before)
        pre_imbalance = self._near_touch_imbalance(price_before)

        self._active_regime = self._next_regime()
        latent = self._next_latent(state.latent)
        intensity = self._next_intensity(latent)
        mdf = self._next_mdf(
            price_before,
            price_grid,
            latent,
            step_index=step_index,
            update_memory=True,
        )

        cancelled_volume = self._cancel_orders(price_before, latent)
        entry_limit_buy, entry_limit_sell = self._entry_limit_volumes(
            intensity,
            mdf,
            latent,
            price_before,
        )
        initial_market_buy, initial_market_sell = self._market_entry_volumes(
            intensity,
            latent,
            pre_imbalance,
        )
        market_buy = initial_market_buy
        market_sell = initial_market_sell
        long_exit_market, long_exit_limit = self._exit_flow(
            "long",
            price_before,
            latent,
            mdf.long_exit_mdf_by_price,
        )
        short_exit_market, short_exit_limit = self._exit_flow(
            "short",
            price_before,
            latent,
            mdf.short_exit_mdf_by_price,
        )

        self._add_liquidity_replenishment(price_before, latent, pre_imbalance)
        self._add_lots(entry_limit_buy, side="bid", kind="buy_entry")
        self._add_lots(entry_limit_sell, side="ask", kind="sell_entry")
        self._add_lots(short_exit_limit, side="bid", kind="short_exit")
        self._add_lots(long_exit_limit, side="ask", kind="long_exit")

        initial_short_exit_market = short_exit_market
        initial_long_exit_market = long_exit_market
        stats = _TradeStats(executed_by_price={})
        execution = self._execute_market_flows(
            price_before=price_before,
            market_buy=market_buy,
            short_exit_market=short_exit_market,
            market_sell=market_sell,
            long_exit_market=long_exit_market,
            mdf=mdf,
            stats=stats,
        )
        self._clean_orderbook()
        self._clean_cohorts()

        price_after = self._next_price_after_trading(price_before, stats)
        price_after = self._snap_price(price_after)
        self._last_return_ticks = (price_after - price_before) / self.gap
        self._last_abs_return_ticks = abs(self._last_return_ticks)
        self._last_execution_volume = stats.total_volume
        self._last_executed_by_price = self._drop_zeroes(stats.executed_by_price)

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
        total_market_buy = initial_market_buy + initial_short_exit_market
        total_market_sell = initial_market_sell + initial_long_exit_market
        residual_market_buy = execution.residual_market_buy + execution.residual_short_exit
        residual_market_sell = execution.residual_market_sell + execution.residual_long_exit
        market_flow_delta = total_market_buy - total_market_sell
        market_flow_total = total_market_buy + total_market_sell
        order_flow_imbalance = (
            self._clamp(market_flow_delta / market_flow_total, -1.0, 1.0)
            if market_flow_total > 1e-12
            else self._near_touch_imbalance(price_after)
        )
        self._last_imbalance = order_flow_imbalance

        entry_volume = self._merge_maps(entry_limit_buy, entry_limit_sell)
        exit_volume = self._merge_maps(long_exit_limit, short_exit_limit)
        buy_volume = self._merge_maps(
            entry_limit_buy,
            short_exit_limit,
            self._single_price_volume(price_before, execution.crossed_market_volume),
            self._single_price_volume(best_ask_before or price_before, residual_market_buy),
        )
        sell_volume = self._merge_maps(
            entry_limit_sell,
            long_exit_limit,
            self._single_price_volume(price_before, execution.crossed_market_volume),
            self._single_price_volume(best_bid_before or price_before, residual_market_sell),
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

    def _next_price_after_trading(self, price_before: float, stats: _TradeStats) -> float:
        if stats.total_volume <= 0 or stats.last_price is None:
            return price_before
        vwap = stats.notional / stats.total_volume
        execution_price = 0.68 * stats.last_price + 0.32 * vwap
        execution_move_ticks = (execution_price - price_before) / self.gap
        if abs(execution_move_ticks) <= 1e-12:
            return price_before
        jitter_ticks = self._clamp(self._rng.gauss(0.0, 0.18), -0.35, 0.35)
        anchor_pull_ticks = self._clamp(
            0.025 * (self._anchor_price - price_before) / self.gap,
            -0.30,
            0.30,
        )
        proposed_ticks = execution_move_ticks + jitter_ticks + anchor_pull_ticks
        if execution_move_ticks > 0:
            proposed_ticks = max(0.0, proposed_ticks)
        else:
            proposed_ticks = min(0.0, proposed_ticks)
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

    def _price_from_relative_tick(self, current_tick: int, relative_tick: int) -> float:
        return self.tick_to_price(current_tick + relative_tick)

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
        mood_noise = self._rng.gauss(0.0, 0.13)
        trend_noise = self._rng.gauss(0.0, 0.08)
        jump_probability = 0.018 * regime["volatility"] * (1.0 + self.augmentation_strength)
        jump = self._rng.gauss(0.0, 0.55) if self._rng.random() < jump_probability else 0.0
        signed_flow = 0.10 * self._last_imbalance + 0.035 * self._last_return_ticks
        anchor_pressure = self._clamp(
            (self._anchor_price - self.state.price) / (self.grid_radius * self.gap),
            -1.0,
            1.0,
        )

        mood = self._clamp(
            0.55 * latent.mood
            + 0.08 * latent.trend
            + signed_flow
            + regime["mood"]
            + 0.12 * anchor_pressure
            + mood_noise
            + 0.25 * jump,
            -1.0,
            1.0,
        )
        trend = self._clamp(
            0.58 * latent.trend
            + 0.08 * mood
            + 0.04 * self._last_return_ticks
            + regime["trend"]
            + 0.10 * anchor_pressure
            + trend_noise,
            -1.0,
            1.0,
        )
        shock = abs(mood_noise) + abs(jump) * 0.75
        realized = 0.12 * self._last_abs_return_ticks + 0.025 * self._last_execution_volume
        volatility = self._clamp(
            0.78 * latent.volatility
            + shock
            + realized
            + 0.03 * abs(self._last_imbalance)
            + regime["volatility_bias"],
            0.04,
            2.2,
        )
        return LatentState(mood=mood, trend=trend, volatility=volatility)

    def _next_intensity(self, latent: LatentState) -> IntensityState:
        regime = self._regime_settings(self._active_regime)
        augmentation = 1.0 + self.augmentation_strength * self._rng.uniform(-0.18, 0.28)
        total = (
            self.popularity * (1.0 + 2.2 * latent.volatility) * regime["intensity"] * augmentation
        )
        buy_ratio = self._clamp(
            0.5 + 0.18 * latent.mood + 0.12 * latent.trend + 0.06 * self._last_imbalance,
            0.12,
            0.88,
        )
        sell_ratio = 1.0 - buy_ratio
        return IntensityState(
            total=total,
            buy=total * buy_ratio,
            sell=total * sell_ratio,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
        )

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

    def _entry_limit_volumes(
        self,
        intensity: IntensityState,
        mdf: MDFState,
        latent: LatentState,
        price: float,
    ) -> tuple[PriceMap, PriceMap]:
        passive_share = 1.0 - self._taker_share(latent, self._last_imbalance)
        bid_ceiling = price - self.gap
        ask_floor = price + self.gap
        buy = {
            price: intensity.buy * passive_share * probability
            for price, probability in mdf.buy_entry_mdf_by_price.items()
            if price <= bid_ceiling
        }
        sell = {
            price: intensity.sell * passive_share * probability
            for price, probability in mdf.sell_entry_mdf_by_price.items()
            if price >= ask_floor
        }
        return self._drop_zeroes(buy), self._drop_zeroes(sell)

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
        anchor = max(logits, default=0.0)
        weights = [exp(self._clamp(logit - anchor, -50.0, 0.0)) for logit in logits]
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
            price = self._price_from_relative_tick(current_tick, relative_tick)
            projected[price] = projected.get(price, 0.0) + probability
        total = sum(projected.values())
        if total <= 0:
            return {}
        return {price: probability / total for price, probability in sorted(projected.items())}

    def _normalize_tick_map(self, values: dict[int, float]) -> dict[int, float]:
        total = sum(max(0.0, value) for value in values.values())
        if total <= 0:
            uniform = 1.0 / len(values)
            return {tick: uniform for tick in values}
        return {tick: max(0.0, value) / total for tick, value in values.items()}

    def _market_entry_volumes(
        self,
        intensity: IntensityState,
        latent: LatentState,
        imbalance: float,
    ) -> tuple[float, float]:
        taker_share = self._taker_share(latent, imbalance)
        burst = 1.0 + (self._rng.random() < 0.06) * self._rng.uniform(0.2, 0.9)
        buy_noise = self._clamp(self._rng.lognormvariate(0.0, 0.16), 0.65, 1.45)
        sell_noise = self._clamp(self._rng.lognormvariate(0.0, 0.16), 0.65, 1.45)
        buy_bias = self._clamp(
            1.0 + 0.24 * max(latent.trend, 0.0) + 0.16 * max(imbalance, 0.0), 0.65, 1.45
        )
        sell_bias = self._clamp(
            1.0 + 0.24 * max(-latent.trend, 0.0) + 0.16 * max(-imbalance, 0.0), 0.65, 1.45
        )
        return (
            intensity.buy * taker_share * burst * buy_bias * buy_noise,
            intensity.sell * taker_share * burst * sell_bias * sell_noise,
        )

    def _taker_share(self, latent: LatentState, imbalance: float) -> float:
        regime = self._regime_settings(self._active_regime)
        base = 0.10 + 0.06 * latent.volatility + 0.05 * abs(imbalance)
        return self._clamp(base * regime["taker"], 0.08, 0.55)

    def _exit_flow(
        self,
        side: str,
        current_price: float,
        latent: LatentState,
        exit_mdf: PriceMap,
    ) -> tuple[float, PriceMap]:
        cohorts = self._long_cohorts if side == "long" else self._short_cohorts
        market_volume = 0.0
        passive_volume = 0.0
        for cohort in cohorts:
            if cohort.mass <= 0:
                continue
            age_pressure = min(0.12, cohort.age * 0.004)
            if side == "long":
                pnl_ticks = (current_price - cohort.entry_price) / self.gap
                stop_pressure = max(0.0, -pnl_ticks - 2.0) * 0.08
                take_profit = max(0.0, pnl_ticks - 3.0) * 0.04
                trend_pressure = max(0.0, -latent.trend) * 0.05
            else:
                pnl_ticks = (cohort.entry_price - current_price) / self.gap
                stop_pressure = max(0.0, -pnl_ticks - 2.0) * 0.08
                take_profit = max(0.0, pnl_ticks - 3.0) * 0.04
                trend_pressure = max(0.0, latent.trend) * 0.05
            fraction = self._clamp(
                0.015 + age_pressure + stop_pressure + take_profit + trend_pressure, 0.0, 0.35
            )
            desired = cohort.mass * fraction
            market_volume += desired * (0.55 + 0.20 * latent.volatility / (1.0 + latent.volatility))
            passive_volume += desired * 0.35

        if side == "long":
            limit_map = {
                price: passive_volume * probability
                for price, probability in exit_mdf.items()
                if passive_volume > 0 and price >= current_price + self.gap
            }
        else:
            limit_map = {
                price: passive_volume * probability
                for price, probability in exit_mdf.items()
                if passive_volume > 0 and price <= current_price - self.gap
            }
        return market_volume, self._drop_zeroes(limit_map)

    def _cancel_orders(self, current_price: float, latent: LatentState) -> PriceMap:
        regime = self._regime_settings(self._active_regime)
        cancelled: PriceMap = {}
        for lots_by_price in (self._orderbook.bid_lots, self._orderbook.ask_lots):
            for price, lots in list(lots_by_price.items()):
                survivors = []
                distance = abs(price - current_price) / self.gap
                for lot in lots:
                    probability = self._clamp(
                        (0.025 + 0.018 * lot.age + 0.012 * distance + 0.035 * latent.volatility)
                        * regime["cancel"],
                        0.02,
                        0.70,
                    )
                    cancel_fraction = probability * self._rng.random()
                    removed = lot.volume * cancel_fraction
                    lot.volume -= removed
                    if removed > 1e-12:
                        cancelled[price] = cancelled.get(price, 0.0) + removed
                    if lot.volume > 1e-12:
                        survivors.append(lot)
                if survivors:
                    lots_by_price[price] = survivors
                else:
                    del lots_by_price[price]
        return self._drop_zeroes(cancelled)

    def _add_liquidity_replenishment(
        self,
        current_price: float,
        latent: LatentState,
        imbalance: float,
    ) -> None:
        regime = self._regime_settings(self._active_regime)
        base = self.popularity * (0.18 + 0.20 / (1.0 + latent.volatility)) * regime["liquidity"]
        for level in range(1, min(7, self.grid_radius + 1)):
            shape = base / (level**1.35)
            bid_volume = shape * self._clamp(1.0 - 0.25 * imbalance, 0.55, 1.45)
            ask_volume = shape * self._clamp(1.0 + 0.25 * imbalance, 0.55, 1.45)
            self._add_lots(
                {self._snap_price(current_price - level * self.gap): bid_volume}, "bid", "buy_entry"
            )
            self._add_lots(
                {self._snap_price(current_price + level * self.gap): ask_volume},
                "ask",
                "sell_entry",
            )

    def _add_lots(self, volume_by_price: PriceMap, side: str, kind: str) -> None:
        self._orderbook.add_lots(volume_by_price, side, kind)

    def _execute_market_flows(
        self,
        *,
        price_before: float,
        market_buy: float,
        short_exit_market: float,
        market_sell: float,
        long_exit_market: float,
        mdf: MDFState,
        stats: _TradeStats,
    ) -> _ExecutionResult:
        market_buy, short_exit_market, market_sell, long_exit_market, crossed_market_volume = (
            self._cross_taker_flow(
                price_before,
                market_buy,
                short_exit_market,
                market_sell,
                long_exit_market,
                mdf,
                stats,
            )
        )
        taker_calls = [
            ("buy", market_buy, "buy_entry"),
            ("sell", market_sell, "sell_entry"),
            ("buy", short_exit_market, "short_exit"),
            ("sell", long_exit_market, "long_exit"),
        ]
        self._rng.shuffle(taker_calls)
        for side, volume, kind in taker_calls:
            self._consume_taker(
                side=side,
                volume=volume,
                kind=kind,
                fallback_price=price_before,
                mdf=mdf,
                stats=stats,
            )
        self._match_crossed_book(mdf, stats)
        return _ExecutionResult(
            residual_market_buy=market_buy,
            residual_short_exit=short_exit_market,
            residual_market_sell=market_sell,
            residual_long_exit=long_exit_market,
            crossed_market_volume=crossed_market_volume,
        )

    def _consume_taker(
        self,
        side: str,
        volume: float,
        kind: str,
        fallback_price: float,
        mdf: MDFState,
        stats: _TradeStats,
    ) -> None:
        if volume <= 0:
            return
        remaining = volume
        max_impact_ticks = max(2, min(6, self.grid_radius // 3))
        while remaining > 1e-12:
            price = self._best_ask() if side == "buy" else self._best_bid()
            if price is not None and abs(price - fallback_price) / self.gap > max_impact_ticks:
                price = None
            if price is None:
                impact_ticks = min(
                    max_impact_ticks,
                    max(1, ceil(remaining / max(self.popularity * 3.0, 0.3))),
                )
                if side == "buy":
                    price = fallback_price + impact_ticks * self.gap
                else:
                    price = max(self._min_price, fallback_price - impact_ticks * self.gap)
                price = self._snap_price(price)
                fill_volume = remaining
                remaining = 0.0
                actual = self._apply_fill(kind, fill_volume, price, mdf)
                if kind == "buy_entry":
                    self._apply_fill("sell_entry", actual, price, mdf)
                elif kind == "sell_entry":
                    self._apply_fill("buy_entry", actual, price, mdf)
                stats.record(price, actual)
                break

            lots_by_price = self._orderbook.lots_for_taker_side(side)
            resting_lot = lots_by_price[price][0]
            fill_volume = min(remaining, resting_lot.volume)
            taker_actual = self._available_fill_volume(kind, fill_volume)
            resting_actual = self._available_fill_volume(resting_lot.kind, fill_volume)
            actual = min(taker_actual, resting_actual)
            if actual <= 1e-12:
                if self._is_exit_kind(resting_lot.kind) and resting_actual <= 1e-12:
                    resting_lot.volume = 0.0
                    self._discard_empty_head(price, "ask" if side == "buy" else "bid")
                    continue
                break
            resting_lot.volume -= actual
            remaining -= actual
            self._apply_fill(kind, actual, price, mdf)
            self._apply_fill(resting_lot.kind, actual, price, mdf)
            stats.record(price, actual)
            self._discard_empty_head(price, "ask" if side == "buy" else "bid")

    def _cross_taker_flow(
        self,
        price: float,
        buy_entry: float,
        short_exit: float,
        sell_entry: float,
        long_exit: float,
        mdf: MDFState,
        stats: _TradeStats,
    ) -> tuple[float, float, float, float, float]:
        buy_total = buy_entry + short_exit
        sell_total = sell_entry + long_exit
        cross_volume = min(buy_total, sell_total) * 0.85
        if cross_volume <= 1e-12:
            return buy_entry, short_exit, sell_entry, long_exit, 0.0

        buy_entry_fill = cross_volume * buy_entry / buy_total if buy_total > 0 else 0.0
        short_exit_fill = cross_volume * short_exit / buy_total if buy_total > 0 else 0.0
        sell_entry_fill = cross_volume * sell_entry / sell_total if sell_total > 0 else 0.0
        long_exit_fill = cross_volume * long_exit / sell_total if sell_total > 0 else 0.0

        actual_buy_entry = self._available_fill_volume("buy_entry", buy_entry_fill)
        actual_short_exit = self._available_fill_volume("short_exit", short_exit_fill)
        actual_sell_entry = self._available_fill_volume("sell_entry", sell_entry_fill)
        actual_long_exit = self._available_fill_volume("long_exit", long_exit_fill)
        actual_buy_side = actual_buy_entry + actual_short_exit
        actual_sell_side = actual_sell_entry + actual_long_exit
        actual_cross = min(actual_buy_side, actual_sell_side)
        if actual_cross <= 1e-12:
            return buy_entry, short_exit, sell_entry, long_exit, 0.0

        buy_scale = actual_cross / actual_buy_side if actual_buy_side > 0 else 0.0
        sell_scale = actual_cross / actual_sell_side if actual_sell_side > 0 else 0.0
        buy_entry_actual = actual_buy_entry * buy_scale
        short_exit_actual = actual_short_exit * buy_scale
        sell_entry_actual = actual_sell_entry * sell_scale
        long_exit_actual = actual_long_exit * sell_scale

        self._apply_fill("buy_entry", buy_entry_actual, price, mdf)
        self._apply_fill("short_exit", short_exit_actual, price, mdf)
        self._apply_fill("sell_entry", sell_entry_actual, price, mdf)
        self._apply_fill("long_exit", long_exit_actual, price, mdf)
        stats.record(price, actual_cross)

        return (
            max(0.0, buy_entry - buy_entry_actual),
            max(0.0, short_exit - short_exit_actual),
            max(0.0, sell_entry - sell_entry_actual),
            max(0.0, long_exit - long_exit_actual),
            actual_cross,
        )

    def _match_crossed_book(self, mdf: MDFState, stats: _TradeStats) -> None:
        while self._best_bid() is not None and self._best_ask() is not None:
            bid_price = self._best_bid()
            ask_price = self._best_ask()
            if bid_price is None or ask_price is None or bid_price < ask_price:
                break

            bid_lot = self._orderbook.bid_lots[bid_price][0]
            ask_lot = self._orderbook.ask_lots[ask_price][0]
            volume = min(bid_lot.volume, ask_lot.volume)
            if volume <= 1e-12:
                self._discard_empty_head(bid_price, "bid")
                self._discard_empty_head(ask_price, "ask")
                continue
            execution_price = ask_price
            bid_actual = self._available_fill_volume(bid_lot.kind, volume)
            ask_actual = self._available_fill_volume(ask_lot.kind, volume)
            actual = min(bid_actual, ask_actual)
            if actual <= 1e-12:
                if self._is_exit_kind(bid_lot.kind) and bid_actual <= 1e-12:
                    bid_lot.volume = 0.0
                if self._is_exit_kind(ask_lot.kind) and ask_actual <= 1e-12:
                    ask_lot.volume = 0.0
                self._discard_empty_head(bid_price, "bid")
                self._discard_empty_head(ask_price, "ask")
                continue
            bid_lot.volume -= actual
            ask_lot.volume -= actual
            self._apply_fill(bid_lot.kind, actual, execution_price, mdf)
            self._apply_fill(ask_lot.kind, actual, execution_price, mdf)
            stats.record(execution_price, actual)
            self._discard_empty_head(bid_price, "bid")
            self._discard_empty_head(ask_price, "ask")

    def _apply_fill(
        self,
        kind: str,
        volume: float,
        execution_price: float,
        mdf: MDFState,
    ) -> float:
        if volume <= 0:
            return 0.0
        if kind == "buy_entry":
            self._long_cohorts.append(_PositionCohort("long", execution_price, volume))
            return volume
        elif kind == "sell_entry":
            self._short_cohorts.append(_PositionCohort("short", execution_price, volume))
            return volume
        elif kind == "long_exit":
            return self._remove_cohort_mass(self._long_cohorts, execution_price, volume)
        elif kind == "short_exit":
            return self._remove_cohort_mass(self._short_cohorts, execution_price, volume)
        return 0.0

    def _remove_cohort_mass(
        self,
        cohorts: list[_PositionCohort],
        execution_price: float,
        volume: float,
    ) -> float:
        remaining = volume
        removed_total = 0.0
        ordered = sorted(
            cohorts,
            key=lambda cohort: (abs(cohort.entry_price - execution_price), -cohort.age),
        )
        for cohort in ordered:
            if remaining <= 1e-12:
                break
            removed = min(cohort.mass, remaining)
            cohort.mass -= removed
            remaining -= removed
            removed_total += removed
        return removed_total

    def _available_fill_volume(self, kind: str, requested: float) -> float:
        if requested <= 0:
            return 0.0
        if kind == "long_exit":
            return min(requested, sum(cohort.mass for cohort in self._long_cohorts))
        if kind == "short_exit":
            return min(requested, sum(cohort.mass for cohort in self._short_cohorts))
        return requested

    def _is_exit_kind(self, kind: str) -> bool:
        return kind in {"long_exit", "short_exit"}

    def _position_mass_from_cohorts(
        self, long_exit_mdf_by_price: PriceMap, short_exit_mdf_by_price: PriceMap
    ) -> PositionMassState:
        long_total = sum(cohort.mass for cohort in self._long_cohorts)
        short_total = sum(cohort.mass for cohort in self._short_cohorts)
        return PositionMassState(
            long_exit_mass_by_price=self._drop_zeroes(
                {
                    price: long_total * probability
                    for price, probability in long_exit_mdf_by_price.items()
                }
            ),
            short_exit_mass_by_price=self._drop_zeroes(
                {
                    price: short_total * probability
                    for price, probability in short_exit_mdf_by_price.items()
                }
            ),
        )

    def _clean_cohorts(self) -> None:
        self._long_cohorts = [cohort for cohort in self._long_cohorts if cohort.mass > 1e-12]
        self._short_cohorts = [cohort for cohort in self._short_cohorts if cohort.mass > 1e-12]

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

    def _price_map_to_relative_ticks(self, reference_price: float, values: PriceMap) -> TickMap:
        reference_tick = self.price_to_tick(reference_price)
        by_tick: TickMap = {}
        min_tick = -self.grid_radius
        max_tick = self.grid_radius
        for price, value in values.items():
            relative_tick = self.price_to_tick(price) - reference_tick
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
