from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from random import Random

from .distribution import DiscreteMixtureDistribution, MixtureComponent
from .state import (
    DistributionState,
    IntensityState,
    LatentState,
    MarketState,
    OrderBookState,
    PositionMassState,
    PriceMap,
    StepInfo,
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


class Market:
    def __init__(
        self,
        initial_price: float,
        gap: float,
        popularity: float = 1.0,
        seed: int | None = None,
        grid_radius: int = 20,
    ) -> None:
        if gap <= 0:
            raise ValueError("gap must be positive")
        if grid_radius < 1:
            raise ValueError("grid_radius must be at least 1")
        if popularity < 0:
            raise ValueError("popularity must be non-negative")

        self.gap = float(gap)
        self.popularity = float(popularity)
        self.grid_radius = int(grid_radius)
        self._rng = Random(seed)
        self._seed = seed
        self.history: list[StepInfo] = []
        self._bid_lots: dict[float, list[_OrderLot]] = {}
        self._ask_lots: dict[float, list[_OrderLot]] = {}
        self._long_cohorts: list[_PositionCohort] = []
        self._short_cohorts: list[_PositionCohort] = []
        self._last_return_ticks = 0.0
        self._last_abs_return_ticks = 0.0
        self._last_imbalance = 0.0
        self._last_execution_volume = 0.0
        self._anchor_price: float

        price = self._snap_price(float(initial_price))
        self._anchor_price = price
        grid = self._price_grid(price)
        intensity = IntensityState(total=0.0, buy=0.0, sell=0.0, buy_ratio=0.5, sell_ratio=0.5)
        latent = LatentState(mood=0.0, trend=0.0, volatility=0.25)
        uniform = 1.0 / len(grid)
        initial_pmf = {price_level: uniform for price_level in grid}
        zero_mass = {price_level: 0.0 for price_level in grid}
        distributions = DistributionState(
            initial_pmf,
            initial_pmf.copy(),
            initial_pmf.copy(),
            initial_pmf.copy(),
        )
        self.state = MarketState(
            price=price,
            step_index=0,
            intensity=intensity,
            latent=latent,
            price_grid=grid,
            distributions=distributions,
            orderbook=OrderBookState(),
            position_mass=PositionMassState(
                long_exit_mass_by_price=zero_mass,
                short_exit_mass_by_price=zero_mass.copy(),
            ),
        )

    @property
    def seed(self) -> int | None:
        return self._seed

    def step(self, n: int) -> list[StepInfo]:
        if n < 0:
            raise ValueError("n must be non-negative")

        steps = [self._step_once() for _ in range(n)]
        self.history.extend(steps)
        return steps

    def plot_history(
        self,
        *,
        ax=None,
        style: str = "market_wave",
        last: int | None = None,
        layout: str = "panel",
    ):
        if not self.history:
            raise ValueError("history is empty; call step(n) before plotting")
        if last is not None and last <= 0:
            raise ValueError("last must be positive")
        if layout not in {"panel", "overlay"}:
            raise ValueError("layout must be 'panel' or 'overlay'")

        import matplotlib.pyplot as plt

        steps = self.history[-last:] if last is not None else self.history
        context = self._plot_style_context(style)
        with plt.style.context(context):
            if layout == "panel" and ax is None:
                return self._plot_history_panel(plt, steps, style)
            return self._plot_history_overlay(plt, steps, ax, style)

    def plot(
        self,
        *,
        ax=None,
        style: str = "market_wave",
        last: int | None = None,
        layout: str = "panel",
    ):
        return self.plot_history(ax=ax, style=style, last=last, layout=layout)

    def _plot_style_context(self, style: str) -> str:
        if style == "market_wave":
            return "default"
        if style == "market_wave_dark":
            return "dark_background"
        return style

    def _plot_history_panel(self, plt, steps: list[StepInfo], style: str):
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
                "buy_imbalance": "#34d399",
                "sell_imbalance": "#fb7185",
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
            "buy_imbalance": "#16a34a",
            "sell_imbalance": "#dc2626",
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
        price_before = state.price
        self._age_state()

        price_grid = self._price_grid(price_before)
        orderbook_before = self._snapshot_orderbook()
        position_before = self._snapshot_position_mass()
        best_bid_before = self._best_bid()
        best_ask_before = self._best_ask()
        spread_before = self._spread(best_bid_before, best_ask_before)
        pre_imbalance = self._near_touch_imbalance(price_before)

        latent = self._next_latent(state.latent)
        intensity = self._next_intensity(latent)
        distributions = self._next_distributions(price_before, price_grid, latent)

        cancelled_volume = self._cancel_orders(price_before, latent)
        entry_limit_buy, entry_limit_sell = self._entry_limit_volumes(intensity, distributions)
        market_buy, market_sell = self._market_entry_volumes(intensity, latent, pre_imbalance)
        long_exit_market, long_exit_limit = self._exit_flow(
            "long",
            price_before,
            latent,
            distributions.long_exit_pmf,
        )
        short_exit_market, short_exit_limit = self._exit_flow(
            "short",
            price_before,
            latent,
            distributions.short_exit_pmf,
        )

        self._add_liquidity_replenishment(price_before, latent, pre_imbalance)
        self._add_lots(entry_limit_buy, side="bid", kind="buy_entry")
        self._add_lots(entry_limit_sell, side="ask", kind="sell_entry")
        self._add_lots(short_exit_limit, side="bid", kind="short_exit")
        self._add_lots(long_exit_limit, side="ask", kind="long_exit")

        stats = _TradeStats(executed_by_price={})
        market_buy, short_exit_market, market_sell, long_exit_market = self._cross_taker_flow(
            price_before,
            market_buy,
            short_exit_market,
            market_sell,
            long_exit_market,
            distributions,
            stats,
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
                distributions=distributions,
                stats=stats,
            )
        self._match_crossed_book(distributions, stats)
        self._clean_orderbook()
        self._clean_cohorts()

        price_after = self._next_price_after_trading(price_before, stats)
        price_after = self._snap_price(price_after)
        self._last_return_ticks = (price_after - price_before) / self.gap
        self._last_abs_return_ticks = abs(self._last_return_ticks)
        self._last_execution_volume = stats.total_volume

        position_after = self._position_mass_from_cohorts(
            distributions.long_exit_pmf,
            distributions.short_exit_pmf,
        )
        orderbook_after = self._snapshot_orderbook()
        best_bid_after = self._best_bid()
        best_ask_after = self._best_ask()
        spread_after = self._spread(best_bid_after, best_ask_after)
        market_flow_delta = (market_buy + short_exit_market) - (market_sell + long_exit_market)
        market_flow_total = market_buy + short_exit_market + market_sell + long_exit_market
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
            self._single_price_volume(
                best_ask_before or price_before, market_buy + short_exit_market
            ),
        )
        sell_volume = self._merge_maps(
            entry_limit_sell,
            long_exit_limit,
            self._single_price_volume(
                best_bid_before or price_before, market_sell + long_exit_market
            ),
        )
        vwap = stats.notional / stats.total_volume if stats.total_volume > 0 else None
        step_index = state.step_index + 1
        step_info = StepInfo(
            step_index=step_index,
            price_before=price_before,
            price_after=price_after,
            price_change=price_after - price_before,
            intensity=intensity,
            mood=latent.mood,
            trend=latent.trend,
            volatility=latent.volatility,
            price_grid=price_grid,
            buy_entry_pmf=distributions.buy_entry_pmf,
            sell_entry_pmf=distributions.sell_entry_pmf,
            long_exit_pmf=distributions.long_exit_pmf,
            short_exit_pmf=distributions.short_exit_pmf,
            buy_volume_by_price=buy_volume,
            sell_volume_by_price=sell_volume,
            entry_volume_by_price=entry_volume,
            exit_volume_by_price=exit_volume,
            cancelled_volume_by_price=cancelled_volume,
            executed_volume_by_price=self._drop_zeroes(stats.executed_by_price),
            total_executed_volume=stats.total_volume,
            market_buy_volume=market_buy + short_exit_market,
            market_sell_volume=market_sell + long_exit_market,
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
            intensity=intensity,
            latent=latent,
            price_grid=self._price_grid(price_after),
            distributions=distributions,
            orderbook=orderbook_after,
            position_mass=position_after,
        )
        return step_info

    def _age_state(self) -> None:
        for lots_by_price in (self._bid_lots, self._ask_lots):
            for lots in lots_by_price.values():
                for lot in lots:
                    lot.age += 1
        for cohort in (*self._long_cohorts, *self._short_cohorts):
            cohort.age += 1

    def _next_price_after_trading(self, price_before: float, stats: _TradeStats) -> float:
        if stats.total_volume <= 0 or stats.last_price is None:
            return price_before
        vwap = stats.notional / stats.total_volume
        micro_noise_ticks = self._clamp(self._rng.gauss(0.0, 0.58), -1.15, 1.15)
        micro_noise = micro_noise_ticks * self.gap
        anchor_pull = 0.04 * (self._anchor_price - price_before)
        return 0.58 * stats.last_price + 0.42 * vwap + micro_noise + anchor_pull

    def _price_grid(self, center_price: float) -> list[float]:
        center = self._snap_price(center_price)
        return [
            self._snap_price(center + offset * self.gap)
            for offset in range(-self.grid_radius, self.grid_radius + 1)
        ]

    def _snap_price(self, price: float) -> float:
        snapped = round(price / self.gap) * self.gap
        return self._clean_number(snapped)

    def _clean_number(self, value: float) -> float:
        rounded = round(value, 10)
        if rounded.is_integer():
            return float(int(rounded))
        return rounded

    def _next_latent(self, latent: LatentState) -> LatentState:
        mood_noise = self._rng.gauss(0.0, 0.13)
        trend_noise = self._rng.gauss(0.0, 0.08)
        jump = self._rng.gauss(0.0, 0.55) if self._rng.random() < 0.018 else 0.0
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
            + 0.10 * anchor_pressure
            + trend_noise,
            -1.0,
            1.0,
        )
        shock = abs(mood_noise) + abs(jump) * 0.75
        realized = 0.12 * self._last_abs_return_ticks + 0.025 * self._last_execution_volume
        volatility = self._clamp(
            0.78 * latent.volatility + shock + realized + 0.03 * abs(self._last_imbalance),
            0.04,
            2.2,
        )
        return LatentState(mood=mood, trend=trend, volatility=volatility)

    def _next_intensity(self, latent: LatentState) -> IntensityState:
        total = self.popularity * (1.0 + 2.2 * latent.volatility)
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

    def _next_distributions(
        self,
        price: float,
        price_grid: list[float],
        latent: LatentState,
    ) -> DistributionState:
        spread_base = self.gap * (1.0 + 5.5 * latent.volatility)
        trend_shift = latent.trend * self.gap * (2.5 + 3.0 * latent.volatility)
        mood_shift = latent.mood * self.gap * 2.0

        buy_entry = DiscreteMixtureDistribution(
            (
                MixtureComponent(
                    0.72 + max(latent.mood, 0.0) * 0.18,
                    price - self.gap + trend_shift,
                    spread_base,
                ),
                MixtureComponent(0.28, price - 4 * self.gap + mood_shift, spread_base * 1.8),
            )
        ).pmf(price_grid)
        sell_entry = DiscreteMixtureDistribution(
            (
                MixtureComponent(
                    0.72 + max(-latent.mood, 0.0) * 0.18,
                    price + self.gap + trend_shift,
                    spread_base,
                ),
                MixtureComponent(0.28, price + 4 * self.gap + mood_shift, spread_base * 1.8),
            )
        ).pmf(price_grid)
        long_exit = DiscreteMixtureDistribution(
            (
                MixtureComponent(0.58, price + self.gap * 3.0, spread_base * 1.2),
                MixtureComponent(0.42, price - self.gap * 2.0, spread_base * 1.1),
            )
        ).pmf(price_grid)
        short_exit = DiscreteMixtureDistribution(
            (
                MixtureComponent(0.58, price - self.gap * 3.0, spread_base * 1.2),
                MixtureComponent(0.42, price + self.gap * 2.0, spread_base * 1.1),
            )
        ).pmf(price_grid)
        return DistributionState(
            buy_entry_pmf=buy_entry,
            sell_entry_pmf=sell_entry,
            long_exit_pmf=long_exit,
            short_exit_pmf=short_exit,
        )

    def _entry_limit_volumes(
        self,
        intensity: IntensityState,
        distributions: DistributionState,
    ) -> tuple[PriceMap, PriceMap]:
        passive_share = 1.0 - self._taker_share(self.state.latent, self._last_imbalance)
        bid_ceiling = self.state.price - self.gap
        ask_floor = self.state.price + self.gap
        buy = {
            price: intensity.buy * passive_share * probability
            for price, probability in distributions.buy_entry_pmf.items()
            if price <= bid_ceiling
        }
        sell = {
            price: intensity.sell * passive_share * probability
            for price, probability in distributions.sell_entry_pmf.items()
            if price >= ask_floor
        }
        return self._drop_zeroes(buy), self._drop_zeroes(sell)

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
        return self._clamp(0.10 + 0.06 * latent.volatility + 0.05 * abs(imbalance), 0.08, 0.38)

    def _exit_flow(
        self,
        side: str,
        current_price: float,
        latent: LatentState,
        exit_pmf: PriceMap,
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

        limit_map = {
            price: passive_volume * probability
            for price, probability in exit_pmf.items()
            if passive_volume > 0
        }
        return market_volume, self._drop_zeroes(limit_map)

    def _cancel_orders(self, current_price: float, latent: LatentState) -> PriceMap:
        cancelled: PriceMap = {}
        for lots_by_price in (self._bid_lots, self._ask_lots):
            for price, lots in list(lots_by_price.items()):
                survivors = []
                distance = abs(price - current_price) / self.gap
                for lot in lots:
                    probability = self._clamp(
                        0.025 + 0.018 * lot.age + 0.012 * distance + 0.035 * latent.volatility,
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
        base = self.popularity * (0.18 + 0.20 / (1.0 + latent.volatility))
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
        lots_by_price = self._bid_lots if side == "bid" else self._ask_lots
        for price, volume in volume_by_price.items():
            if volume <= 0:
                continue
            lots_by_price.setdefault(price, []).append(_OrderLot(volume=volume, kind=kind))

    def _consume_taker(
        self,
        side: str,
        volume: float,
        kind: str,
        fallback_price: float,
        distributions: DistributionState,
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
                    price = fallback_price - impact_ticks * self.gap
                price = self._snap_price(price)
                fill_volume = remaining
                remaining = 0.0
                self._apply_fill(kind, fill_volume, price, distributions)
                stats.record(price, fill_volume)
                break

            lots_by_price = self._ask_lots if side == "buy" else self._bid_lots
            resting_lot = lots_by_price[price][0]
            fill_volume = min(remaining, resting_lot.volume)
            resting_lot.volume -= fill_volume
            remaining -= fill_volume
            self._apply_fill(kind, fill_volume, price, distributions)
            self._apply_fill(resting_lot.kind, fill_volume, price, distributions)
            stats.record(price, fill_volume)
            self._discard_empty_head(price, "ask" if side == "buy" else "bid")

    def _cross_taker_flow(
        self,
        price: float,
        buy_entry: float,
        short_exit: float,
        sell_entry: float,
        long_exit: float,
        distributions: DistributionState,
        stats: _TradeStats,
    ) -> tuple[float, float, float, float]:
        buy_total = buy_entry + short_exit
        sell_total = sell_entry + long_exit
        cross_volume = min(buy_total, sell_total) * 0.85
        if cross_volume <= 1e-12:
            return buy_entry, short_exit, sell_entry, long_exit

        buy_entry_fill = cross_volume * buy_entry / buy_total if buy_total > 0 else 0.0
        short_exit_fill = cross_volume * short_exit / buy_total if buy_total > 0 else 0.0
        sell_entry_fill = cross_volume * sell_entry / sell_total if sell_total > 0 else 0.0
        long_exit_fill = cross_volume * long_exit / sell_total if sell_total > 0 else 0.0

        self._apply_fill("buy_entry", buy_entry_fill, price, distributions)
        self._apply_fill("short_exit", short_exit_fill, price, distributions)
        self._apply_fill("sell_entry", sell_entry_fill, price, distributions)
        self._apply_fill("long_exit", long_exit_fill, price, distributions)
        stats.record(price, cross_volume)

        return (
            max(0.0, buy_entry - buy_entry_fill),
            max(0.0, short_exit - short_exit_fill),
            max(0.0, sell_entry - sell_entry_fill),
            max(0.0, long_exit - long_exit_fill),
        )

    def _match_crossed_book(self, distributions: DistributionState, stats: _TradeStats) -> None:
        while self._best_bid() is not None and self._best_ask() is not None:
            bid_price = self._best_bid()
            ask_price = self._best_ask()
            if bid_price is None or ask_price is None or bid_price < ask_price:
                break

            bid_lot = self._bid_lots[bid_price][0]
            ask_lot = self._ask_lots[ask_price][0]
            volume = min(bid_lot.volume, ask_lot.volume)
            if volume <= 1e-12:
                self._discard_empty_head(bid_price, "bid")
                self._discard_empty_head(ask_price, "ask")
                continue
            execution_price = ask_price
            bid_lot.volume -= volume
            ask_lot.volume -= volume
            self._apply_fill(bid_lot.kind, volume, execution_price, distributions)
            self._apply_fill(ask_lot.kind, volume, execution_price, distributions)
            stats.record(execution_price, volume)
            self._discard_empty_head(bid_price, "bid")
            self._discard_empty_head(ask_price, "ask")

    def _apply_fill(
        self,
        kind: str,
        volume: float,
        execution_price: float,
        distributions: DistributionState,
    ) -> None:
        if volume <= 0:
            return
        if kind == "buy_entry":
            self._long_cohorts.append(_PositionCohort("long", execution_price, volume))
        elif kind == "sell_entry":
            self._short_cohorts.append(_PositionCohort("short", execution_price, volume))
        elif kind == "long_exit":
            self._remove_cohort_mass(self._long_cohorts, execution_price, volume)
        elif kind == "short_exit":
            self._remove_cohort_mass(self._short_cohorts, execution_price, volume)

    def _remove_cohort_mass(
        self,
        cohorts: list[_PositionCohort],
        execution_price: float,
        volume: float,
    ) -> None:
        remaining = volume
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

    def _position_mass_from_cohorts(
        self, long_exit_pmf: PriceMap, short_exit_pmf: PriceMap
    ) -> PositionMassState:
        long_total = sum(cohort.mass for cohort in self._long_cohorts)
        short_total = sum(cohort.mass for cohort in self._short_cohorts)
        return PositionMassState(
            long_exit_mass_by_price=self._drop_zeroes(
                {price: long_total * probability for price, probability in long_exit_pmf.items()}
            ),
            short_exit_mass_by_price=self._drop_zeroes(
                {price: short_total * probability for price, probability in short_exit_pmf.items()}
            ),
        )

    def _clean_cohorts(self) -> None:
        self._long_cohorts = [cohort for cohort in self._long_cohorts if cohort.mass > 1e-12]
        self._short_cohorts = [cohort for cohort in self._short_cohorts if cohort.mass > 1e-12]

    def _best_bid(self) -> float | None:
        prices = [
            price
            for price, lots in self._bid_lots.items()
            if sum(lot.volume for lot in lots) > 1e-12
        ]
        return max(prices) if prices else None

    def _best_ask(self) -> float | None:
        prices = [
            price
            for price, lots in self._ask_lots.items()
            if sum(lot.volume for lot in lots) > 1e-12
        ]
        return min(prices) if prices else None

    def _spread(self, bid: float | None, ask: float | None) -> float | None:
        if bid is None or ask is None:
            return None
        return ask - bid

    def _near_touch_imbalance(self, current_price: float) -> float:
        bid_depth = 0.0
        ask_depth = 0.0
        for price, lots in self._bid_lots.items():
            distance = max(1.0, abs(current_price - price) / self.gap)
            if distance <= 5:
                bid_depth += sum(lot.volume for lot in lots) / distance
        for price, lots in self._ask_lots.items():
            distance = max(1.0, abs(price - current_price) / self.gap)
            if distance <= 5:
                ask_depth += sum(lot.volume for lot in lots) / distance
        total = bid_depth + ask_depth
        if total <= 1e-12:
            return 0.0
        return self._clamp((bid_depth - ask_depth) / total, -1.0, 1.0)

    def _discard_empty_head(self, price: float, side: str) -> None:
        lots_by_price = self._bid_lots if side == "bid" else self._ask_lots
        lots = lots_by_price.get(price, [])
        while lots and lots[0].volume <= 1e-12:
            lots.pop(0)
        if not lots and price in lots_by_price:
            del lots_by_price[price]

    def _clean_orderbook(self) -> None:
        for lots_by_price in (self._bid_lots, self._ask_lots):
            for price in list(lots_by_price):
                lots_by_price[price] = [lot for lot in lots_by_price[price] if lot.volume > 1e-12]
                if not lots_by_price[price]:
                    del lots_by_price[price]

    def _snapshot_orderbook(self) -> OrderBookState:
        return OrderBookState(
            bid_volume_by_price=self._aggregate_lots(self._bid_lots),
            ask_volume_by_price=self._aggregate_lots(self._ask_lots),
        )

    def _snapshot_position_mass(self) -> PositionMassState:
        return PositionMassState(
            long_exit_mass_by_price=dict(self.state.position_mass.long_exit_mass_by_price),
            short_exit_mass_by_price=dict(self.state.position_mass.short_exit_mass_by_price),
        )

    def _aggregate_lots(self, lots_by_price: dict[float, list[_OrderLot]]) -> PriceMap:
        return self._drop_zeroes(
            {price: sum(lot.volume for lot in lots) for price, lots in lots_by_price.items()}
        )

    def _merge_maps(self, *maps: PriceMap) -> PriceMap:
        merged: PriceMap = {}
        for values in maps:
            for price, volume in values.items():
                merged[price] = merged.get(price, 0.0) + volume
        return self._drop_zeroes(merged)

    def _single_price_volume(self, price: float, volume: float) -> PriceMap:
        if volume <= 0:
            return {}
        return {self._snap_price(price): volume}

    def _drop_zeroes(self, values: PriceMap) -> PriceMap:
        return {price: value for price, value in sorted(values.items()) if value > 1e-12}

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
