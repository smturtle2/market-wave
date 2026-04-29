from __future__ import annotations

from .state import StepInfo


class _MarketPlottingMixin:
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
