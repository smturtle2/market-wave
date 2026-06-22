from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .corpus import read_public_corpus

BID_BLUE = "#1d5fd1"
ASK_RED = "#dc2626"
TRADE_DARK = "#0f1b2d"
GRID = "#e4ebf3"
BACKGROUND = "#f6f9fc"
TERMINAL_BG = "#f4f7fb"
PANEL_BG = "#ffffff"
PANEL_EDGE = "#d7e0ea"
GRID_DARK = "#e8eef6"
TEXT = "#0f1b2d"
MUTED = "#66758a"
UP = "#1d5fd1"
DOWN = "#dc2626"
TABLE_LINE = "#e4ebf3"


def plot_all(root: Path, output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    market, snapshots, _ = read_public_corpus(root)
    if not snapshots:
        raise ValueError("visible_snapshot_stream is required for plots")
    paths = [
        plot_market_screen(market, snapshots, output_dir / "market_screen.png"),
        plot_price_chart(market, snapshots, output_dir / "price_chart.png"),
        plot_orderbook_panel(market, snapshots, output_dir / "orderbook_panel.png"),
        plot_top10_orderbook_heatmap(snapshots, output_dir / "top10_orderbook_heatmap.png"),
        plot_event_tape(market, output_dir / "event_tape.png"),
        plot_mid_spread_depth(snapshots, output_dir / "mid_spread_depth.png"),
        plot_depth_heatmap(snapshots, output_dir / "depth_heatmap.png"),
    ]
    return paths


def plot_market_screen(market: list[dict[str, Any]], snapshots: list[dict[str, Any]], path: Path) -> Path:
    snap = snapshots[-1]
    trades = [row for row in market if row.get("record_type") == "TRADE"][-30:]
    fig = plt.figure(figsize=(16, 9), facecolor=TERMINAL_BG)
    gs = fig.add_gridspec(
        12,
        16,
        left=0.012,
        right=0.992,
        top=0.982,
        bottom=0.022,
        wspace=0.16,
        hspace=0.18,
    )
    ax_header = fig.add_subplot(gs[0:2, :])
    ax_chart = fig.add_subplot(gs[2:9, 0:10])
    ax_heatmap = fig.add_subplot(gs[9:12, 0:10])
    ax_ladder = fig.add_subplot(gs[2:12, 10:14])
    ax_tape = fig.add_subplot(gs[2:8, 14:16])
    ax_depth = fig.add_subplot(gs[8:12, 14:16])

    _draw_terminal_header(ax_header, market, snapshots)
    _draw_terminal_price_chart(ax_chart, market, snapshots)
    _draw_level2_ladder(ax_ladder, snap, trades)
    _draw_time_and_sales(ax_tape, trades)
    _draw_terminal_depth_strip(ax_depth, snapshots)
    _draw_terminal_heatmap(ax_heatmap, snapshots)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_price_chart(market: list[dict[str, Any]], snapshots: list[dict[str, Any]], path: Path) -> Path:
    fig = plt.figure(figsize=(14, 7), facecolor=TERMINAL_BG)
    ax_header = fig.add_axes([0.018, 0.805, 0.964, 0.165])
    ax = fig.add_axes([0.035, 0.065, 0.94, 0.715])
    _draw_terminal_header(ax_header, market, snapshots)
    _draw_terminal_price_chart(ax, market, snapshots)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_orderbook_panel(market: list[dict[str, Any]], snapshots: list[dict[str, Any]], path: Path) -> Path:
    snap = snapshots[-1]
    trades = [row for row in market if row.get("record_type") == "TRADE"][-30:]
    fig = plt.figure(figsize=(12, 7), facecolor=TERMINAL_BG)
    ax_ladder = fig.add_axes([0.035, 0.055, 0.58, 0.89])
    ax_depth = fig.add_axes([0.65, 0.055, 0.315, 0.89])
    _draw_level2_ladder(ax_ladder, snap, trades)
    _draw_terminal_depth_strip(ax_depth, snapshots)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def trade_price_bars(market: list[dict[str, Any]], snapshots: list[dict[str, Any]], *, max_bars: int = 120) -> list[dict[str, float]]:
    start = _timestamp_seconds(snapshots[0]["timestamp"])
    end = _timestamp_seconds(snapshots[-1]["timestamp"])
    duration = max(1.0, end - start)
    bucket_seconds = max(1.0, float(np.ceil(duration / max(1, max_bars))))
    trades = []
    for row in market:
        if row.get("record_type") != "TRADE":
            continue
        elapsed = _timestamp_seconds(row["timestamp"]) - start
        trades.append(
            {
                "time": elapsed,
                "price": float(row["payload"]["price"]),
                "volume": float(row["payload"]["volume"]),
            }
        )
    if not trades:
        return []

    buckets: dict[int, list[dict[str, float]]] = {}
    for trade in trades:
        bucket = int(trade["time"] // bucket_seconds)
        buckets.setdefault(bucket, []).append(trade)

    bars: list[dict[str, float]] = []
    for bucket in sorted(buckets):
        rows = buckets[bucket]
        prices = [row["price"] for row in rows]
        volumes = [row["volume"] for row in rows]
        bars.append(
            {
                "x": float(bucket * bucket_seconds + bucket_seconds / 2),
                "width": float(bucket_seconds),
                "open": prices[0],
                "high": max(prices),
                "low": min(prices),
                "close": prices[-1],
                "volume": sum(volumes),
            }
        )
    return bars


def plot_top10_orderbook_heatmap(snapshots: list[dict[str, Any]], path: Path) -> Path:
    fig = plt.figure(figsize=(14, 6), facecolor=TERMINAL_BG)
    ax = fig.add_axes([0.05, 0.08, 0.92, 0.84])
    _draw_terminal_heatmap(ax, snapshots)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_event_tape(market: list[dict[str, Any]], path: Path) -> Path:
    fig = plt.figure(figsize=(14, 4), facecolor=TERMINAL_BG)
    ax = fig.add_axes([0.025, 0.08, 0.95, 0.84])
    trades = [row for row in market if row.get("record_type") == "TRADE"][-36:]
    _draw_time_and_sales(ax, trades)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_mid_spread_depth(snapshots: list[dict[str, Any]], path: Path) -> Path:
    x = np.arange(len(snapshots))
    mid = np.array([float(s["mid"]) for s in snapshots])
    spread = np.array([float(s["spread"]) for s in snapshots])
    bid_depth = np.array([float(s["total_visible_bid_depth"]) for s in snapshots])
    ask_depth = np.array([float(s["total_visible_ask_depth"]) for s in snapshots])
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, facecolor=BACKGROUND)
    for ax in axes:
        _style_axis(ax)
    axes[0].plot(x, mid, color=TRADE_DARK, linewidth=1.1)
    axes[0].set_title("Mid", fontsize=11)
    _plain_price_axis(axes[0])
    axes[1].plot(x, spread, color="#6d7280", linewidth=1.0)
    axes[1].set_title("Spread", fontsize=11)
    axes[2].plot(x, bid_depth, color=BID_BLUE, linewidth=1.0)
    axes[2].plot(x, ask_depth, color=ASK_RED, linewidth=1.0)
    axes[2].set_title("Visible Depth", fontsize=11)
    axes[2].set_xlabel("snapshot")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_depth_heatmap(snapshots: list[dict[str, Any]], path: Path) -> Path:
    fig = plt.figure(figsize=(14, 7), facecolor=TERMINAL_BG)
    ax = fig.add_axes([0.05, 0.07, 0.92, 0.86])
    _draw_terminal_heatmap(ax, snapshots)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _draw_terminal_header(ax: plt.Axes, market: list[dict[str, Any]], snapshots: list[dict[str, Any]]) -> None:
    _style_terminal_panel(ax)
    ax.axis("off")
    snap = snapshots[-1]
    summary = _quote_summary(market, snapshots)
    change_color = UP if summary["change"] >= 0 else DOWN
    ax.text(0.015, 0.68, "MARKET WAVE", color=TEXT, fontsize=22, fontweight="bold", family="DejaVu Sans Mono", transform=ax.transAxes)
    ax.text(0.015, 0.26, "CONTINUOUS  L2  09:00:00 KST", color=MUTED, fontsize=9, family="DejaVu Sans Mono", transform=ax.transAxes)
    ax.text(0.185, 0.61, _format_price(summary["last"]), color=change_color, fontsize=32, fontweight="bold", family="DejaVu Sans Mono", transform=ax.transAxes)
    ax.text(0.188, 0.25, f"{summary['change']:+.2f}  {summary['change_pct']:+.2f}%", color=change_color, fontsize=12, family="DejaVu Sans Mono", transform=ax.transAxes)

    fields = [
        ("BID", snap["best_bid"], BID_BLUE),
        ("ASK", snap["best_ask"], ASK_RED),
        ("SPREAD", snap["spread"], TEXT),
        ("VOL", _format_compact(summary["volume"]), TEXT),
        ("TRADES", str(summary["trade_count"]), TEXT),
        ("TIME", _clock(summary["timestamp"]), TEXT),
    ]
    x0 = 0.42
    width = 0.09
    for i, (label, value, color) in enumerate(fields):
        left = x0 + i * width
        ax.add_patch(Rectangle((left, 0.18), width - 0.012, 0.62, transform=ax.transAxes, facecolor="#eef4fa", edgecolor=PANEL_EDGE, linewidth=0.8))
        ax.text(left + 0.010, 0.58, label, color=MUTED, fontsize=8, family="DejaVu Sans Mono", transform=ax.transAxes)
        ax.text(left + 0.010, 0.34, str(value), color=color, fontsize=12, fontweight="bold", family="DejaVu Sans Mono", transform=ax.transAxes)


def _draw_terminal_price_chart(ax: plt.Axes, market: list[dict[str, Any]], snapshots: list[dict[str, Any]]) -> None:
    _style_terminal_panel(ax)
    start = _timestamp_seconds(snapshots[0]["timestamp"])
    x = np.array([_timestamp_seconds(row["timestamp"]) - start for row in snapshots])
    mid = np.array([float(row["mid"]) for row in snapshots])
    best_bid = np.array([float(row["best_bid"]) for row in snapshots])
    best_ask = np.array([float(row["best_ask"]) for row in snapshots])
    bars = trade_price_bars(market, snapshots)
    trade_x, trade_price, trade_volume, trade_colors = _trade_points(market, snapshots)

    ax.fill_between(x, best_bid, best_ask, color="#e6edf6", alpha=0.92, linewidth=0)
    ax.plot(x, mid, color="#64748b", linewidth=0.75, alpha=0.80)
    if len(trade_x):
        ax.step(trade_x, trade_price, where="post", color=TEXT, linewidth=1.05, alpha=0.95)
        sizes = np.clip(np.sqrt(trade_volume) * 3.1, 10, 52)
        ax.scatter(trade_x, trade_price, c=trade_colors, s=sizes, alpha=0.82, linewidths=0)
    _draw_ohlc_bars(ax, bars)
    _plain_price_axis(ax)
    ax.yaxis.tick_right()
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    ax.text(0.012, 0.972, "PRICE / LAST TRADE", color=TEXT, fontsize=9, fontweight="bold", family="DejaVu Sans Mono", transform=ax.transAxes, va="top")
    ax.text(0.988, 0.972, _clock(snapshots[-1]["timestamp"]), color=MUTED, fontsize=9, family="DejaVu Sans Mono", transform=ax.transAxes, ha="right", va="top")

    volume_ax = ax.inset_axes([0.018, 0.035, 0.965, 0.17])
    _draw_terminal_volume_inset(volume_ax, bars)


def _draw_level2_ladder(ax: plt.Axes, snap: dict[str, Any], trades: list[dict[str, Any]]) -> None:
    _style_terminal_panel(ax)
    ax.axis("off")
    asks = list(reversed(snap["asks"][:10]))
    bids = snap["bids"][:10]
    rows = [("ASK", level) for level in asks] + [("BID", level) for level in bids]
    volumes = [float(level["volume"]) for _, level in rows]
    max_volume = max(volumes) if volumes else 1.0
    ax.text(0.035, 0.965, "LEVEL 2 BOOK", color=TEXT, fontsize=10, fontweight="bold", family="DejaVu Sans Mono", transform=ax.transAxes, va="top")
    ax.text(0.050, 0.905, "PRICE", color=MUTED, fontsize=8.8, family="DejaVu Sans Mono", transform=ax.transAxes)
    ax.text(0.470, 0.905, "SIZE", color=MUTED, fontsize=8.8, family="DejaVu Sans Mono", transform=ax.transAxes)
    ax.text(0.735, 0.905, "SUM", color=MUTED, fontsize=8.8, family="DejaVu Sans Mono", transform=ax.transAxes)

    ask_cum = _cumulative([float(level["volume"]) for level in reversed(asks)])
    bid_cum = _cumulative([float(level["volume"]) for level in bids])
    ask_cum_by_row = list(reversed(ask_cum))
    bid_cum_by_row = bid_cum
    y_top = 0.862
    row_h = 0.0365
    for index, (side_name, level) in enumerate(rows):
        y = y_top - index * row_h
        color = ASK_RED if side_name == "ASK" else BID_BLUE
        fill = min(0.96, float(level["volume"]) / max_volume)
        ax.add_patch(Rectangle((0.022, y - 0.021), 0.956, row_h * 0.92, transform=ax.transAxes, facecolor="#f8fbfe", edgecolor=TABLE_LINE, linewidth=0.35))
        ax.add_patch(Rectangle((0.978 - 0.56 * fill, y - 0.021), 0.56 * fill, row_h * 0.92, transform=ax.transAxes, facecolor=color, alpha=0.24, edgecolor="none"))
        if index == 9:
            ax.add_patch(Rectangle((0.022, y - 0.021), 0.956, row_h * 0.92, transform=ax.transAxes, facecolor="#fee2e2", alpha=0.95, edgecolor=ASK_RED, linewidth=0.65))
        if index == 10:
            ax.add_patch(Rectangle((0.022, y - 0.021), 0.956, row_h * 0.92, transform=ax.transAxes, facecolor="#eaf1ff", alpha=0.98, edgecolor=BID_BLUE, linewidth=0.65))
        cum = ask_cum_by_row[index] if side_name == "ASK" else bid_cum_by_row[index - 10]
        row_font = 9.5 if index in {9, 10} else 8.8
        weight = "bold" if index in {9, 10} else "normal"
        ax.text(0.050, y - 0.004, _format_price(float(level["price"])), color=color, fontsize=row_font, fontweight=weight, family="DejaVu Sans Mono", transform=ax.transAxes, va="center")
        ax.text(0.485, y - 0.004, _format_compact(float(level["volume"])), color=TEXT, fontsize=row_font, fontweight=weight, family="DejaVu Sans Mono", transform=ax.transAxes, va="center", ha="right")
        ax.text(0.900, y - 0.004, _format_compact(cum), color=MUTED, fontsize=row_font, family="DejaVu Sans Mono", transform=ax.transAxes, va="center", ha="right")

    spread_y = y_top - 9.5 * row_h
    ax.plot([0.022, 0.978], [spread_y, spread_y], color="#475569", linewidth=0.85, alpha=0.70, transform=ax.transAxes)
    last_side = trades[-1]["payload"]["side"] if trades else ""
    last_color = BID_BLUE if last_side == "BUY" else ASK_RED
    if trades:
        ax.text(0.035, 0.050, f"LAST {trades[-1]['payload']['price']}  {trades[-1]['payload']['volume']}  {last_side}", color=last_color, fontsize=9, family="DejaVu Sans Mono", transform=ax.transAxes)


def _draw_time_and_sales(ax: plt.Axes, trades: list[dict[str, Any]]) -> None:
    _style_terminal_panel(ax)
    ax.axis("off")
    ax.text(0.055, 0.955, "TIME & SALES", color=TEXT, fontsize=9.5, fontweight="bold", family="DejaVu Sans Mono", transform=ax.transAxes, va="top")
    ax.text(0.060, 0.885, "TIME", color=MUTED, fontsize=7.5, family="DejaVu Sans Mono", transform=ax.transAxes)
    ax.text(0.350, 0.885, "PRICE", color=MUTED, fontsize=7.5, family="DejaVu Sans Mono", transform=ax.transAxes)
    ax.text(0.760, 0.885, "QTY", color=MUTED, fontsize=7.5, family="DejaVu Sans Mono", transform=ax.transAxes)
    rows = trades[-16:]
    row_h = 0.047
    y_top = 0.825
    for index, row in enumerate(rows):
        y = y_top - index * row_h
        side = row["payload"]["side"]
        color = BID_BLUE if side == "BUY" else ASK_RED
        ax.add_patch(Rectangle((0.040, y - 0.020), 0.920, row_h * 0.82, transform=ax.transAxes, facecolor="#f8fbfe" if index % 2 == 0 else "#eef4fa", edgecolor="none"))
        ax.text(0.060, y, _clock(row["timestamp"]), color=MUTED, fontsize=7.6, family="DejaVu Sans Mono", transform=ax.transAxes, va="center")
        ax.text(0.350, y, str(row["payload"]["price"]), color=color, fontsize=7.8, family="DejaVu Sans Mono", transform=ax.transAxes, va="center")
        ax.text(0.905, y, str(row["payload"]["volume"]), color=TEXT, fontsize=7.8, family="DejaVu Sans Mono", transform=ax.transAxes, va="center", ha="right")


def _draw_terminal_depth_strip(ax: plt.Axes, snapshots: list[dict[str, Any]]) -> None:
    _style_terminal_panel(ax)
    tail = snapshots[-160:]
    x = np.arange(len(tail))
    bid_depth = np.array([float(row["total_visible_bid_depth"]) for row in tail])
    ask_depth = np.array([float(row["total_visible_ask_depth"]) for row in tail])
    imbalance = (bid_depth - ask_depth) / np.maximum(1.0, bid_depth + ask_depth)
    ax.plot(x, bid_depth, color=BID_BLUE, linewidth=1.05)
    ax.plot(x, ask_depth, color=ASK_RED, linewidth=1.05)
    ax.fill_between(x, 0, np.maximum(bid_depth, ask_depth), color="#e6edf6", alpha=0.70)
    ax2 = ax.twinx()
    ax2.plot(x, imbalance, color="#0f766e", linewidth=0.65, alpha=0.82)
    ax2.set_ylim(-1, 1)
    ax2.set_yticks([])
    ax.text(0.045, 0.940, "DEPTH / IMBAL", color=TEXT, fontsize=8.5, fontweight="bold", family="DejaVu Sans Mono", transform=ax.transAxes, va="top")
    _terminal_ticks(ax)


def _draw_terminal_heatmap(ax: plt.Axes, snapshots: list[dict[str, Any]]) -> None:
    _style_terminal_panel(ax)
    volume, side = _top10_level_matrices(snapshots)
    image = _terminal_level_depth_rgba(volume, side)
    ax.imshow(image, aspect="auto", interpolation="nearest")
    ax.axhline(9.5, color="#475569", linewidth=0.7, alpha=0.62)
    ax.set_yticks([0, 4, 9, 10, 15, 19])
    ax.set_yticklabels(["ask10", "ask6", "ask1", "bid1", "bid6", "bid10"], color=MUTED, fontsize=8, family="DejaVu Sans Mono")
    _set_time_tick_labels(ax, snapshots)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.text(0.010, 0.955, "TOP-10 QUEUE HEAT", color=TEXT, fontsize=8.5, fontweight="bold", family="DejaVu Sans Mono", transform=ax.transAxes, va="top")


def _draw_terminal_volume_inset(ax: plt.Axes, bars: list[dict[str, float]]) -> None:
    ax.set_facecolor("#f8fbfe")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if not bars:
        return
    colors = [ASK_RED if bar["close"] >= bar["open"] else BID_BLUE for bar in bars]
    ax.bar([bar["x"] for bar in bars], [bar["volume"] for bar in bars], width=[max(0.8, bar["width"] * 0.62) for bar in bars], color=colors, alpha=0.68)
    ax.text(0.006, 0.88, "VOL", color=MUTED, fontsize=7.5, family="DejaVu Sans Mono", transform=ax.transAxes, va="top")


def _draw_orderbook(ax: plt.Axes, snap: dict[str, Any]) -> None:
    _style_axis(ax)
    asks = list(reversed(snap["asks"][:10]))
    bids = snap["bids"][:10]
    labels = [level["price"] for level in asks] + [level["price"] for level in bids]
    volumes = [float(level["volume"]) for level in asks] + [float(level["volume"]) for level in bids]
    colors = [ASK_RED] * len(asks) + [BID_BLUE] * len(bids)
    y = np.arange(len(labels))
    ax.barh(y, volumes, color=colors, alpha=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axhline(len(asks) - 0.5, color="#30343b", linewidth=0.8)
    ax.set_title("Top-10 Orderbook", fontsize=11)
    ax.set_xlabel("visible volume")


def _draw_recent_trades(ax: plt.Axes, trades: list[dict[str, Any]]) -> None:
    _style_axis(ax)
    if not trades:
        ax.set_title("Recent Trades", fontsize=11)
        return
    prices = [float(row["payload"]["price"]) for row in trades]
    colors = [BID_BLUE if row["payload"]["side"] == "BUY" else ASK_RED for row in trades]
    ax.scatter(np.arange(len(trades)), prices, c=colors, s=22)
    ax.plot(np.arange(len(trades)), prices, color="#8b909a", linewidth=0.7, alpha=0.55)
    ax.set_title("Recent Trades", fontsize=11)
    _plain_price_axis(ax)


def _draw_price_chart(
    ax: plt.Axes,
    market: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    *,
    title: str,
    volume_axis: plt.Axes | None = None,
) -> None:
    _style_axis(ax)
    start = _timestamp_seconds(snapshots[0]["timestamp"])
    x = np.array([_timestamp_seconds(row["timestamp"]) - start for row in snapshots])
    mid = np.array([float(row["mid"]) for row in snapshots])
    best_bid = np.array([float(row["best_bid"]) for row in snapshots])
    best_ask = np.array([float(row["best_ask"]) for row in snapshots])
    bars = trade_price_bars(market, snapshots)
    trade_x, trade_price, trade_volume, trade_colors = _trade_points(market, snapshots)

    ax.fill_between(x, best_bid, best_ask, color="#dfe5ee", alpha=0.55, linewidth=0, label="spread")
    ax.plot(x, mid, color="#30343b", linewidth=0.75, alpha=0.62, label="mid")
    if len(trade_x):
        ax.step(trade_x, trade_price, where="post", color=TRADE_DARK, linewidth=1.25, alpha=0.86, label="last trade")
        sizes = np.clip(np.sqrt(trade_volume) * 3.6, 12, 72)
        ax.scatter(trade_x, trade_price, c=trade_colors, s=sizes, alpha=0.72, linewidths=0)
    _draw_ohlc_bars(ax, bars)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("seconds from open")
    ax.set_ylabel("price")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _plain_price_axis(ax)
    if volume_axis is not None:
        _draw_volume_bars(volume_axis, bars)


def _draw_ohlc_bars(ax: plt.Axes, bars: list[dict[str, float]]) -> None:
    if not bars:
        return
    lows = [bar["low"] for bar in bars]
    highs = [bar["high"] for bar in bars]
    body_floor = max((max(highs) - min(lows)) * 0.002, 0.01)
    for index, bar in enumerate(bars):
        color = ASK_RED if bar["close"] >= bar["open"] else BID_BLUE
        width = max(0.55, bar["width"] * 0.62)
        body_low = min(bar["open"], bar["close"])
        body_height = max(abs(bar["close"] - bar["open"]), body_floor)
        ax.vlines(bar["x"], bar["low"], bar["high"], color=color, linewidth=0.8, alpha=0.95)
        ax.add_patch(
            Rectangle(
                (bar["x"] - width / 2, body_low),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
                alpha=0.74,
                label="trade OHLC" if index == 0 else None,
            )
        )


def _draw_volume_bars(ax: plt.Axes, bars: list[dict[str, float]]) -> None:
    _style_axis(ax)
    if not bars:
        ax.set_ylabel("volume")
        return
    colors = [ASK_RED if bar["close"] >= bar["open"] else BID_BLUE for bar in bars]
    ax.bar([bar["x"] for bar in bars], [bar["volume"] for bar in bars], width=[max(0.8, bar["width"] * 0.62) for bar in bars], color=colors, alpha=0.62)
    ax.set_ylabel("volume")
    ax.set_xlabel("seconds from open")


def _draw_visible_depth_strip(ax: plt.Axes, snapshots: list[dict[str, Any]]) -> None:
    _style_axis(ax)
    tail = snapshots[-120:]
    x = np.arange(len(tail))
    bid_depth = [float(row["total_visible_bid_depth"]) for row in tail]
    ask_depth = [float(row["total_visible_ask_depth"]) for row in tail]
    ax.plot(x, bid_depth, color=BID_BLUE, linewidth=1.0)
    ax.plot(x, ask_depth, color=ASK_RED, linewidth=1.0)
    ax.set_title("Visible Depth", fontsize=11)


def _draw_depth_bars(ax: plt.Axes, snap: dict[str, Any]) -> None:
    _style_axis(ax)
    bids = [float(level["volume"]) for level in snap["bids"][:10]]
    asks = [float(level["volume"]) for level in snap["asks"][:10]]
    y = np.arange(10)
    ax.barh(y - 0.18, bids, height=0.32, color=BID_BLUE, alpha=0.75, label="bid")
    ax.barh(y + 0.18, asks, height=0.32, color=ASK_RED, alpha=0.75, label="ask")
    ax.set_yticks(y)
    ax.set_yticklabels([str(i + 1) for i in y])
    ax.invert_yaxis()
    ax.set_title("Depth by Level", fontsize=11)
    ax.legend(frameon=False, fontsize=8)


def _draw_top10_level_heatmap(ax: plt.Axes, snapshots: list[dict[str, Any]], *, title: str) -> None:
    _style_axis(ax)
    volume, side = _top10_level_matrices(snapshots)
    image = _level_depth_rgba(volume, side)
    ax.imshow(image, aspect="auto", interpolation="nearest")
    ax.axhline(9.5, color="#30343b", linewidth=0.8)
    ax.set_yticks([0, 4, 9, 10, 15, 19])
    ax.set_yticklabels(["ask 10", "ask 6", "ask 1", "bid 1", "bid 6", "bid 10"])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("time")
    ax.set_ylabel("level")
    _set_time_tick_labels(ax, snapshots)


def _top10_level_matrices(snapshots: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    volume = np.zeros((20, len(snapshots)), dtype=float)
    side = np.zeros((20, len(snapshots)), dtype=float)
    for col, snap in enumerate(snapshots):
        asks = list(reversed(snap["asks"][:10]))
        bids = snap["bids"][:10]
        for row, level in enumerate(asks):
            volume[row, col] = float(level["volume"])
            side[row, col] = 1.0
        for offset, level in enumerate(bids):
            row = 10 + offset
            volume[row, col] = float(level["volume"])
            side[row, col] = -1.0
    return volume, side


def _level_depth_rgba(volume: np.ndarray, side: np.ndarray) -> np.ndarray:
    if volume.size == 0:
        return np.zeros((0, 0, 4))
    positive = np.log1p(volume[volume > 0])
    high = max(1.0, float(np.percentile(positive, 98))) if positive.size else 1.0
    normalized = np.clip(np.log1p(volume) / high, 0.0, 1.0)
    base = np.array([1.0, 1.0, 1.0])
    ask = _hex_rgb(ASK_RED)
    bid = _hex_rgb(BID_BLUE)
    image = np.zeros((volume.shape[0], volume.shape[1], 4), dtype=float)
    for row in range(volume.shape[0]):
        for col in range(volume.shape[1]):
            color = ask if side[row, col] > 0 else bid
            strength = normalized[row, col]
            rgb = base * (1.0 - strength) + color * strength
            image[row, col, :3] = rgb
            image[row, col, 3] = 1.0
    return image


def _trade_points(market: list[dict[str, Any]], snapshots: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    start = _timestamp_seconds(snapshots[0]["timestamp"])
    rows = [row for row in market if row.get("record_type") == "TRADE"]
    if not rows:
        return np.array([]), np.array([]), np.array([]), []
    x = np.array([_timestamp_seconds(row["timestamp"]) - start for row in rows], dtype=float)
    prices = np.array([float(row["payload"]["price"]) for row in rows], dtype=float)
    volumes = np.array([float(row["payload"]["volume"]) for row in rows], dtype=float)
    colors = [BID_BLUE if row["payload"]["side"] == "BUY" else ASK_RED for row in rows]
    return x, prices, volumes, colors


def _set_time_tick_labels(ax: plt.Axes, snapshots: list[dict[str, Any]]) -> None:
    if not snapshots:
        return
    start = _timestamp_seconds(snapshots[0]["timestamp"])
    count = min(6, len(snapshots))
    rows = np.linspace(0, len(snapshots) - 1, count, dtype=int)
    labels = [str(int(_timestamp_seconds(snapshots[int(row)]["timestamp"]) - start)) for row in rows]
    ax.set_xticks(rows)
    ax.set_xticklabels(labels)


def _quote_summary(market: list[dict[str, Any]], snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    trades = [row for row in market if row.get("record_type") == "TRADE"]
    first_price = float(trades[0]["payload"]["price"]) if trades else float(snapshots[0]["mid"])
    last_price = float(trades[-1]["payload"]["price"]) if trades else float(snapshots[-1]["mid"])
    volume = sum(float(row["payload"]["volume"]) for row in trades)
    change = last_price - first_price
    return {
        "last": last_price,
        "change": change,
        "change_pct": 100.0 * change / first_price if first_price else 0.0,
        "volume": volume,
        "trade_count": len(trades),
        "timestamp": snapshots[-1]["timestamp"],
    }


def _terminal_level_depth_rgba(volume: np.ndarray, side: np.ndarray) -> np.ndarray:
    if volume.size == 0:
        return np.zeros((0, 0, 4))
    positive = np.log1p(volume[volume > 0])
    high = max(1.0, float(np.percentile(positive, 98))) if positive.size else 1.0
    normalized = np.clip(np.log1p(volume) / high, 0.0, 1.0)
    base = _hex_rgb(PANEL_BG)
    ask = _hex_rgb(ASK_RED)
    bid = _hex_rgb(BID_BLUE)
    image = np.zeros((volume.shape[0], volume.shape[1], 4), dtype=float)
    for row in range(volume.shape[0]):
        color = ask if row < 10 else bid
        for col in range(volume.shape[1]):
            strength = normalized[row, col] * 0.92
            image[row, col, :3] = base * (1.0 - strength) + color * strength
            image[row, col, 3] = 1.0
    return image


def _style_terminal_panel(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID_DARK, linewidth=0.55, alpha=0.72)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.8)


def _terminal_ticks(ax: plt.Axes) -> None:
    ax.tick_params(axis="x", colors=MUTED, labelsize=7)
    ax.tick_params(axis="y", colors=MUTED, labelsize=7)
    ax.yaxis.tick_right()
    _plain_price_axis(ax)


def _cumulative(values: list[float]) -> list[float]:
    total = 0.0
    out = []
    for value in values:
        total += value
        out.append(total)
    return out


def _clock(value: str) -> str:
    return datetime.fromisoformat(value).strftime("%H:%M:%S")


def _format_price(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _format_compact(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if absolute >= 10_000:
        return f"{value / 1_000:.1f}K"
    if absolute >= 100:
        return f"{value:,.0f}"
    if absolute >= 1:
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _timestamp_seconds(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=float)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.tick_params(labelsize=8, colors="#424957")
    for spine in ax.spines.values():
        spine.set_color("#d9dee7")
        spine.set_linewidth(0.8)


def _plain_price_axis(ax: plt.Axes) -> None:
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
