from __future__ import annotations

import matplotlib
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from market_wave import Market

matplotlib.use("Agg", force=True)


def _market_with_history(step_count: int = 5) -> Market:
    market = Market(initial_price=100.0, gap=1.0, popularity=1.1, seed=2026, grid_radius=8)

    steps = market.step(step_count)

    assert len(steps) == step_count
    assert len(market.history) == step_count
    return market


def _close(fig: Figure) -> None:
    from matplotlib import pyplot as plt

    plt.close(fig)


def _line_lengths(ax: Axes) -> list[int]:
    return [len(line.get_ydata()) for line in ax.lines]


def _relative_luminance(color) -> float:
    red, green, blue, *_ = color
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _axes_same_position(left: Axes, right: Axes) -> bool:
    left_bounds = left.get_position().bounds
    right_bounds = right.get_position().bounds
    return all(abs(a - b) <= 1e-6 for a, b in zip(left_bounds, right_bounds, strict=False))


def _has_separate_panel(fig: Figure, ax: Axes) -> bool:
    return any(not _axes_same_position(ax, other) for other in fig.axes if other is not ax)


def test_plot_history_uses_matplotlib_agg_backend_and_returns_figure_axes():
    assert "agg" in matplotlib.get_backend().lower()
    market = _market_with_history(4)

    fig, ax = market.plot_history()

    try:
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax in fig.axes
        assert any(length == len(market.history) for length in _line_lengths(ax))
    finally:
        _close(fig)


def test_plot_history_defaults_to_light_market_wave_panel_layout():
    market = _market_with_history(5)

    fig, ax = market.plot_history()

    try:
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert _relative_luminance(fig.get_facecolor()) >= 0.7
        assert _relative_luminance(ax.get_facecolor()) >= 0.7
        assert len(fig.axes) >= 2
        assert _has_separate_panel(fig, ax)
    finally:
        _close(fig)


def test_plot_history_supports_dark_market_wave_style():
    market = _market_with_history(5)

    fig, ax = market.plot_history(style="market_wave_dark")

    try:
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert _relative_luminance(fig.get_facecolor()) <= 0.35
        assert _relative_luminance(ax.get_facecolor()) <= 0.35
        assert any(length == len(market.history) for length in _line_lengths(ax))
    finally:
        _close(fig)


def test_plot_is_alias_for_plot_history_and_accepts_plot_options():
    market = _market_with_history(3)

    fig, ax = market.plot(style="market_wave_dark", layout="overlay")

    try:
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert any(length == len(market.history) for length in _line_lengths(ax))
    finally:
        _close(fig)


def test_plot_history_empty_history_raises_value_error():
    market = Market(initial_price=100.0, gap=1.0, seed=7, grid_radius=4)

    with pytest.raises(ValueError, match="history"):
        market.plot_history()


def test_plot_history_last_limits_plotted_history_window():
    market = _market_with_history(6)

    full_fig, full_ax = market.plot_history()
    last_fig, last_ax = market.plot_history(last=2)

    try:
        assert any(length == len(market.history) for length in _line_lengths(full_ax))
        assert any(length == 2 for length in _line_lengths(last_ax))
    finally:
        _close(full_fig)
        _close(last_fig)


def test_plot_history_overlay_layout_keeps_single_primary_axes_contract():
    market = _market_with_history(5)

    fig, ax = market.plot_history(layout="overlay", style="market_wave_dark")

    try:
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert all(_axes_same_position(ax, other) for other in fig.axes if other is not ax)
        assert any(length == len(market.history) for length in _line_lengths(ax))
    finally:
        _close(fig)


def test_plot_history_invalid_layout_raises_value_error():
    market = _market_with_history(2)

    with pytest.raises(ValueError, match="layout"):
        market.plot_history(layout="stacked")
