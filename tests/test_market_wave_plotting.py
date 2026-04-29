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


def _image_shapes(ax: Axes) -> list[tuple[int, int]]:
    return [image.get_array().shape for image in ax.images]


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
        assert len(fig.axes) >= 5
        assert _has_separate_panel(fig, ax)
        assert fig.axes[1].get_ylabel() == "ask level"
        assert fig.axes[2].get_ylabel() == "bid level"
        expected_depth = min(market.grid_radius, 20)
        assert _image_shapes(fig.axes[1]) == [(expected_depth, len(market.history))]
        assert _image_shapes(fig.axes[2]) == [(expected_depth, len(market.history))]
    finally:
        _close(fig)


def test_plot_history_can_disable_orderbook_heatmap_for_legacy_panel_layout():
    market = _market_with_history(5)

    fig, ax = market.plot_history(orderbook=False)

    try:
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(fig.axes) == 3
        assert not any(axis.images for axis in fig.axes)
        assert [axis.get_ylabel() for axis in fig.axes] == ["price", "volume", "imbalance"]
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
        assert _image_shapes(last_fig.axes[1]) == [(market.grid_radius, 2)]
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
        assert not any(axis.images for axis in fig.axes)
    finally:
        _close(fig)


def test_plot_history_rejects_panel_layout_with_existing_axes():
    from matplotlib import pyplot as plt

    market = _market_with_history(2)
    fig, ax = plt.subplots()

    try:
        with pytest.raises(ValueError, match="ax.*layout='overlay'"):
            market.plot_history(ax=ax, layout="panel")
    finally:
        _close(fig)


def test_plot_history_overlay_rejects_orderbook_heatmap():
    market = _market_with_history(2)

    with pytest.raises(ValueError, match="orderbook"):
        market.plot_history(layout="overlay", orderbook=True)


def test_plot_history_invalid_layout_raises_value_error():
    market = _market_with_history(2)

    with pytest.raises(ValueError, match="layout"):
        market.plot_history(layout="stacked")


@pytest.mark.parametrize("snapshot", ["middle", "open", "close", "", None])
def test_plot_history_invalid_orderbook_snapshot_raises_value_error(snapshot):
    market = _market_with_history(2)

    with pytest.raises(ValueError, match="orderbook_snapshot|snapshot"):
        market.plot_history(orderbook_snapshot=snapshot)


@pytest.mark.parametrize("depth", [0, -1])
def test_plot_history_invalid_orderbook_depth_raises_value_error(depth):
    market = _market_with_history(2)

    with pytest.raises(ValueError, match="orderbook_depth|depth"):
        market.plot_history(orderbook_depth=depth)


def test_orderbook_heatmap_uses_level_depth_not_price_axis():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.1, seed=2026, grid_radius=30)
    market.step(4)

    fig, ax = market.plot_history(orderbook_depth=6)

    try:
        ask_ax = fig.axes[1]
        bid_ax = fig.axes[2]
        assert ax.get_ylabel() == "price"
        assert bid_ax.get_ylabel() == "bid level"
        assert ask_ax.get_ylabel() == "ask level"
        assert _image_shapes(bid_ax) == [(6, 4)]
        assert _image_shapes(ask_ax) == [(6, 4)]
        assert max(bid_ax.get_yticks()) <= 6
        assert max(ask_ax.get_yticks()) <= 6
        assert 100.0 not in bid_ax.get_yticks()
        assert 100.0 not in ask_ax.get_yticks()
        assert not ask_ax.yaxis_inverted()
        assert bid_ax.yaxis_inverted()
    finally:
        _close(fig)


def test_orderbook_heatmap_maps_price_gaps_to_actual_tick_levels():
    market = Market(initial_price=100.0, gap=1.0, popularity=0.0, seed=2027, grid_radius=6)
    step = market.step(1)[0]
    step.orderbook_after.bid_volume_by_price.clear()
    step.orderbook_after.ask_volume_by_price.clear()
    step.orderbook_after.bid_volume_by_price.update({98.0: 2.0, 95.0: 5.0})
    step.orderbook_after.ask_volume_by_price.update({102.0: 3.0, 105.0: 7.0})

    _x_values, levels, bid_matrix, ask_matrix = market._orderbook_heatmap_matrices(
        [step],
        snapshot="after",
        depth=6,
    )

    assert levels == [1, 2, 3, 4, 5, 6]
    assert [row[0] for row in bid_matrix] == [0.0, 2.0, 0.0, 0.0, 5.0, 0.0]
    assert [row[0] for row in ask_matrix] == [0.0, 3.0, 0.0, 0.0, 7.0, 0.0]


def test_orderbook_heatmap_default_depth_is_min_grid_radius_and_twenty():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.1, seed=2026, grid_radius=30)
    market.step(4)

    fig, _ax = market.plot_history()

    try:
        expected_depth = min(market.grid_radius, 20)
        assert _image_shapes(fig.axes[1]) == [(expected_depth, len(market.history))]
        assert _image_shapes(fig.axes[2]) == [(expected_depth, len(market.history))]
    finally:
        _close(fig)
