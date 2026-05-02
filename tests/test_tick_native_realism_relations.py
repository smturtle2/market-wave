from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev

from market_wave import Market

_TAIL_TICK_THRESHOLD = 2.0
_THIN_ALIVE_ACTIVITY_FLOOR = 0.50


def _corr(left: list[float], right: list[float]) -> float:
    if len(left) < 3:
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var <= 1e-12 or right_var <= 1e-12:
        return 0.0
    return sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=False)
    ) / sqrt(left_var * right_var)


def _q99_abs_tick(returns: list[float]) -> float:
    values = sorted(abs(value) for value in returns)
    return values[int((len(values) - 1) * 0.99)] if values else 0.0


def _submitted_volume(step) -> float:
    return sum(step.buy_volume_by_price.values()) + sum(step.sell_volume_by_price.values())


def _rested_volume(step) -> float:
    return step.residual_market_buy_volume + step.residual_market_sell_volume


def _step_intent_imbalance(step) -> float:
    buy = sum(step.buy_volume_by_price.values())
    sell = sum(step.sell_volume_by_price.values())
    total = buy + sell
    return 0.0 if total <= 1e-12 else (buy - sell) / total


def _step_rested_imbalance(step) -> float:
    buy = step.residual_market_buy_volume
    sell = step.residual_market_sell_volume
    total = buy + sell
    return 0.0 if total <= 1e-12 else (buy - sell) / total


def _family_stats(config: dict, *, seeds: range = range(401, 405), steps: int = 450) -> dict:
    rows = []
    for seed in seeds:
        market = Market(**config, seed=seed)
        path = market.step(steps)
        returns = [float(step.tick_change) for step in path]
        abs_returns = [abs(value) for value in returns]
        volumes = [step.total_executed_volume for step in path]
        submitted = [_submitted_volume(step) for step in path]
        rested = [_rested_volume(step) for step in path]
        cancelled = [sum(step.cancelled_volume_by_price.values()) for step in path]
        event_pressure = [
            (step.total_executed_volume + sum(step.cancelled_volume_by_price.values()))
            * abs(step.order_flow_imbalance)
            for step in path
        ]
        execution_pressure = [
            step.total_executed_volume * abs(step.order_flow_imbalance) for step in path
        ]
        anchor_moves = [
            abs((current.mdf_price_basis - previous.mdf_price_basis) / config["gap"])
            for previous, current in zip(path, path[1:], strict=False)
        ]
        spread_ticks = [
            step.spread_after / market.gap for step in path if step.spread_after is not None
        ]
        near_depth_share = []
        far_depth_share = []
        one_sided_book = []
        pre_near_depth_on_execution = []
        low_pre_depth_execution = []
        for step in path:
            bid = step.orderbook_after.bid_volume_by_price
            ask = step.orderbook_after.ask_volume_by_price
            bid_depth = sum(bid.values())
            ask_depth = sum(ask.values())
            one_sided_book.append((bid_depth <= 1e-12) != (ask_depth <= 1e-12))
            near_depth = 0.0
            far_depth = 0.0
            for book in (bid, ask):
                for price, volume in book.items():
                    distance = abs(price - step.price_after) / market.gap
                    if distance <= 3:
                        near_depth += volume
                    if distance >= 8:
                        far_depth += volume
            measured_depth = near_depth + far_depth
            if measured_depth > 1e-12:
                near_depth_share.append(near_depth / measured_depth)
                far_depth_share.append(far_depth / measured_depth)
            pre_near_depth = 0.0
            for book in (
                step.orderbook_before.bid_volume_by_price,
                step.orderbook_before.ask_volume_by_price,
            ):
                for price, volume in book.items():
                    distance = abs(price - step.price_before) / market.gap
                    if distance <= 3:
                        pre_near_depth += volume
            if step.total_executed_volume > 1e-12:
                pre_near_depth_on_execution.append(pre_near_depth)
                low_pre_depth_execution.append(pre_near_depth < 0.5)
        depth = [
            sum(step.orderbook_after.bid_volume_by_price.values())
            + sum(step.orderbook_after.ask_volume_by_price.values())
            for step in path
        ]
        intent_signal = [
            0.65 * _step_intent_imbalance(step) + 0.35 * _step_rested_imbalance(step)
            for step in path[:-1]
        ]
        rows.append(
            {
                "activity_rate": mean(
                    submitted_volume > 1e-12
                    or rested_volume > 1e-12
                    or cancelled_volume > 1e-12
                    or step.trade_count > 0
                    for step, submitted_volume, rested_volume, cancelled_volume in zip(
                        path, submitted, rested, cancelled, strict=False
                    )
                ),
                "execution_rate": mean(volume > 1e-12 for volume in volumes),
                "trade_count": mean(float(step.trade_count) for step in path),
                "submitted_volume": mean(submitted),
                "rested_volume": mean(rested),
                "spread_ticks": mean(spread_ticks) if spread_ticks else 0.0,
                "depth": mean(depth),
                "near_depth_share": mean(near_depth_share) if near_depth_share else 0.0,
                "far_depth_share": mean(far_depth_share) if far_depth_share else 0.0,
                "one_sided_book_rate": mean(one_sided_book),
                "pre_near_depth_on_execution": (
                    mean(pre_near_depth_on_execution) if pre_near_depth_on_execution else 0.0
                ),
                "low_pre_depth_execution_rate": (
                    mean(low_pre_depth_execution) if low_pre_depth_execution else 0.0
                ),
                "zero_volume_rate": mean(volume <= 1e-12 for volume in volumes),
                "tick_std": pstdev(returns),
                "q99_abs_tick": _q99_abs_tick(returns),
                "tail_event_rate": mean(
                    value >= _TAIL_TICK_THRESHOLD for value in abs_returns
                ),
                "nonzero_tick_rate": mean(value > 0.0 for value in abs_returns),
                "signed_lag1": _corr(returns[:-1], returns[1:]),
                "abs_lag1": _corr(abs_returns[:-1], abs_returns[1:]),
                "volume_lag1": _corr(volumes[:-1], volumes[1:]),
                "cancel_rate": mean(volume > 1e-12 for volume in cancelled),
                "abs_volume_corr": _corr(abs_returns, volumes),
                "abs_execution_pressure_corr": _corr(abs_returns, execution_pressure),
                "anchor_event_pressure_corr": _corr(anchor_moves, event_pressure[:-1]),
                "intent_next_tick_corr": _corr(intent_signal, returns[1:]),
                "intent_next_abs_tick_corr": _corr(
                    [abs(value) for value in intent_signal],
                    abs_returns[1:],
                ),
                "cumulative_tick": sum(returns),
                "positive_tick_rate": mean(value > 0.0 for value in returns),
                "negative_tick_rate": mean(value < 0.0 for value in returns),
            }
        )
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def test_normal_and_thin_markets_are_not_fully_frozen_but_inactive_is() -> None:
    normal = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0})
    thin = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 0.25})
    inactive = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 0.0})

    assert normal["activity_rate"] > 0.0
    assert normal["activity_rate"] > thin["activity_rate"]
    assert thin["activity_rate"] > _THIN_ALIVE_ACTIVITY_FLOOR
    assert thin["execution_rate"] > 0.0
    assert inactive["activity_rate"] == 0.0


def test_busy_market_has_more_tick_native_activity_than_base() -> None:
    base = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0})
    busy = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 3.0})

    assert busy["submitted_volume"] > base["submitted_volume"]
    assert busy["execution_rate"] > base["execution_rate"]
    assert busy["trade_count"] > base["trade_count"]
    assert busy["depth"] > base["depth"]
    assert busy["spread_ticks"] <= base["spread_ticks"] + 0.25
    assert busy["zero_volume_rate"] < base["zero_volume_rate"]
    assert busy["volume_lag1"] > 0.0


def test_high_vol_market_has_larger_tick_variance_and_tail_over_paired_seeds() -> None:
    variance_wins = 0
    tail_wins = 0
    variance_deltas = []
    tail_deltas = []

    for seed in range(18, 23):
        base = _family_stats(
            {"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0},
            seeds=range(seed, seed + 1),
            steps=200,
        )
        high_vol = _family_stats(
            {
                "initial_price": 10_000.0,
                "gap": 10.0,
                "popularity": 1.0,
                "regime": "high_vol",
            },
            seeds=range(seed, seed + 1),
            steps=200,
        )
        variance_wins += high_vol["tick_std"] > base["tick_std"]
        tail_wins += high_vol["tail_event_rate"] > base["tail_event_rate"]
        variance_deltas.append(high_vol["tick_std"] - base["tick_std"])
        tail_deltas.append(high_vol["tail_event_rate"] - base["tail_event_rate"])

    assert variance_wins >= 4
    assert tail_wins >= 3
    assert mean(variance_deltas) > 0.0
    assert mean(tail_deltas) > 0.0

    high_vol = _family_stats(
        {"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0, "regime": "high_vol"}
    )
    assert high_vol["abs_lag1"] > 0.0


def test_thin_liquidity_is_less_active_alive_and_wider_than_base_over_paired_seeds() -> None:
    activity_wins = 0
    spread_wins = 0
    for seed in range(14, 19):
        base = _family_stats(
            {"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0},
            seeds=range(seed, seed + 1),
            steps=200,
        )
        thin = _family_stats(
            {"initial_price": 10_000.0, "gap": 10.0, "popularity": 0.25},
            seeds=range(seed, seed + 1),
            steps=200,
        )
        activity_wins += 0.0 < thin["activity_rate"] < base["activity_rate"]
        spread_wins += thin["spread_ticks"] > base["spread_ticks"]
        assert thin["activity_rate"] > _THIN_ALIVE_ACTIVITY_FLOOR

    assert activity_wins == 5
    assert spread_wins >= 4


def test_directional_regimes_separate_in_cumulative_ticks() -> None:
    normal = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0})
    trend_up = _family_stats(
        {"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0, "regime": "trend_up"}
    )
    trend_down = _family_stats(
        {"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0, "regime": "trend_down"}
    )
    high_vol = _family_stats(
        {"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0, "regime": "high_vol"}
    )

    assert trend_up["cumulative_tick"] > trend_down["cumulative_tick"]
    assert trend_up["cumulative_tick"] > 0.0
    assert trend_down["cumulative_tick"] < 0.0
    assert normal["cumulative_tick"] < trend_up["cumulative_tick"]
    assert (
        abs(trend_up["cumulative_tick"]) / max(abs(trend_down["cumulative_tick"]), 1e-12)
        < 8.0
    )
    assert (
        abs(trend_down["cumulative_tick"]) / max(abs(trend_up["cumulative_tick"]), 1e-12)
        < 8.0
    )
    assert abs(high_vol["cumulative_tick"]) < max(
        abs(trend_up["cumulative_tick"]),
        abs(trend_down["cumulative_tick"]),
    )


def test_submitted_and_rested_intent_does_not_become_a_strong_predictor() -> None:
    normal = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0})

    assert abs(normal["intent_next_tick_corr"]) < 0.60
    assert abs(normal["intent_next_abs_tick_corr"]) < 0.60


def test_execution_volume_is_not_strongly_inverse_to_tick_movement() -> None:
    normal = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0})
    busy = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 3.0})
    high_vol = _family_stats(
        {"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0, "regime": "high_vol"}
    )
    thin = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 0.25})

    assert normal["abs_volume_corr"] > -0.12
    assert busy["abs_volume_corr"] > -0.12
    assert busy["abs_execution_pressure_corr"] > 0.0
    assert high_vol["abs_volume_corr"] > -0.18
    assert high_vol["abs_execution_pressure_corr"] > 0.0
    assert thin["abs_volume_corr"] > 0.0


def test_microstructure_artifacts_stay_below_synthetic_noise_ceiling() -> None:
    normal = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0})
    busy = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 3.0})
    high_vol = _family_stats(
        {"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0, "regime": "high_vol"}
    )

    assert normal["execution_rate"] < 0.92
    assert busy["execution_rate"] < 0.96
    assert high_vol["execution_rate"] < 0.93
    assert normal["tail_event_rate"] < 0.48
    assert busy["tail_event_rate"] < 0.52
    assert high_vol["tail_event_rate"] < 0.58
    assert normal["signed_lag1"] > -0.12
    assert high_vol["signed_lag1"] > -0.16
    assert normal["cancel_rate"] < 0.72
    assert busy["cancel_rate"] < 0.72
    assert high_vol["cancel_rate"] < 0.74


def test_mdf_anchor_responds_to_event_pressure() -> None:
    normal = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0})
    busy = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 3.0})
    high_vol = _family_stats(
        {"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0, "regime": "high_vol"}
    )

    assert normal["anchor_event_pressure_corr"] > 0.0
    assert busy["anchor_event_pressure_corr"] > 0.0
    assert high_vol["anchor_event_pressure_corr"] > 0.0


def test_book_topology_does_not_collapse_to_far_or_one_sided_depth() -> None:
    normal = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0})
    busy = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 3.0})
    trend_down = _family_stats(
        {"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0, "regime": "trend_down"}
    )

    assert normal["near_depth_share"] > 0.24
    assert busy["near_depth_share"] > 0.20
    assert trend_down["near_depth_share"] > 0.20
    assert normal["far_depth_share"] < 0.76
    assert busy["far_depth_share"] < 0.80
    assert trend_down["far_depth_share"] < 0.80
    assert normal["one_sided_book_rate"] < 0.20
    assert busy["one_sided_book_rate"] < 0.20
    assert trend_down["one_sided_book_rate"] < 0.26


def test_executions_are_supported_by_preexisting_near_book_depth() -> None:
    normal = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0})
    busy = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 3.0})
    high_vol = _family_stats(
        {"initial_price": 10_000.0, "gap": 10.0, "popularity": 1.0, "regime": "high_vol"}
    )
    thin = _family_stats({"initial_price": 10_000.0, "gap": 10.0, "popularity": 0.25})

    assert normal["pre_near_depth_on_execution"] > 4.0
    assert busy["pre_near_depth_on_execution"] > normal["pre_near_depth_on_execution"]
    assert high_vol["pre_near_depth_on_execution"] > 3.0
    assert thin["pre_near_depth_on_execution"] > 0.8
    assert normal["low_pre_depth_execution_rate"] < 0.04
    assert busy["low_pre_depth_execution_rate"] < 0.01
    assert high_vol["low_pre_depth_execution_rate"] < 0.04
    assert thin["low_pre_depth_execution_rate"] < 0.22
