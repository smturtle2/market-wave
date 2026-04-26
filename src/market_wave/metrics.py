from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

from .generation import GeneratedPath


@dataclass(frozen=True)
class ValidationMetrics:
    path_count: int
    total_steps: int
    min_horizon: int
    max_horizon: int
    return_mean: float
    return_std: float
    return_tail_ratio: float
    volume_mean: float
    volume_std: float
    volatility_clustering_score: float
    max_drawdown: float
    price_move_ratio: float
    zero_volume_ratio: float
    mean_final_price: float | None
    mean_price_change: float
    mean_abs_price_change: float
    price_change_volatility: float
    nonzero_price_change_rate: float
    paths_with_price_moves: int
    min_price: float | None
    max_price: float | None
    total_executed_volume: float
    mean_executed_volume: float
    execution_rate: float
    mean_trade_count: float
    mean_order_flow_imbalance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "ValidationMetrics.to_dataframe() requires pandas; "
                "install market-wave[dataframe] to use it"
            ) from exc
        return pd.DataFrame.from_records([self.to_dict()])


def compute_metrics(paths: Iterable[GeneratedPath]) -> ValidationMetrics:
    path_list = list(paths)
    horizons = [len(path.steps) for path in path_list]
    steps = [step for path in path_list for step in path.steps]
    price_changes = [step.price_change for step in steps]
    tick_returns = [float(step.tick_change) for step in steps]
    abs_price_changes = [abs(change) for change in price_changes]
    abs_tick_returns_by_path = [
        [abs(float(step.tick_change)) for step in path.steps] for path in path_list
    ]
    executed_volumes = [step.total_executed_volume for step in steps]
    trade_counts = [float(step.trade_count) for step in steps]
    imbalances = [step.order_flow_imbalance for step in steps]
    prices = [step.price_before for step in steps] + [step.price_after for step in steps]
    final_prices = [path.final_price for path in path_list]
    drawdowns = [_max_drawdown(path) for path in path_list]

    total_steps = len(steps)
    total_executed_volume = sum(executed_volumes)
    return_std = _population_stddev(tick_returns)
    volume_mean = total_executed_volume / total_steps if total_steps else 0.0
    return ValidationMetrics(
        path_count=len(path_list),
        total_steps=total_steps,
        min_horizon=min(horizons) if horizons else 0,
        max_horizon=max(horizons) if horizons else 0,
        return_mean=_mean(tick_returns),
        return_std=return_std,
        return_tail_ratio=_tail_ratio(tick_returns, return_std),
        volume_mean=volume_mean,
        volume_std=_population_stddev(executed_volumes),
        volatility_clustering_score=_path_weighted_lag1_correlation(abs_tick_returns_by_path),
        max_drawdown=max(drawdowns) if drawdowns else 0.0,
        price_move_ratio=_rate(abs(value) > 1e-12 for value in tick_returns),
        zero_volume_ratio=_rate(volume <= 1e-12 for volume in executed_volumes),
        mean_final_price=_mean_or_none(final_prices),
        mean_price_change=_mean(price_changes),
        mean_abs_price_change=_mean(abs_price_changes),
        price_change_volatility=_population_stddev(price_changes),
        nonzero_price_change_rate=_rate(abs(change) > 1e-12 for change in price_changes),
        paths_with_price_moves=sum(
            any(abs(step.price_change) > 1e-12 for step in path.steps) for path in path_list
        ),
        min_price=min(prices) if prices else None,
        max_price=max(prices) if prices else None,
        total_executed_volume=total_executed_volume,
        mean_executed_volume=volume_mean,
        execution_rate=_rate(volume > 1e-12 for volume in executed_volumes),
        mean_trade_count=_mean(trade_counts),
        mean_order_flow_imbalance=_mean(imbalances),
    )


def _max_drawdown(path: GeneratedPath) -> float:
    prices = [path.metadata.initial_price, *(step.price_after for step in path.steps)]
    if not prices:
        return 0.0
    peak = prices[0]
    max_drawdown = 0.0
    for price in prices:
        peak = max(peak, price)
        max_drawdown = max(max_drawdown, peak - price)
    return max_drawdown


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_or_none(values: Sequence[float]) -> float | None:
    return _mean(values) if values else None


def _population_stddev(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _rate(flags: Iterable[bool]) -> float:
    values = list(flags)
    return sum(values) / len(values) if values else 0.0


def _tail_ratio(values: Sequence[float], stddev: float) -> float:
    if not values or stddev <= 1e-12:
        return 0.0
    threshold = 2.0 * stddev
    return sum(abs(value) >= threshold for value in values) / len(values)


def _path_weighted_lag1_correlation(paths: Sequence[Sequence[float]]) -> float:
    weighted_total = 0.0
    total_pairs = 0
    for values in paths:
        if len(values) < 3:
            continue
        pair_count = max(0, len(values) - 1)
        weighted_total += _lag1_correlation(values) * pair_count
        total_pairs += pair_count
    return weighted_total / total_pairs if total_pairs else 0.0


def _lag1_correlation(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=False)
    )
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    denominator = sqrt(left_var * right_var)
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator
