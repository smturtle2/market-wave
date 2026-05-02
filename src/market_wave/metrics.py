from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass
from math import sqrt
from os import PathLike
from pathlib import Path
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class MetricComparisonSpec:
    """Configuration for one field in ``compare_metrics``."""

    field: str
    weight: float = 1.0
    scale_floor: float = 1e-9
    kind: str = "relative"


DEFAULT_COMPARISON_SPECS = (
    MetricComparisonSpec("tick_return_mean", 0.5),
    MetricComparisonSpec("tick_return_std", 1.5),
    MetricComparisonSpec("tick_return_tail_ratio", 1.25, kind="absolute"),
    MetricComparisonSpec("volatility_clustering_score", 1.25, kind="absolute"),
    MetricComparisonSpec("mean_abs_tick_change", 1.5),
    MetricComparisonSpec("nonzero_tick_change_rate", 1.0, kind="absolute"),
    MetricComparisonSpec("zero_volume_ratio", 1.0, kind="absolute"),
    MetricComparisonSpec("mean_executed_volume", 1.0),
    MetricComparisonSpec("execution_rate", 1.0, kind="absolute"),
    MetricComparisonSpec("mean_cancelled_volume", 1.0),
    MetricComparisonSpec("cancellation_rate", 1.25, kind="absolute"),
    MetricComparisonSpec("mean_abs_position_change_ticks", 1.25),
    MetricComparisonSpec("position_change_rate", 1.0, kind="absolute"),
    MetricComparisonSpec("mean_abs_mdf_anchor_change_ticks", 1.25),
    MetricComparisonSpec("mdf_anchor_change_rate", 1.0, kind="absolute"),
    MetricComparisonSpec("mdf_anchor_event_pressure_corr", 1.0, kind="absolute"),
    MetricComparisonSpec("mean_spread_ticks", 1.25),
    MetricComparisonSpec("mean_near_depth_share", 1.0, kind="absolute"),
    MetricComparisonSpec("mean_far_depth_share", 1.0, kind="absolute"),
    MetricComparisonSpec("one_sided_book_rate", 1.0, kind="absolute"),
    MetricComparisonSpec("mean_trade_count", 0.75),
    MetricComparisonSpec("mean_order_flow_imbalance", 0.5, kind="absolute"),
)
DEFAULT_COMPARISON_FIELDS = tuple(spec.field for spec in DEFAULT_COMPARISON_SPECS)

REFERENCE_RECORD_REQUIRED_FIELDS = (
    "tick_change",
    "tick_before",
    "tick_after",
    "total_executed_volume",
    "cancelled_volume_by_price",
    "trade_count",
    "order_flow_imbalance",
    "mdf_price_basis",
    "spread_after",
    "orderbook_after",
    "price_after",
)
REFERENCE_RECORD_OPTIONAL_FIELDS = (
    "path_id",
    "mean_quote_age",
)
REFERENCE_ORDERBOOK_REQUIRED_FIELDS = (
    "bid_volume_by_price",
    "ask_volume_by_price",
)


@dataclass(frozen=True)
class ValidationMetrics:
    """Tick-native summary statistics for one or more simulated paths."""

    path_count: int
    total_steps: int
    min_horizon: int
    max_horizon: int
    tick_return_mean: float
    tick_return_std: float
    tick_return_tail_ratio: float
    volume_mean: float
    volume_std: float
    volatility_clustering_score: float
    max_drawdown_ticks: float
    max_runup_ticks: float
    cumulative_tick_range: float
    mean_abs_tick_change: float
    tick_change_volatility: float
    nonzero_tick_change_rate: float
    zero_volume_ratio: float
    total_executed_volume: float
    mean_executed_volume: float
    execution_rate: float
    total_cancelled_volume: float
    mean_cancelled_volume: float
    cancellation_rate: float
    mean_abs_position_change_ticks: float
    position_change_rate: float
    mean_abs_mdf_anchor_change_ticks: float
    mdf_anchor_change_rate: float
    mdf_anchor_event_pressure_corr: float
    mean_spread_ticks: float
    mean_near_depth_share: float
    mean_far_depth_share: float
    one_sided_book_rate: float
    mean_trade_count: float
    mean_order_flow_imbalance: float
    mean_quote_age: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a JSON-friendly dictionary."""

        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        """Serialize metrics to JSON."""

        return json.dumps(self.to_dict(), **kwargs)

    def to_dataframe(self):
        """Return a one-row pandas DataFrame.

        Requires the optional ``market-wave[dataframe]`` dependency.
        """

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "ValidationMetrics.to_dataframe() requires pandas; "
                "install market-wave[dataframe] to use it"
            ) from exc
        return pd.DataFrame.from_records([self.to_dict()])


@dataclass(frozen=True)
class MetricsProfile:
    """Named metrics profile used for persistence and comparisons."""

    name: str
    metrics: ValidationMetrics
    fields: tuple[str, ...] = DEFAULT_COMPARISON_FIELDS

    def to_dict(self) -> dict[str, Any]:
        """Return the profile as a JSON-friendly dictionary."""

        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        """Serialize the profile to JSON."""

        return json.dumps(self.to_dict(), **kwargs)


@dataclass(frozen=True)
class MetricComparisonRow:
    """Per-field contribution to a metrics comparison score."""

    field: str
    synthetic: float
    reference: float
    delta: float
    normalized_error: float
    weight: float
    contribution: float


@dataclass(frozen=True)
class MetricsComparison:
    """Result returned by ``compare_metrics``."""

    synthetic: MetricsProfile
    reference: MetricsProfile
    score: float
    rows: tuple[MetricComparisonRow, ...]

    @property
    def total_distance(self) -> float:
        return self.score

    @property
    def field_distances(self) -> dict[str, float]:
        return {row.field: row.normalized_error for row in self.rows}

    @property
    def field_deltas(self) -> dict[str, float]:
        return {row.field: row.delta for row in self.rows}

    @property
    def field_weights(self) -> dict[str, float]:
        return {row.field: row.weight for row in self.rows}

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison as a JSON-friendly dictionary."""

        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        """Serialize the comparison to JSON."""

        return json.dumps(self.to_dict(), **kwargs)


MetricComparison = MetricsComparison


def compute_metrics_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    gap: float = 1.0,
    path_id_field: str = "path_id",
) -> ValidationMetrics:
    """Compute metrics from exported ``StepInfo.to_dict()``-style rows."""

    if gap <= 0.0:
        raise ValueError("gap must be positive")
    grouped: dict[Any, list[Any]] = {}
    for _row_index, record in _validated_reference_records(records):
        path_id = record.get(path_id_field, 0)
        step_record = dict(record)
        step_record.pop(path_id_field, None)
        grouped.setdefault(path_id, []).append(_record_to_namespace(step_record))

    paths = [
        SimpleNamespace(
            path_id=path_id,
            steps=tuple(steps),
            metadata=SimpleNamespace(config={"gap": float(gap)}),
        )
        for path_id, steps in grouped.items()
    ]
    return compute_metrics(paths)


def validate_reference_records(records: Iterable[Mapping[str, Any]]) -> None:
    """Validate StepInfo-like reference records without computing metrics."""

    for _row_index, _record in _validated_reference_records(records):
        pass


def load_reference_metrics_profile(
    source: str | PathLike[str] | Iterable[Mapping[str, Any]],
    *,
    name: str = "reference",
    format: str = "auto",
    gap: float = 1.0,
    path_id_field: str = "path_id",
) -> MetricsProfile:
    """Load a reference profile from metrics JSON or StepInfo-like JSONL/CSV records."""

    if isinstance(source, str | PathLike):
        records_or_profile = _load_reference_source(Path(source), format=format)
        if isinstance(records_or_profile, Mapping):
            return load_metrics_profile(records_or_profile, name=name)
        records = records_or_profile
    else:
        records = source
    return MetricsProfile(
        name=name,
        metrics=compute_metrics_from_records(records, gap=gap, path_id_field=path_id_field),
    )


def compare_metrics(
    synthetic: ValidationMetrics | MetricsProfile,
    reference: ValidationMetrics | MetricsProfile,
    *,
    specs: Sequence[MetricComparisonSpec] = DEFAULT_COMPARISON_SPECS,
    fields: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
) -> MetricsComparison:
    """Compare synthetic metrics against a reference metrics profile.

    By default, fields are compared with ``DEFAULT_COMPARISON_SPECS``. Pass
    ``fields`` to compare a custom subset, and ``weights`` to override field
    weights.
    """

    synthetic_profile = _as_metrics_profile(synthetic, "synthetic")
    reference_profile = _as_metrics_profile(reference, "reference")
    if fields is not None:
        weight_map = weights or {}
        specs = tuple(
            _field_comparison_spec(field, float(weight_map.get(field, 1.0)))
            for field in fields
        )
    elif weights:
        specs = tuple(
            MetricComparisonSpec(
                spec.field,
                float(weights.get(spec.field, spec.weight)),
                spec.scale_floor,
                spec.kind,
            )
            for spec in specs
        )
    rows: list[MetricComparisonRow] = []
    total_contribution = 0.0
    total_weight = 0.0
    for spec in specs:
        if not hasattr(synthetic_profile.metrics, spec.field) or not hasattr(
            reference_profile.metrics,
            spec.field,
        ):
            raise ValueError(f"unknown metric field: {spec.field}")
        if spec.kind not in {"relative", "absolute"}:
            raise ValueError(f"unknown comparison kind for {spec.field}: {spec.kind}")
        weight = float(spec.weight)
        if weight <= 0.0:
            continue
        synthetic_value = float(getattr(synthetic_profile.metrics, spec.field))
        reference_value = float(getattr(reference_profile.metrics, spec.field))
        delta = synthetic_value - reference_value
        if spec.kind == "absolute":
            normalized_error = abs(delta)
        else:
            normalized_error = abs(delta) / max(abs(reference_value), float(spec.scale_floor))
        contribution = weight * normalized_error
        rows.append(
            MetricComparisonRow(
                field=spec.field,
                synthetic=synthetic_value,
                reference=reference_value,
                delta=delta,
                normalized_error=normalized_error,
                weight=weight,
                contribution=contribution,
            )
        )
        total_contribution += contribution
        total_weight += weight
    return MetricsComparison(
        synthetic=synthetic_profile,
        reference=reference_profile,
        score=total_contribution / total_weight if total_weight else 0.0,
        rows=tuple(rows),
    )


def load_metrics_profile(data: Mapping[str, Any], *, name: str = "reference") -> MetricsProfile:
    """Build a named metrics profile from a mapping of ``ValidationMetrics`` fields."""

    required_fields = ValidationMetrics.__dataclass_fields__
    missing = [
        field
        for field, field_def in required_fields.items()
        if field not in data
        and field_def.default is MISSING
        and field_def.default_factory is MISSING
    ]
    if missing:
        raise ValueError(f"missing metric field: {missing[0]}")
    metrics = ValidationMetrics(
        **{field: data[field] for field in required_fields if field in data}
    )
    return MetricsProfile(name=name, metrics=metrics)


def load_metrics_profile_json(
    path: str | PathLike[str],
    *,
    name: str | None = None,
) -> MetricsProfile:
    """Load a metrics profile from a JSON file."""

    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, Mapping):
        raise ValueError("metrics profile JSON must contain an object")
    profile_name = name
    metrics_data: Mapping[str, Any]
    if isinstance(data.get("metrics"), Mapping):
        metrics_data = data["metrics"]
        if profile_name is None:
            raw_name = data.get("name", "reference")
            profile_name = str(raw_name)
    else:
        metrics_data = data
    return load_metrics_profile(metrics_data, name=profile_name or "reference")


def save_metrics_profile_json(
    profile: ValidationMetrics | MetricsProfile,
    path: str | PathLike[str],
    *,
    name: str = "reference",
    **json_kwargs: Any,
) -> None:
    """Write a metrics profile JSON file."""

    metrics_profile = _as_metrics_profile(profile, name)
    kwargs = {"indent": 2, "sort_keys": True, **json_kwargs}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics_profile.to_dict(), file, **kwargs)
        file.write("\n")


def _load_reference_source(
    path: Path,
    *,
    format: str,
) -> Mapping[str, Any] | list[Mapping[str, Any]]:
    source_format = _infer_reference_format(path, format)
    if source_format == "json":
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, Mapping) and isinstance(data.get("metrics"), Mapping):
            return data["metrics"]
        if isinstance(data, Mapping):
            return data
        if isinstance(data, list) and all(isinstance(item, Mapping) for item in data):
            return data
        raise ValueError("JSON reference source must contain an object or list of objects")
    if source_format == "jsonl":
        records: list[Mapping[str, Any]] = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if not isinstance(record, Mapping):
                    raise ValueError(f"JSONL line {line_number} must contain an object")
                records.append(record)
        return records
    if source_format == "csv":
        with path.open(newline="", encoding="utf-8") as file:
            return [_decode_csv_record(record) for record in csv.DictReader(file)]
    raise ValueError("format must be one of: auto, json, jsonl, csv")


def _infer_reference_format(path: Path, format: str) -> str:
    if format != "auto":
        return format
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    return "json"


def _decode_csv_record(record: Mapping[str, str]) -> dict[str, Any]:
    return {key: _decode_csv_value(value) for key, value in record.items()}


def _decode_csv_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return stripped
    if stripped[0] in "[{":
        return json.loads(stripped)
    try:
        if any(marker in stripped.lower() for marker in (".", "e")):
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def _validate_step_record(record: Mapping[str, Any], row_index: int) -> None:
    for field in REFERENCE_RECORD_REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"record {row_index} missing required field: {field}")
    orderbook = record["orderbook_after"]
    if not isinstance(orderbook, Mapping):
        raise ValueError(f"record {row_index} field orderbook_after must be a mapping")
    for field in REFERENCE_ORDERBOOK_REQUIRED_FIELDS:
        if field not in orderbook:
            raise ValueError(f"record {row_index} missing required field: orderbook_after.{field}")
        if not isinstance(orderbook[field], Mapping):
            raise ValueError(
                f"record {row_index} field orderbook_after.{field} must be a mapping"
            )


def _validated_reference_records(
    records: Iterable[Mapping[str, Any]],
) -> Iterable[tuple[int, Mapping[str, Any]]]:
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {row_index} must be a mapping")
        _validate_step_record(record, row_index)
        yield row_index, record


def _record_to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        if _is_numeric_mapping(value):
            return {
                _float_key_if_numeric(key): float(item)
                for key, item in value.items()
                if item is not None
            }
        return SimpleNamespace(
            **{str(key): _record_to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_record_to_namespace(item) for item in value]
    return value


def _is_numeric_mapping(value: Mapping[str, Any]) -> bool:
    return not value or all(_is_number_like(item) for item in value.values())


def _is_number_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return False
        return True
    return False


def _float_key_if_numeric(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _as_metrics_profile(
    value: ValidationMetrics | MetricsProfile,
    default_name: str,
) -> MetricsProfile:
    if isinstance(value, MetricsProfile):
        return value
    return MetricsProfile(name=default_name, metrics=value)


def _field_comparison_spec(field: str, weight: float) -> MetricComparisonSpec:
    if field == "mean_quote_age":
        return MetricComparisonSpec(field, weight, scale_floor=1.0)
    return MetricComparisonSpec(field, weight)


def compute_metrics(paths: Iterable[Any]) -> ValidationMetrics:
    """Compute validation metrics from step paths.

    ``paths`` may be an iterable of ``StepInfo`` objects for one path, an
    iterable of step iterables, or objects with a ``steps`` attribute. Metrics are
    tick-native and infer ``gap`` from each step's ``price_grid`` unless path
    metadata provides one.
    """

    path_list = _normalize_metric_paths(paths)
    horizons = [len(path.steps) for path in path_list]
    steps = [step for path in path_list for step in path.steps]
    tick_returns = [float(step.tick_change) for step in steps]
    abs_tick_returns = [abs(value) for value in tick_returns]
    abs_tick_returns_by_path = [
        [abs(float(step.tick_change)) for step in path.steps] for path in path_list
    ]
    executed_volumes = [step.total_executed_volume for step in steps]
    cancelled_volumes = [_cancelled_volume(step) for step in steps]
    position_changes = [float(step.tick_after - step.tick_before) for step in steps]
    abs_position_changes = [abs(value) for value in position_changes]
    trade_counts = [float(step.trade_count) for step in steps]
    imbalances = [step.order_flow_imbalance for step in steps]
    quote_ages = [float(getattr(step, "mean_quote_age", 0.0)) for step in steps]
    tick_drawdowns = [_max_tick_drawdown(path) for path in path_list]
    tick_runups = [_max_tick_runup(path) for path in path_list]
    tick_ranges = [_cumulative_tick_range(path) for path in path_list]
    anchor_changes_by_path = [_mdf_anchor_change_ticks(path) for path in path_list]
    anchor_changes = [value for path_values in anchor_changes_by_path for value in path_values]
    abs_anchor_changes = [abs(value) for value in anchor_changes]
    event_pressure_by_path = [_event_pressure(path)[1:] for path in path_list]
    spread_ticks = [
        step.spread_after / _path_gap(path)
        for path in path_list
        for step in path.steps
        if step.spread_after is not None
    ]
    book_topology = [
        _book_topology(step, _path_gap(path))
        for path in path_list
        for step in path.steps
    ]
    near_depth_shares = [item["near_depth_share"] for item in book_topology]
    far_depth_shares = [item["far_depth_share"] for item in book_topology]
    one_sided_book = [item["one_sided_book"] for item in book_topology]

    total_steps = len(steps)
    total_executed_volume = sum(executed_volumes)
    total_cancelled_volume = sum(cancelled_volumes)
    tick_return_std = _population_stddev(tick_returns)
    volume_mean = total_executed_volume / total_steps if total_steps else 0.0
    return ValidationMetrics(
        path_count=len(path_list),
        total_steps=total_steps,
        min_horizon=min(horizons) if horizons else 0,
        max_horizon=max(horizons) if horizons else 0,
        tick_return_mean=_mean(tick_returns),
        tick_return_std=tick_return_std,
        tick_return_tail_ratio=_tail_ratio(tick_returns, tick_return_std),
        volume_mean=volume_mean,
        volume_std=_population_stddev(executed_volumes),
        volatility_clustering_score=_path_weighted_lag1_correlation(abs_tick_returns_by_path),
        max_drawdown_ticks=max(tick_drawdowns) if tick_drawdowns else 0.0,
        max_runup_ticks=max(tick_runups) if tick_runups else 0.0,
        cumulative_tick_range=max(tick_ranges) if tick_ranges else 0.0,
        mean_abs_tick_change=_mean(abs_tick_returns),
        tick_change_volatility=tick_return_std,
        nonzero_tick_change_rate=_rate(abs(value) > 1e-12 for value in tick_returns),
        zero_volume_ratio=_rate(volume <= 1e-12 for volume in executed_volumes),
        total_executed_volume=total_executed_volume,
        mean_executed_volume=volume_mean,
        execution_rate=_rate(volume > 1e-12 for volume in executed_volumes),
        total_cancelled_volume=total_cancelled_volume,
        mean_cancelled_volume=total_cancelled_volume / total_steps if total_steps else 0.0,
        cancellation_rate=_rate(volume > 1e-12 for volume in cancelled_volumes),
        mean_abs_position_change_ticks=_mean(abs_position_changes),
        position_change_rate=_rate(value > 1e-12 for value in abs_position_changes),
        mean_abs_mdf_anchor_change_ticks=_mean(abs_anchor_changes),
        mdf_anchor_change_rate=_rate(value > 1e-12 for value in abs_anchor_changes),
        mdf_anchor_event_pressure_corr=_paired_path_weighted_correlation(
            anchor_changes_by_path,
            event_pressure_by_path,
            absolute_left=True,
        ),
        mean_spread_ticks=_mean(spread_ticks),
        mean_near_depth_share=_mean(near_depth_shares),
        mean_far_depth_share=_mean(far_depth_shares),
        one_sided_book_rate=_rate(one_sided_book),
        mean_trade_count=_mean(trade_counts),
        mean_order_flow_imbalance=_mean(imbalances),
        mean_quote_age=_mean(quote_ages),
    )


def _normalize_metric_paths(paths: Iterable[Any]) -> list[Any]:
    path_list = list(paths)
    if not path_list:
        return []
    if all(_looks_like_step(item) for item in path_list):
        return [_path_namespace(path_list)]
    return [_path_namespace(path) for path in path_list]


def _path_namespace(path: Any) -> Any:
    if hasattr(path, "steps"):
        steps = tuple(path.steps)
        metadata = getattr(path, "metadata", None)
        if metadata is None:
            metadata = SimpleNamespace(config={"gap": _infer_gap(steps)})
        return SimpleNamespace(steps=steps, metadata=metadata)
    steps = tuple(path)
    return SimpleNamespace(steps=steps, metadata=SimpleNamespace(config={"gap": _infer_gap(steps)}))


def _looks_like_step(value: Any) -> bool:
    return hasattr(value, "tick_change") and hasattr(value, "total_executed_volume")


def _path_gap(path: Any) -> float:
    metadata = getattr(path, "metadata", None)
    config = getattr(metadata, "config", {}) if metadata is not None else {}
    gap = float(config.get("gap", _infer_gap(path.steps)))
    return gap if gap > 0.0 else 1.0


def _infer_gap(steps: Sequence[Any]) -> float:
    for step in steps:
        price_grid = getattr(step, "price_grid", None)
        if price_grid and len(price_grid) >= 2:
            gap = abs(float(price_grid[1]) - float(price_grid[0]))
            if gap > 0.0:
                return gap
    return 1.0


def _cancelled_volume(step: Any) -> float:
    return sum(float(volume) for volume in step.cancelled_volume_by_price.values())


def _event_pressure(path: Any) -> list[float]:
    return [
        (step.total_executed_volume + _cancelled_volume(step)) * abs(step.order_flow_imbalance)
        for step in path.steps
    ]


def _mdf_anchor_change_ticks(path: Any) -> list[float]:
    if len(path.steps) < 2:
        return []
    gap = _path_gap(path)
    return [
        (current.mdf_price_basis - previous.mdf_price_basis) / gap
        for previous, current in zip(path.steps, path.steps[1:], strict=False)
    ]


def _book_topology(step: Any, gap: float) -> dict[str, float | bool]:
    bid = step.orderbook_after.bid_volume_by_price
    ask = step.orderbook_after.ask_volume_by_price
    bid_depth = sum(bid.values())
    ask_depth = sum(ask.values())
    near_depth = 0.0
    far_depth = 0.0
    for book in (bid, ask):
        for price, volume in book.items():
            distance = abs(float(price) - step.price_after) / gap
            if distance <= 3.0:
                near_depth += float(volume)
            if distance >= 8.0:
                far_depth += float(volume)
    measured_depth = near_depth + far_depth
    return {
        "near_depth_share": near_depth / measured_depth if measured_depth > 1e-12 else 0.0,
        "far_depth_share": far_depth / measured_depth if measured_depth > 1e-12 else 0.0,
        "one_sided_book": (bid_depth <= 1e-12) != (ask_depth <= 1e-12),
    }


def _cumulative_ticks(path: Any) -> list[float]:
    ticks = [0.0]
    current = 0.0
    for step in path.steps:
        current += float(step.tick_change)
        ticks.append(current)
    return ticks


def _max_tick_drawdown(path: Any) -> float:
    ticks = _cumulative_ticks(path)
    if not ticks:
        return 0.0
    peak = ticks[0]
    max_drawdown_ticks = 0.0
    for tick in ticks:
        peak = max(peak, tick)
        max_drawdown_ticks = max(max_drawdown_ticks, peak - tick)
    return max_drawdown_ticks


def _max_tick_runup(path: Any) -> float:
    ticks = _cumulative_ticks(path)
    if not ticks:
        return 0.0
    trough = ticks[0]
    max_runup_ticks = 0.0
    for tick in ticks:
        trough = min(trough, tick)
        max_runup_ticks = max(max_runup_ticks, tick - trough)
    return max_runup_ticks


def _cumulative_tick_range(path: Any) -> float:
    ticks = _cumulative_ticks(path)
    return max(ticks) - min(ticks) if ticks else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def _paired_path_weighted_correlation(
    left_paths: Sequence[Sequence[float]],
    right_paths: Sequence[Sequence[float]],
    *,
    absolute_left: bool = False,
) -> float:
    weighted_total = 0.0
    total_pairs = 0
    for left, right in zip(left_paths, right_paths, strict=False):
        if absolute_left:
            left = [abs(value) for value in left]
        pair_count = min(len(left), len(right))
        if pair_count < 3:
            continue
        corr = _correlation(left[:pair_count], right[:pair_count])
        weighted_total += corr * pair_count
        total_pairs += pair_count
    return weighted_total / total_pairs if total_pairs else 0.0


def _lag1_correlation(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    return _correlation(left, right)


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 3 or len(right) < 3:
        return 0.0
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
