from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import market_wave as mw
from market_wave.metrics import (
    DEFAULT_COMPARISON_SPECS,
    REFERENCE_ORDERBOOK_REQUIRED_FIELDS,
    REFERENCE_RECORD_OPTIONAL_FIELDS,
    REFERENCE_RECORD_REQUIRED_FIELDS,
    compare_metrics,
    compute_metrics,
    compute_metrics_from_records,
    load_metrics_profile,
    load_metrics_profile_json,
    load_reference_metrics_profile,
    save_metrics_profile_json,
    validate_reference_records,
)


@dataclass(frozen=True)
class _StepPath:
    path_id: int
    steps: tuple[object, ...]
    metadata: object

    @property
    def tick_returns(self) -> tuple[float, ...]:
        return tuple(float(step.tick_change) for step in self.steps)

    def to_records(self) -> list[dict]:
        records = []
        for step in self.steps:
            record = step.to_dict()
            record["path_id"] = self.path_id
            records.append(record)
        return records


def _step_paths(n_paths: int, horizon: int, config: dict | None = None) -> list[_StepPath]:
    paths = []
    for path_id in range(n_paths):
        kwargs = dict(config or {})
        seed = kwargs.get("seed")
        if seed is None:
            kwargs["seed"] = path_id
        elif isinstance(seed, int) and not isinstance(seed, bool):
            kwargs["seed"] = seed + path_id
        market = mw.Market(**kwargs)
        steps = tuple(market.step(horizon))
        metadata = SimpleNamespace(config=_market_config(market))
        paths.append(_StepPath(path_id=path_id, steps=steps, metadata=metadata))
    return paths


def _market_config(market: mw.Market) -> dict:
    config = {
        "initial_price": market.state.price,
        "gap": market.gap,
        "popularity": market.popularity,
        "seed": market.seed,
        "grid_radius": market.grid_radius,
        "initial_regime": market.initial_regime,
        "augmentation_strength": market.augmentation_strength,
    }
    return config


def test_market_constructor_uses_plain_parameters() -> None:
    busy = mw.Market(
        initial_price=250.0,
        gap=5.0,
        popularity=3.0,
        regime="normal",
        seed=77,
    )
    assert busy.state.price == 250.0
    assert busy.gap == 5.0
    assert busy.seed == 77
    assert busy.popularity == 3.0
    assert busy.initial_regime == "normal"

    thin_paths = _step_paths(
        2,
        30,
        {"popularity": 0.25, "regime": "normal", "seed": 88},
    )
    metrics = compute_metrics(thin_paths)
    comparison = compare_metrics(metrics, metrics)

    assert all(path.metadata.config["popularity"] == 0.25 for path in thin_paths)
    assert comparison.score == 0.0


def test_metrics_are_invariant_to_absolute_price_scale() -> None:
    base = _step_paths(
        4,
        200,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.5, "seed": 123},
    )
    scaled = _step_paths(
        4,
        200,
        {"initial_price": 10_000.0, "gap": 100.0, "popularity": 1.5, "seed": 123},
    )

    base_ticks = [step.tick_change for path in base for step in path.steps]
    scaled_ticks = [step.tick_change for path in scaled for step in path.steps]

    assert base_ticks == scaled_ticks
    assert compute_metrics(base).to_dict() == pytest.approx(compute_metrics(scaled).to_dict())


def test_validation_metrics_do_not_expose_price_based_evaluation_keys() -> None:
    paths = _step_paths(
        2,
        50,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 321},
    )
    keys = set(compute_metrics(paths).to_dict())

    assert all("price" not in key for key in keys)
    assert "max_drawdown" not in keys
    assert "max_drawdown_ticks" in keys
    assert "tick_return_std" in keys
    assert "total_cancelled_volume" in keys
    assert "cancellation_rate" in keys
    assert "mean_abs_position_change_ticks" in keys
    assert "position_change_rate" in keys
    assert "mean_abs_mdf_anchor_change_ticks" in keys
    assert "mdf_anchor_change_rate" in keys
    assert "mdf_anchor_event_pressure_corr" in keys
    assert "mean_spread_ticks" in keys
    assert "mean_near_depth_share" in keys
    assert "mean_far_depth_share" in keys
    assert "one_sided_book_rate" in keys
    assert "mean_quote_age" in keys


def test_validation_metrics_expose_cancel_and_position_change_activity() -> None:
    paths = _step_paths(
        3,
        120,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.5, "seed": 654},
    )
    metrics = compute_metrics(paths)
    records = [step for path in paths for step in path.steps]
    cancelled = [sum(step.cancelled_volume_by_price.values()) for step in records]
    position_changes = [abs(step.tick_after - step.tick_before) for step in records]

    assert metrics.total_cancelled_volume == pytest.approx(sum(cancelled))
    assert metrics.mean_cancelled_volume == pytest.approx(sum(cancelled) / len(cancelled))
    assert metrics.cancellation_rate == pytest.approx(
        sum(volume > 1e-12 for volume in cancelled) / len(cancelled)
    )
    assert metrics.mean_abs_position_change_ticks == pytest.approx(
        sum(position_changes) / len(position_changes)
    )
    assert metrics.position_change_rate == pytest.approx(
        sum(change > 1e-12 for change in position_changes) / len(position_changes)
    )


def test_validation_metrics_expose_mdf_anchor_activity() -> None:
    paths = _step_paths(
        3,
        120,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.5, "seed": 765},
    )
    metrics = compute_metrics(paths)
    anchor_changes = [
        abs(current.mdf_price_basis - previous.mdf_price_basis) / path.metadata.config["gap"]
        for path in paths
        for previous, current in zip(path.steps, path.steps[1:], strict=False)
    ]

    assert metrics.mean_abs_mdf_anchor_change_ticks == pytest.approx(
        sum(anchor_changes) / len(anchor_changes)
    )
    assert metrics.mdf_anchor_change_rate == pytest.approx(
        sum(change > 1e-12 for change in anchor_changes) / len(anchor_changes)
    )
    assert -1.0 <= metrics.mdf_anchor_event_pressure_corr <= 1.0


def test_validation_metrics_expose_book_topology_activity() -> None:
    paths = _step_paths(
        3,
        120,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.5, "seed": 876},
    )
    metrics = compute_metrics(paths)
    spread_ticks = [
        step.spread_after / path.metadata.config["gap"]
        for path in paths
        for step in path.steps
        if step.spread_after is not None
    ]
    near_depth_shares = []
    far_depth_shares = []
    one_sided_books = []
    for path in paths:
        gap = path.metadata.config["gap"]
        for step in path.steps:
            bid = step.orderbook_after.bid_volume_by_price
            ask = step.orderbook_after.ask_volume_by_price
            bid_depth = sum(bid.values())
            ask_depth = sum(ask.values())
            near_depth = 0.0
            far_depth = 0.0
            for book in (bid, ask):
                for price, volume in book.items():
                    distance = abs(price - step.price_after) / gap
                    if distance <= 3.0:
                        near_depth += volume
                    if distance >= 8.0:
                        far_depth += volume
            measured_depth = near_depth + far_depth
            near_depth_shares.append(near_depth / measured_depth if measured_depth else 0.0)
            far_depth_shares.append(far_depth / measured_depth if measured_depth else 0.0)
            one_sided_books.append((bid_depth <= 1e-12) != (ask_depth <= 1e-12))

    assert metrics.mean_spread_ticks == pytest.approx(sum(spread_ticks) / len(spread_ticks))
    assert metrics.mean_near_depth_share == pytest.approx(
        sum(near_depth_shares) / len(near_depth_shares)
    )
    assert metrics.mean_far_depth_share == pytest.approx(
        sum(far_depth_shares) / len(far_depth_shares)
    )
    assert metrics.one_sided_book_rate == pytest.approx(
        sum(one_sided_books) / len(one_sided_books)
    )


def test_validation_metrics_expose_quote_age_lifecycle() -> None:
    paths = _step_paths(
        3,
        120,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.5, "seed": 246},
    )
    metrics = compute_metrics(paths)
    quote_ages = [step.mean_quote_age for path in paths for step in path.steps]

    assert metrics.mean_quote_age == pytest.approx(sum(quote_ages) / len(quote_ages))
    assert metrics.mean_quote_age >= 0.0
    assert any(age > 0.0 for age in quote_ages)


def test_step_path_exposes_tick_returns_not_price_returns() -> None:
    path = _step_paths(
        1,
        50,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 456},
    )[0]

    assert not hasattr(path, "returns")
    assert path.tick_returns == tuple(float(step.tick_change) for step in path.steps)


def test_metric_comparison_is_zero_for_identical_profiles() -> None:
    paths = _step_paths(
        3,
        120,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.5, "seed": 987},
    )
    metrics = compute_metrics(paths)
    comparison = compare_metrics(metrics, metrics)

    assert comparison.total_distance == 0.0
    assert all(distance == 0.0 for distance in comparison.field_distances.values())
    assert set(comparison.field_deltas) == set(comparison.field_distances)
    assert comparison.score == 0.0


def test_metric_comparison_detects_different_market_profiles() -> None:
    reference = compute_metrics(
        _step_paths(
            3,
            120,
            {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 111},
        )
    )
    candidate = compute_metrics(
        _step_paths(
            3,
            120,
            {"initial_price": 100.0, "gap": 1.0, "popularity": 3.0, "seed": 111},
        )
    )
    comparison = compare_metrics(candidate, reference)

    assert comparison.total_distance > 0.0
    assert comparison.field_distances["mean_executed_volume"] > 0.0
    assert comparison.field_deltas["mean_executed_volume"] > 0.0


def test_generated_market_families_have_distinct_validation_profiles() -> None:
    def profile(config: dict) -> object:
        return compute_metrics(_step_paths(4, 220, config))

    normal = profile({"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 900})
    busy = profile({"initial_price": 100.0, "gap": 1.0, "popularity": 3.0, "seed": 900})
    thin = profile({"initial_price": 100.0, "gap": 1.0, "popularity": 0.25, "seed": 900})
    high_vol = profile(
        {
            "initial_price": 100.0,
            "gap": 1.0,
            "popularity": 1.0,
            "seed": 900,
            "regime": "high_vol",
        }
    )
    trend_up = profile(
        {
            "initial_price": 100.0,
            "gap": 1.0,
            "popularity": 1.0,
            "seed": 900,
            "regime": "trend_up",
        }
    )
    trend_down = profile(
        {
            "initial_price": 100.0,
            "gap": 1.0,
            "popularity": 1.0,
            "seed": 900,
            "regime": "trend_down",
        }
    )
    inactive = profile({"initial_price": 100.0, "gap": 1.0, "popularity": 0.0, "seed": 900})

    assert busy.mean_executed_volume > normal.mean_executed_volume * 3.0
    assert busy.mean_trade_count > normal.mean_trade_count * 2.5
    assert busy.zero_volume_ratio < normal.zero_volume_ratio
    assert busy.mean_near_depth_share > normal.mean_near_depth_share

    assert thin.mean_executed_volume < normal.mean_executed_volume * 0.10
    assert thin.execution_rate < normal.execution_rate * 0.35
    assert thin.zero_volume_ratio > normal.zero_volume_ratio * 3.0
    assert thin.mean_spread_ticks > normal.mean_spread_ticks * 3.0

    assert high_vol.tick_return_std > normal.tick_return_std * 1.15
    assert high_vol.mean_abs_tick_change > normal.mean_abs_tick_change * 1.25

    assert trend_up.tick_return_mean > normal.tick_return_mean
    assert trend_up.tick_return_mean > trend_down.tick_return_mean
    assert abs(trend_up.tick_return_mean - trend_down.tick_return_mean) > 0.15

    assert inactive.execution_rate == 0.0
    assert inactive.mean_executed_volume == 0.0
    assert inactive.mean_trade_count == 0.0
    assert inactive.mean_abs_tick_change == 0.0
    assert inactive.zero_volume_ratio == 1.0


def test_metric_comparison_supports_field_weights() -> None:
    reference = compute_metrics(
        _step_paths(
            2,
            80,
            {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 222},
        )
    )
    candidate = compute_metrics(
        _step_paths(
            2,
            80,
            {"initial_price": 100.0, "gap": 1.0, "popularity": 2.0, "seed": 222},
        )
    )
    comparison = compare_metrics(
        candidate,
        reference,
        fields=("volume_mean", "execution_rate"),
        weights={"volume_mean": 2.0, "execution_rate": 0.0},
    )

    assert set(comparison.field_distances) == {"volume_mean"}
    assert comparison.total_distance == comparison.field_distances["volume_mean"]
    assert comparison.rows[0].contribution == pytest.approx(
        comparison.rows[0].normalized_error * comparison.rows[0].weight
    )


def test_metric_comparison_is_invariant_to_absolute_price_scale() -> None:
    base = compute_metrics(
        _step_paths(
            3,
            120,
            {"initial_price": 100.0, "gap": 1.0, "popularity": 1.5, "seed": 333},
        )
    )
    scaled = compute_metrics(
        _step_paths(
            3,
            120,
            {"initial_price": 10_000.0, "gap": 100.0, "popularity": 1.5, "seed": 333},
        )
    )

    assert compare_metrics(scaled, base).score == pytest.approx(0.0)


def test_default_comparison_specs_are_tick_native_and_valid_metric_fields() -> None:
    paths = _step_paths(
        1,
        20,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 444},
    )
    metric_keys = set(compute_metrics(paths).to_dict())

    assert all("price" not in spec.field for spec in DEFAULT_COMPARISON_SPECS)
    assert all(spec.field in metric_keys for spec in DEFAULT_COMPARISON_SPECS)


def test_load_metrics_profile_rejects_missing_required_fields() -> None:
    paths = _step_paths(
        1,
        20,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 555},
    )
    data = compute_metrics(paths).to_dict()
    data.pop("tick_return_std")

    with pytest.raises(ValueError, match="tick_return_std"):
        load_metrics_profile(data)


def test_load_metrics_profile_allows_missing_optional_quote_age_field() -> None:
    paths = _step_paths(
        1,
        20,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 556},
    )
    data = compute_metrics(paths).to_dict()
    data.pop("mean_quote_age")

    profile = load_metrics_profile(data)

    assert profile.metrics.mean_quote_age == 0.0


def test_explicit_quote_age_comparison_uses_step_scale_floor() -> None:
    paths = _step_paths(
        2,
        120,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.5, "seed": 557},
    )
    metrics = compute_metrics(paths)
    data = metrics.to_dict()
    data.pop("mean_quote_age")
    old_profile = load_metrics_profile(data)

    comparison = compare_metrics(metrics, old_profile, fields=("mean_quote_age",))

    assert metrics.mean_quote_age > 0.0
    assert comparison.rows[0].normalized_error == pytest.approx(metrics.mean_quote_age)


def test_load_metrics_profile_can_feed_metric_comparison() -> None:
    paths = _step_paths(
        1,
        50,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 666},
    )
    metrics = compute_metrics(paths)
    profile = load_metrics_profile(metrics.to_dict(), name="reference")
    comparison = compare_metrics(metrics, profile)

    assert comparison.reference.name == "reference"
    assert comparison.score == 0.0


def test_compute_metrics_from_records_matches_step_records() -> None:
    paths = _step_paths(
        2,
        40,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 667},
    )
    records = [record for path in paths for record in path.to_records()]

    assert compute_metrics_from_records(records).to_dict() == pytest.approx(
        compute_metrics(paths).to_dict()
    )


def test_reference_metrics_profile_round_trips_json(tmp_path) -> None:
    paths = _step_paths(
        1,
        30,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 668},
    )
    metrics = compute_metrics(paths)
    profile_path = tmp_path / "reference-metrics.json"

    save_metrics_profile_json(metrics, profile_path, name="real-reference")
    profile = load_metrics_profile_json(profile_path)

    assert profile.name == "real-reference"
    assert profile.metrics.to_dict() == pytest.approx(metrics.to_dict())


def test_load_reference_metrics_profile_from_jsonl_records(tmp_path) -> None:
    paths = _step_paths(
        2,
        30,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 669},
    )
    records = [record for path in paths for record in path.to_records()]
    record_path = tmp_path / "reference-records.jsonl"
    record_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    profile = load_reference_metrics_profile(record_path, name="jsonl-reference")

    assert profile.name == "jsonl-reference"
    assert profile.metrics.to_dict() == pytest.approx(compute_metrics(paths).to_dict())


def test_load_reference_metrics_profile_from_csv_records(tmp_path) -> None:
    paths = _step_paths(
        1,
        20,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 670},
    )
    records = paths[0].to_records()
    record_path = tmp_path / "reference-records.csv"
    with record_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, dict | list) else value
                    for key, value in record.items()
                }
            )

    profile = load_reference_metrics_profile(record_path, name="csv-reference")

    assert profile.name == "csv-reference"
    assert profile.metrics.to_dict() == pytest.approx(compute_metrics(paths).to_dict())


def test_compute_metrics_from_records_rejects_missing_required_fields() -> None:
    paths = _step_paths(
        1,
        5,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 671},
    )
    record = paths[0].to_records()[0]
    record.pop("orderbook_after")

    with pytest.raises(ValueError, match="orderbook_after"):
        compute_metrics_from_records([record])


def test_reference_record_schema_constants_match_loader_contract() -> None:
    paths = _step_paths(
        1,
        5,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 672},
    )
    record = paths[0].to_records()[0]

    assert set(REFERENCE_RECORD_REQUIRED_FIELDS).issubset(record)
    assert set(REFERENCE_ORDERBOOK_REQUIRED_FIELDS).issubset(record["orderbook_after"])
    assert "path_id" in REFERENCE_RECORD_OPTIONAL_FIELDS
    assert "mean_quote_age" in REFERENCE_RECORD_OPTIONAL_FIELDS
    for field in REFERENCE_RECORD_REQUIRED_FIELDS:
        partial = dict(record)
        partial.pop(field)
        with pytest.raises(ValueError, match=field):
            compute_metrics_from_records([partial])

    for field in REFERENCE_ORDERBOOK_REQUIRED_FIELDS:
        partial = dict(record)
        partial["orderbook_after"] = dict(record["orderbook_after"])
        partial["orderbook_after"].pop(field)
        with pytest.raises(ValueError, match=f"orderbook_after.{field}"):
            compute_metrics_from_records([partial])


def test_compute_metrics_from_records_rejects_malformed_orderbook_record() -> None:
    paths = _step_paths(
        1,
        5,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 673},
    )
    record = paths[0].to_records()[0]
    record["orderbook_after"] = "not an orderbook"

    with pytest.raises(ValueError, match="orderbook_after must be a mapping"):
        compute_metrics_from_records([record])


def test_validate_reference_records_checks_schema_without_computing_metrics() -> None:
    paths = _step_paths(
        1,
        5,
        {"initial_price": 100.0, "gap": 1.0, "popularity": 1.0, "seed": 674},
    )
    records = paths[0].to_records()

    validate_reference_records(records)

    invalid = dict(records[0])
    invalid["orderbook_after"] = {"bid_volume_by_price": {}}
    with pytest.raises(ValueError, match="orderbook_after.ask_volume_by_price"):
        validate_reference_records([invalid])
