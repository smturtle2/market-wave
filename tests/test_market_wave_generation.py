from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from market_wave import GeneratedPath, Market, ValidationMetrics, compute_metrics, generate_paths


def _metric_path(tick_changes):
    steps = []
    price = 100.0
    for index, tick_change in enumerate(tick_changes, start=1):
        price_after = price + tick_change
        steps.append(
            SimpleNamespace(
                step_index=index,
                price_before=price,
                price_after=price_after,
                price_change=tick_change,
                tick_change=tick_change,
                total_executed_volume=1.0,
                trade_count=1,
                order_flow_imbalance=0.0,
            )
        )
        price = price_after
    return SimpleNamespace(
        steps=tuple(steps),
        final_price=price,
        metadata=SimpleNamespace(initial_price=100.0),
    )


def test_generate_paths_returns_batch_of_generated_paths():
    paths = generate_paths(
        3,
        8,
        lambda path_id: {
            "initial_price": 100.0,
            "gap": 1.0,
            "popularity": 1.0,
            "seed": 100 + path_id,
            "grid_radius": 8,
            "regime": "auto",
            "augmentation_strength": 0.25,
        },
    )

    assert len(paths) == 3
    assert all(isinstance(path, GeneratedPath) for path in paths)
    assert [path.path_id for path in paths] == [0, 1, 2]
    assert all(len(path.steps) == 8 for path in paths)
    assert all(path.metadata.step_count == 8 for path in paths)
    assert all(path.hidden_states is None for path in paths)
    assert all(path.metadata.config_hash for path in paths)
    assert all(path.metadata.version for path in paths)
    assert all(path.metadata.regime == "auto" for path in paths)
    assert all(path.metadata.augmentation_strength == pytest.approx(0.25) for path in paths)


def test_generate_paths_can_return_iterator_and_hidden_states():
    paths = generate_paths(
        2,
        4,
        {"initial_price": 50.0, "gap": 0.5, "popularity": 1.2, "seed": 42, "grid_radius": 6},
        include_hidden=True,
        as_iterator=True,
    )

    assert not isinstance(paths, list)
    materialized = list(paths)

    assert len(materialized) == 2
    for path in materialized:
        assert path.hidden_states is not None
        assert len(path.hidden_states) == 5
        assert path.hidden_states[0].step_index == 0
        assert path.hidden_states[-1].step_index == 4


def test_static_batch_config_gets_traceable_path_seeds():
    paths = generate_paths(
        3,
        1,
        {"initial_price": 50.0, "gap": 0.5, "popularity": 1.2, "seed": 42, "grid_radius": 6},
    )

    assert [path.metadata.seed for path in paths] == [42, 43, 44]
    assert len({path.metadata.config_hash for path in paths}) == 3


def test_generate_paths_accepts_prebuilt_market_sampler():
    def sampler() -> Market:
        return Market(initial_price=25.0, gap=0.25, popularity=0.8, seed=7, grid_radius=5)

    path = generate_paths(1, 3, sampler)[0]

    assert path.metadata.initial_price == 25.0
    assert path.metadata.seed == 7
    assert len(path.prices) == 3
    assert len(path.returns) == 3


def test_generate_paths_rejects_reusing_one_market_for_multiple_paths():
    market = Market(initial_price=25.0, gap=0.25, popularity=0.8, seed=7, grid_radius=5)

    with pytest.raises(ValueError, match="prebuilt Market"):
        generate_paths(2, 3, market)


def test_config_hash_is_stable_for_equivalent_custom_models():
    class PlainModel:
        def __init__(self, scale):
            self.scale = scale

        def scores(self, side, intent, relative_ticks, context, signals=None):
            del side, intent, context, signals
            weights = [self.scale / (1.0 + abs(tick)) for tick in relative_ticks]
            return weights

    left = generate_paths(1, 0, {"initial_price": 100.0, "gap": 1.0, "mdf_model": PlainModel(2.0)})[
        0
    ]
    right = generate_paths(
        1, 0, {"initial_price": 100.0, "gap": 1.0, "mdf_model": PlainModel(2.0)}
    )[0]

    assert left.metadata.config_hash == right.metadata.config_hash
    json.loads(left.metadata.to_json())


def test_config_hash_captures_slotted_custom_model_state():
    class SlottedModel:
        __slots__ = ("scale",)

        def __init__(self, scale):
            self.scale = scale

        def scores(self, side, intent, relative_ticks, context, signals=None):
            del side, intent, context, signals
            return [self.scale / (1.0 + abs(tick)) for tick in relative_ticks]

    left = generate_paths(
        1, 0, {"initial_price": 100.0, "gap": 1.0, "mdf_model": SlottedModel(1.0)}
    )[0]
    right = generate_paths(
        1, 0, {"initial_price": 100.0, "gap": 1.0, "mdf_model": SlottedModel(2.0)}
    )[0]

    assert left.metadata.config_hash != right.metadata.config_hash
    assert json.loads(left.metadata.to_json())["config"]["mdf_model"]["attrs"]["scale"] == 1.0


def test_prebuilt_market_config_hash_includes_custom_mdf_model_identity():
    class PlainModel:
        def __init__(self, scale):
            self.scale = scale

        def scores(self, side, intent, relative_ticks, context, signals=None):
            del side, intent, context, signals
            return [self.scale / (1.0 + abs(tick)) for tick in relative_ticks]

    left_market = Market(initial_price=100.0, gap=1.0, seed=5, mdf_model=PlainModel(1.0))
    right_market = Market(initial_price=100.0, gap=1.0, seed=5, mdf_model=PlainModel(2.0))

    left = generate_paths(1, 0, left_market)[0]
    right = generate_paths(1, 0, right_market)[0]

    assert left.metadata.config_hash != right.metadata.config_hash


def test_generated_path_records_are_exportable():
    path = generate_paths(1, 2)[0]

    records = path.to_records()
    metadata_payload = json.loads(path.metadata.to_json())

    assert len(records) == 2
    assert {record["path_id"] for record in records} == {0}
    assert metadata_payload["path_id"] == 0
    assert metadata_payload["step_count"] == 2


def test_compute_metrics_summarizes_paths():
    paths = generate_paths(3, 12)

    metrics = compute_metrics(paths)

    assert isinstance(metrics, ValidationMetrics)
    assert metrics.path_count == 3
    assert metrics.total_steps == 36
    assert metrics.min_horizon == 12
    assert metrics.max_horizon == 12
    assert metrics.mean_final_price is not None
    assert metrics.min_price is not None
    assert metrics.max_price is not None
    assert metrics.max_price >= metrics.min_price
    assert metrics.total_executed_volume >= 0.0
    assert metrics.volume_mean >= 0.0
    assert metrics.volume_std >= 0.0
    assert metrics.return_std >= 0.0
    assert metrics.max_drawdown >= 0.0
    assert 0.0 <= metrics.return_tail_ratio <= 1.0
    assert 0.0 <= metrics.price_move_ratio <= 1.0
    assert 0.0 <= metrics.zero_volume_ratio <= 1.0
    assert 0.0 <= metrics.execution_rate <= 1.0
    assert 0.0 <= metrics.nonzero_price_change_rate <= 1.0
    assert json.loads(metrics.to_json())["path_count"] == 3


def test_compute_metrics_volatility_clustering_ignores_path_boundaries():
    paths = [
        _metric_path([1.0, 2.0, 3.0]),
        _metric_path([9.0, 8.0, 7.0]),
    ]

    metrics = compute_metrics(paths)

    assert metrics.volatility_clustering_score == pytest.approx(1.0)


def test_compute_metrics_volatility_clustering_handles_short_paths():
    metrics = compute_metrics([_metric_path([1.0]), _metric_path([])])

    assert metrics.volatility_clustering_score == 0.0


def test_compute_metrics_handles_empty_input():
    metrics = compute_metrics([])

    assert metrics.path_count == 0
    assert metrics.total_steps == 0
    assert metrics.min_horizon == 0
    assert metrics.max_horizon == 0
    assert metrics.return_mean == 0.0
    assert metrics.return_std == 0.0
    assert metrics.zero_volume_ratio == 0.0
    assert metrics.mean_final_price is None
    assert metrics.min_price is None
    assert metrics.max_price is None


def test_to_dataframe_errors_name_dataframe_extra(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("blocked pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    path = generate_paths(1, 1)[0]
    metrics = compute_metrics([path])

    with pytest.raises(ImportError, match=r"market-wave\[dataframe\]"):
        path.to_dataframe()
    with pytest.raises(ImportError, match=r"market-wave\[dataframe\]"):
        metrics.to_dataframe()


def test_generation_validates_inputs():
    with pytest.raises(ValueError, match="n_paths"):
        generate_paths(-1, 1)
    with pytest.raises(ValueError, match="horizon"):
        generate_paths(1, -1)
    with pytest.raises(TypeError, match="config_sampler"):
        generate_paths(1, 1, lambda _: object())
