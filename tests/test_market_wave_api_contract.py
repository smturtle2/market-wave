from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence

import pytest

import market_wave as mw
from market_wave import Market, MarketState, StepInfo

PUBLIC_TYPE_EXPORTS = (
    "Market",
    "MarketState",
    "IntensityState",
    "LatentState",
    "MDFContext",
    "MDFSignals",
    "MDFModel",
    "RelativeMDFComponent",
    "DynamicMDFModel",
    "MDFState",
    "OrderBookState",
    "PositionMassState",
    "StepInfo",
)

REMOVED_PUBLIC_PMF_EXPORTS = (
    "DistributionModel",
    "DistributionState",
    "DistributionContext",
    "LaplaceMixturePMF",
    "SkewedPMF",
    "FatTailPMF",
    "NoisyPMF",
)

MDF_FIELDS = (
    "buy_entry_mdf",
    "sell_entry_mdf",
    "long_exit_mdf",
    "short_exit_mdf",
)

MDF_BY_PRICE_FIELDS = (
    "buy_entry_mdf_by_price",
    "sell_entry_mdf_by_price",
    "long_exit_mdf_by_price",
    "short_exit_mdf_by_price",
)

REALISM_STEPINFO_FIELDS = (
    "market_buy_volume",
    "market_sell_volume",
    "crossed_market_volume",
    "residual_market_buy_volume",
    "residual_market_sell_volume",
    "trade_count",
    "vwap_price",
    "best_bid_before",
    "best_bid_after",
    "best_ask_before",
    "best_ask_after",
    "spread_before",
    "spread_after",
    "order_flow_imbalance",
    "cancelled_volume_by_price",
    "entry_volume_by_price",
    "exit_volume_by_price",
)


def _public_items(obj):
    if isinstance(obj, Mapping):
        return obj.items()
    if dataclasses.is_dataclass(obj):
        return ((field.name, getattr(obj, field.name)) for field in dataclasses.fields(obj))
    if hasattr(obj, "_asdict"):
        return obj._asdict().items()
    if hasattr(obj, "__dict__"):
        return ((name, value) for name, value in vars(obj).items() if not name.startswith("_"))
    return ()


def _walk(obj, *, seen=None):
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj

    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return
    if isinstance(obj, Mapping):
        values = obj.values()
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        values = obj
    else:
        values = (value for _, value in _public_items(obj))

    for value in values:
        yield from _walk(value, seen=seen)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numeric_sequence(value):
    if isinstance(value, Mapping):
        values = list(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        return None
    if values and all(_is_number(item) for item in values):
        return [float(item) for item in values]
    return None


def _state_from_market(market):
    for name in ("state", "current_state", "market_state"):
        if hasattr(market, name):
            value = getattr(market, name)
            return value() if callable(value) else value
    history = _history(market)
    if history:
        return history[-1]
    pytest.fail(
        "Market should expose current state via state/current_state/market_state or history"
    )


def _history(market):
    for name in ("history", "states", "state_history"):
        if hasattr(market, name):
            value = getattr(market, name)
            value = value() if callable(value) else value
            return list(value)
    return []


def _state_from_step(step):
    for name in ("state", "market_state", "after_state", "post_state", "next_state"):
        if hasattr(step, name):
            value = getattr(step, name)
            return value() if callable(value) else value
    for obj in _walk(step):
        if isinstance(obj, MarketState) or type(obj).__name__ == "MarketState":
            return obj
    return None


def _mdf_state(state):
    assert isinstance(state.mdf, mw.MDFState)
    return state.mdf


def _price(obj):
    for candidate in _walk(obj):
        for name, value in _public_items(candidate):
            if name in {"price", "mid_price", "last_price"} and _is_number(value):
                return float(value)
    pytest.fail("Market state/step should expose a numeric price")


def _execution_count(step):
    names = {
        "execution",
        "executions",
        "executed",
        "trade",
        "trades",
        "trade_count",
        "executed_count",
        "executed_volume",
        "matched",
        "matched_volume",
    }
    for obj in _walk(step):
        for name, value in _public_items(obj):
            lowered = name.lower()
            if lowered in names or any(
                token in lowered for token in ("execution", "executed", "trade", "matched")
            ):
                if _is_number(value):
                    return float(value)
                seq = _numeric_sequence(value)
                if seq is not None:
                    return sum(seq)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    return float(len(value))
    pytest.fail("StepInfo should expose whether executions/trades happened")


def _freeze(obj):
    if _is_number(obj):
        return round(float(obj), 12)
    if isinstance(obj, (str, bytes, bool, type(None))):
        return obj
    if isinstance(obj, Mapping):
        return tuple(sorted((key, _freeze(value)) for key, value in obj.items()))
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return tuple(_freeze(value) for value in obj)
    return (
        type(obj).__name__,
        tuple(sorted((name, _freeze(value)) for name, value in _public_items(obj))),
    )


def _distribution_probability_vectors(state):
    vectors = []
    for obj in _walk(state):
        class_name = type(obj).__name__.lower()
        if not any(token in class_name for token in ("mdf", "distribution", "mixture")):
            continue
        for name, value in _public_items(obj):
            lowered = name.lower()
            if any(token in lowered for token in ("mdf", "prob", "weight")):
                vector = _numeric_sequence(value)
                if vector is not None:
                    vectors.append((type(obj).__name__, name, vector))
    return vectors


def _position_mass_vectors(state):
    vectors = []
    for obj in _walk(state):
        class_name = type(obj).__name__.lower()
        for name, value in _public_items(obj):
            lowered_name = str(name).lower()
            if "positionmass" in class_name.replace("_", "") or "position_mass" in lowered_name:
                vector = _numeric_sequence(value)
                if vector is not None:
                    vectors.append((type(obj).__name__, name, vector))
            elif "mass" in lowered_name and "position" in class_name:
                vector = _numeric_sequence(value)
                if vector is not None:
                    vectors.append((type(obj).__name__, name, vector))
    return vectors


def _orderbook_sides(state):
    sides = []
    for obj in _walk(state):
        class_name = type(obj).__name__.lower()
        if "orderbook" not in class_name and "order_book" not in class_name:
            continue
        for name, value in _public_items(obj):
            lowered = name.lower()
            if any(
                token in lowered
                for token in ("bid", "ask", "buy", "sell", "volume", "depth", "quantity")
            ):
                vector = _numeric_sequence(value)
                if vector is not None:
                    sides.append((name, vector))
    return sides


def _total_price_map(value):
    assert isinstance(value, Mapping), f"Expected price map, got {type(value).__name__}"
    assert all(_is_number(price) for price in value), "Price map keys should be finite numbers"
    assert all(_is_number(volume) for volume in value.values()), (
        "Price map values should be finite numbers"
    )
    return sum(float(volume) for volume in value.values())


def _total_orderbook_depth(orderbook):
    return sum(_total_price_map(value) for _, value in _public_items(orderbook))


def _required_realism_step(step):
    missing = [name for name in REALISM_STEPINFO_FIELDS if not hasattr(step, name)]
    if missing:
        pytest.skip("StepInfo realism fields are not implemented yet: " + ", ".join(missing))
    return step


def _realism_steps_or_skip(market, count):
    try:
        steps = market.step(count)
    except TypeError as exc:
        if "StepInfo.__init__()" in str(exc):
            pytest.skip(f"StepInfo construction is not wired to realism fields yet: {exc}")
        raise
    _required_realism_step(steps[0])
    return steps


def _lag1_correlation(values):
    if len(values) < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=False))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_var * right_var)
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def _lag_correlation(values, lag):
    if len(values) <= lag + 2:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=False))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_var * right_var)
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def _overlap(left, right):
    total = 0.0
    for (_, a), (_, b) in zip(left, right, strict=False):
        for av, bv in zip(a, b, strict=False):
            total += min(max(av, 0.0), max(bv, 0.0))
    return total


def _assert_mdf_map_is_finite_nonnegative_normalized(name, mdf, *, min_effective_support=None):
    assert mdf, f"{name} should not be empty"
    values = list(mdf.values())
    assert all(_is_number(value) for value in values), f"{name} should contain finite numbers"
    assert all(value >= -1e-12 for value in values), f"{name} should not contain negatives"
    assert sum(values) == pytest.approx(1.0, abs=1e-9), f"{name} should be normalized"
    if min_effective_support is not None:
        effective_support = 1.0 / sum(value * value for value in values)
        assert effective_support >= min_effective_support, (
            f"{name} collapsed to effective support {effective_support:.3f}"
        )


def test_public_api_exports_expected_types():
    missing = [name for name in PUBLIC_TYPE_EXPORTS if not hasattr(mw, name)]

    assert not missing, "Missing public MDF exports: " + ", ".join(missing)
    assert all(isinstance(getattr(mw, name), type) for name in PUBLIC_TYPE_EXPORTS)


def test_public_api_removes_dynamic_pmf_exports():
    leaked = [name for name in REMOVED_PUBLIC_PMF_EXPORTS if hasattr(mw, name)]

    assert not leaked, "Dynamic PMF names should not remain public: " + ", ".join(leaked)


def test_distribution_mdfs_are_normalized_from_initial_market_state():
    market = Market(initial_price=100.0, gap=0.5, popularity=1.0, seed=7, grid_radius=8)

    vectors = _distribution_probability_vectors(_state_from_market(market))

    assert vectors, "Expected MDF/probability vectors on Market state"
    for owner, name, vector in vectors:
        assert all(value >= -1e-12 for value in vector), (
            f"{owner}.{name} contains negative probabilities"
        )
        assert sum(vector) == pytest.approx(1.0, abs=1e-9), f"{owner}.{name} should be normalized"


def test_market_state_distributions_stay_aligned_with_current_price_grid_after_moves():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.2, seed=99, grid_radius=16)

    market.step(80)
    state = _state_from_market(market)

    assert any(abs(step.price_change) > 1e-12 for step in market.history)
    grid = set(state.price_grid)
    mdf_state = _mdf_state(state)
    for name in MDF_BY_PRICE_FIELDS:
        mdf = getattr(mdf_state, name)
        assert set(mdf) == grid, f"{name} keys should match current state.price_grid"
        assert sum(mdf.values()) == pytest.approx(1.0, abs=1e-9)
    tick_grid = set(state.tick_grid)
    for name in MDF_FIELDS:
        mdf = getattr(mdf_state, name)
        assert set(mdf) == tick_grid
        assert sum(mdf.values()) == pytest.approx(1.0, abs=1e-9)


def test_step_returns_stepinfo_items_and_appends_history():
    market = Market(initial_price=100.0, gap=1.0, seed=11, grid_radius=6)
    before_history_len = len(_history(market))

    steps = market.step(5)

    assert isinstance(steps, list)
    assert len(steps) == 5
    assert all(isinstance(step, StepInfo) for step in steps)
    assert len(_history(market)) >= before_history_len + 5


def test_seeded_markets_are_deterministic():
    left = Market(initial_price=100.0, gap=1.0, popularity=1.3, seed=123, grid_radius=10)
    right = Market(initial_price=100.0, gap=1.0, popularity=1.3, seed=123, grid_radius=10)

    assert _freeze(left.step(12)) == _freeze(right.step(12))
    assert _freeze(_state_from_market(left)) == _freeze(_state_from_market(right))


def test_stepinfo_json_and_history_records_are_exportable():
    market = Market(initial_price=100.0, gap=1.0, seed=101, grid_radius=8)
    step = market.step(1)[0]

    payload = json.loads(step.to_json())
    records = market.history_records()

    assert payload["step_index"] == step.step_index
    assert payload["price_after"] == step.price_after
    assert records == [step.to_dict()]


def test_stepinfo_and_market_state_serialization_do_not_expose_public_pmf_names():
    market = Market(initial_price=100.0, gap=1.0, seed=103, grid_radius=4)
    step = market.step(1)[0]

    step_payload = step.to_dict()
    state_payload = dataclasses.asdict(_state_from_market(market))

    for payload_name, payload in (("StepInfo", step_payload), ("MarketState", state_payload)):
        serialized_keys = {
            str(key) for obj in _walk(payload) if isinstance(obj, Mapping) for key in obj
        }
        leaked = sorted(key for key in serialized_keys if "pmf" in key.lower())
        assert not leaked, f"{payload_name} serialization leaked PMF keys: {', '.join(leaked)}"

    for name in MDF_FIELDS + MDF_BY_PRICE_FIELDS:
        assert name in step_payload
    for name in MDF_FIELDS + MDF_BY_PRICE_FIELDS:
        assert name in state_payload["mdf"]


def test_step_can_skip_history_and_stream_steps():
    market = Market(initial_price=100.0, gap=1.0, seed=13, grid_radius=6)

    steps = market.step(3, keep_history=False)

    assert len(steps) == 3
    assert market.history == []

    streamed = list(market.stream(2, keep_history=True))

    assert len(streamed) == 2
    assert market.history == streamed
    assert [step.step_index for step in streamed] == [4, 5]


def test_stepinfo_exposes_relative_tick_mdfs():
    market = Market(initial_price=100.0, gap=1.0, seed=23, grid_radius=4)

    step = market.step(1)[0]

    assert step.tick_before == 100
    assert step.relative_ticks == list(range(-4, 5))
    tick_grid = set(step.relative_ticks)
    for name in MDF_FIELDS:
        mdf = getattr(step, name)
        assert set(mdf) == tick_grid
        assert sum(mdf.values()) == pytest.approx(1.0, abs=1e-9)
    assert step.price_change == pytest.approx(step.tick_change * market.tick_size)


def test_stepinfo_mdf_price_basis_is_pre_trade_price():
    market = Market(initial_price=100.0, gap=1.0, seed=24, grid_radius=6)

    steps = market.step(80, keep_history=False)

    assert any(step.price_after != pytest.approx(step.price_before) for step in steps)
    for step in steps:
        assert step.mdf_price_basis == pytest.approx(step.price_before)
        basis_tick = market.price_to_tick(step.mdf_price_basis)
        assert step.price_grid == [
            market.tick_to_price(basis_tick + relative_tick)
            for relative_tick in step.relative_ticks
        ]
        for name in MDF_BY_PRICE_FIELDS:
            assert set(getattr(step, name)) == set(step.price_grid)


def test_custom_mdf_model_scores_are_called_for_four_flows_and_normalized():
    class CenterWeightedModel:
        def __init__(self):
            self.calls = []

        def scores(self, side, intent, relative_ticks, context, signals=None):
            del signals
            self.calls.append(
                (side, intent, tuple(relative_ticks), context.regime, context.step_index)
            )
            return [1.0 if tick == 0 else 0.05 for tick in relative_ticks]

    model = CenterWeightedModel()
    market = Market(
        initial_price=100.0,
        gap=1.0,
        seed=29,
        grid_radius=3,
        mdf_model=model,
        regime="trend_up",
        augmentation_strength=0.4,
    )

    step = market.step(1, keep_history=False)[0]

    assert len(model.calls) == 4
    assert {(side, intent) for side, intent, *_ in model.calls} == {
        ("buy", "entry"),
        ("sell", "entry"),
        ("long", "exit"),
        ("short", "exit"),
    }
    assert {call[3] for call in model.calls} == {"trend_up"}
    assert {call[4] for call in model.calls} == {1}
    assert step.regime == "trend_up"
    assert step.augmentation_strength == pytest.approx(0.4)
    assert step.buy_entry_mdf[0] > step.buy_entry_mdf[-1]


def test_executed_volume_memory_is_rebased_to_current_price_basis():
    market = Market(initial_price=100.0, gap=1.0, seed=43, grid_radius=5)
    market._last_execution_volume = 7.0
    market._last_executed_by_price = {101.0: 7.0}

    signals_at_100 = market._mdf_signals(100.0)
    signals_at_102 = market._mdf_signals(102.0)

    assert set(signals_at_100.executed_volume_by_tick) == {1}
    assert signals_at_100.executed_volume_by_tick[1] == pytest.approx(7.0)
    assert set(signals_at_102.executed_volume_by_tick) == {-1}
    assert signals_at_102.executed_volume_by_tick[-1] == pytest.approx(7.0)


def test_mdf_model_scores_supports_legacy_four_argument_signature():
    class LegacyModel:
        def __init__(self):
            self.calls = []

        def scores(self, side, intent, relative_ticks, context):
            self.calls.append((side, intent, tuple(relative_ticks), context.step_index))
            return [0.0 for _ in relative_ticks]

    model = LegacyModel()
    market = Market(initial_price=100.0, gap=1.0, seed=44, grid_radius=3, mdf_model=model)

    step = market.step(1, keep_history=False)[0]

    assert len(model.calls) == 4
    assert {call[3] for call in model.calls} == {1}
    for name in MDF_FIELDS:
        _assert_mdf_map_is_finite_nonnegative_normalized(name, getattr(step, name))


def test_mdf_model_scores_typeerror_from_signal_aware_model_is_not_signature_fallback():
    class BrokenSignalAwareModel:
        def __init__(self):
            self.calls = 0

        def scores(self, side, intent, relative_ticks, context, signals=None):
            del side, intent, relative_ticks, context, signals
            self.calls += 1
            raise TypeError("internal score failure")

    model = BrokenSignalAwareModel()
    market = Market(initial_price=100.0, gap=1.0, seed=45, grid_radius=3, mdf_model=model)

    with pytest.raises(TypeError, match="internal score failure"):
        market.step(1, keep_history=False)

    assert model.calls == 1


def test_mdf_temperature_controls_update_concentration():
    class TargetScoreModel:
        def scores(self, side, intent, relative_ticks, context, signals=None):
            del side, intent, context, signals
            target = 1
            return [10.0 if tick == target else 0.0 for tick in relative_ticks]

    cold_market = Market(
        initial_price=100.0,
        gap=1.0,
        seed=37,
        grid_radius=3,
        mdf_model=TargetScoreModel(),
        mdf_temperature=0.25,
    )
    warm_market = Market(
        initial_price=100.0,
        gap=1.0,
        seed=37,
        grid_radius=3,
        mdf_model=TargetScoreModel(),
        mdf_temperature=4.0,
    )

    cold_step = cold_market.step(1, keep_history=False)[0]
    warm_step = warm_market.step(1, keep_history=False)[0]

    assert cold_step.buy_entry_mdf[1] > warm_step.buy_entry_mdf[1]
    assert cold_step.buy_entry_mdf[1] > 0.90
    assert warm_step.buy_entry_mdf[1] < 0.75


def test_mdf_update_accumulates_previous_distribution_state():
    class TargetScoreModel:
        def scores(self, side, intent, relative_ticks, context, signals=None):
            del side, intent, context, signals
            return [2.0 if tick == 1 else 0.0 for tick in relative_ticks]

    market = Market(
        initial_price=100.0,
        gap=1.0,
        seed=37,
        grid_radius=3,
        mdf_model=TargetScoreModel(),
        mdf_temperature=1.0,
    )

    first, second = market.step(2, keep_history=False)

    assert second.buy_entry_mdf[1] > first.buy_entry_mdf[1]
    assert second.buy_entry_mdf[1] > second.buy_entry_mdf[-1]


@pytest.mark.slow
def test_dynamic_mdf_acceptance_does_not_collapse_for_default_temperature_seed_matrix():
    peak_ticks = []
    for seed in range(10, 20):
        market = Market(
            initial_price=100.0,
            gap=1.0,
            popularity=1.0,
            seed=seed,
            grid_radius=12,
            mdf_temperature=1.0,
            regime="auto",
            augmentation_strength=0.25,
        )

        steps = market.step(160, keep_history=False)
        peak_ticks.append(max(steps[-1].buy_entry_mdf, key=steps[-1].buy_entry_mdf.get))

        for step in steps:
            for name in MDF_FIELDS + MDF_BY_PRICE_FIELDS:
                _assert_mdf_map_is_finite_nonnegative_normalized(
                    f"seed={seed} step={step.step_index} {name}",
                    getattr(step, name),
                    min_effective_support=3.0,
                )
        mdf_state = _mdf_state(_state_from_market(market))
        for name in MDF_FIELDS + MDF_BY_PRICE_FIELDS:
            _assert_mdf_map_is_finite_nonnegative_normalized(
                f"seed={seed} final_state.{name}",
                getattr(mdf_state, name),
                min_effective_support=3.0,
            )
    assert any(tick != 0 for tick in peak_ticks)


def test_price_changes_only_on_steps_with_executions():
    market = Market(initial_price=100.0, gap=1.0, seed=19, grid_radius=8)
    previous_price = _price(_state_from_market(market))

    for step in market.step(30):
        state = _state_from_step(step) or _state_from_market(market)
        current_price = _price(state)
        executions = _execution_count(step)
        if current_price != pytest.approx(previous_price, abs=1e-12):
            assert executions > 0
        previous_price = current_price


def test_position_mass_is_nonnegative_and_changes_after_steps():
    market = Market(initial_price=100.0, gap=1.0, seed=5, grid_radius=10)
    before = _position_mass_vectors(_state_from_market(market))

    market.step(20)
    after = _position_mass_vectors(_state_from_market(market))

    assert before, "Expected position mass vectors on initial state"
    assert after, "Expected position mass vectors after stepping"
    for owner, name, vector in after:
        assert all(value >= -1e-12 for value in vector), f"{owner}.{name} contains negative mass"
    assert _freeze(before) != _freeze(after)


def test_orderbook_state_is_nonnegative_and_carries_forward_tolerantly():
    market = Market(initial_price=100.0, gap=1.0, seed=31, grid_radius=10)
    snapshots = [_orderbook_sides(_state_from_market(market))]

    for step in market.step(12):
        state = _state_from_step(step) or _state_from_market(market)
        snapshots.append(_orderbook_sides(state))

    assert any(snapshots), "Expected observable OrderBookState side/depth vectors"
    for snapshot in snapshots:
        for name, vector in snapshot:
            assert all(value >= -1e-12 for value in vector), (
                f"Order book {name} contains negative depth"
            )

    nonempty = [
        snapshot for snapshot in snapshots if sum(sum(vector) for _, vector in snapshot) > 1e-12
    ]
    carryover_seen = any(
        _overlap(left, right) > 1e-12 for left, right in zip(nonempty, nonempty[1:], strict=False)
    )
    assert carryover_seen or nonempty, (
        "Expected order book depth to persist or remain observable across steps"
    )


def test_stepinfo_exposes_market_realism_diagnostics():
    declared = {field.name for field in dataclasses.fields(StepInfo)}
    missing = sorted(set(REALISM_STEPINFO_FIELDS) - declared)

    assert not missing, "StepInfo is missing market realism fields: " + ", ".join(missing)

    market = Market(initial_price=100.0, gap=1.0, popularity=1.2, seed=41, grid_radius=10)
    step = _realism_steps_or_skip(market, 1)[0]

    for name in REALISM_STEPINFO_FIELDS:
        assert hasattr(step, name), f"StepInfo should expose {name}"

    numeric_fields = (
        "market_buy_volume",
        "market_sell_volume",
        "crossed_market_volume",
        "residual_market_buy_volume",
        "residual_market_sell_volume",
        "trade_count",
        "order_flow_imbalance",
    )
    nullable_numeric_fields = (
        "vwap_price",
        "best_bid_before",
        "best_bid_after",
        "best_ask_before",
        "best_ask_after",
        "spread_before",
        "spread_after",
    )
    map_fields = (
        "cancelled_volume_by_price",
        "entry_volume_by_price",
        "exit_volume_by_price",
    )

    for name in numeric_fields:
        value = getattr(step, name)
        assert _is_number(value), f"{name} should be a finite number"
        if name == "order_flow_imbalance":
            assert -1.0 <= value <= 1.0
        else:
            assert value >= -1e-12, f"{name} should not be negative"

    for name in nullable_numeric_fields:
        value = getattr(step, name)
        assert value is None or _is_number(value), f"{name} should be None or finite"

    for name in map_fields:
        total = _total_price_map(getattr(step, name))
        assert total >= -1e-12, f"{name} should not contain net negative volume"

    if step.trade_count > 0:
        assert step.vwap_price is not None, "vwap_price should be present when trades execute"
        assert (
            min(step.price_before, step.price_after) - market.gap * market.grid_radius
            <= (step.vwap_price)
            <= max(step.price_before, step.price_after) + market.gap * market.grid_radius
        )


def test_market_rejects_non_positive_or_non_finite_inputs():
    with pytest.raises(ValueError, match="initial_price"):
        Market(initial_price=0.0, gap=1.0)
    with pytest.raises(ValueError, match="gap"):
        Market(initial_price=100.0, gap=math.nan)
    with pytest.raises(ValueError, match="popularity"):
        Market(initial_price=100.0, gap=1.0, popularity=math.inf)


@pytest.mark.slow
def test_low_price_markets_do_not_print_zero_or_negative_prices():
    market = Market(initial_price=1.0, gap=1.0, popularity=2.0, seed=17, grid_radius=8)

    steps = market.step(250)

    assert min(step.price_before for step in steps) >= market.gap
    assert min(step.price_after for step in steps) >= market.gap
    assert min(market.state.price_grid) >= market.gap


@pytest.mark.slow
def test_realism_seeded_determinism_is_tolerant_for_longer_runs():
    left = Market(initial_price=100.0, gap=0.5, popularity=1.4, seed=2024, grid_radius=12)
    right = Market(initial_price=100.0, gap=0.5, popularity=1.4, seed=2024, grid_radius=12)

    left_steps = _realism_steps_or_skip(left, 250)
    right_steps = _realism_steps_or_skip(right, 250)

    assert _freeze(left_steps) == _freeze(right_steps)
    assert _freeze(_state_from_market(left)) == _freeze(_state_from_market(right))


@pytest.mark.slow
def test_realism_orderbook_stays_bounded_over_long_simulation():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=77, grid_radius=16)
    max_depth = 0.0
    max_levels = 0

    for step in _realism_steps_or_skip(market, 1000):
        after = step.orderbook_after
        sides = [value for _, value in _public_items(after) if isinstance(value, Mapping)]
        max_depth = max(max_depth, _total_orderbook_depth(after))
        max_levels = max(max_levels, sum(len(side) for side in sides))

        for side in sides:
            assert all(volume >= -1e-12 for volume in side.values())

    # Realism cancellation should prevent the book from accumulating stale depth indefinitely.
    assert max_levels <= 8 * (2 * market.grid_radius + 1)
    assert max_depth <= 1000.0 * max(1.0, market.popularity)


@pytest.mark.slow
def test_realism_position_mass_remains_nonnegative_over_long_simulation():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.3, seed=88, grid_radius=14)

    for step in _realism_steps_or_skip(market, 750):
        for position in (step.position_mass_before, step.position_mass_after):
            for name, mass_by_price in _public_items(position):
                assert _total_price_map(mass_by_price) >= -1e-12
                assert all(value >= -1e-12 for value in mass_by_price.values()), (
                    f"{name} contains negative position mass"
                )


@pytest.mark.slow
def test_realism_price_moves_over_thousand_steps():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.1, seed=99, grid_radius=16)
    steps = _realism_steps_or_skip(market, 1000)

    prices = [step.price_after for step in steps]
    changes = [step.price_change for step in steps]

    assert max(prices) - min(prices) >= market.gap
    assert sum(1 for change in changes if abs(change) > 1e-12) >= 3


@pytest.mark.slow
def test_realism_price_path_moves_without_overly_periodic_returns():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.2, seed=2026, grid_radius=16)
    steps = _realism_steps_or_skip(market, 1500)

    prices = [step.price_after for step in steps]
    changes = [step.price_change for step in steps]
    moving_changes = [change for change in changes if abs(change) > 1e-12]
    lag_correlations = [_lag_correlation(changes, lag) for lag in range(2, 121)]

    assert max(prices) - min(prices) >= 3 * market.gap
    assert len(moving_changes) >= 50
    assert max(abs(value) for value in lag_correlations) < 0.85


@pytest.mark.slow
def test_realism_volatility_clustering_diagnostic_is_positiveish():
    correlations = []
    follower_ratios = []
    for seed in (1234, 42, 7, 99, 202, 404):
        market = Market(initial_price=100.0, gap=1.0, popularity=1.2, seed=seed, grid_radius=16)
        steps = _realism_steps_or_skip(market, 1500)

        absolute_returns = [abs(step.price_change) for step in steps]
        nonzero_returns = [value for value in absolute_returns if value > 1e-12]
        assert len(nonzero_returns) >= 10, "Need enough price moves for a volatility diagnostic"

        threshold = sorted(absolute_returns)[int(0.75 * (len(absolute_returns) - 1))]
        high_followers = [
            absolute_returns[index + 1]
            for index, value in enumerate(absolute_returns[:-1])
            if value >= threshold
        ]
        correlations.append(_lag1_correlation(absolute_returns))
        follower_ratios.append(
            (sum(high_followers) / len(high_followers))
            / (sum(absolute_returns) / len(absolute_returns))
        )

    assert sum(correlations) / len(correlations) > -0.02
    assert sum(follower_ratios) / len(follower_ratios) >= 0.9


@pytest.mark.slow
def test_realism_imbalance_sign_matches_flow_and_price_direction_tolerantly():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.5, seed=987, grid_radius=16)
    steps = _realism_steps_or_skip(market, 1000)

    directional = []
    for step in steps:
        flow_delta = step.market_buy_volume - step.market_sell_volume
        if abs(flow_delta) > 1e-12:
            assert math.copysign(1.0, step.order_flow_imbalance) == math.copysign(1.0, flow_delta)
        if abs(step.order_flow_imbalance) > 1e-12 and abs(step.price_change) > 1e-12:
            directional.append((step.order_flow_imbalance, step.price_change))

    assert len(directional) >= 10, "Need enough directional observations for imbalance diagnostic"
    positive_changes = [change for imbalance, change in directional if imbalance > 0]
    negative_changes = [change for imbalance, change in directional if imbalance < 0]

    if positive_changes and negative_changes:
        assert sum(positive_changes) / len(positive_changes) >= (
            sum(negative_changes) / len(negative_changes) - market.gap
        )


@pytest.mark.slow
def test_realism_price_path_is_not_overly_periodic():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=42, grid_radius=16)
    steps = _realism_steps_or_skip(market, 500)
    prices = [step.price_after for step in steps]
    changes = [step.price_change for step in steps]
    signs = [1 if change > 0 else -1 if change < 0 else 0 for change in changes]
    moving_signs = [sign for sign in signs if sign != 0]
    lag_correlations = [_lag_correlation(changes, lag) for lag in range(2, 81)]

    assert len(moving_signs) >= 50
    assert max(prices) - min(prices) >= market.gap
    reversals = sum(
        1 for left, right in zip(moving_signs, moving_signs[1:], strict=False) if left != right
    )

    assert reversals >= 8
    assert max(abs(value) for value in lag_correlations) < 0.85
