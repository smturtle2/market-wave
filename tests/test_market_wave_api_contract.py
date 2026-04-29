from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence

import pytest

import market_wave as mw
from market_wave import Market, MarketState, StepInfo
from market_wave.market import (
    _ExecutionResult,
    _IncomingOrder,
    _MicrostructureState,
    _StepComputationCache,
    _TradeStats,
)
from market_wave.state import PriceMap

PUBLIC_TYPE_EXPORTS = (
    "Market",
    "MarketState",
    "IntensityState",
    "LatentState",
    "MDFState",
    "OrderBookState",
    "StepInfo",
)

PUBLIC_CONTRACT_EXPORTS = (
    "GeneratedPath",
    "GenerationMetadata",
    "IntensityState",
    "LatentState",
    "MDFState",
    "Market",
    "MarketState",
    "OrderBookState",
    "StepInfo",
    "ValidationMetrics",
    "compute_metrics",
    "generate_paths",
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

REMOVED_PUBLIC_MDF_EXPORTS = (
    "DynamicMDFModel",
    "MDFContext",
    "MDFModel",
    "MDFSignals",
    "RelativeMDFComponent",
)

MDF_FIELDS = (
    "buy_entry_mdf",
    "sell_entry_mdf",
)

MDF_BY_PRICE_FIELDS = (
    "buy_entry_mdf_by_price",
    "sell_entry_mdf_by_price",
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


def _near_touch_depth(orderbook, levels=2):
    bid = sorted(orderbook.bid_volume_by_price.items(), reverse=True)[:levels]
    ask = sorted(orderbook.ask_volume_by_price.items())[:levels]
    return sum(volume for _, volume in (*bid, *ask))


def _assert_internal_orderbook_is_aggregate_only(market):
    orderbook = market._orderbook
    assert not hasattr(orderbook, "bid_lots")
    assert not hasattr(orderbook, "ask_lots")
    assert isinstance(orderbook.bid_volume_by_price, Mapping)
    assert isinstance(orderbook.ask_volume_by_price, Mapping)


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


def _weighted_tick_mean(mdf):
    return sum(tick * probability for tick, probability in mdf.items())


def _tail_mass(mdf, *, min_abs_tick):
    return sum(probability for tick, probability in mdf.items() if abs(tick) >= min_abs_tick)


def _mdf_l1_distance(left, right):
    ticks = set(left) | set(right)
    return sum(abs(left.get(tick, 0.0) - right.get(tick, 0.0)) for tick in ticks)


def _single_mdf_overlap(left, right):
    ticks = set(left) | set(right)
    return sum(min(left.get(tick, 0.0), right.get(tick, 0.0)) for tick in ticks)


def _mirrored_mdf(mdf):
    return {-tick: probability for tick, probability in mdf.items()}


def test_public_api_exports_expected_types():
    missing = [name for name in PUBLIC_TYPE_EXPORTS if not hasattr(mw, name)]

    assert not missing, "Missing public MDF exports: " + ", ".join(missing)
    assert all(isinstance(getattr(mw, name), type) for name in PUBLIC_TYPE_EXPORTS)


def test_public_api_contract_exports_are_inventoried():
    assert tuple(mw.__all__) == PUBLIC_CONTRACT_EXPORTS
    assert "_quote_texture" not in mw.__all__
    assert "_refresh_post_event_deep_quotes" not in mw.__all__


def test_public_api_removes_dynamic_pmf_exports():
    leaked = [name for name in REMOVED_PUBLIC_PMF_EXPORTS if hasattr(mw, name)]

    assert not leaked, "Dynamic PMF names should not remain public: " + ", ".join(leaked)


def test_public_api_removes_custom_mdf_exports():
    leaked = [name for name in REMOVED_PUBLIC_MDF_EXPORTS if hasattr(mw, name)]

    assert not leaked, "Custom MDF names should not remain public: " + ", ".join(leaked)


@pytest.mark.parametrize("removed_kwarg", ("mdf_model", "mdf_temperature"))
def test_market_rejects_removed_custom_mdf_constructor_kwargs(removed_kwarg):
    with pytest.raises(TypeError):
        Market(initial_price=100.0, gap=1.0, **{removed_kwarg: object()})


def test_public_snapshots_do_not_expose_private_liquidity_stress_state():
    legacy_prefix = "wa" + "ll"
    forbidden = {
        "liquidity_stress",
        "stress_side",
        "spread_pressure",
        f"{legacy_prefix}_pressure_by_absolute_tick",
        f"{legacy_prefix}_strength",
        f"{legacy_prefix}_persistence",
    }
    market = Market(initial_price=100.0, gap=1.0, seed=111, grid_radius=4)
    step = market.step(1)[0]
    public_payload = {
        "step": step.to_dict(),
        "state": dataclasses.asdict(market.snapshot()),
        "exports": list(mw.__all__),
    }
    encoded = json.dumps(public_payload)

    assert not forbidden & {field.name for field in dataclasses.fields(StepInfo)}
    assert not forbidden & {field.name for field in dataclasses.fields(MarketState)}
    assert all(name not in encoded for name in forbidden)


def test_state_snapshots_are_attribute_frozen_but_nested_containers_are_plain_mutable():
    market = Market(initial_price=100.0, gap=1.0, seed=107, grid_radius=4)
    step = market.step(1)[0]

    with pytest.raises(dataclasses.FrozenInstanceError):
        step.price_after = 0.0

    step.buy_entry_mdf[0] = 0.0

    assert step.buy_entry_mdf[0] == 0.0


def test_market_snapshot_returns_mutation_safe_copy_of_current_state():
    market = Market(initial_price=100.0, gap=1.0, seed=109, grid_radius=4)
    market.step(3)

    snapshot = market.snapshot()
    snapshot.mdf.buy_entry_mdf[0] = 0.0
    snapshot.price_grid.clear()

    assert market.state.mdf.buy_entry_mdf[0] > 0.0
    assert market.state.price_grid


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
    for relative_name, price_name in (
        ("buy_entry_mdf", "buy_entry_mdf_by_price"),
        ("sell_entry_mdf", "sell_entry_mdf_by_price"),
    ):
        projected = {
            market.tick_to_price(state.current_tick + relative_tick): probability
            for relative_tick, probability in getattr(mdf_state, relative_name).items()
            if state.current_tick + relative_tick >= 1
        }
        assert getattr(mdf_state, price_name) == pytest.approx(projected)


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


def test_seeded_regime_microstructure_is_deterministic():
    for regime in ("normal", "high_vol", "thin_liquidity", "squeeze"):
        left = Market(
            initial_price=100.0,
            gap=1.0,
            popularity=1.2,
            seed=314,
            grid_radius=10,
            regime=regime,
        )
        right = Market(
            initial_price=100.0,
            gap=1.0,
            popularity=1.2,
            seed=314,
            grid_radius=10,
            regime=regime,
        )

        left_steps = left.step(36)
        right_steps = right.step(36)

        assert _freeze(left_steps) == _freeze(right_steps)
        assert _freeze(_state_from_market(left)) == _freeze(_state_from_market(right))


def test_mdf_judgment_sampling_is_seed_reproducible_and_seed_sensitive():
    left = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=210, grid_radius=10)
    right = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=210, grid_radius=10)
    other = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=211, grid_radius=10)

    left_steps = left.step(24, keep_history=False)
    right_steps = right.step(24, keep_history=False)
    other_steps = other.step(24, keep_history=False)

    for left_step, right_step in zip(left_steps, right_steps, strict=True):
        assert left_step.buy_entry_mdf == right_step.buy_entry_mdf
        assert left_step.sell_entry_mdf == right_step.sell_entry_mdf
        assert left_step.buy_entry_mdf_by_price == right_step.buy_entry_mdf_by_price
        assert left_step.sell_entry_mdf_by_price == right_step.sell_entry_mdf_by_price

    buy_distances = [
        _mdf_l1_distance(left_step.buy_entry_mdf, other_step.buy_entry_mdf)
        for left_step, other_step in zip(left_steps, other_steps, strict=True)
    ]
    sell_distances = [
        _mdf_l1_distance(left_step.sell_entry_mdf, other_step.sell_entry_mdf)
        for left_step, other_step in zip(left_steps, other_steps, strict=True)
    ]
    assert max(buy_distances) > 0.05
    assert max(sell_distances) > 0.05


def test_buy_and_sell_mdfs_are_not_generated_as_mirrors():
    distances = []
    mirrored_peak_count = 0
    sample_count = 0
    for seed in range(20, 28):
        market = Market(
            initial_price=100.0,
            gap=1.0,
            popularity=1.0,
            seed=seed,
            grid_radius=12,
            regime="auto",
            augmentation_strength=0.35,
        )
        for step in market.step(45, keep_history=False)[10::5]:
            mirrored_sell = _mirrored_mdf(step.sell_entry_mdf)
            distances.append(_mdf_l1_distance(step.buy_entry_mdf, mirrored_sell))
            buy_peak = max(step.buy_entry_mdf, key=step.buy_entry_mdf.get)
            sell_peak = max(step.sell_entry_mdf, key=step.sell_entry_mdf.get)
            mirrored_peak_count += int(buy_peak == -sell_peak)
            sample_count += 1

    assert sum(distances) / len(distances) > 0.10
    assert mirrored_peak_count < sample_count * 0.80


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


def test_public_mdf_serialization_does_not_expose_raw_or_effective_internal_names():
    market = Market(initial_price=100.0, gap=1.0, seed=104, grid_radius=4)
    step = market.step(3)[-1]

    assert {field.name for field in dataclasses.fields(mw.MDFState)} == {
        "relative_ticks",
        "buy_entry_mdf",
        "sell_entry_mdf",
        "buy_entry_mdf_by_price",
        "sell_entry_mdf_by_price",
    }
    payload = {
        "step": step.to_dict(),
        "state": dataclasses.asdict(_state_from_market(market)),
        "exports": list(mw.__all__),
    }
    encoded = json.dumps(payload).lower()

    for internal_name in (
        "raw_mdf",
        "raw_entry_mdf",
        "effective_mdf",
        "overlap",
        "fifo",
        "noise_mdf",
        "entry_noise",
        "centered_noise",
    ):
        assert internal_name not in encoded


def test_step_can_skip_history_and_stream_steps():
    market = Market(initial_price=100.0, gap=1.0, seed=13, grid_radius=6)

    steps = market.step(3, keep_history=False)

    assert len(steps) == 3
    assert market.history == []

    streamed = list(market.stream(2, keep_history=True))

    assert len(streamed) == 2
    assert market.history == streamed
    assert [step.step_index for step in streamed] == [4, 5]


def test_default_step_produces_active_seeded_price_discovery():
    market = Market(initial_price=10_000.0, gap=10.0, popularity=1.0, seed=42)

    steps = market.step(160)
    prices = [step.price_after for step in steps]
    changes = [step.price_change for step in steps]

    assert not hasattr(market, "_anchor_price")
    assert not hasattr(market, "_price_pressure_ticks")
    assert len(set(prices)) > 2
    assert max(prices) > min(prices)
    assert any(change > 0 for change in changes)
    assert any(change < 0 for change in changes)


def test_inactive_market_has_no_executions_or_price_changes():
    market = Market(initial_price=100.0, gap=1.0, popularity=0.0, seed=9)
    steps = market.step(100)

    assert all(step.total_executed_volume == 0 for step in steps)
    assert {step.price_after for step in steps} == {100.0}


def test_price_update_stays_flat_when_executions_print_at_previous_price():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=44)
    stats = _TradeStats(executed_by_price={})
    stats.record(100.0, 1.0)
    execution = _ExecutionResult(
        residual_market_buy=0.0,
        residual_market_sell=0.0,
        crossed_market_volume=0.0,
    )

    assert market._next_price_after_trading(100.0, stats, execution) == 100.0


def test_latent_mdf_context_is_seeded_and_evolves_over_steps():
    left = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=91)
    right = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=91)
    other = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=92)

    assert left.state.latent == right.state.latent
    assert left.state.latent != other.state.latent

    steps = left.step(80)
    latent_path = {
        (
            round(step.mood, 4),
            round(step.trend, 4),
            round(step.volatility, 4),
        )
        for step in steps
    }

    assert len(latent_path) > 1
    assert any(step.mood != left.state.latent.mood for step in steps)
    assert any(step.trend != left.state.latent.trend for step in steps)
    assert any(step.volatility != left.state.latent.volatility for step in steps)


def test_default_step_keeps_full_history_and_updates_live_aggregates():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.2, seed=13, grid_radius=16)

    steps = market.step(160)

    assert len(steps) == 160
    assert market.history == steps
    assert [step.step_index for step in steps[:3]] == [1, 2, 3]
    assert steps[-1].step_index == 160

    _assert_internal_orderbook_is_aggregate_only(market)
    bid_volume = market._orderbook.bid_volume_by_price
    ask_volume = market._orderbook.ask_volume_by_price
    assert market.state.orderbook.bid_volume_by_price == pytest.approx(bid_volume)
    assert market.state.orderbook.ask_volume_by_price == pytest.approx(ask_volume)
    assert steps[-1].orderbook_after.bid_volume_by_price == pytest.approx(bid_volume)
    assert steps[-1].orderbook_after.ask_volume_by_price == pytest.approx(ask_volume)
    assert market._best_bid() == (max(bid_volume) if bid_volume else None)
    assert market._best_ask() == (min(ask_volume) if ask_volume else None)


def test_internal_orderbook_exposes_aggregate_price_levels_only():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=93, grid_radius=8)
    market._add_lots({99.0: 3.0}, "bid", "buy_entry")
    market._add_lots({99.0: 2.0}, "bid", "buy_entry")
    market._add_lots({101.0: 4.0}, "ask", "sell_entry")

    _assert_internal_orderbook_is_aggregate_only(market)
    assert market._orderbook.bid_volume_by_price == pytest.approx({99.0: 5.0})
    assert market._orderbook.ask_volume_by_price == pytest.approx({101.0: 4.0})
    assert market._orderbook.snapshot().bid_volume_by_price == pytest.approx({99.0: 5.0})
    assert market._orderbook.snapshot().ask_volume_by_price == pytest.approx({101.0: 4.0})


def test_orderbook_aggregate_maps_match_public_snapshot_after_long_run():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.2, seed=13, grid_radius=16)

    market.step(160, keep_history=False)

    assert market.history == []
    _assert_internal_orderbook_is_aggregate_only(market)
    bid_volume = market._orderbook.bid_volume_by_price
    ask_volume = market._orderbook.ask_volume_by_price
    assert market._orderbook.snapshot().bid_volume_by_price == pytest.approx(bid_volume)
    assert market._orderbook.snapshot().ask_volume_by_price == pytest.approx(ask_volume)
    assert market.state.orderbook.bid_volume_by_price == pytest.approx(bid_volume)
    assert market.state.orderbook.ask_volume_by_price == pytest.approx(ask_volume)
    assert market._best_bid() == (max(bid_volume) if bid_volume else None)
    assert market._best_ask() == (min(ask_volume) if ask_volume else None)


def test_entry_fills_do_not_create_hidden_position_inventory():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.2, seed=13, grid_radius=16)

    market.step(160, keep_history=False)

    assert not hasattr(market, "_long_cohorts")
    assert not hasattr(market, "_short_cohorts")
    assert not hasattr(market, "_long_mass_total")
    assert not hasattr(market, "_short_mass_total")


def test_buy_entry_mdf_order_matches_resting_asks_and_rests_leftover_volume():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=93, grid_radius=8)
    market._add_lots({101.0: 1.0, 102.0: 2.0, 103.0: 4.0}, "ask", "sell_entry")
    stats = _TradeStats(executed_by_price={})
    execution = market._execute_market_flows(
        entry_orders=[_IncomingOrder(side="buy", kind="buy_entry", price=102.0, volume=5.0)],
        stats=stats,
    )

    assert stats.executed_by_price == pytest.approx({101.0: 1.0, 102.0: 2.0})
    assert execution.residual_market_buy == pytest.approx(2.0)
    assert execution.market_buy_volume == pytest.approx(3.0)
    assert market._orderbook.ask_volume_by_price == pytest.approx({103.0: 4.0})
    assert market._orderbook.bid_volume_by_price == pytest.approx({102.0: 2.0})


def test_execution_consumes_aggregate_price_level_volume_only():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=93, grid_radius=8)
    market._add_lots({101.0: 3.0}, "ask", "sell_entry")
    market._add_lots({101.0: 2.0}, "ask", "sell_entry")
    stats = _TradeStats(executed_by_price={})

    execution = market._execute_market_flows(
        entry_orders=[_IncomingOrder(side="buy", kind="buy_entry", price=101.0, volume=4.0)],
        stats=stats,
    )

    _assert_internal_orderbook_is_aggregate_only(market)
    assert stats.executed_by_price == pytest.approx({101.0: 4.0})
    assert stats.total_volume == pytest.approx(4.0)
    assert execution.market_buy_volume == pytest.approx(4.0)
    assert execution.residual_market_buy == 0.0
    assert market._orderbook.ask_volume_by_price == pytest.approx({101.0: 1.0})
    assert market._orderbook.bid_volume_by_price == {}


def test_same_step_entry_quotes_do_not_execute_against_each_other():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=94, grid_radius=8)
    stats = _TradeStats(executed_by_price={})

    execution = market._execute_market_flows(
        entry_orders=[
            _IncomingOrder(side="buy", kind="buy_entry", price=101.0, volume=2.0),
            _IncomingOrder(side="sell", kind="sell_entry", price=99.0, volume=3.0),
        ],
        stats=stats,
    )

    assert stats.executed_by_price == {}
    assert stats.total_volume == 0.0
    assert execution.market_buy_volume == 0.0
    assert execution.market_sell_volume == 0.0
    assert execution.residual_market_buy == pytest.approx(2.0)
    assert execution.residual_market_sell == pytest.approx(3.0)
    assert market._orderbook.bid_volume_by_price == pytest.approx({101.0: 2.0})
    assert market._orderbook.ask_volume_by_price == pytest.approx({99.0: 3.0})


def test_entry_flow_samples_entry_prices_from_mdf_instead_of_splitting_by_ratio():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=105, grid_radius=8)
    mdf = mw.MDFState(
        buy_entry_mdf_by_price={98.0: 0.70, 101.0: 0.30},
        sell_entry_mdf_by_price={99.0: 0.25, 103.0: 0.75},
    )
    intensity = mw.IntensityState(total=20.0, buy=8.0, sell=12.0, buy_ratio=0.4, sell_ratio=0.6)

    flow = market._entry_flow(intensity, mdf)

    assert flow.buy_intent_by_price != pytest.approx({98.0: 5.6, 101.0: 2.4})
    assert flow.sell_intent_by_price != pytest.approx({99.0: 3.0, 103.0: 9.0})
    assert set(flow.buy_intent_by_price) <= {98.0, 101.0}
    assert set(flow.sell_intent_by_price) <= {99.0, 103.0}
    assert flow.orders
    assert sum(order.volume for order in flow.orders if order.side == "buy") != pytest.approx(
        intensity.buy
    )
    assert sum(order.volume for order in flow.orders if order.side == "sell") != pytest.approx(
        intensity.sell
    )


def test_entry_flow_can_sample_zero_orders_even_with_single_price_mdf():
    class ZeroRandom:
        def random(self):
            return 0.0

    market = Market(initial_price=100.0, gap=1.0, popularity=0.01, seed=105, grid_radius=8)
    market._rng = ZeroRandom()
    mdf = mw.MDFState(buy_entry_mdf_by_price={99.0: 1.0})
    intensity = mw.IntensityState(
        total=0.01,
        buy=0.01,
        sell=0.0,
        buy_ratio=1.0,
        sell_ratio=0.0,
    )

    flow = market._entry_flow(intensity, mdf)

    assert flow.orders == []
    assert flow.buy_intent_by_price == {}


def test_entry_flow_keeps_duplicate_price_orders_as_separate_events():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=105, grid_radius=8)
    mdf = mw.MDFState(buy_entry_mdf_by_price={99.0: 1.0})
    intensity = mw.IntensityState(total=20.0, buy=20.0, sell=0.0, buy_ratio=1.0, sell_ratio=0.0)

    flow = market._entry_flow(intensity, mdf)

    assert len(flow.orders) > 1
    assert {order.price for order in flow.orders} == {99.0}
    assert all(order.side == "buy" for order in flow.orders)
    assert flow.buy_intent_by_price[99.0] == pytest.approx(
        sum(order.volume for order in flow.orders)
    )


def test_entry_flow_sampling_converges_toward_mdf_over_many_events():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=105, grid_radius=8)
    mdf = mw.MDFState(buy_entry_mdf_by_price={98.0: 0.70, 101.0: 0.30})
    intensity = mw.IntensityState(total=500.0, buy=500.0, sell=0.0, buy_ratio=1.0, sell_ratio=0.0)

    flow = market._entry_flow(intensity, mdf)
    sampled_share = sum(1 for order in flow.orders if order.price == 98.0) / len(flow.orders)

    assert sampled_share == pytest.approx(0.70, abs=0.06)


def test_sell_entry_mdf_order_matches_resting_bids_and_rests_leftover_volume():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=94, grid_radius=8)
    market._add_lots({99.0: 1.0, 98.0: 2.0, 97.0: 4.0}, "bid", "buy_entry")
    stats = _TradeStats(executed_by_price={})
    execution = market._execute_market_flows(
        entry_orders=[_IncomingOrder(side="sell", kind="sell_entry", price=98.0, volume=5.0)],
        stats=stats,
    )

    assert stats.executed_by_price == pytest.approx({99.0: 1.0, 98.0: 2.0})
    assert execution.residual_market_sell == pytest.approx(2.0)
    assert execution.market_sell_volume == pytest.approx(3.0)
    assert market._orderbook.bid_volume_by_price == pytest.approx({97.0: 4.0})
    assert market._orderbook.ask_volume_by_price == pytest.approx({98.0: 2.0})


def test_entry_mdf_order_rests_when_it_does_not_overlap_opposite_quotes():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=95, grid_radius=8)
    market._add_lots({101.0: 3.0}, "ask", "sell_entry")
    stats = _TradeStats(executed_by_price={})
    execution = market._execute_market_flows(
        entry_orders=[_IncomingOrder(side="buy", kind="buy_entry", price=100.0, volume=4.0)],
        stats=stats,
    )

    assert stats.total_volume == 0.0
    assert execution.residual_market_buy == pytest.approx(4.0)
    assert execution.market_buy_volume == 0.0
    assert market._orderbook.bid_volume_by_price == pytest.approx({100.0: 4.0})
    assert market._orderbook.ask_volume_by_price == pytest.approx({101.0: 3.0})


def test_liquidity_stress_is_side_specific_without_memory_reinforcement():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=98, grid_radius=10)
    market._add_lots({99.0: 10.0}, "bid", "buy_entry")
    market._add_lots({101.0: 10.0}, "ask", "sell_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.sell_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({99.0: 1.0})
    market.state.mdf.sell_entry_mdf_by_price.update({101.0: 1.0})
    ask_stress = _MicrostructureState(liquidity_stress=1.0, stress_side=1.0)
    bid_stress = _MicrostructureState(liquidity_stress=1.0, stress_side=-1.0)
    calm = _MicrostructureState(liquidity_stress=0.0, stress_side=0.0)

    calm_ask = market._cancel_price_pressure(
        "ask",
        101.0,
        100.0,
        calm,
        market.state.mdf,
    )
    stressed_ask = market._cancel_price_pressure(
        "ask",
        101.0,
        100.0,
        ask_stress,
        market.state.mdf,
    )
    calm_bid = market._cancel_price_pressure(
        "bid",
        99.0,
        100.0,
        calm,
        market.state.mdf,
    )
    stressed_bid = market._cancel_price_pressure(
        "bid",
        99.0,
        100.0,
        bid_stress,
        market.state.mdf,
    )

    assert stressed_ask > calm_ask
    assert stressed_bid > calm_bid
    assert market._cancel_price_pressure(
        "bid",
        99.0,
        100.0,
        ask_stress,
        market.state.mdf,
    ) == pytest.approx(calm_bid)


def test_liquidity_stress_reduces_replenishment_without_compensating_depth():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=99, grid_radius=10)
    market._book_level_noise = lambda sigma: 1.0
    neutral = mw.LatentState(mood=0.0, trend=0.0, volatility=0.2)
    calm = _MicrostructureState(liquidity_stress=0.0, stress_side=0.0)
    ask_stress = _MicrostructureState(liquidity_stress=1.0, stress_side=1.0)

    calm_ask = market._replenishment_volume_for_level(
        1,
        "ask",
        1.0,
        0.0,
        neutral,
        calm,
    )
    stressed_ask = market._replenishment_volume_for_level(
        1,
        "ask",
        1.0,
        0.0,
        neutral,
        ask_stress,
    )
    stressed_bid = market._replenishment_volume_for_level(
        1,
        "bid",
        1.0,
        0.0,
        neutral,
        ask_stress,
    )

    assert stressed_ask < calm_ask
    assert stressed_bid == pytest.approx(calm_ask)


def test_replenishment_samples_passive_liquidity_from_public_entry_mdfs():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=106, grid_radius=8)
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({99.0: 1.0})
    market.state.mdf.sell_entry_mdf_by_price.clear()
    market.state.mdf.sell_entry_mdf_by_price.update({101.0: 1.0})

    market._add_sampled_replenishment(
        "bid",
        "buy_entry",
        100.0,
        20.0,
        market.state.mdf,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        _MicrostructureState(resiliency=1.0),
    )
    market._add_sampled_replenishment(
        "ask",
        "sell_entry",
        100.0,
        20.0,
        market.state.mdf,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        _MicrostructureState(resiliency=1.0),
    )

    assert set(market._orderbook.bid_volume_by_price) <= {99.0}
    assert set(market._orderbook.ask_volume_by_price) <= {101.0}
    assert market._orderbook.bid_volume_by_price[99.0] > 0.0
    assert market._orderbook.ask_volume_by_price[101.0] > 0.0
    assert market._best_bid() < market._best_ask()


def test_replenishment_sampling_weights_follow_entry_mdf_with_aggregate_depth_tilt():
    near_shortage_market = Market(
        initial_price=100.0, gap=1.0, popularity=1.0, seed=106, grid_radius=8
    )
    deep_shortage_market = Market(
        initial_price=100.0, gap=1.0, popularity=1.0, seed=106, grid_radius=8
    )
    for market in (near_shortage_market, deep_shortage_market):
        market.state.mdf.buy_entry_mdf_by_price.clear()
        market.state.mdf.buy_entry_mdf_by_price.update({99.0: 0.70, 97.0: 0.30})
    near_shortage_market._add_lots(
        {98.0: 5.0, 97.0: 5.0, 96.0: 5.0, 95.0: 5.0},
        "bid",
        "buy_entry",
    )
    deep_shortage_market._add_lots(
        {99.0: 5.0, 98.0: 5.0, 97.0: 5.0, 96.0: 5.0},
        "bid",
        "buy_entry",
    )

    near_weights = near_shortage_market._replenishment_sampling_weights(
        "bid",
        100.0,
        near_shortage_market.state.mdf,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        _MicrostructureState(resiliency=1.0),
    )
    deep_weights = deep_shortage_market._replenishment_sampling_weights(
        "bid",
        100.0,
        deep_shortage_market.state.mdf,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        _MicrostructureState(resiliency=1.0),
    )

    assert set(near_weights) == {99.0, 97.0}
    assert set(deep_weights) == {99.0, 97.0}
    assert near_weights[99.0] > deep_weights[99.0]
    assert deep_weights[97.0] > near_weights[97.0]
    assert abs(near_weights[99.0] - deep_weights[99.0]) < 0.05
    assert near_weights[99.0] > 0.80
    assert near_weights[97.0] < 0.20


def test_spread_pressure_modulates_replenishment_without_adding_a_new_mdf():
    def market_with_spread(best_bid: float, best_ask: float) -> Market:
        market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=106, grid_radius=8)
        market.state.mdf.buy_entry_mdf_by_price.clear()
        market.state.mdf.buy_entry_mdf_by_price.update({99.0: 0.55, 98.0: 0.25, 95.0: 0.20})
        market._add_lots({best_bid: 1.0}, "bid", "buy_entry")
        market._add_lots({best_ask: 1.0}, "ask", "sell_entry")
        return market

    latent = mw.LatentState(mood=0.0, trend=0.0, volatility=0.2)
    calm = _MicrostructureState(resiliency=1.0)
    stressed = _MicrostructureState(
        resiliency=0.75,
        liquidity_stress=1.4,
        stress_side=1.0,
        spread_pressure=1.8,
    )
    tight_market = market_with_spread(99.0, 101.0)
    wide_market = market_with_spread(96.0, 104.0)

    tight_weights = tight_market._replenishment_sampling_weights(
        "bid",
        100.0,
        tight_market.state.mdf,
        latent,
        calm,
    )
    wide_calm_weights = wide_market._replenishment_sampling_weights(
        "bid",
        100.0,
        wide_market.state.mdf,
        latent,
        calm,
    )
    wide_stressed_weights = wide_market._replenishment_sampling_weights(
        "bid",
        100.0,
        wide_market.state.mdf,
        latent,
        stressed,
    )

    assert set(wide_calm_weights) == {99.0, 98.0, 95.0}
    assert wide_calm_weights[99.0] > tight_weights[99.0]
    assert wide_stressed_weights[99.0] < wide_calm_weights[99.0]
    assert wide_stressed_weights[95.0] > wide_calm_weights[95.0]


def test_stressed_replenishment_penalizes_overfilled_deep_tail():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=106, grid_radius=10)
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update(
        {99.0: 0.25, 96.0: 0.25, 94.0: 0.25, 93.0: 0.25}
    )
    market._add_lots({94.0: 20.0, 93.0: 20.0}, "bid", "buy_entry")

    weights = market._replenishment_sampling_weights(
        "bid",
        100.0,
        market.state.mdf,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        _MicrostructureState(
            resiliency=0.75,
            liquidity_stress=1.4,
            stress_side=-1.0,
            spread_pressure=1.8,
        ),
    )

    far_tail = sum(weight for price, weight in weights.items() if 100.0 - price >= 5.0)

    assert weights[96.0] > weights[94.0]
    assert weights[99.0] > weights[93.0]
    assert far_tail < 0.08


def test_wide_spread_quote_arrivals_do_not_over_repair_spread(monkeypatch):
    def market_with_spread(best_bid: float, best_ask: float) -> Market:
        market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=106, grid_radius=8)
        market._add_lots({best_bid: 1.0}, "bid", "buy_entry")
        market._add_lots({best_ask: 1.0}, "ask", "sell_entry")
        return market

    def captured_targets(market: Market) -> list[tuple[str, float, int | None]]:
        captured: list[tuple[str, float, int | None]] = []

        def capture_replenishment(
            side,
            kind,
            current_price,
            target_volume,
            mdf,
            latent,
            micro,
            cache=None,
            max_passive_distance=None,
        ):
            del kind, current_price, mdf, latent, micro, cache
            captured.append((side, target_volume, max_passive_distance))

        monkeypatch.setattr(market, "_add_sampled_replenishment", capture_replenishment)
        latent = mw.LatentState(mood=0.0, trend=0.0, volatility=0.2)
        micro = _MicrostructureState(resiliency=1.0)
        market._add_post_event_quote_arrivals(
            100.0,
            latent,
            micro,
            _TradeStats(executed_by_price={}),
            0.0,
            market.state.mdf,
            _StepComputationCache(100.0, mdf=market.state.mdf, micro=micro),
        )
        return captured

    tight_targets = captured_targets(market_with_spread(99.0, 101.0))
    wide_targets = captured_targets(market_with_spread(96.0, 104.0))
    tight_total = sum(target for _, target, _ in tight_targets)
    wide_total = sum(target for _, target, _ in wide_targets)

    assert [(side, max_distance) for side, _, max_distance in tight_targets] == [
        ("bid", None),
        ("ask", None),
        ("bid", 2),
        ("ask", 2),
    ]
    assert [(side, max_distance) for side, _, max_distance in wide_targets] == [
        ("bid", None),
        ("ask", None),
        ("bid", 2),
        ("ask", 2),
    ]
    assert wide_total > tight_total
    assert wide_total <= tight_total * 1.70


def test_cancellations_sample_book_prices_with_weak_entry_mdf_support():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=107, grid_radius=8)
    market._add_lots({99.0: 10.0, 94.0: 10.0}, "bid", "buy_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({99.0: 0.99, 94.0: 0.01})

    weights = market._cancel_sampling_weights(
        "bid",
        100.0,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        _MicrostructureState(),
        market.state.mdf,
    )

    assert weights[94.0] > weights[99.0]

    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=108, grid_radius=8)
    market._add_lots({101.0: 10.0, 106.0: 10.0}, "ask", "sell_entry")
    market.state.mdf.sell_entry_mdf_by_price.clear()
    market.state.mdf.sell_entry_mdf_by_price.update({101.0: 0.99, 106.0: 0.01})

    weights = market._cancel_sampling_weights(
        "ask",
        100.0,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        _MicrostructureState(),
        market.state.mdf,
    )

    assert weights[106.0] > weights[101.0]


def test_cancel_pressure_is_small_for_depth_supported_by_entry_mdf():
    aligned = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=109, grid_radius=8)
    aligned._add_lots({99.0: 20.0}, "bid", "buy_entry")
    aligned.state.mdf.buy_entry_mdf_by_price.clear()
    aligned.state.mdf.buy_entry_mdf_by_price.update({99.0: 1.0})

    mismatched = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=110, grid_radius=8)
    mismatched._add_lots({99.0: 10.0, 94.0: 10.0}, "bid", "buy_entry")
    mismatched.state.mdf.buy_entry_mdf_by_price.clear()
    mismatched.state.mdf.buy_entry_mdf_by_price.update({99.0: 0.99, 94.0: 0.01})

    aligned_pressure = aligned._cancel_side_pressure(
        "bid",
        100.0,
        _MicrostructureState(),
        aligned.state.mdf,
    )
    mismatched_pressure = mismatched._cancel_side_pressure(
        "bid",
        100.0,
        _MicrostructureState(),
        mismatched.state.mdf,
    )

    assert aligned_pressure < 0.06
    assert mismatched_pressure > aligned_pressure * 8.0


def test_cancel_pressure_protects_supported_near_depth_and_targets_unsupported_deep_depth():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=111, grid_radius=8)
    market._add_lots({99.0: 10.0, 94.0: 10.0}, "bid", "buy_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({99.0: 0.95, 94.0: 0.05})
    micro = _MicrostructureState(
        activity=1.0,
        activity_event=0.8,
        cancel_pressure=1.1,
        liquidity_stress=0.8,
        stress_side=0.6,
        spread_pressure=0.9,
    )

    near_pressure = market._cancel_price_pressure(
        "bid",
        99.0,
        100.0,
        micro,
        market.state.mdf,
    )
    stale_deep_pressure = market._cancel_price_pressure(
        "bid",
        94.0,
        100.0,
        micro,
        market.state.mdf,
    )
    mid_deep_pressure = market._cancel_price_pressure(
        "bid",
        96.0,
        100.0,
        micro,
        market.state.mdf,
    )
    weights = market._cancel_sampling_weights(
        "bid",
        100.0,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        micro,
        market.state.mdf,
    )

    assert near_pressure < 0.05
    assert stale_deep_pressure > mid_deep_pressure > near_pressure + 0.30
    assert stale_deep_pressure > near_pressure + 0.50
    assert weights[94.0] > 0.90


def test_cancel_orders_reuses_cached_mdf_signals_across_price_levels(monkeypatch):
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=111, grid_radius=12)
    market._add_lots({99.0 - level: 1.0 for level in range(8)}, "bid", "buy_entry")
    market._add_lots({101.0 + level: 1.0 for level in range(8)}, "ask", "sell_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({99.0: 0.70, 94.0: 0.30})
    market.state.mdf.sell_entry_mdf_by_price.clear()
    market.state.mdf.sell_entry_mdf_by_price.update({101.0: 0.70, 106.0: 0.30})
    micro = _MicrostructureState(cancel_pressure=1.0, liquidity_stress=0.8)
    signal_prices: list[float] = []
    original_mdf_signals = market._mdf_signals

    def counted_mdf_signals(price: float):
        signal_prices.append(price)
        return original_mdf_signals(price)

    monkeypatch.setattr(market, "_mdf_signals", counted_mdf_signals)
    monkeypatch.setattr(market, "_cancel_event_probability", lambda *args: 0.0)
    monkeypatch.setattr(market, "_deep_refresh_event_probability", lambda *args: 0.0)

    cancelled = market._cancel_orders(
        100.0,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        micro,
        market.state.mdf,
        _StepComputationCache(100.0, mdf=market.state.mdf, micro=micro),
    )

    assert cancelled == {}
    assert signal_prices == [100.0]


def test_supported_cancellations_have_low_expected_event_rate(monkeypatch):
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=111, grid_radius=8)
    market._add_lots({99.0: 10.0}, "bid", "buy_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({99.0: 1.0})
    expected_values: list[float] = []

    def sample_poisson(expected: float) -> int:
        expected_values.append(expected)
        return 0

    monkeypatch.setattr(market, "_sample_poisson", sample_poisson)

    cancelled = market._cancel_orders(
        100.0,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        _MicrostructureState(),
        market.state.mdf,
    )

    assert cancelled == {}
    assert not expected_values or max(expected_values) < 0.30
    pressure = market._cancel_side_pressure(
        "bid",
        100.0,
        _MicrostructureState(),
        market.state.mdf,
    )
    gate = market._cancel_event_probability(10.0, pressure, _MicrostructureState())
    assert gate < 0.20
    assert market._orderbook.bid_volume_by_price[99.0] == pytest.approx(10.0)


def test_cancel_event_probability_is_smooth_and_monotonic():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=112, grid_radius=8)
    micro = _MicrostructureState()

    assert market._cancel_event_probability(0.0, 0.5, micro) == 0.0

    by_pressure = [
        market._cancel_event_probability(8.0, pressure, micro)
        for pressure in (0.0, 0.1, 0.3, 0.6, 1.0)
    ]
    by_volume = [
        market._cancel_event_probability(volume, 0.4, micro)
        for volume in (1.0, 3.0, 8.0, 15.0)
    ]

    assert by_pressure == sorted(by_pressure)
    assert by_volume == sorted(by_volume)
    assert max(b - a for a, b in zip(by_pressure, by_pressure[1:], strict=False)) < 0.12
    assert max(by_pressure) < 0.50
    assert max(by_volume) < 0.50


def test_quote_texture_is_deterministic_rng_free_and_near_protective():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=123, grid_radius=12)
    rng_state = market._rng.getstate()

    textures = [
        market._quote_texture("bid", 100.0 - distance, 100.0, float(distance))
        for distance in range(1, 13)
    ] + [
        market._quote_texture("ask", 100.0 + distance, 100.0, float(distance))
        for distance in range(1, 13)
    ]
    clone = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=123, grid_radius=12)
    clone_textures = [
        clone._quote_texture("bid", 100.0 - distance, 100.0, float(distance))
        for distance in range(1, 13)
    ] + [
        clone._quote_texture("ask", 100.0 + distance, 100.0, float(distance))
        for distance in range(1, 13)
    ]

    assert market._rng.getstate() == rng_state
    assert textures == pytest.approx(clone_textures)
    assert min(textures[:2]) > 0.90
    assert min(textures) < 0.50
    assert max(textures) > 1.05


def test_deep_refresh_turns_over_excess_depth_without_touching_near(monkeypatch):
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=112, grid_radius=8)
    market._add_lots({99.0: 6.0, 94.0: 10.0, 93.0: 8.0}, "bid", "buy_entry")
    market._add_lots({101.0: 6.0, 106.0: 10.0, 107.0: 8.0}, "ask", "sell_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({99.0: 0.85, 94.0: 0.10, 93.0: 0.05})
    market.state.mdf.sell_entry_mdf_by_price.clear()
    market.state.mdf.sell_entry_mdf_by_price.update({101.0: 0.85, 106.0: 0.10, 107.0: 0.05})
    micro = _MicrostructureState(
        cancel_pressure=1.0,
        liquidity_stress=0.8,
        spread_pressure=0.8,
        activity_event=1.0,
    )
    near_bid_before = market._orderbook.bid_volume_by_price[99.0]
    near_ask_before = market._orderbook.ask_volume_by_price[101.0]

    monkeypatch.setattr(market, "_cancel_event_probability", lambda *args: 0.0)
    monkeypatch.setattr(market, "_deep_refresh_event_probability", lambda *args: 1.0)
    monkeypatch.setattr(market, "_cancel_requote_probability", lambda *args: 0.0)
    monkeypatch.setattr(market, "_unit_random", lambda: 0.0)

    cancelled = market._cancel_orders(
        100.0,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        micro,
        market.state.mdf,
        _StepComputationCache(100.0, mdf=market.state.mdf, micro=micro),
    )

    assert cancelled.get(99.0, 0.0) == 0.0
    assert cancelled.get(101.0, 0.0) == 0.0
    assert market._orderbook.bid_volume_by_price[99.0] == pytest.approx(near_bid_before)
    assert market._orderbook.ask_volume_by_price[101.0] == pytest.approx(near_ask_before)
    assert sum(cancelled.get(price, 0.0) for price in (94.0, 93.0)) > 0.0
    assert sum(cancelled.get(price, 0.0) for price in (106.0, 107.0)) > 0.0
    assert market._best_bid() < market._best_ask()


def test_post_event_deep_refresh_records_texture_fade_without_crossing(monkeypatch):
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=114, grid_radius=8)
    market._add_lots({99.0: 5.0, 96.0: 8.0, 93.0: 8.0}, "bid", "buy_entry")
    market._add_lots({101.0: 5.0, 104.0: 8.0, 107.0: 8.0}, "ask", "sell_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({99.0: 0.84, 96.0: 0.08, 93.0: 0.08})
    market.state.mdf.sell_entry_mdf_by_price.clear()
    market.state.mdf.sell_entry_mdf_by_price.update({101.0: 0.84, 104.0: 0.08, 107.0: 0.08})
    micro = _MicrostructureState(cancel_pressure=1.0, liquidity_stress=0.8, activity_event=1.0)
    cancelled: PriceMap = {}

    monkeypatch.setattr(market, "_deep_refresh_event_probability", lambda *args: 1.0)
    monkeypatch.setattr(market, "_cancel_requote_probability", lambda *args: 0.0)
    monkeypatch.setattr(market, "_unit_random", lambda: 0.0)

    market._refresh_post_event_deep_quotes(
        cancelled,
        100.0,
        micro,
        market.state.mdf,
        _StepComputationCache(100.0, mdf=market.state.mdf, micro=micro),
    )

    assert sum(cancelled.get(price, 0.0) for price in (96.0, 93.0)) > 0.0
    assert sum(cancelled.get(price, 0.0) for price in (104.0, 107.0)) > 0.0
    assert cancelled.get(99.0, 0.0) == 0.0
    assert cancelled.get(101.0, 0.0) == 0.0
    assert market._best_bid() < market._best_ask()


def test_deep_refresh_can_requote_without_recording_destination_as_cancel(monkeypatch):
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=113, grid_radius=8)
    market._add_lots({99.0: 2.0, 94.0: 10.0}, "bid", "buy_entry")
    market._add_lots({101.0: 2.0}, "ask", "sell_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({99.0: 0.90, 94.0: 0.10})
    micro = _MicrostructureState(
        cancel_pressure=1.0,
        liquidity_stress=0.8,
        spread_pressure=0.8,
        activity_event=1.0,
        resiliency=1.0,
    )
    near_before = market._orderbook.bid_volume_by_price[99.0]

    monkeypatch.setattr(market, "_cancel_event_probability", lambda *args: 0.0)
    monkeypatch.setattr(market, "_deep_refresh_event_probability", lambda *args: 1.0)
    monkeypatch.setattr(market, "_cancel_requote_probability", lambda *args: 1.0)
    monkeypatch.setattr(market, "_unit_random", lambda: 0.0)
    monkeypatch.setattr(market, "_sample_price", lambda weights: max(weights))

    cancelled = market._cancel_orders(
        100.0,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        micro,
        market.state.mdf,
        _StepComputationCache(100.0, mdf=market.state.mdf, micro=micro),
    )

    assert cancelled.get(94.0, 0.0) > 0.0
    assert cancelled.get(99.0, 0.0) == 0.0
    assert market._orderbook.bid_volume_by_price[99.0] > near_before
    assert micro.last_cancelled_volume == pytest.approx(sum(cancelled.values()))
    assert market._best_bid() < market._best_ask()


def test_cancel_event_can_remain_pure_cancel_without_requote(monkeypatch):
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=112, grid_radius=8)
    market._add_lots({99.0: 10.0}, "bid", "buy_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({98.0: 1.0})
    monkeypatch.setattr(market, "_cancel_event_probability", lambda *args: 1.0)
    monkeypatch.setattr(market, "_sample_poisson", lambda expected: 1)
    monkeypatch.setattr(market, "_sample_order_size", lambda: 2.0)
    monkeypatch.setattr(market, "_cancel_requote_probability", lambda *args: 0.0)

    cancelled = market._cancel_orders(
        100.0,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        _MicrostructureState(),
        market.state.mdf,
    )

    assert cancelled == pytest.approx({99.0: 2.0})
    assert market._orderbook.bid_volume_by_price == pytest.approx({99.0: 8.0})


def test_stressed_cancellations_are_more_frequent_but_smaller_per_event(monkeypatch):
    def run_cancel(micro: _MicrostructureState) -> tuple[list[float], PriceMap]:
        market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=112, grid_radius=8)
        market._add_lots({94.0: 5.0}, "bid", "buy_entry")
        market.state.mdf.buy_entry_mdf_by_price.clear()
        market.state.mdf.buy_entry_mdf_by_price.update({99.0: 1.0})
        expected_values: list[float] = []

        def sample_poisson(expected: float) -> int:
            expected_values.append(expected)
            return 1

        monkeypatch.setattr(market, "_unit_random", lambda: 0.0)
        monkeypatch.setattr(market, "_deep_refresh_event_probability", lambda *args: 0.0)
        monkeypatch.setattr(market, "_cancel_requote_probability", lambda *args: 0.0)
        monkeypatch.setattr(market, "_sample_order_size", lambda: 1.0)
        monkeypatch.setattr(market, "_sample_poisson", sample_poisson)

        cancelled = market._cancel_orders(
            100.0,
            mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
            micro,
            market.state.mdf,
            _StepComputationCache(100.0, mdf=market.state.mdf, micro=micro),
        )
        return expected_values, cancelled

    calm_expected, calm_cancelled = run_cancel(_MicrostructureState())
    stressed_expected, stressed_cancelled = run_cancel(
        _MicrostructureState(
            cancel_pressure=1.5,
            liquidity_stress=1.4,
            stress_side=1.0,
            spread_pressure=1.3,
        )
    )

    assert stressed_expected[0] > calm_expected[0]
    assert stressed_cancelled[94.0] < calm_cancelled[94.0]
    assert stressed_cancelled[94.0] > 0.75


def test_cancel_event_reduces_aggregate_price_level_volume_only(monkeypatch):
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=112, grid_radius=8)
    market._add_lots({99.0: 3.0}, "bid", "buy_entry")
    market._add_lots({99.0: 2.0}, "bid", "buy_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({98.0: 1.0})
    monkeypatch.setattr(market, "_cancel_event_probability", lambda *args: 1.0)
    monkeypatch.setattr(market, "_sample_poisson", lambda expected: 1)
    monkeypatch.setattr(market, "_sample_order_size", lambda: 2.0)
    monkeypatch.setattr(market, "_cancel_requote_probability", lambda *args: 0.0)

    cancelled = market._cancel_orders(
        100.0,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        _MicrostructureState(),
        market.state.mdf,
    )

    _assert_internal_orderbook_is_aggregate_only(market)
    assert cancelled == pytest.approx({99.0: 2.0})
    assert market._orderbook.bid_volume_by_price == pytest.approx({99.0: 3.0})
    assert market._orderbook.ask_volume_by_price == {}


def test_cancel_event_can_requote_volume_to_mdf_sampled_passive_price(monkeypatch):
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=113, grid_radius=8)
    market._add_lots({98.0: 5.0}, "bid", "buy_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({98.0: 0.05, 99.0: 0.95})
    monkeypatch.setattr(market, "_cancel_event_probability", lambda *args: 1.0)
    monkeypatch.setattr(market, "_sample_poisson", lambda expected: 1)
    monkeypatch.setattr(market, "_sample_order_size", lambda: 2.0)
    monkeypatch.setattr(market, "_cancel_requote_probability", lambda *args: 1.0)

    cancelled = market._cancel_orders(
        100.0,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        _MicrostructureState(resiliency=1.0),
        market.state.mdf,
    )

    assert cancelled == pytest.approx({98.0: 2.0})
    assert market._orderbook.bid_volume_by_price == pytest.approx({98.0: 3.0, 99.0: 2.0})
    assert market._best_bid() < 100.0


def test_cancel_requote_falls_back_to_pure_cancel_without_destination(monkeypatch):
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=114, grid_radius=8)
    market._add_lots({99.0: 5.0}, "bid", "buy_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({99.0: 1.0})
    monkeypatch.setattr(market, "_cancel_event_probability", lambda *args: 1.0)
    monkeypatch.setattr(market, "_sample_poisson", lambda expected: 1)
    monkeypatch.setattr(market, "_sample_order_size", lambda: 2.0)
    monkeypatch.setattr(market, "_cancel_requote_probability", lambda *args: 1.0)

    cancelled = market._cancel_orders(
        100.0,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        _MicrostructureState(resiliency=1.0),
        market.state.mdf,
    )

    assert cancelled == pytest.approx({99.0: 2.0})
    assert market._orderbook.bid_volume_by_price == pytest.approx({99.0: 3.0})


def test_cancel_requote_destination_uses_same_side_passive_mdf_only():
    bid_market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=115, grid_radius=8)
    bid_market.state.mdf.buy_entry_mdf_by_price.clear()
    bid_market.state.mdf.buy_entry_mdf_by_price.update(
        {98.0: 0.20, 99.0: 0.50, 100.0: 0.20, 101.0: 0.10}
    )
    bid_weights = bid_market._cancel_requote_sampling_weights(
        "bid",
        100.0,
        99.0,
        bid_market.state.mdf,
        _MicrostructureState(),
    )

    ask_market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=116, grid_radius=8)
    ask_market.state.mdf.sell_entry_mdf_by_price.clear()
    ask_market.state.mdf.sell_entry_mdf_by_price.update(
        {99.0: 0.10, 100.0: 0.20, 101.0: 0.50, 102.0: 0.20}
    )
    ask_weights = ask_market._cancel_requote_sampling_weights(
        "ask",
        100.0,
        101.0,
        ask_market.state.mdf,
        _MicrostructureState(),
    )

    assert set(bid_weights) == {98.0}
    assert set(ask_weights) == {102.0}


def test_wide_spread_requote_weights_prefer_near_touch_passive_depth():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=118, grid_radius=8)
    market._add_lots({96.0: 1.0}, "bid", "buy_entry")
    market._add_lots({104.0: 1.0}, "ask", "sell_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({99.0: 0.34, 98.0: 0.33, 95.0: 0.33})

    weights = market._cancel_requote_sampling_weights(
        "bid",
        100.0,
        98.0,
        market.state.mdf,
        _MicrostructureState(resiliency=1.0, liquidity_stress=0.5, spread_pressure=1.0),
    )

    assert set(weights) == {95.0, 99.0}
    assert weights[99.0] > weights[95.0]
    assert weights[99.0] < 0.90
    assert weights[95.0] > 0.10


def test_cancel_requote_probability_keeps_pure_cancel_floor():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=117, grid_radius=8)
    market._add_lots({99.0: 10.0}, "bid", "buy_entry")
    market.state.mdf.buy_entry_mdf_by_price.clear()
    market.state.mdf.buy_entry_mdf_by_price.update({99.0: 0.80, 98.0: 0.20})

    calm = market._cancel_requote_probability(
        "bid",
        99.0,
        100.0,
        _MicrostructureState(resiliency=1.4),
        market.state.mdf,
    )
    stressed = market._cancel_requote_probability(
        "bid",
        99.0,
        100.0,
        _MicrostructureState(
            resiliency=0.45,
            cancel_pressure=1.5,
            liquidity_stress=1.4,
            stress_side=-1.0,
            spread_pressure=1.3,
        ),
        market.state.mdf,
    )

    assert stressed < calm - 0.05
    assert stressed >= 0.05
    assert calm <= 0.35


def test_price_move_trim_preserves_non_crossed_quotes_through_last_price():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=112, grid_radius=8)
    market._add_lots({101.0: 2.0}, "bid", "buy_entry")
    market._add_lots({103.0: 2.0}, "ask", "sell_entry")

    market._trim_orderbook_through_last_price(101.0)

    assert market._best_bid() == 101.0
    assert market._best_ask() == 103.0
    assert market._orderbook.bid_volume_by_price[101.0] == pytest.approx(2.0)
    assert market._orderbook.ask_volume_by_price[103.0] == pytest.approx(2.0)


def test_price_move_trim_resolves_crossed_quotes_without_full_last_price_sweep():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=113, grid_radius=8)
    market._add_lots({103.0: 2.0}, "bid", "buy_entry")
    market._add_lots({101.0: 2.0}, "ask", "sell_entry")

    market._trim_orderbook_through_last_price(102.0)

    best_bid = market._best_bid()
    best_ask = market._best_ask()
    assert best_bid is None or best_ask is None or best_bid < best_ask


def test_same_price_entry_orders_match_after_first_order_rests():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=96, grid_radius=8)
    orders = [
        _IncomingOrder(side="buy", kind="buy_entry", price=100.0, volume=4.0),
        _IncomingOrder(side="sell", kind="sell_entry", price=100.0, volume=6.0),
    ]
    stats = _TradeStats(executed_by_price={})

    buy_result = market._process_incoming_order(
        orders[0],
        stats=stats,
    )
    sell_result = market._process_incoming_order(
        orders[1],
        stats=stats,
    )

    assert stats.executed_by_price == pytest.approx({100.0: 4.0})
    assert buy_result.executed == 0.0
    assert buy_result.rested == pytest.approx(4.0)
    assert sell_result.executed == pytest.approx(4.0)
    assert sell_result.rested == pytest.approx(2.0)
    assert market._orderbook.bid_volume_by_price == {}
    assert market._orderbook.ask_volume_by_price == pytest.approx({100.0: 2.0})


def test_orderbook_does_not_stay_crossed_after_steps():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.3, seed=97, grid_radius=12)
    steps = market.step(250)

    for step in steps:
        best_bid = step.best_bid_after
        best_ask = step.best_ask_after
        assert best_bid is None or best_ask is None or best_bid < best_ask


def test_post_trade_replenishment_restores_near_touch_depth_without_crossing():
    market = Market(initial_price=100.0, gap=1.0, popularity=20.0, seed=104, grid_radius=8)
    stats = _TradeStats(executed_by_price={})
    stats.record(100.0, 20.0)

    market._add_post_trade_replenishment(
        100.0,
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.4),
        market._microstructure,
        stats,
        flow_imbalance=0.0,
    )

    assert market._orderbook.bid_volume_by_price
    assert market._orderbook.ask_volume_by_price
    assert all(price < 100.0 for price in market._orderbook.bid_volume_by_price)
    assert all(price > 100.0 for price in market._orderbook.ask_volume_by_price)
    assert market._best_bid() < market._best_ask()


def test_default_plot_path_renders_panel_with_orderbook_heatmaps():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    market = Market(initial_price=100.0, gap=1.0, popularity=1.1, seed=2026, grid_radius=8)
    market.step(5)

    fig, ax = market.plot()

    try:
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax in fig.axes
        assert len(fig.axes) >= 5
        assert fig.axes[1].get_ylabel() == "ask level"
        assert fig.axes[2].get_ylabel() == "bid level"
        assert [image.get_array().shape for image in fig.axes[1].images] == [
            (market.grid_radius, len(market.history))
        ]
        assert [image.get_array().shape for image in fig.axes[2].images] == [
            (market.grid_radius, len(market.history))
        ]
    finally:
        plt.close(fig)


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


def test_relative_mdf_excludes_impossible_prices_at_lower_bound():
    market = Market(initial_price=1.0, gap=1.0, seed=29, grid_radius=5)

    step = market.step(1)[0]

    for name in MDF_FIELDS:
        mdf = getattr(step, name)
        impossible_mass = sum(
            probability
            for relative_tick, probability in mdf.items()
            if step.tick_before + relative_tick < 1
        )
        assert impossible_mass == pytest.approx(0.0, abs=1e-12)

    state = _state_from_market(market)
    current_tick = state.current_tick
    for name in MDF_FIELDS:
        mdf = getattr(state.mdf, name)
        impossible_mass = sum(
            probability
            for relative_tick, probability in mdf.items()
            if current_tick + relative_tick < 1
        )
        assert impossible_mass == pytest.approx(0.0, abs=1e-12)


def test_default_entry_mdf_shape_changes_materially_by_regime_state():
    up_market = Market(
        initial_price=100.0,
        gap=1.0,
        seed=11,
        grid_radius=12,
        regime="trend_up",
        augmentation_strength=0.0,
    )
    down_market = Market(
        initial_price=100.0,
        gap=1.0,
        seed=11,
        grid_radius=12,
        regime="trend_down",
        augmentation_strength=0.0,
    )
    high_vol_market = Market(
        initial_price=100.0,
        gap=1.0,
        seed=11,
        grid_radius=12,
        regime="high_vol",
        augmentation_strength=0.0,
    )
    normal_market = Market(
        initial_price=100.0,
        gap=1.0,
        seed=11,
        grid_radius=12,
        regime="normal",
        augmentation_strength=0.0,
    )

    up_mdf = up_market._next_mdf(
        100.0,
        up_market._price_grid(100.0),
        mw.LatentState(mood=0.55, trend=0.55, volatility=0.25),
        step_index=1,
        update_memory=False,
    )
    down_mdf = down_market._next_mdf(
        100.0,
        down_market._price_grid(100.0),
        mw.LatentState(mood=-0.55, trend=-0.55, volatility=0.25),
        step_index=1,
        update_memory=False,
    )
    high_vol_mdf = high_vol_market._next_mdf(
        100.0,
        high_vol_market._price_grid(100.0),
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.80),
        step_index=1,
        update_memory=False,
    )
    normal_mdf = normal_market._next_mdf(
        100.0,
        normal_market._price_grid(100.0),
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.12),
        step_index=1,
        update_memory=False,
    )

    up_buy_marketable = sum(value for tick, value in up_mdf.buy_entry_mdf.items() if tick >= 1)
    down_buy_marketable = sum(value for tick, value in down_mdf.buy_entry_mdf.items() if tick >= 1)
    up_sell_marketable = sum(value for tick, value in up_mdf.sell_entry_mdf.items() if tick <= -1)
    down_sell_marketable = sum(
        value for tick, value in down_mdf.sell_entry_mdf.items() if tick <= -1
    )

    assert up_buy_marketable > down_buy_marketable + 0.015
    assert down_sell_marketable > up_sell_marketable + 0.015
    assert _mdf_l1_distance(high_vol_mdf.buy_entry_mdf, normal_mdf.buy_entry_mdf) > 0.08
    assert _mdf_l1_distance(high_vol_mdf.sell_entry_mdf, normal_mdf.sell_entry_mdf) > 0.08


def test_default_entry_mdfs_keep_passive_reservation_price_zones():
    market = Market(
        initial_price=100.0,
        gap=1.0,
        seed=7,
        grid_radius=12,
        regime="normal",
        augmentation_strength=0.0,
    )

    step = market.step(1, keep_history=False)[0]
    buy_passive = sum(volume for tick, volume in step.buy_entry_mdf.items() if tick <= -2)
    sell_passive = sum(volume for tick, volume in step.sell_entry_mdf.items() if tick >= 2)
    buy_marketish = sum(volume for tick, volume in step.buy_entry_mdf.items() if tick >= 1)
    sell_marketish = sum(volume for tick, volume in step.sell_entry_mdf.items() if tick <= -1)
    buy_arrival = sum(volume for tick, volume in step.buy_entry_mdf.items() if -1 <= tick <= 0)
    sell_arrival = sum(volume for tick, volume in step.sell_entry_mdf.items() if 0 <= tick <= 1)

    assert buy_passive > 0.50
    assert sell_passive > 0.50
    assert buy_marketish < 0.30
    assert sell_marketish < 0.30
    assert buy_arrival > 0.08
    assert sell_arrival > 0.08


def test_centered_entry_noise_mdf_is_normalized_and_centered():
    market = Market(initial_price=100.0, gap=1.0, seed=108, grid_radius=8)

    noise = market._centered_entry_noise_mdf(
        market.relative_tick_grid(),
        market.price_to_tick(100.0),
    )

    _assert_mdf_map_is_finite_nonnegative_normalized("entry_noise", noise)
    assert noise[0] == max(noise.values())
    assert noise[-1] == pytest.approx(noise[1])
    assert noise[-2] == pytest.approx(noise[2])
    assert noise[-3] < noise[-2] < noise[-1] < noise[0]


def test_centered_entry_noise_mdf_respects_lower_price_bound():
    market = Market(initial_price=1.0, gap=1.0, seed=109, grid_radius=5)

    noise = market._centered_entry_noise_mdf(
        market.relative_tick_grid(),
        current_tick=1,
    )

    _assert_mdf_map_is_finite_nonnegative_normalized("entry_noise", noise)
    assert all(noise[tick] == 0.0 for tick in noise if 1 + tick < 1)
    assert noise[0] == max(noise.values())


def test_entry_noise_mix_adds_near_current_mass_without_replacing_raw_shape():
    market = Market(initial_price=100.0, gap=1.0, seed=110, grid_radius=8)
    relative_ticks = market.relative_tick_grid()
    raw = {tick: 0.0 for tick in relative_ticks}
    raw.update({-6: 0.5, 6: 0.5})
    noise = market._centered_entry_noise_mdf(
        relative_ticks,
        market.price_to_tick(100.0),
    )

    mixed = market._mix_entry_noise_mdf(raw, noise, mix=0.10)

    _assert_mdf_map_is_finite_nonnegative_normalized("mixed_entry_noise", mixed)
    assert sum(mixed[tick] for tick in (-1, 0, 1)) > sum(raw[tick] for tick in (-1, 0, 1))
    assert mixed[-6] > 0.0
    assert mixed[6] > 0.0
    assert mixed[0] > mixed[-4]


def test_default_entry_noise_increases_near_current_effective_mdf_mass():
    default_market = Market(
        initial_price=100.0,
        gap=1.0,
        popularity=1.0,
        seed=111,
        grid_radius=12,
        regime="normal",
        augmentation_strength=0.0,
    )
    no_noise_market = Market(
        initial_price=100.0,
        gap=1.0,
        popularity=1.0,
        seed=111,
        grid_radius=12,
        regime="normal",
        augmentation_strength=0.0,
    )
    no_noise_market._entry_noise_mix = 0.0
    latent = mw.LatentState(mood=0.0, trend=0.0, volatility=0.20)

    default_mdf = default_market._next_mdf(
        100.0,
        default_market._price_grid(100.0),
        latent,
        step_index=1,
        update_memory=False,
    )
    no_noise_mdf = no_noise_market._next_mdf(
        100.0,
        no_noise_market._price_grid(100.0),
        latent,
        step_index=1,
        update_memory=False,
    )

    default_buy_near = sum(
        value for tick, value in default_mdf.buy_entry_mdf.items() if -1 <= tick <= 1
    )
    no_noise_buy_near = sum(
        value for tick, value in no_noise_mdf.buy_entry_mdf.items() if -1 <= tick <= 1
    )
    default_sell_near = sum(
        value for tick, value in default_mdf.sell_entry_mdf.items() if -1 <= tick <= 1
    )
    no_noise_sell_near = sum(
        value for tick, value in no_noise_mdf.sell_entry_mdf.items() if -1 <= tick <= 1
    )

    _assert_mdf_map_is_finite_nonnegative_normalized("default_buy", default_mdf.buy_entry_mdf)
    _assert_mdf_map_is_finite_nonnegative_normalized("default_sell", default_mdf.sell_entry_mdf)
    assert default_buy_near > no_noise_buy_near + 0.006
    assert default_sell_near > no_noise_sell_near + 0.006


def test_centered_entry_noise_is_applied_after_overlap_resolution(monkeypatch):
    market = Market(initial_price=100.0, gap=1.0, seed=112, grid_radius=4)
    market._entry_noise_mix = 0.20
    relative_ticks = market.relative_tick_grid()

    def edge_only_resolver(
        passed_ticks,
        current_tick,
        raw_buy,
        raw_sell,
        *,
        buy_aggression,
        sell_aggression,
    ):
        del current_tick, raw_buy, raw_sell, buy_aggression, sell_aggression
        buy = {tick: 0.0 for tick in passed_ticks}
        sell = {tick: 0.0 for tick in passed_ticks}
        buy[min(passed_ticks)] = 1.0
        sell[max(passed_ticks)] = 1.0
        return buy, sell

    monkeypatch.setattr(market, "_resolve_entry_mdf_overlap", edge_only_resolver)

    mdf = market._next_mdf(
        100.0,
        market._price_grid(100.0),
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.20),
        step_index=1,
        update_memory=False,
    )

    _assert_mdf_map_is_finite_nonnegative_normalized("buy", mdf.buy_entry_mdf)
    _assert_mdf_map_is_finite_nonnegative_normalized("sell", mdf.sell_entry_mdf)
    assert mdf.buy_entry_mdf[0] > 0.0
    assert mdf.sell_entry_mdf[0] > 0.0
    assert set(mdf.buy_entry_mdf) == set(relative_ticks)
    assert set(mdf.sell_entry_mdf) == set(relative_ticks)


def test_entry_mdf_overlap_factor_is_tick_agnostic_and_monotone():
    market = Market(initial_price=100.0, gap=1.0, seed=113, grid_radius=4)

    low_overlap = market._entry_overlap_factor(0.25, 0.20)
    high_overlap = market._entry_overlap_factor(0.25, 0.80)
    calm = market._entry_overlap_factor(0.10, 0.65)
    aggressive = market._entry_overlap_factor(0.90, 0.65)

    assert high_overlap < low_overlap
    assert aggressive > calm
    assert 0.0 < high_overlap < 1.0
    assert 0.0 < calm < aggressive <= 1.0


def test_entry_mdf_overlap_resolver_is_symmetric_for_mirrored_inputs():
    market = Market(initial_price=100.0, gap=1.0, seed=114, grid_radius=6)
    relative_ticks = market.relative_tick_grid()
    raw_buy = {tick: 0.0 for tick in relative_ticks}
    raw_buy.update({-5: 0.12, -3: 0.34, -1: 0.22, 2: 0.19, 5: 0.13})
    raw_sell = _mirrored_mdf(raw_buy)

    buy, sell = market._resolve_entry_mdf_overlap(
        relative_ticks,
        market.price_to_tick(100.0),
        raw_buy,
        raw_sell,
        buy_aggression=0.35,
        sell_aggression=0.35,
    )

    _assert_mdf_map_is_finite_nonnegative_normalized("buy", buy)
    _assert_mdf_map_is_finite_nonnegative_normalized("sell", sell)
    mirrored_sell = _mirrored_mdf(sell)
    for tick, probability in buy.items():
        assert probability == pytest.approx(mirrored_sell[tick])


def test_entry_mdf_overlap_resolver_reduces_shared_ticks_without_marketability_cap():
    market = Market(initial_price=100.0, gap=1.0, seed=8, grid_radius=4)
    relative_ticks = market.relative_tick_grid()
    raw_buy = {tick: 0.0 for tick in relative_ticks}
    raw_sell = {tick: 0.0 for tick in relative_ticks}
    raw_buy.update({-4: 0.50, -1: 0.50})
    raw_sell.update({-1: 0.50, 4: 0.50})
    normalized_buy = market._normalize_tick_map(raw_buy)
    normalized_sell = market._normalize_tick_map(raw_sell)

    buy, sell = market._resolve_entry_mdf_overlap(
        relative_ticks,
        market.price_to_tick(100.0),
        raw_buy,
        raw_sell,
        buy_aggression=0.25,
        sell_aggression=0.25,
    )

    _assert_mdf_map_is_finite_nonnegative_normalized("buy", buy)
    _assert_mdf_map_is_finite_nonnegative_normalized("sell", sell)
    assert buy[-1] < normalized_buy[-1]
    assert sell[-1] < normalized_sell[-1]
    assert buy[-4] > normalized_buy[-4]
    assert sell[4] > normalized_sell[4]
    assert _single_mdf_overlap(buy, sell) < _single_mdf_overlap(normalized_buy, normalized_sell)


def test_public_entry_mdfs_are_effective_overlap_resolved_for_seed_matrix():
    overlaps = []
    buy_passive = []
    sell_passive = []
    buy_cross = []
    sell_cross = []
    for seed in (31, 123, 987):
        market = Market(initial_price=100.0, gap=1.0, popularity=1.0, seed=seed, grid_radius=16)
        for step in market.step(120, keep_history=False)[20::5]:
            _assert_mdf_map_is_finite_nonnegative_normalized(
                f"seed={seed} step={step.step_index} buy_entry_mdf",
                step.buy_entry_mdf,
                min_effective_support=3.0,
            )
            _assert_mdf_map_is_finite_nonnegative_normalized(
                f"seed={seed} step={step.step_index} sell_entry_mdf",
                step.sell_entry_mdf,
                min_effective_support=3.0,
            )
            overlaps.append(_single_mdf_overlap(step.buy_entry_mdf, step.sell_entry_mdf))
            buy_passive.append(sum(v for tick, v in step.buy_entry_mdf.items() if tick < 0))
            sell_passive.append(sum(v for tick, v in step.sell_entry_mdf.items() if tick > 0))
            buy_cross.append(sum(v for tick, v in step.buy_entry_mdf.items() if tick > 0))
            sell_cross.append(sum(v for tick, v in step.sell_entry_mdf.items() if tick < 0))

    assert sum(overlaps) / len(overlaps) < 0.65
    assert max(overlaps) < 0.75
    assert sum(buy_passive) / len(buy_passive) > 0.55
    assert sum(sell_passive) / len(sell_passive) > 0.55
    assert 0.15 < sum(buy_cross) / len(buy_cross) < 0.45
    assert 0.15 < sum(sell_cross) / len(sell_cross) < 0.45


def test_entry_mdfs_show_diverse_shapes_across_market_conditions():
    scenarios = (
        ("normal", 11, 12, 0.08, 80),
        ("trend_up", 17, 12, 0.08, 80),
        ("trend_down", 23, 12, 0.08, 80),
        ("high_vol", 29, 12, 0.08, 80),
        ("thin_liquidity", 31, 5, 0.006, 120),
        ("squeeze", 37, 12, 0.08, 80),
    )
    shapes: dict[tuple[str, str], dict[int, float]] = {}

    for regime, seed, grid_radius, popularity, steps_count in scenarios:
        market = Market(
            initial_price=100.0,
            gap=1.0,
            popularity=popularity,
            seed=seed,
            grid_radius=grid_radius,
            regime=regime,
        )

        step = market.step(steps_count, keep_history=False)[-1]

        for name in ("buy_entry_mdf", "sell_entry_mdf"):
            shapes[(regime, name)] = getattr(step, name)

    for name in ("buy_entry_mdf", "sell_entry_mdf"):
        normal_shape = shapes[("normal", name)]
        distances = [
            _mdf_l1_distance(normal_shape, shapes[(regime, name)])
            for regime, *_ in scenarios
            if regime != "normal"
        ]
        pairwise_distances = [
            _mdf_l1_distance(left, right)
            for (left_regime, left_name), left in shapes.items()
            for (right_regime, right_name), right in shapes.items()
            if left_name == right_name and left_regime < right_regime
        ]

        assert max(distances) > 0.30
        assert sum(distance > 0.08 for distance in distances) >= 3
        assert sum(distance > 0.12 for distance in pairwise_distances) >= 6


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
        for relative_name, price_name in (
            ("buy_entry_mdf", "buy_entry_mdf_by_price"),
            ("sell_entry_mdf", "sell_entry_mdf_by_price"),
        ):
            projected = {
                market.tick_to_price(basis_tick + relative_tick): probability
                for relative_tick, probability in getattr(step, relative_name).items()
                if basis_tick + relative_tick >= 1
            }
            assert getattr(step, price_name) == pytest.approx(projected)


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


def test_mdf_signals_include_book_shortage_gap_and_front_observables():
    market = Market(initial_price=100.0, gap=1.0, seed=46, grid_radius=5)
    market._add_lots({99.0: 12.0}, "bid", "buy_entry")
    market._add_lots({101.0: 12.0}, "ask", "sell_entry")

    signals = market._mdf_signals(100.0)

    assert set(signals.bid_shortage_by_tick) == {-5, -4, -3, -2, -1}
    assert set(signals.ask_shortage_by_tick) == {1, 2, 3, 4, 5}
    assert signals.bid_shortage_by_tick[-2] > signals.bid_shortage_by_tick[-1]
    assert signals.ask_shortage_by_tick[2] > signals.ask_shortage_by_tick[1]
    assert signals.bid_front_by_tick[-2] > signals.bid_front_by_tick[-5]
    assert signals.ask_front_by_tick[2] > signals.ask_front_by_tick[5]
    assert all(value >= 0.0 for value in signals.bid_gap_by_tick.values())
    assert all(value >= 0.0 for value in signals.ask_gap_by_tick.values())


def test_entry_mdf_shape_changes_with_side_shortage_observables():
    near_shortage_market = Market(initial_price=100.0, gap=1.0, seed=46, grid_radius=6)
    deep_shortage_market = Market(initial_price=100.0, gap=1.0, seed=46, grid_radius=6)
    near_shortage_market._add_lots(
        {98.0: 12.0, 97.0: 12.0, 96.0: 12.0, 95.0: 12.0, 94.0: 12.0},
        "bid",
        "buy_entry",
    )
    deep_shortage_market._add_lots(
        {99.0: 12.0, 98.0: 12.0, 97.0: 12.0, 96.0: 12.0, 94.0: 12.0},
        "bid",
        "buy_entry",
    )

    near_mdf = near_shortage_market._next_mdf(
        100.0,
        near_shortage_market._price_grid(100.0),
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        step_index=1,
        update_memory=False,
    ).buy_entry_mdf
    deep_mdf = deep_shortage_market._next_mdf(
        100.0,
        deep_shortage_market._price_grid(100.0),
        mw.LatentState(mood=0.0, trend=0.0, volatility=0.2),
        step_index=1,
        update_memory=False,
    ).buy_entry_mdf

    assert _mdf_l1_distance(near_mdf, deep_mdf) > 0.05


def test_effective_mdf_reflects_raw_memory_without_freezing_initial_shape():
    market = Market(initial_price=100.0, gap=1.0, seed=37, grid_radius=6)

    first = market.step(1, keep_history=False)[0]
    market._add_lots(
        {99.0: 20.0, 98.0: 20.0, 97.0: 20.0, 96.0: 20.0, 94.0: 20.0},
        "bid",
        "buy_entry",
    )
    second = market.step(1, keep_history=False)[0]

    assert first.buy_entry_mdf != second.buy_entry_mdf
    assert _mdf_l1_distance(first.buy_entry_mdf, second.buy_entry_mdf) > 0.03


@pytest.mark.slow
def test_dynamic_mdf_acceptance_does_not_collapse_for_seed_matrix():
    peak_ticks = []
    for seed in range(10, 20):
        market = Market(
            initial_price=100.0,
            gap=1.0,
            popularity=1.0,
            seed=seed,
            grid_radius=12,
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
        current_price = step.price_after
        executions = _execution_count(step)
        if current_price != pytest.approx(previous_price, abs=1e-12):
            assert executions > 0
        previous_price = current_price


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


def test_cancellation_no_longer_erases_depth_on_nearly_every_step():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.2, seed=123, grid_radius=10)
    steps = _realism_steps_or_skip(market, 360)
    entry = [sum(step.entry_volume_by_price.values()) for step in steps]
    cancelled = [sum(step.cancelled_volume_by_price.values()) for step in steps]
    executed = [sum(step.executed_volume_by_price.values()) for step in steps]
    depth = [_total_orderbook_depth(step.orderbook_after) for step in steps]
    near_share = [
        _near_touch_depth(step.orderbook_after) / total
        for step, total in zip(steps, depth, strict=False)
        if total > 1e-12
    ]

    assert 125 <= sum(value > 1e-12 for value in cancelled) <= 300
    assert 180 <= sum(value > 1e-12 for value in executed) <= 310
    assert sum(cancelled) / len(cancelled) < 0.38 * (sum(entry) / len(entry))
    assert sum(near_share) / len(near_share) > 0.16
    assert sum(depth[-120:]) / 120.0 > 8.0


def test_default_entry_mdfs_build_passive_depth_without_killing_executions():
    market = Market(
        initial_price=100.0,
        gap=1.0,
        popularity=1.0,
        seed=31,
        grid_radius=12,
        regime="normal",
        augmentation_strength=0.0,
    )

    deep_bid_seen = False
    deep_ask_seen = False
    total_executed = 0.0
    for step in market.step(40, keep_history=False):
        price = step.price_after
        deep_bid = sum(
            volume
            for bid_price, volume in step.orderbook_after.bid_volume_by_price.items()
            if (price - bid_price) / market.gap >= 3
        )
        deep_ask = sum(
            volume
            for ask_price, volume in step.orderbook_after.ask_volume_by_price.items()
            if (ask_price - price) / market.gap >= 3
        )
        deep_bid_seen = deep_bid_seen or deep_bid > 0.5
        deep_ask_seen = deep_ask_seen or deep_ask > 0.5
        total_executed += step.total_executed_volume

    assert deep_bid_seen
    assert deep_ask_seen
    assert total_executed > 0.0


def test_stepinfo_exposes_market_realism_diagnostics():
    declared = {field.name for field in dataclasses.fields(StepInfo)}
    missing = sorted(set(REALISM_STEPINFO_FIELDS) - declared)

    assert not missing, "StepInfo is missing market realism fields: " + ", ".join(missing)

    market = Market(initial_price=100.0, gap=1.0, popularity=1.2, seed=41, grid_radius=10)
    step = _realism_steps_or_skip(market, 1)[0]

    for name in REALISM_STEPINFO_FIELDS:
        assert hasattr(step, name), f"StepInfo should expose {name}"

    assert step.residual_market_buy_volume >= 0.0
    assert step.residual_market_sell_volume >= 0.0
    assert step.crossed_market_volume == 0.0

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

    cancelled_total = _total_price_map(step.cancelled_volume_by_price)
    for price in step.cancelled_volume_by_price:
        assert price == market._snap_price(price)
    if _total_orderbook_depth(step.orderbook_before) <= 1e-12:
        assert cancelled_total == 0.0

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
def test_microstructure_activity_clusters_intensity():
    correlations = []
    follower_ratios = []
    for seed in (42, 77, 123, 404):
        market = Market(initial_price=100.0, gap=1.0, popularity=1.2, seed=seed, grid_radius=16)
        steps = _realism_steps_or_skip(market, 900)
        intensities = [step.intensity.total for step in steps]
        threshold = sorted(intensities)[int(0.75 * (len(intensities) - 1))]
        high_followers = [
            intensities[index + 1]
            for index, value in enumerate(intensities[:-1])
            if value >= threshold
        ]
        correlations.append(_lag1_correlation(intensities))
        follower_ratios.append(
            (sum(high_followers) / len(high_followers))
            / (sum(intensities) / len(intensities))
        )

    assert sum(correlations) / len(correlations) > 0.25
    assert sum(follower_ratios) / len(follower_ratios) > 1.05


@pytest.mark.slow
def test_microstructure_cancellation_is_bursty_but_nonnegative():
    burst_ratios = []
    for seed in (42, 77, 123, 404):
        market = Market(initial_price=100.0, gap=1.0, popularity=1.2, seed=seed, grid_radius=16)
        steps = _realism_steps_or_skip(market, 900)
        cancelled = [sum(step.cancelled_volume_by_price.values()) for step in steps]
        nonzero = [value for value in cancelled if value > 1e-12]

        assert len(nonzero) >= 100
        assert all(value >= -1e-12 for value in cancelled)
        top_decile = sorted(nonzero)[int(0.90 * (len(nonzero) - 1)) :]
        burst_ratios.append((sum(top_decile) / len(top_decile)) / (sum(nonzero) / len(nonzero)))

    assert sum(burst_ratios) / len(burst_ratios) > 1.20


@pytest.mark.slow
def test_l2_replenishment_keeps_near_touch_depth_without_overfilling_book():
    entry_total = 0.0
    cancel_total = 0.0
    near_shares = []
    near_depths = []
    late_depths = []
    late_far_shares = []
    late_top_shares = []
    for seed in (42, 77, 123, 404):
        market = Market(initial_price=100.0, gap=1.0, popularity=1.2, seed=seed, grid_radius=16)
        steps = _realism_steps_or_skip(market, 700)
        entry_total += sum(sum(step.entry_volume_by_price.values()) for step in steps)
        cancel_total += sum(sum(step.cancelled_volume_by_price.values()) for step in steps)
        late_depths.extend(_total_orderbook_depth(step.orderbook_after) for step in steps[-120:])
        for step in steps[-120:]:
            by_level = [
                step.orderbook_after.bid_volume_by_price.get(
                    step.price_after - level * market.gap, 0.0
                )
                + step.orderbook_after.ask_volume_by_price.get(
                    step.price_after + level * market.gap, 0.0
                )
                for level in range(1, 9)
            ]
            level_total = sum(by_level)
            if level_total > 1e-12:
                late_top_shares.append(sum(by_level[:2]) / level_total)
                late_far_shares.append(sum(by_level[4:]) / level_total)
        for step in steps:
            total_depth = _total_orderbook_depth(step.orderbook_after)
            if total_depth <= 1e-12:
                continue
            near_depth = _near_touch_depth(step.orderbook_after)
            near_depths.append(near_depth)
            near_shares.append(near_depth / total_depth)

    assert 0.15 <= cancel_total / max(entry_total, 1e-12) <= 0.45
    assert sum(near_shares) / len(near_shares) > 0.09
    assert sum(near_depths) / len(near_depths) > 18.0
    assert sum(late_depths) / len(late_depths) > 100.0
    assert sum(late_top_shares) / len(late_top_shares) > 0.20
    assert sum(late_far_shares) / len(late_far_shares) < 0.55


@pytest.mark.slow
def test_microstructure_regimes_change_activity_and_depth_shape():
    def summarize(regime):
        intensity = []
        volatility = []
        cancel = []
        depth = []
        spread = []
        near_share = []
        near_depth = []
        abs_change = []
        for seed in (11, 42, 77):
            market = Market(
                initial_price=100.0,
                gap=1.0,
                popularity=1.2,
                seed=seed,
                grid_radius=16,
                regime=regime,
            )
            steps = _realism_steps_or_skip(market, 700)
            intensity.extend(step.intensity.total for step in steps)
            volatility.extend(step.volatility for step in steps)
            cancel.extend(sum(step.cancelled_volume_by_price.values()) for step in steps)
            abs_change.extend(abs(step.price_change) for step in steps)
            for step in steps:
                total_depth = _total_orderbook_depth(step.orderbook_after)
                depth.append(total_depth)
                if total_depth > 1e-12:
                    near_touch_depth = _near_touch_depth(step.orderbook_after)
                    near_depth.append(near_touch_depth)
                    near_share.append(near_touch_depth / total_depth)
                if step.spread_after is not None:
                    spread.append(step.spread_after)
        return {
            "intensity": sum(intensity) / len(intensity),
            "volatility": sum(volatility) / len(volatility),
            "cancel": sum(cancel) / len(cancel),
            "cancel_rate": sum(cancel) / max(sum(depth), 1e-12),
            "depth": sum(depth) / len(depth),
            "spread": sum(spread) / len(spread),
            "near_depth": sum(near_depth) / len(near_depth),
            "near_share": sum(near_share) / len(near_share),
            "abs_change": sum(abs_change) / len(abs_change),
        }

    normal = summarize("normal")
    high_vol = summarize("high_vol")
    thin = summarize("thin_liquidity")

    assert high_vol["intensity"] > normal["intensity"]
    assert high_vol["volatility"] > normal["volatility"]
    assert high_vol["cancel_rate"] > normal["cancel_rate"]
    assert high_vol["abs_change"] > normal["abs_change"]
    assert thin["depth"] < normal["depth"]
    assert thin["spread"] >= normal["spread"]
    assert thin["near_depth"] < normal["near_depth"]


@pytest.mark.slow
def test_microstructure_events_do_not_lock_private_state_at_caps():
    for regime in ("high_vol", "squeeze"):
        sample_count = 0
        activity_cap_hits = 0
        cancel_cap_hits = 0
        spread_cap_hits = 0
        activity_events = []
        squeeze_pressure = []
        spread_pressure = []
        for seed in (42, 77, 123):
            market = Market(
                initial_price=100.0,
                gap=1.0,
                popularity=1.2,
                seed=seed,
                grid_radius=16,
                regime=regime,
            )

            for _ in range(700):
                market.step(1, keep_history=False)
                micro = market._microstructure
                sample_count += 1
                activity_cap_hits += micro.activity > 1.95
                cancel_cap_hits += micro.cancel_pressure > 1.95
                spread_cap_hits += micro.spread_pressure > 1.75
                activity_events.append(micro.activity_event)
                squeeze_pressure.append(micro.squeeze_pressure)
                spread_pressure.append(micro.spread_pressure)

        assert max(activity_events) > 0.6
        assert max(spread_pressure) > 0.05
        assert activity_cap_hits / sample_count < 0.01
        assert cancel_cap_hits / sample_count < 0.01
        assert spread_cap_hits / sample_count < 0.01
        if regime == "squeeze":
            assert max(squeeze_pressure) > 0.02
            assert sum(value > 0.01 for value in squeeze_pressure) >= 5


@pytest.mark.slow
def test_realism_imbalance_sign_matches_flow_and_price_direction_tolerantly():
    market = Market(initial_price=100.0, gap=1.0, popularity=1.5, seed=987, grid_radius=16)
    steps = _realism_steps_or_skip(market, 1000)

    flow_sign_matches = []
    directional = []
    for step in steps:
        flow_delta = step.market_buy_volume - step.market_sell_volume
        if abs(flow_delta) > 1e-12 and abs(step.order_flow_imbalance) > 0.03:
            flow_sign_matches.append(
                math.copysign(1.0, step.order_flow_imbalance)
                == math.copysign(1.0, flow_delta)
            )
        if abs(step.order_flow_imbalance) > 1e-12 and abs(step.price_change) > 1e-12:
            directional.append((step.order_flow_imbalance, step.price_change))

    assert len(flow_sign_matches) >= 20
    assert sum(flow_sign_matches) / len(flow_sign_matches) >= 0.60
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
