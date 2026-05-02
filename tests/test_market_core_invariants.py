from __future__ import annotations

from dataclasses import fields
from math import isfinite

from market_wave import Market, MDFState


def _assert_probability_map(values: dict[float, float] | dict[int, float]) -> None:
    assert values
    assert all(isfinite(value) and value >= 0.0 for value in values.values())
    assert abs(sum(values.values()) - 1.0) < 1e-9


def test_mdf_state_has_only_the_two_authoritative_entry_distributions() -> None:
    market = Market(100.0, 1.0, popularity=2.0, seed=7)
    step = market.step(1)[0]

    assert [field.name for field in fields(MDFState)] == [
        "buy_entry_mdf",
        "sell_entry_mdf",
    ]
    _assert_probability_map(step.buy_entry_mdf)
    _assert_probability_map(step.sell_entry_mdf)
    assert "buy_entry_mdf_by_price" not in step.to_dict()
    assert "sell_entry_mdf_by_price" not in step.to_dict()


def test_sampled_entry_and_cancel_prices_are_valid_book_ticks() -> None:
    market = Market(100.0, 0.5, popularity=3.0, seed=11)

    for step in market.step(30):
        sampled = (
            step.buy_volume_by_price
            | step.sell_volume_by_price
            | step.cancelled_volume_by_price
            | step.executed_volume_by_price
        )
        for price, volume in sampled.items():
            assert price >= market.tick_size
            assert price == market.tick_to_price(market.price_to_tick(price))
            assert isfinite(volume) and volume > 0.0


def test_cancel_events_remove_only_existing_book_volume() -> None:
    market = Market(100.0, 1.0, popularity=2.0, seed=19)
    market.step(20)

    for _ in range(20):
        before = market.snapshot().orderbook
        step = market.step(1)[0]
        for price, cancelled in step.cancelled_volume_by_price.items():
            available = 0.0
            available += before.bid_volume_by_price.get(price, 0.0)
            available += before.ask_volume_by_price.get(price, 0.0)
            assert cancelled <= available + 1e-9


def test_realized_flow_snapshot_matches_the_last_step_observation() -> None:
    market = Market(100.0, 1.0, popularity=2.0, seed=23)
    step = market.step(5)[-1]
    realized = market._realized_flow

    assert realized.return_ticks == step.tick_change
    assert realized.abs_return_ticks == abs(step.tick_change)
    assert realized.execution_volume == step.total_executed_volume
    assert realized.executed_by_price == step.executed_volume_by_price
    assert realized.cancelled_by_price == step.cancelled_volume_by_price
    assert realized.flow_imbalance == step.order_flow_imbalance
    assert realized.best_bid == step.best_bid_after
    assert realized.best_ask == step.best_ask_after
    assert realized.spread == step.spread_after
    assert realized.submitted_buy_volume == sum(step.buy_volume_by_price.values())
    assert realized.submitted_sell_volume == sum(step.sell_volume_by_price.values())


def test_hidden_participant_pressure_does_not_expand_public_state() -> None:
    market = Market(100.0, 1.0, popularity=2.0, seed=29)
    step = market.step(10)[-1]

    assert hasattr(market, "_participant_pressure")
    assert not hasattr(market.snapshot(), "participant_pressure")
    assert not hasattr(market.snapshot().mdf, "participant_pressure")
    assert "participant" not in step.to_dict()


def test_public_step_snapshot_does_not_expose_private_cluster_state() -> None:
    market = Market(100.0, 1.0, popularity=2.0, seed=29)
    step = market.step(10)[-1]
    record = step.to_dict()

    assert [key for key in record if key.endswith("_mdf")] == [
        "buy_entry_mdf",
        "sell_entry_mdf",
    ]
    assert all("clock" not in key for key in record)
    assert all("noise" not in key for key in record)
    assert all("arrival_cluster" not in key for key in record)


def test_mdf_center_is_public_basis_not_an_extra_mdf_distribution() -> None:
    market = Market(100.0, 1.0, popularity=2.0, seed=44)
    step = market.step(1)[0]
    snapshot = market.snapshot()

    assert hasattr(market, "_mdf_anchor_tick")
    assert hasattr(snapshot, "mdf_price_basis")
    assert [field.name for field in fields(MDFState)] == [
        "buy_entry_mdf",
        "sell_entry_mdf",
    ]
    assert step.mdf_price_basis == market.tick_to_price(market.price_to_tick(step.mdf_price_basis))
    assert snapshot.mdf_price_basis == market.tick_to_price(
        market.price_to_tick(snapshot.mdf_price_basis)
    )
    assert not hasattr(snapshot.mdf, "center")
    assert not hasattr(snapshot.mdf, "noise")


def test_mdf_anchor_moves_as_state_and_stays_near_current_price() -> None:
    market = Market(100.0, 1.0, popularity=3.0, seed=45, regime="high_vol")
    steps = market.step(200)
    max_lag = max(1.0, 0.70 * market.grid_radius)

    for step in steps:
        basis_tick = market.price_to_tick(step.mdf_price_basis)
        assert abs(basis_tick - step.tick_before) <= max_lag + 1.0

    moved = [step for step in steps if step.tick_change != 0]
    assert moved
    assert any(step.mdf_price_basis != step.price_before for step in steps[1:])


def test_submitted_entry_prices_are_sampled_from_public_mdf_support() -> None:
    market = Market(100.0, 1.0, popularity=4.0, seed=46)

    for step in market.step(80):
        basis_tick = market.price_to_tick(step.mdf_price_basis)
        buy_support = {
            market.tick_to_price(basis_tick + int(round(offset)))
            for offset, probability in step.buy_entry_mdf.items()
            if probability > 0.0 and basis_tick + int(round(offset)) >= 1
        }
        sell_support = {
            market.tick_to_price(basis_tick + int(round(offset)))
            for offset, probability in step.sell_entry_mdf.items()
            if probability > 0.0 and basis_tick + int(round(offset)) >= 1
        }

        assert set(step.buy_volume_by_price) <= buy_support
        assert set(step.sell_volume_by_price) <= sell_support


def test_unexecuted_intent_feeds_private_pressure_memory() -> None:
    from market_wave.market import _MicrostructureState, _RealizedFlow

    market = Market(100.0, 1.0, popularity=2.0, seed=30)
    market._realized_flow = _RealizedFlow(
        submitted_buy_volume=4.0,
        submitted_sell_volume=1.0,
        rested_buy_volume=3.0,
        rested_sell_volume=0.5,
        intent_imbalance=0.6,
        rested_imbalance=0.7,
    )
    pressure = market._next_participant_pressure(
        market.state.latent,
        _MicrostructureState(flow_persistence=0.4, meta_order_side=0.5),
        pre_imbalance=0.0,
    )

    assert pressure.signed_intent_memory > 0.0
    assert pressure.absorption > 0.0


def test_aligned_intent_and_execution_feeds_private_continuation() -> None:
    from market_wave.market import _MicrostructureState, _RealizedFlow

    market = Market(100.0, 1.0, popularity=2.0, seed=36)
    market._realized_flow = _RealizedFlow(
        return_ticks=1.0,
        submitted_buy_volume=4.0,
        submitted_sell_volume=1.0,
        intent_imbalance=0.6,
        flow_imbalance=0.55,
    )
    pressure = market._next_participant_pressure(
        market.state.latent,
        _MicrostructureState(flow_persistence=0.4, meta_order_side=0.5),
        pre_imbalance=0.0,
    )

    assert pressure.signed_intent_memory > 0.0
    assert pressure.flow_continuation > 0.0


def test_price_impact_direction_follows_executed_side() -> None:
    from market_wave.market import _MarketEvent, _TradeStats

    buy_market = Market(100.0, 1.0, popularity=2.0, seed=32)
    buy_market._orderbook.add_lot(101.0, 3.0, "ask", "seed")
    buy_stats = _TradeStats(executed_by_price={})
    buy_execution = buy_market._execute_market_flows(
        events=[_MarketEvent("buy_marketable", "buy", 101.0, 1.0)],
        stats=buy_stats,
    )
    assert buy_market._next_price_after_trading(100.0, buy_stats, buy_execution) > 100.0

    sell_market = Market(100.0, 1.0, popularity=2.0, seed=33)
    sell_market._orderbook.add_lot(99.0, 3.0, "bid", "seed")
    sell_stats = _TradeStats(executed_by_price={})
    sell_execution = sell_market._execute_market_flows(
        events=[_MarketEvent("sell_marketable", "sell", 99.0, 1.0)],
        stats=sell_stats,
    )
    assert sell_market._next_price_after_trading(100.0, sell_stats, sell_execution) < 100.0


def test_price_does_not_move_when_all_trades_print_at_previous_price() -> None:
    from market_wave.market import _MarketEvent, _TradeStats

    buy_market = Market(100.0, 1.0, popularity=2.0, seed=34)
    buy_market._orderbook.add_lot(100.0, 3.0, "ask", "seed")
    buy_stats = _TradeStats(executed_by_price={})
    buy_execution = buy_market._execute_market_flows(
        events=[_MarketEvent("buy_marketable", "buy", 100.0, 1.0)],
        stats=buy_stats,
    )
    assert buy_market._next_price_after_trading(100.0, buy_stats, buy_execution) == 100.0

    sell_market = Market(100.0, 1.0, popularity=2.0, seed=35)
    sell_market._orderbook.add_lot(100.0, 3.0, "bid", "seed")
    sell_stats = _TradeStats(executed_by_price={})
    sell_execution = sell_market._execute_market_flows(
        events=[_MarketEvent("sell_marketable", "sell", 100.0, 1.0)],
        stats=sell_stats,
    )
    assert sell_market._next_price_after_trading(100.0, sell_stats, sell_execution) == 100.0


def test_price_mark_follows_dominant_executed_side_not_last_print_only() -> None:
    from market_wave.market import _ExecutionResult, _TradeStats

    market = Market(100.0, 1.0, popularity=2.0, seed=38)
    stats = _TradeStats(executed_by_price={})
    stats.record(99.0, 5.0, "sell")
    stats.record(101.0, 1.0, "buy")
    execution = _ExecutionResult(
        residual_market_buy=0.0,
        residual_market_sell=0.0,
        crossed_market_volume=0.0,
        market_buy_volume=1.0,
        market_sell_volume=5.0,
    )

    assert stats.last_price == 101.0
    assert market._next_price_after_trading(100.0, stats, execution) < 100.0


def test_side_specific_cancels_sum_to_public_cancel_map() -> None:
    market = Market(100.0, 1.0, popularity=4.0, seed=31)
    market.step(100)
    realized = market._realized_flow

    merged: dict[float, float] = {}
    for values in (realized.bid_cancelled_by_price, realized.ask_cancelled_by_price):
        for price, volume in values.items():
            merged[price] = merged.get(price, 0.0) + volume
    assert merged == realized.cancelled_by_price


def test_same_seed_remains_deterministic_after_event_interleaving() -> None:
    left = Market(100.0, 1.0, popularity=3.0, seed=37).step(50)
    right = Market(100.0, 1.0, popularity=3.0, seed=37).step(50)

    assert [step.to_dict() for step in left] == [step.to_dict() for step in right]


def test_price_does_not_move_without_execution() -> None:
    market = Market(100.0, 1.0, popularity=3.0, seed=41)

    for step in market.step(100):
        if step.total_executed_volume <= 1e-12:
            assert step.price_after == step.price_before


def test_cancel_interleaving_does_not_cancel_same_step_new_liquidity() -> None:
    market = Market(100.0, 1.0, popularity=1.0, seed=43)
    market._orderbook.add_lot(99.0, 1.0, "bid", "seed")

    from market_wave.market import _MarketEvent, _TradeStats

    stats = _TradeStats(executed_by_price={})
    execution = market._execute_market_flows(
        events=[
            _MarketEvent("sell_marketable", "sell", 99.0, 1.0),
            _MarketEvent("buy_limit_add", "buy", 99.0, 2.0),
            _MarketEvent("bid_cancel", "bid", 99.0, 1.0),
        ],
        stats=stats,
    )

    assert execution.bid_cancelled_volume_by_price == {}
    assert execution.cancelled_volume_by_price == {}
    assert market._snapshot_orderbook().bid_volume_by_price == {99.0: 2.0}


def test_same_step_limit_add_rests_before_becoming_executable_next_step() -> None:
    market = Market(100.0, 1.0, popularity=1.0, seed=48)

    from market_wave.market import _MarketEvent, _TradeStats

    stats = _TradeStats(executed_by_price={})
    execution = market._execute_market_flows(
        events=[
            _MarketEvent("buy_limit_add", "buy", 99.0, 2.0),
            _MarketEvent("sell_marketable", "sell", 99.0, 1.25),
        ],
        stats=stats,
    )

    assert stats.executed_by_price == {}
    assert execution.market_sell_volume == 0.0
    assert execution.residual_market_buy == 2.0
    assert market._snapshot_orderbook().bid_volume_by_price == {99.0: 2.0}

    next_stats = _TradeStats(executed_by_price={})
    next_execution = market._execute_market_flows(
        events=[
            _MarketEvent("sell_marketable", "sell", 99.0, 1.25),
        ],
        stats=next_stats,
    )

    assert next_stats.executed_by_price == {99.0: 1.25}
    assert next_execution.market_sell_volume == 1.25
    assert market._snapshot_orderbook().bid_volume_by_price == {99.0: 0.75}


def test_ioc_order_unfilled_remainder_does_not_rest_as_liquidity() -> None:
    market = Market(100.0, 1.0, popularity=1.0, seed=50)
    market._orderbook.add_lot(99.0, 1.0, "bid", "seed")

    from market_wave.market import _MarketEvent, _TradeStats

    stats = _TradeStats(executed_by_price={})
    execution = market._execute_market_flows(
        events=[
            _MarketEvent("sell_ioc", "sell", 98.0, 3.0),
        ],
        stats=stats,
    )

    assert stats.executed_by_price == {99.0: 1.0}
    assert execution.market_sell_volume == 1.0
    assert execution.residual_market_sell == 0.0
    assert market._snapshot_orderbook().ask_volume_by_price == {}
    assert market._snapshot_orderbook().bid_volume_by_price == {}


def test_aggressive_order_does_not_rest_when_opposite_touch_is_missing() -> None:
    market = Market(100.0, 1.0, popularity=1.0, seed=51)

    from market_wave.market import _IncomingOrder

    buy_event = market._event_type_for_incoming_order(
        _IncomingOrder("buy", "buy_entry", 101.0, 1.0)
    )
    sell_event = market._event_type_for_incoming_order(
        _IncomingOrder("sell", "sell_entry", 99.0, 1.0)
    )

    assert buy_event == "buy_marketable"
    assert sell_event == "sell_marketable"


def test_cancel_budget_tracks_starting_liquidity_after_same_step_execution() -> None:
    market = Market(100.0, 1.0, popularity=1.0, seed=49)
    market._orderbook.add_lot(99.0, 2.0, "bid", "seed")

    from market_wave.market import _MarketEvent, _TradeStats

    stats = _TradeStats(executed_by_price={})
    execution = market._execute_market_flows(
        events=[
            _MarketEvent("sell_marketable", "sell", 99.0, 1.25),
            _MarketEvent("buy_limit_add", "buy", 99.0, 4.0),
            _MarketEvent("bid_cancel", "bid", 99.0, 3.0),
        ],
        stats=stats,
    )

    assert stats.executed_by_price == {99.0: 1.25}
    assert execution.bid_cancelled_volume_by_price == {99.0: 0.75}
    assert execution.cancelled_volume_by_price == {99.0: 0.75}
    assert market._snapshot_orderbook().bid_volume_by_price == {99.0: 4.0}


def test_same_price_addition_blends_quote_age_without_fifo_queue() -> None:
    market = Market(100.0, 1.0, popularity=1.0, seed=47)
    market._add_resting_lot("bid", 99.0, 9.0, "seed")
    market._set_quote_age("bid", 99.0, 10)

    market._add_resting_lot("bid", 99.0, 1.0, "buy_entry")

    assert market._snapshot_orderbook().bid_volume_by_price == {99.0: 10.0}
    assert market._quote_age("bid", 99.0) == 9


def test_stale_quote_refresh_restarts_age_only_for_requoted_volume() -> None:
    market = Market(100.0, 1.0, popularity=3.0, seed=52)
    market._add_resting_lot("bid", 90.0, 10.0, "seed")
    market._add_resting_lot("ask", 110.0, 8.0, "seed")
    market._set_quote_age("bid", 90.0, 7)
    market._set_quote_age("ask", 110.0, 8)
    mean_age_before = market._mean_quote_age()

    market._repair_missing_touch(100.0)
    book = market._snapshot_orderbook()
    mean_age_after = market._mean_quote_age(book)

    assert book.bid_volume_by_price[90.0] < 10.0
    assert book.bid_volume_by_price[97.0] > 0.0
    assert book.ask_volume_by_price[110.0] < 8.0
    assert book.ask_volume_by_price[103.0] > 0.0
    assert mean_age_after < mean_age_before
    assert market._quote_age("bid", 90.0) == 7
    assert market._quote_age("ask", 110.0) == 8
    assert market._quote_age("bid", 97.0) == 0
    assert market._quote_age("ask", 103.0) == 0

    step = market.step(1)[0]
    assert step.mean_quote_age >= 0.0
    assert step.to_dict()["mean_quote_age"] == step.mean_quote_age


def test_sparse_book_boundary_reprice_preserves_quote_age() -> None:
    market = Market(100.0, 1.0, popularity=0.1, seed=53)
    market._add_resting_lot("bid", 99.0, 6.0, "seed")
    market._add_resting_lot("bid", 98.0, 2.0, "seed")
    market._add_resting_lot("ask", 101.0, 4.0, "seed")
    market._set_quote_age("bid", 99.0, 6)
    market._set_quote_age("bid", 98.0, 2)
    market._set_quote_age("ask", 101.0, 5)

    market._repair_missing_touch(100.0)
    book = market._snapshot_orderbook()

    assert 99.0 not in book.bid_volume_by_price
    assert 98.0 not in book.bid_volume_by_price
    assert book.bid_volume_by_price[97.0] == 8.0
    assert market._quote_age("bid", 97.0) == 5
    assert 101.0 not in book.ask_volume_by_price
    assert book.ask_volume_by_price[103.0] == 4.0
    assert market._quote_age("ask", 103.0) == 5


def test_stale_outer_book_expiry_reduces_aged_far_quotes_symmetrically() -> None:
    market = Market(100.0, 1.0, popularity=1.0, seed=55)
    market._add_resting_lot("bid", 75.0, 10.0, "seed")
    market._add_resting_lot("ask", 125.0, 12.0, "seed")
    market._add_resting_lot("bid", 96.0, 3.0, "seed")
    market._add_resting_lot("ask", 104.0, 4.0, "seed")
    market._set_quote_age("bid", 75.0, 30)
    market._set_quote_age("ask", 125.0, 30)
    market._set_quote_age("bid", 96.0, 2)
    market._set_quote_age("ask", 104.0, 2)

    market._expire_stale_outer_book(100.0)
    book = market._snapshot_orderbook()

    assert book.bid_volume_by_price[75.0] < 10.0
    assert book.ask_volume_by_price[125.0] < 12.0
    assert book.bid_volume_by_price[96.0] == 3.0
    assert book.ask_volume_by_price[104.0] == 4.0


def test_extreme_outer_book_quotes_expire_instead_of_widening_visible_range() -> None:
    market = Market(100.0, 1.0, popularity=3.0, seed=56)
    market._add_resting_lot("bid", 40.0, 10.0, "seed")
    market._add_resting_lot("ask", 160.0, 12.0, "seed")
    market._set_quote_age("bid", 40.0, 1)
    market._set_quote_age("ask", 160.0, 1)

    market._expire_stale_outer_book(100.0)
    book = market._snapshot_orderbook()

    assert 40.0 not in book.bid_volume_by_price
    assert 160.0 not in book.ask_volume_by_price
    assert market._quote_age("bid", 40.0) == 0
    assert market._quote_age("ask", 160.0) == 0


def test_outer_book_dust_expires_before_it_sets_visible_range() -> None:
    market = Market(100.0, 1.0, popularity=1.0, seed=57)
    market._add_resting_lot("bid", 75.0, 0.02, "seed")
    market._add_resting_lot("ask", 125.0, 0.02, "seed")
    market._add_resting_lot("bid", 96.0, 0.02, "seed")
    market._add_resting_lot("ask", 104.0, 0.02, "seed")

    market._expire_stale_outer_book(100.0)
    book = market._snapshot_orderbook()

    assert 75.0 not in book.bid_volume_by_price
    assert 125.0 not in book.ask_volume_by_price
    assert book.bid_volume_by_price[96.0] == 0.02
    assert book.ask_volume_by_price[104.0] == 0.02


def test_multi_market_family_invariants_hold_without_exact_path_targets() -> None:
    cases = [
        dict(initial_price=100.0, gap=1.0, popularity=1.0, seed=101),
        dict(initial_price=100.0, gap=1.0, popularity=3.0, seed=101),
        dict(initial_price=100.0, gap=1.0, popularity=0.25, seed=101),
        dict(initial_price=1.0, gap=1.0, popularity=2.0, seed=101),
        dict(initial_price=10_000.0, gap=5.0, popularity=1.5, seed=101),
        dict(initial_price=100.0, gap=1.0, popularity=1.0, seed=101, regime="high_vol"),
        dict(initial_price=100.0, gap=1.0, popularity=1.0, seed=101, regime="trend_up"),
        dict(initial_price=100.0, gap=1.0, popularity=0.0, seed=101),
    ]

    for config in cases:
        market = Market(**config)
        for step in market.step(120):
            assert step.price_after >= market.tick_size
            assert not market._is_crossed()
            _assert_probability_map(step.buy_entry_mdf)
            _assert_probability_map(step.sell_entry_mdf)
            for volume in (
                *step.orderbook_after.bid_volume_by_price.values(),
                *step.orderbook_after.ask_volume_by_price.values(),
            ):
                assert isfinite(volume) and volume > 0.0
            if step.total_executed_volume <= 1e-12:
                assert step.price_after == step.price_before


def test_busy_market_has_more_activity_than_thin_market_for_same_seed() -> None:
    thin = Market(100.0, 1.0, popularity=0.25, seed=211).step(200)
    busy = Market(100.0, 1.0, popularity=3.0, seed=211).step(200)

    thin_exec = sum(step.total_executed_volume for step in thin)
    busy_exec = sum(step.total_executed_volume for step in busy)
    thin_trades = sum(step.trade_count for step in thin)
    busy_trades = sum(step.trade_count for step in busy)

    assert busy_exec > thin_exec
    assert busy_trades > thin_trades


def test_child_lots_are_small_and_busy_activity_comes_from_more_draws() -> None:
    normal = Market(100.0, 1.0, popularity=1.0, seed=233)
    busy = Market(100.0, 1.0, popularity=3.0, seed=233)

    assert busy._mean_child_order_size() < normal._mean_child_order_size() * 1.25
    assert normal._mean_child_order_size() < 0.14
    assert busy._mean_child_order_size() < 0.16

    normal_steps = normal.step(200)
    busy_steps = busy.step(200)

    normal_submitted = sum(
        sum(step.buy_volume_by_price.values()) + sum(step.sell_volume_by_price.values())
        for step in normal_steps
    )
    busy_submitted = sum(
        sum(step.buy_volume_by_price.values()) + sum(step.sell_volume_by_price.values())
        for step in busy_steps
    )
    normal_events = sum(
        len(step.buy_volume_by_price) + len(step.sell_volume_by_price) for step in normal_steps
    )
    busy_events = sum(
        len(step.buy_volume_by_price) + len(step.sell_volume_by_price) for step in busy_steps
    )

    assert busy_events > normal_events
    assert busy_submitted > normal_submitted


def test_fixed_time_slice_generation_allows_silent_steps() -> None:
    base = Market(100.0, 1.0, popularity=1.0, seed=211).step(300)
    busy = Market(100.0, 1.0, popularity=3.0, seed=211).step(300)
    thin = Market(100.0, 1.0, popularity=0.25, seed=211).step(300)

    def frozen_steps(path) -> int:
        return sum(
            step.total_executed_volume <= 1e-12
            and sum(step.entry_volume_by_price.values()) <= 1e-12
            and sum(step.cancelled_volume_by_price.values()) <= 1e-12
            and step.residual_market_buy_volume <= 1e-12
            and step.residual_market_sell_volume <= 1e-12
            and step.trade_count == 0
            for step in path
        )

    base_silent = frozen_steps(base)
    busy_silent = frozen_steps(busy)
    thin_silent = frozen_steps(thin)

    assert 0 < base_silent < len(base)
    assert busy_silent < len(busy)
    assert 0 < thin_silent < len(thin)
    assert thin_silent > busy_silent


def test_inactive_market_stays_flat_and_empty() -> None:
    steps = Market(100.0, 1.0, popularity=0.0, seed=307).step(50)

    assert all(step.price_after == step.price_before for step in steps)
    assert all(step.total_executed_volume == 0.0 for step in steps)
    assert all(step.entry_volume_by_price == {} for step in steps)
