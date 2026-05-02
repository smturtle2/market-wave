from __future__ import annotations

from copy import deepcopy
from math import isfinite, log1p
from random import Random

from ._conditions import _MarketConditionsMixin
from ._execution import _MarketExecutionMixin
from ._mdf import _MarketMDFMixin
from ._microstructure import _MarketMicrostructureMixin
from ._orderbook import _OrderBook as _OrderBook
from ._plotting import _MarketPlottingMixin
from ._types import (
    _EntryFlow,
    _EventSizeState,
    _ExecutionResult,
    _IncomingOrder,
    _MarketConditionInputs,
    _MarketConditionState,
    _MarketEvent,
    _MicrostructureInputs,
    _MicrostructureState,
    _ParticipantPressureState,
    _ProcessedOrder,
    _RealizedFlow,
    _StepComputationCache,
    _TradeStats,
)
from .state import (
    IntensityState,
    MarketState,
    MDFState,
    OrderBookState,
    StepInfo,
)

_PRIVATE_COMPAT_EXPORTS = (
    _EntryFlow,
    _EventSizeState,
    _ExecutionResult,
    _IncomingOrder,
    _MarketEvent,
    _MarketConditionInputs,
    _MarketConditionState,
    _MicrostructureInputs,
    _MicrostructureState,
    _ParticipantPressureState,
    _ProcessedOrder,
    _RealizedFlow,
    _StepComputationCache,
    _TradeStats,
)


class Market(
    _MarketConditionsMixin,
    _MarketPlottingMixin,
    _MarketMicrostructureMixin,
    _MarketMDFMixin,
    _MarketExecutionMixin,
):
    """Synthetic order-book market driven by buy/sell entry MDFs.

    ``Market`` is the main public entry point. Configure it with plain
    parameters, call :meth:`step`, and inspect the returned ``StepInfo`` objects
    or the live ``state`` snapshot.

    Args:
        initial_price: Positive starting price. It is snapped to the configured
            ``gap`` tick grid.
        gap: Positive price distance for one simulator tick.
        popularity: Non-negative activity scale. Higher values create more
            submitted volume, trades, and book replenishment. ``0`` produces an
            inactive market.
        seed: Optional random seed for reproducible paths.
        grid_radius: Number of gap-unit offsets on each side of the MDF grid.
        augmentation_strength: Non-negative extra texture/noise scale.
        regime: Initial market condition only. Supported values are
            ``"normal"``, ``"trend_up"``, ``"trend_down"``, ``"high_vol"``,
            ``"thin_liquidity"``, ``"squeeze"``, and ``"auto"``. Later
            ``StepInfo.regime`` values are active condition labels produced by
            simulator state transitions, not a fixed constructor label.

    Attributes:
        state: Live ``MarketState`` snapshot for the current step.
        history: List of ``StepInfo`` objects returned by previous calls to
            :meth:`step`.
        initial_regime: Constructor regime used to seed the initial market
            condition.
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        gap: float = 1.0,
        popularity: float = 1.0,
        seed: int | None = None,
        grid_radius: int = 20,
        augmentation_strength: float = 0.0,
        regime: str = "normal",
    ) -> None:
        if not isfinite(initial_price) or initial_price <= 0:
            raise ValueError("initial_price must be a positive finite number")
        if not isfinite(gap) or gap <= 0:
            raise ValueError("gap must be a positive finite number")
        if grid_radius < 1:
            raise ValueError("grid_radius must be at least 1")
        if not isfinite(popularity) or popularity < 0:
            raise ValueError("popularity must be a non-negative finite number")
        if not isfinite(augmentation_strength) or augmentation_strength < 0:
            raise ValueError("augmentation_strength must be a non-negative finite number")
        if regime not in self._regime_names() | {"auto"}:
            raise ValueError("regime must be one of the supported regimes or 'auto'")

        self.gap = float(gap)
        self._gap_is_integer = self.gap.is_integer()
        self.popularity = float(popularity)
        self._min_price = self.gap
        self.grid_radius = int(grid_radius)
        self.augmentation_strength = float(augmentation_strength)
        self.initial_regime = regime
        self._market_condition = self._condition_preset(regime)
        self._active_regime = self._condition_label(self._market_condition)
        self._active_settings = self._condition_settings(self._market_condition)
        self._rng = Random(seed)
        self._seed = seed
        self.history: list[StepInfo] = []
        self._orderbook = _OrderBook()
        self._realized_flow = _RealizedFlow()
        self._microstructure = _MicrostructureState()
        self._participant_pressure = _ParticipantPressureState()
        self._quote_age_by_side: dict[str, dict[float, int]] = {"bid": {}, "ask": {}}

        price = self._snap_price(float(initial_price))
        self._reference_price = price
        tick_grid = self.relative_tick_grid()
        current_tick = self.price_to_tick(price)
        self._mdf_anchor_tick = float(current_tick)
        mdf_price_basis = self._mdf_anchor_price()
        grid = self._price_grid(mdf_price_basis)
        intensity = IntensityState(total=0.0, buy=0.0, sell=0.0, buy_ratio=0.5, sell_ratio=0.5)
        latent = self._initial_latent()
        initial_mdf = {tick: 1.0 / len(tick_grid) for tick in tick_grid}
        mdf = MDFState(
            buy_entry_mdf=initial_mdf,
            sell_entry_mdf=initial_mdf.copy(),
        )
        self.state = MarketState(
            price=price,
            step_index=0,
            current_tick=current_tick,
            tick_grid=tick_grid,
            intensity=intensity,
            latent=latent,
            price_grid=grid,
            mdf_price_basis=mdf_price_basis,
            mdf=mdf,
            orderbook=OrderBookState(),
        )


    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def tick_size(self) -> float:
        return self.gap

    def snapshot(self) -> MarketState:
        """Return a deep copy of the current public market state.

        ``Market.state`` is retained as the current alpha-compatible state object
        and contains mutable plain containers. Use this method when downstream
        code needs to inspect state without risking accidental mutation of the
        live market object.
        """

        return deepcopy(self.state)

    def step(self, n: int) -> list[StepInfo]:
        """Advance the market by ``n`` steps and return per-step observations.

        The returned ``StepInfo`` objects are also appended to ``history``.
        ``n`` must be non-negative.
        """

        if n < 0:
            raise ValueError("n must be non-negative")

        steps = [self._step_once() for _ in range(n)]
        self.history.extend(steps)
        return steps

    def history_records(self) -> list[dict]:
        """Return ``history`` as JSON-friendly dictionaries."""

        return [step.to_dict() for step in self.history]

    def _step_once(self) -> StepInfo:
        state = self.state
        step_index = state.step_index + 1
        price_before = state.price
        self._clean_orderbook()
        self._age_state()

        self._mdf_anchor_tick = self._constrain_mdf_anchor_tick(
            self._mdf_anchor_tick,
            self.price_to_tick(price_before),
        )
        mdf_price_basis = self._mdf_anchor_price()
        price_grid = self._price_grid(mdf_price_basis)
        orderbook_before = self._snapshot_orderbook()
        best_bid_before = self._best_bid()
        best_ask_before = self._best_ask()
        spread_before = self._spread(best_bid_before, best_ask_before)
        pre_imbalance = self._near_touch_imbalance(price_before)

        condition_inputs = self._market_condition_inputs(state.latent, pre_imbalance)
        self._market_condition = self._next_market_condition(
            self._market_condition,
            condition_inputs,
        )
        self._active_settings = self._condition_settings(self._market_condition)
        self._active_regime = self._condition_label(self._market_condition)
        latent = self._next_latent(state.latent)
        micro = self._next_microstructure_state(state.latent, latent, pre_imbalance)
        self._participant_pressure = self._next_participant_pressure(latent, micro, pre_imbalance)
        intensity = self._next_intensity(latent, micro)
        pre_trade_cache = _StepComputationCache(mdf_price_basis, micro=micro)
        mdf = self._next_mdf(
            mdf_price_basis,
            price_grid,
            latent,
            step_index=step_index,
            cache=pre_trade_cache,
            current_price=price_before,
        )
        pre_trade_cache.mdf = mdf

        entry_flow = self._entry_flow(
            intensity,
            mdf,
            mdf_price_basis=mdf_price_basis,
            current_price=price_before,
        )

        stats = _TradeStats(executed_by_price={})
        execution = self._execute_market_flows(
            entry_orders=entry_flow.orders,
            events=entry_flow.events,
            stats=stats,
        )
        self._clean_orderbook()

        price_after = self._next_price_after_trading(price_before, stats, execution)
        price_after = self._snap_price(price_after)
        total_market_buy = execution.market_buy_volume
        total_market_sell = execution.market_sell_volume
        residual_market_buy = execution.residual_market_buy
        residual_market_sell = execution.residual_market_sell
        cancelled_volume = self._drop_zeroes(execution.cancelled_volume_by_price)
        order_flow_imbalance = self._step_order_flow_imbalance(
            price_after,
            execution,
        )
        self._clean_orderbook()
        self._repair_missing_touch(price_after)
        self._clean_orderbook()

        orderbook_after = self._snapshot_orderbook()
        mean_quote_age = self._mean_quote_age(orderbook_after)
        best_bid_after = self._best_bid()
        best_ask_after = self._best_ask()
        spread_after = self._spread(best_bid_after, best_ask_after)
        buy_volume = self._drop_zeroes(entry_flow.buy_intent_by_price)
        sell_volume = self._drop_zeroes(entry_flow.sell_intent_by_price)
        entry_volume = self._merge_maps(buy_volume, sell_volume)
        submitted_buy = sum(buy_volume.values())
        submitted_sell = sum(sell_volume.values())
        submitted_total = submitted_buy + submitted_sell
        rested_buy = residual_market_buy
        rested_sell = residual_market_sell
        rested_total = rested_buy + rested_sell
        residual_total = residual_market_buy + residual_market_sell
        self._realized_flow = _RealizedFlow(
            return_ticks=(price_after - price_before) / self.gap,
            abs_return_ticks=abs((price_after - price_before) / self.gap),
            execution_volume=stats.total_volume,
            submitted_buy_volume=submitted_buy,
            submitted_sell_volume=submitted_sell,
            rested_buy_volume=rested_buy,
            rested_sell_volume=rested_sell,
            executed_by_price=self._drop_zeroes(stats.executed_by_price),
            cancelled_by_price=cancelled_volume,
            bid_cancelled_by_price=self._drop_zeroes(execution.bid_cancelled_volume_by_price),
            ask_cancelled_by_price=self._drop_zeroes(execution.ask_cancelled_volume_by_price),
            flow_imbalance=order_flow_imbalance,
            intent_imbalance=(
                0.0
                if submitted_total <= 1e-12
                else self._clamp((submitted_buy - submitted_sell) / submitted_total, -1.0, 1.0)
            ),
            rested_imbalance=(
                0.0
                if rested_total <= 1e-12
                else self._clamp((rested_buy - rested_sell) / rested_total, -1.0, 1.0)
            ),
            residual_imbalance=(
                0.0
                if residual_total <= 1e-12
                else self._clamp(
                    (residual_market_buy - residual_market_sell) / residual_total,
                    -1.0,
                    1.0,
                )
            ),
            best_bid=best_bid_after,
            best_ask=best_ask_after,
            spread=spread_after,
        )

        vwap = stats.notional / stats.total_volume if stats.total_volume > 0 else None
        step_info = StepInfo(
            step_index=step_index,
            price_before=price_before,
            price_after=price_after,
            price_change=price_after - price_before,
            tick_before=self.price_to_tick(price_before),
            tick_after=self.price_to_tick(price_after),
            tick_change=self.price_to_tick(price_after) - self.price_to_tick(price_before),
            intensity=intensity,
            mood=latent.mood,
            trend=latent.trend,
            volatility=latent.volatility,
            regime=self._active_regime,
            augmentation_strength=self.augmentation_strength,
            price_grid=price_grid,
            mdf_price_basis=mdf_price_basis,
            buy_entry_mdf=mdf.buy_entry_mdf,
            sell_entry_mdf=mdf.sell_entry_mdf,
            buy_volume_by_price=buy_volume,
            sell_volume_by_price=sell_volume,
            entry_volume_by_price=entry_volume,
            cancelled_volume_by_price=cancelled_volume,
            executed_volume_by_price=self._drop_zeroes(stats.executed_by_price),
            total_executed_volume=stats.total_volume,
            market_buy_volume=total_market_buy,
            market_sell_volume=total_market_sell,
            crossed_market_volume=execution.crossed_market_volume,
            residual_market_buy_volume=residual_market_buy,
            residual_market_sell_volume=residual_market_sell,
            trade_count=stats.trade_count,
            vwap_price=vwap,
            best_bid_before=best_bid_before,
            best_ask_before=best_ask_before,
            best_bid_after=best_bid_after,
            best_ask_after=best_ask_after,
            spread_before=spread_before,
            spread_after=spread_after,
            mean_quote_age=mean_quote_age,
            order_flow_imbalance=order_flow_imbalance,
            orderbook_before=orderbook_before,
            orderbook_after=orderbook_after,
        )

        reference_pull = self._clamp(
            0.010 + 0.006 * self._microstructure.resiliency,
            0.006,
            0.030,
        )
        next_reference_price = max(
            self._min_price,
            (1.0 - reference_pull) * self._reference_price
            + reference_pull * price_after,
        )
        self._reference_price = next_reference_price
        self._mdf_anchor_tick = self._next_mdf_anchor_tick(price_after)
        next_mdf_price_basis = self._mdf_anchor_price()
        next_price_grid = self._price_grid(next_mdf_price_basis)
        next_mdf = self._next_mdf(
            next_mdf_price_basis,
            next_price_grid,
            latent,
            step_index=step_index,
            current_price=price_after,
        )

        self.state = MarketState(
            price=price_after,
            step_index=step_index,
            current_tick=self.price_to_tick(price_after),
            tick_grid=self.relative_tick_grid(),
            intensity=intensity,
            latent=latent,
            price_grid=next_price_grid,
            mdf_price_basis=next_mdf_price_basis,
            mdf=next_mdf,
            orderbook=orderbook_after,
        )
        return step_info

    def _age_state(self) -> None:
        self._sync_quote_ages()
        for side in ("bid", "ask"):
            for price in list(self._quote_age_by_side[side]):
                self._quote_age_by_side[side][price] += 1

    def _step_order_flow_imbalance(
        self,
        current_price: float,
        execution: _ExecutionResult,
    ) -> float:
        del current_price
        trade_delta = execution.market_buy_volume - execution.market_sell_volume
        trade_total = execution.market_buy_volume + execution.market_sell_volume
        executed = 0.0 if trade_total <= 1e-12 else trade_delta / trade_total
        residual_total = execution.residual_market_buy + execution.residual_market_sell
        residual = (
            0.0
            if residual_total <= 1e-12
            else (execution.residual_market_buy - execution.residual_market_sell) / residual_total
        )
        bid_cancel = sum(execution.bid_cancelled_volume_by_price.values())
        ask_cancel = sum(execution.ask_cancelled_volume_by_price.values())
        cancel_total = bid_cancel + ask_cancel
        cancel_imbalance = (
            0.0
            if cancel_total <= 1e-12
            else (ask_cancel - bid_cancel) / cancel_total
        )
        return self._clamp(0.72 * executed + 0.16 * residual + 0.12 * cancel_imbalance, -1.0, 1.0)

    def _next_price_after_trading(
        self,
        price_before: float,
        stats: _TradeStats,
        execution: _ExecutionResult,
    ) -> float:
        cancel_total = sum(execution.cancelled_volume_by_price.values())
        cancel_imbalance = 0.0
        if cancel_total > 1e-12:
            bid_cancel = sum(execution.bid_cancelled_volume_by_price.values())
            ask_cancel = sum(execution.ask_cancelled_volume_by_price.values())
            cancel_imbalance = self._clamp((ask_cancel - bid_cancel) / cancel_total, -1.0, 1.0)
        position_mark = self._book_position_mark(price_before)
        position_confirmation = self._clamp(
            log1p(cancel_total / max(0.7, self.popularity + 0.1)) / 1.8,
            0.0,
            1.0,
        )
        if stats.total_volume <= 1e-12:
            return price_before
        if stats.last_price is None or stats.notional <= 1e-12:
            return price_before
        execution_vwap = stats.notional / stats.total_volume
        range_ticks = (
            0.0
            if stats.min_price is None or stats.max_price is None
            else (stats.max_price - stats.min_price) / self.gap
        )
        range_pressure = self._clamp(range_ticks / 8.0, 0.0, 1.0) * self._clamp(
            log1p(stats.trade_count) / 5.2,
            0.0,
            1.0,
        )
        if stats.min_price is None or stats.max_price is None:
            range_mark = execution_vwap
        else:
            down_distance = abs(stats.min_price - price_before)
            up_distance = abs(stats.max_price - price_before)
            range_mark = stats.max_price if up_distance >= down_distance else stats.min_price
        range_direction = self._clamp((range_mark - price_before) / max(self.gap, 1e-12), -1.0, 1.0)
        high_participation = self._clamp((self.popularity - 1.0) / 2.5, 0.0, 1.0)
        volatility_context = self._clamp(
            (self._active_settings["volatility"] - 1.0) / 0.75,
            0.0,
            1.0,
        )
        volume_confirmation = self._clamp(
            (stats.total_volume + 0.55 * cancel_total)
            / max(1.0, self.popularity * self._active_settings["liquidity"] * 2.8),
            0.0,
            1.0,
        )
        range_weight = range_pressure * (
            0.22 * high_participation + 0.18 * volatility_context
        )
        signed_total = stats.buy_volume + stats.sell_volume
        execution_imbalance = (
            0.0 if signed_total <= 1e-12 else (stats.buy_volume - stats.sell_volume) / signed_total
        )
        residual_total = execution.residual_market_buy + execution.residual_market_sell
        residual_imbalance = (
            0.0
            if residual_total <= 1e-12
            else (execution.residual_market_buy - execution.residual_market_sell) / residual_total
        )
        direct_impact_ticks = self._clamp(
            0.72 * execution_imbalance + 0.28 * residual_imbalance,
            -1.0,
            1.0,
        ) * (0.16 + 0.75 * volume_confirmation + 0.28 * range_pressure)
        range_impact_ticks = (
            range_direction
            * range_pressure
            * (0.13 + 0.50 * volume_confirmation + 0.18 * high_participation)
        )
        book_release = self._clamp(
            0.18 * volume_confirmation + 0.22 * abs(execution_imbalance) + 0.18 * range_pressure,
            0.0,
            0.46,
        )
        if signed_total <= 1e-12:
            mark = (1.0 - range_weight) * execution_vwap + range_weight * range_mark
            mark = price_before + (mark - price_before) * (
                0.45 + 0.55 * volume_confirmation + 0.12 * high_participation
            )
            mark += direct_impact_ticks * self.gap
            mark += range_impact_ticks * self.gap
            mark += cancel_imbalance * position_confirmation * self.gap
            return self._finalize_execution_mark(
                price_before,
                mark,
                stats,
                position_mark=position_mark,
                position_confirmation=position_confirmation,
                volume_confirmation=volume_confirmation,
                range_pressure=range_pressure,
                book_release=book_release,
            )

        if abs(execution_imbalance) <= 0.15:
            mark = (1.0 - range_weight) * execution_vwap + range_weight * range_mark
            mark = price_before + (mark - price_before) * (
                0.45 + 0.55 * volume_confirmation + 0.12 * high_participation
            )
            mark += direct_impact_ticks * self.gap
            mark += range_impact_ticks * self.gap
            mark += cancel_imbalance * position_confirmation * self.gap
            return self._finalize_execution_mark(
                price_before,
                mark,
                stats,
                position_mark=position_mark,
                position_confirmation=position_confirmation,
                volume_confirmation=volume_confirmation,
                range_pressure=range_pressure,
                book_release=book_release,
            )

        side_prices = (
            stats.buy_executed_by_price
            if execution_imbalance > 0.0
            else stats.sell_executed_by_price
        )
        side_volume = stats.buy_volume if execution_imbalance > 0.0 else stats.sell_volume
        if side_volume <= 1e-12:
            return max(self._min_price, execution_vwap)
        side_vwap = sum(price * volume for price, volume in side_prices.items()) / side_volume
        extreme_price = stats.max_price if execution_imbalance > 0.0 else stats.min_price
        if extreme_price is None:
            return max(self._min_price, side_vwap)
        sweep_pressure = self._clamp((abs(execution_imbalance) - 0.15) / 0.85, 0.0, 1.0)
        mark = (
            (0.36 - 0.05 * sweep_pressure - 0.08 * range_weight) * execution_vwap
            + (0.46 - 0.11 * sweep_pressure - 0.06 * range_weight) * side_vwap
            + (0.18 + 0.16 * sweep_pressure - 0.04 * range_weight) * extreme_price
            + 0.18 * range_weight * range_mark
        )
        mark = price_before + (mark - price_before) * (
            0.45 + 0.55 * volume_confirmation + 0.12 * high_participation
        )
        mark += direct_impact_ticks * self.gap
        mark += range_impact_ticks * self.gap
        mark += cancel_imbalance * position_confirmation * self.gap
        return self._finalize_execution_mark(
            price_before,
            mark,
            stats,
            position_mark=position_mark,
            position_confirmation=position_confirmation,
            volume_confirmation=volume_confirmation,
            range_pressure=range_pressure,
            book_release=book_release,
        )

    def _finalize_execution_mark(
        self,
        price_before: float,
        mark: float,
        stats: _TradeStats,
        *,
        position_mark: float | None,
        position_confirmation: float,
        volume_confirmation: float,
        range_pressure: float,
        book_release: float,
    ) -> float:
        mark = self._blend_position_mark(mark, position_mark, position_confirmation)
        mark = self._directional_mark_bias(mark, volume_confirmation, range_pressure)
        return max(
            self._min_price,
            self._book_consistent_mark(
                self._liquid_mark_with_confirmation(price_before, mark, stats),
                release=book_release,
            ),
        )

    def _liquid_mark_with_confirmation(
        self,
        price_before: float,
        mark: float,
        stats: _TradeStats,
    ) -> float:
        expected_depth = max(self._expected_nearby_depth(), 1.0)
        confirmation = self._clamp(stats.total_volume / expected_depth, 0.0, 1.0)
        mark_weight = self._clamp(0.35 + 0.65 * confirmation, 0.35, 1.0)
        confirmed_mark = price_before + (mark - price_before) * mark_weight
        volatility_context = self._clamp(
            (self._active_settings["volatility"] - 1.0) / 0.75,
            0.0,
            1.0,
        )
        recent_shock = self._clamp(self._realized_flow.abs_return_ticks / 4.0, 0.0, 1.0)
        mark_weight = self._clamp(
            mark_weight + 0.15 * volatility_context * recent_shock,
            0.35,
            1.0,
        )
        confirmed_mark = price_before + (mark - price_before) * mark_weight
        max_move_ticks = 1.0 + 3.8 * confirmation + 1.2 * volatility_context
        max_move_ticks += 1.2 * volatility_context * recent_shock
        max_move = max_move_ticks * self.gap
        confirmed_mark = self._clamp(
            confirmed_mark,
            price_before - max_move,
            price_before + max_move,
        )
        high_participation = self._clamp((self.popularity - 1.0) / 2.5, 0.0, 1.0)
        if high_participation <= 1e-12:
            return confirmed_mark
        high_participation_weight = self._clamp(0.30 + 0.70 * confirmation, 0.30, 1.0)
        return (
            1.0 - high_participation * (1.0 - high_participation_weight)
        ) * confirmed_mark + (
            high_participation * (1.0 - high_participation_weight)
        ) * price_before

    def _book_position_mark(self, price_before: float) -> float | None:
        best_bid = self._best_bid()
        best_ask = self._best_ask()
        if best_bid is not None and best_ask is not None:
            return 0.5 * (best_bid + best_ask)
        if best_bid is not None:
            return max(self._min_price, best_bid + self.gap)
        if best_ask is not None:
            return max(self._min_price, best_ask - self.gap)
        return None

    def _blend_position_mark(
        self,
        mark: float,
        position_mark: float | None,
        position_confirmation: float,
    ) -> float:
        if position_mark is None or position_confirmation <= 1e-12:
            return mark
        weight = self._clamp(0.18 + 0.42 * position_confirmation, 0.0, 0.54)
        return (1.0 - weight) * mark + weight * position_mark

    def _directional_mark_bias(
        self,
        mark: float,
        volume_confirmation: float,
        range_pressure: float,
    ) -> float:
        regime_direction = self._clamp(self._active_settings["trend"] / 0.08, -1.0, 1.0)
        if abs(regime_direction) <= 1e-12:
            return mark
        directional_ticks = regime_direction * (
            0.18 + 0.46 * volume_confirmation + 0.22 * range_pressure
        )
        return mark + directional_ticks * self.gap

    def _repair_missing_touch(self, current_price: float) -> None:
        high_participation = self._clamp((self.popularity - 1.0) / 2.5, 0.0, 1.0)
        trend_pressure = abs(self._clamp(self._active_settings["trend"] / 0.30, -1.0, 1.0))
        volatility_refresh = self._clamp(
            (self._active_settings["volatility"] - 1.0) / 0.75,
            0.0,
            1.0,
        )
        stale_refresh = max(
            0.38 * high_participation,
            0.30 * trend_pressure,
            0.18 * volatility_refresh,
        )
        if stale_refresh > 1e-12:
            refresh_share = stale_refresh
            bid_target = self._snap_price(current_price - 3.0 * self.gap)
            ask_target = self._snap_price(current_price + 3.0 * self.gap)
            if bid_target >= self._min_price:
                refreshed_bid = 0.0
                for price, volume in list(self._orderbook.bid_volume_by_price.items()):
                    if current_price - price > 5.0 * self.gap:
                        moved = volume * refresh_share
                        self._orderbook.adjust_volume("bid", price, -moved)
                        if self._orderbook.bid_volume_by_price.get(price, 0.0) <= 1e-12:
                            self._quote_age_by_side["bid"].pop(price, None)
                        refreshed_bid += moved
                if refreshed_bid > 1e-12:
                    self._add_resting_lot("bid", bid_target, refreshed_bid, "stale_refresh")
            refreshed_ask = 0.0
            for price, volume in list(self._orderbook.ask_volume_by_price.items()):
                if price - current_price > 5.0 * self.gap:
                    moved = volume * refresh_share
                    self._orderbook.adjust_volume("ask", price, -moved)
                    if self._orderbook.ask_volume_by_price.get(price, 0.0) <= 1e-12:
                        self._quote_age_by_side["ask"].pop(price, None)
                    refreshed_ask += moved
            if refreshed_ask > 1e-12:
                self._add_resting_lot("ask", ask_target, refreshed_ask, "stale_refresh")
        self._expire_stale_outer_book(current_price)
        sparse_book = self._clamp((0.80 - self.popularity) / 0.80, 0.0, 1.0)
        if sparse_book > 1e-12:
            target_spread_ticks = 3.5 + 3.0 * sparse_book
            half_spread = 0.5 * target_spread_ticks * self.gap
            bid_boundary = self._snap_price(current_price - half_spread)
            ask_boundary = self._snap_price(current_price + half_spread)
            if bid_boundary < ask_boundary:
                moved_bid = 0.0
                moved_bid_age = 0.0
                for price, volume in list(self._orderbook.bid_volume_by_price.items()):
                    if price > bid_boundary:
                        moved_bid += volume
                        moved_bid_age += volume * self._quote_age("bid", price)
                        del self._orderbook.bid_volume_by_price[price]
                        self._quote_age_by_side["bid"].pop(price, None)
                if moved_bid > 1e-12 and bid_boundary >= self._min_price:
                    existing_bid = self._orderbook.bid_volume_by_price.get(bid_boundary, 0.0)
                    existing_bid_age = self._quote_age("bid", bid_boundary)
                    self._orderbook.bid_volume_by_price[bid_boundary] = (
                        existing_bid + moved_bid
                    )
                    self._set_quote_age(
                        "bid",
                        bid_boundary,
                        round(
                            (existing_bid_age * existing_bid + moved_bid_age)
                            / (existing_bid + moved_bid)
                        ),
                    )
                moved_ask = 0.0
                moved_ask_age = 0.0
                for price, volume in list(self._orderbook.ask_volume_by_price.items()):
                    if price < ask_boundary:
                        moved_ask += volume
                        moved_ask_age += volume * self._quote_age("ask", price)
                        del self._orderbook.ask_volume_by_price[price]
                        self._quote_age_by_side["ask"].pop(price, None)
                if moved_ask > 1e-12:
                    existing_ask = self._orderbook.ask_volume_by_price.get(ask_boundary, 0.0)
                    existing_ask_age = self._quote_age("ask", ask_boundary)
                    self._orderbook.ask_volume_by_price[ask_boundary] = (
                        existing_ask + moved_ask
                    )
                    self._set_quote_age(
                        "ask",
                        ask_boundary,
                        round(
                            (existing_ask_age * existing_ask + moved_ask_age)
                            / (existing_ask + moved_ask)
                        ),
                    )
                if moved_bid > 1e-12:
                    self._orderbook.invalidate("bid")
                if moved_ask > 1e-12:
                    self._orderbook.invalidate("ask")
        bid_missing = self._best_bid() is None
        ask_missing = self._best_ask() is None
        if not bid_missing and not ask_missing:
            return
        repair_base = max(
            0.05,
            0.16 * self.popularity * self._active_settings["liquidity"],
        )
        repair_distance = self.gap * (1.0 + 3.0 * sparse_book)
        trend = self._clamp(self._active_settings["trend"] / 0.30, -1.0, 1.0)
        if bid_missing:
            best_ask = self._best_ask()
            price = self._snap_price(current_price - repair_distance)
            if best_ask is not None:
                price = min(price, self._snap_price(best_ask - self.gap))
            if price >= self._min_price:
                volume = repair_base * (1.0 + 0.55 * max(0.0, -trend))
                self._add_resting_lot("bid", price, volume, "touch_repair")
        if ask_missing:
            best_bid = self._best_bid()
            price = self._snap_price(current_price + repair_distance)
            if best_bid is not None:
                price = max(price, self._snap_price(best_bid + self.gap))
            volume = repair_base * (1.0 + 0.55 * max(0.0, trend))
            self._add_resting_lot("ask", price, volume, "touch_repair")

    def _expire_stale_outer_book(self, current_price: float) -> None:
        if self.popularity <= 1e-12:
            return
        resiliency = self._clamp(self._microstructure.resiliency, 0.2, 1.4)
        activity = self._clamp(
            self._microstructure.activity
            + 0.70 * self._microstructure.activity_event
            + 0.50 * self._microstructure.cancel_burst,
            0.0,
            4.0,
        ) / 4.0
        max_resting_distance = self._clamp(
            15.0
            + 0.26 * self.grid_radius
            - 4.0 * activity
            - 2.0 * self._clamp(self.popularity - 1.0, 0.0, 3.0) / 3.0,
            10.0,
            22.0,
        )
        stale_age = self._clamp(10.0 + 5.0 * resiliency - 3.0 * activity, 7.0, 17.0)
        hard_expire_distance = self._clamp(
            max_resting_distance + 8.0 + 0.20 * self.grid_radius,
            24.0,
            36.0,
        )
        outer_dust_volume = max(0.015, 0.06 * self.popularity)
        for side in ("bid", "ask"):
            for price, volume in list(self._orderbook.volumes_for_side(side).items()):
                distance = abs(price - current_price) / max(self.gap, 1e-12)
                age = self._quote_age(side, price)
                if distance > hard_expire_distance or (
                    distance > max_resting_distance and volume <= outer_dust_volume
                ):
                    self._orderbook.adjust_volume(side, price, -volume)
                    self._quote_age_by_side[side].pop(price, None)
                    continue
                if distance <= max_resting_distance and age <= stale_age:
                    continue
                distance_pressure = self._clamp(
                    (distance - max_resting_distance) / max(3.0, 0.25 * self.grid_radius),
                    0.0,
                    1.0,
                )
                age_pressure = self._clamp((age - stale_age) / max(4.0, stale_age), 0.0, 1.0)
                expiry_share = self._clamp(
                    0.22 * distance_pressure
                    + 0.30 * age_pressure
                    + 0.36 * distance_pressure * age_pressure
                    + 0.18 * activity * max(distance_pressure, age_pressure),
                    0.0,
                    0.92,
                )
                if expiry_share <= 1e-12:
                    continue
                self._orderbook.adjust_volume(side, price, -volume * expiry_share)
                if self._orderbook.volumes_for_side(side).get(price, 0.0) <= 1e-12:
                    self._quote_age_by_side[side].pop(price, None)

    def _book_consistent_mark(self, mark: float, *, release: float = 0.0) -> float:
        best_bid = self._best_bid()
        best_ask = self._best_ask()
        if best_bid is None and best_ask is None:
            return mark
        if best_bid is not None and best_ask is not None:
            if best_bid <= mark <= best_ask:
                return mark
            mid = 0.5 * (best_bid + best_ask)
            if mark < best_bid:
                clamped = self._snap_price(0.72 * best_bid + 0.28 * mid)
                return self._snap_price((1.0 - release) * clamped + release * mark)
            clamped = self._snap_price(0.72 * best_ask + 0.28 * mid)
            return self._snap_price((1.0 - release) * clamped + release * mark)
        if best_bid is not None and mark < best_bid:
            return self._snap_price((1.0 - release) * best_bid + release * mark)
        if best_ask is not None and mark > best_ask:
            return self._snap_price((1.0 - release) * best_ask + release * mark)
        return mark

    def _price_grid(self, center_price: float) -> list[float]:
        center = self._snap_price(center_price)
        return self._dedupe_prices(
            self._snap_price(center + offset * self.gap)
            for offset in range(-self.grid_radius, self.grid_radius + 1)
        )

    def price_to_tick(self, price: float) -> int:
        return max(1, int(round(price / self.gap)))

    def tick_to_price(self, tick: int) -> float:
        snapped = max(1, int(tick)) * self.gap
        if self._gap_is_integer:
            return float(int(snapped))
        return self._clean_number(snapped)

    def _price_from_tick_float(self, tick: float) -> float:
        return self.tick_to_price(int(round(max(1.0, tick))))

    def _mdf_anchor_price(self) -> float:
        return self._price_from_tick_float(self._mdf_anchor_tick)

    def _constrain_mdf_anchor_tick(self, anchor_tick: float, current_tick: int) -> float:
        max_lag = max(1.0, 0.70 * self.grid_radius)
        constrained = self._clamp(
            float(anchor_tick),
            current_tick - max_lag,
            current_tick + max_lag,
        )
        return max(1.0, constrained)

    def _next_mdf_anchor_tick(self, price_after: float) -> float:
        current_tick = self.price_to_tick(price_after)
        previous = self._constrain_mdf_anchor_tick(self._mdf_anchor_tick, current_tick)
        distance = current_tick - previous
        micro = self._microstructure
        realized = self._realized_flow
        book_imbalance = self._near_touch_imbalance(price_after)
        book_asymmetry = abs(book_imbalance)
        cancelled = sum(realized.cancelled_by_price.values())
        bid_cancelled = sum(realized.bid_cancelled_by_price.values())
        ask_cancelled = sum(realized.ask_cancelled_by_price.values())
        cancel_total = bid_cancelled + ask_cancelled
        cancel_confirmation = self._clamp(
            log1p(cancelled / max(0.7, self.popularity + 0.1)) / 1.8,
            0.0,
            1.0,
        )
        cancel_imbalance = (
            0.0
            if cancel_total <= 1e-12
            else self._clamp((ask_cancelled - bid_cancelled) / cancel_total, -1.0, 1.0)
        )
        activity = self._clamp(
            micro.activity + micro.activity_event + 0.55 * micro.arrival_cluster,
            0.0,
            4.6,
        ) / 4.6
        stress = self._clamp(
            micro.liquidity_stress
            + 0.65 * micro.volatility_cluster
            + 0.45 * micro.displacement_pressure,
            0.0,
            4.2,
        ) / 4.2
        confirmation = self._clamp(
            log1p(max(0.0, realized.execution_volume) / max(0.7, self.popularity + 0.1)) / 2.0,
            0.0,
            1.0,
        )
        event_pressure = self._clamp(
            log1p(
                (realized.execution_volume + cancelled)
                * abs(realized.flow_imbalance)
                / max(0.7, self.popularity + 0.1)
            )
            / 1.6,
            0.0,
            1.0,
        )
        event_confirmation = max(confirmation, cancel_confirmation, event_pressure)
        intent_confirmation = self._clamp(
            abs(realized.intent_imbalance) * 0.65 + abs(realized.rested_imbalance) * 0.35,
            0.0,
            1.0,
        )
        signed_flow = self._clamp(
            0.48 * realized.flow_imbalance
            + 0.30 * realized.intent_imbalance
            + 0.14 * realized.rested_imbalance,
            -1.0,
            1.0,
        )
        position_flow = self._clamp(
            0.50 * signed_flow + 0.24 * cancel_imbalance + 0.26 * book_imbalance,
            -1.0,
            1.0,
        )
        execution_position_impulse = self._clamp(
            realized.return_ticks / 4.0,
            -1.0,
            1.0,
        ) * (0.24 + 0.76 * confirmation)
        event_position_impulse = (
            signed_flow
            * confirmation
            * (0.36 + 0.64 * max(intent_confirmation, book_asymmetry))
        )
        cancel_position_impulse = cancel_imbalance * cancel_confirmation * (
            0.28 + 0.72 * book_asymmetry
        )
        drought = self._clamp(micro.liquidity_drought, 0.0, 2.0) / 2.0
        follow = self._clamp(
            0.010
            + 0.200 * confirmation
            + 0.130 * cancel_confirmation
            + 0.050 * intent_confirmation
            + 0.220 * book_asymmetry * max(event_confirmation, intent_confirmation)
            + 0.230 * event_pressure
            + 0.130 * event_pressure * book_asymmetry
            + 0.018 * activity
            + 0.026 * stress
            - 0.026 * drought,
            0.006,
            0.520,
        )
        event_catchup_impulse = (
            self._clamp(distance, -1.0, 1.0)
            * event_pressure
            * (0.32 + 0.36 * book_asymmetry)
            * (0.35 + 0.65 * stress)
        )
        event_direction = self._clamp(
            0.55 * self._clamp(realized.return_ticks / 4.0, -1.0, 1.0)
            + 0.45 * position_flow,
            -1.0,
            1.0,
        )
        event_direction_impulse = event_direction * event_pressure * (
            0.28 + 0.42 * stress + 0.24 * book_asymmetry
        )
        impulse = (
            0.58
            * position_flow
            * (0.32 + 0.68 * max(event_confirmation, intent_confirmation))
            * (0.45 + 0.55 * stress)
            + 0.62 * execution_position_impulse
            + 0.52 * event_position_impulse
            + 0.36 * cancel_position_impulse
            + 0.72 * book_imbalance * book_asymmetry * event_confirmation
            + event_catchup_impulse
            + event_direction_impulse
        )
        proposed = previous + follow * distance + impulse
        max_step = (
            0.85
            + 1.35 * stress
            + 1.95 * max(event_confirmation, intent_confirmation)
            + 1.15 * book_asymmetry
            + 1.10 * event_pressure
        )
        proposed = previous + self._clamp(proposed - previous, -max_step, max_step)
        return self._constrain_mdf_anchor_tick(proposed, current_tick)

    def relative_tick_grid(self) -> list[int]:
        return list(range(-self.grid_radius, self.grid_radius + 1))

    def _snap_price(self, price: float) -> float:
        tick = max(1, int(round(price / self.gap)))
        snapped = tick * self.gap
        if self._gap_is_integer:
            return float(int(snapped))
        return self._clean_number(snapped)

    def _clean_number(self, value: float) -> float:
        rounded = round(value, 10)
        if rounded.is_integer():
            return float(int(rounded))
        return rounded
