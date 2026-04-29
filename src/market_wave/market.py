from __future__ import annotations

from copy import deepcopy
from math import isfinite
from random import Random

from ._conditions import _MarketConditionsMixin
from ._execution import _MarketExecutionMixin
from ._mdf import _MarketMDFMixin
from ._microstructure import _MarketMicrostructureMixin
from ._orderbook import _OrderBook as _OrderBook
from ._plotting import _MarketPlottingMixin
from ._types import (
    _EntryFlow,
    _ExecutionResult,
    _IncomingOrder,
    _MarketConditionInputs,
    _MarketConditionState,
    _MDFJudgmentSample,
    _MDFSideJudgment,
    _MicrostructureInputs,
    _MicrostructureState,
    _ProcessedOrder,
    _StepComputationCache,
    _TradeStats,
)
from .state import (
    IntensityState,
    LatentState,
    MarketState,
    MDFState,
    OrderBookState,
    PriceMap,
    StepInfo,
)

_PRIVATE_COMPAT_EXPORTS = (
    _EntryFlow,
    _ExecutionResult,
    _IncomingOrder,
    _MarketConditionInputs,
    _MarketConditionState,
    _MDFJudgmentSample,
    _MDFSideJudgment,
    _MicrostructureInputs,
    _MicrostructureState,
    _ProcessedOrder,
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
    def __init__(
        self,
        initial_price: float,
        gap: float,
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
        self._entry_mdf_memory = 0.18
        self._mdf_floor_mix = 0.012
        self._entry_noise_mix = 0.023
        self._entry_noise_sigma_ticks = self._clamp(0.14 * self.grid_radius, 1.25, 2.50)
        self.regime = regime
        self._market_condition = self._condition_preset(regime)
        self._active_regime = self._condition_label(self._market_condition)
        self._active_settings = self._condition_settings(self._market_condition)
        self._rng = Random(seed)
        self._seed = seed
        self.history: list[StepInfo] = []
        self._orderbook = _OrderBook()
        self._last_return_ticks = 0.0
        self._last_abs_return_ticks = 0.0
        self._last_imbalance = 0.0
        self._last_execution_volume = 0.0
        self._last_executed_by_price: PriceMap = {}
        self._price_residual_ticks = 0.0
        self._microstructure = _MicrostructureState()
        self._mdf_memory: dict[str, dict[int, float]] = {}
        self._mdf_judgment_memory: dict[str, _MDFSideJudgment] = {}

        price = self._snap_price(float(initial_price))
        grid = self._price_grid(price)
        tick_grid = self.relative_tick_grid()
        current_tick = self.price_to_tick(price)
        intensity = IntensityState(total=0.0, buy=0.0, sell=0.0, buy_ratio=0.5, sell_ratio=0.5)
        latent = self._initial_latent()
        uniform = 1.0 / len(grid)
        initial_mdf_by_price = {price_level: uniform for price_level in grid}
        initial_mdf = {tick: 1.0 / len(tick_grid) for tick in tick_grid}
        mdf = MDFState(
            relative_ticks=tick_grid,
            buy_entry_mdf=initial_mdf,
            sell_entry_mdf=initial_mdf.copy(),
            buy_entry_mdf_by_price=initial_mdf_by_price,
            sell_entry_mdf_by_price=initial_mdf_by_price.copy(),
        )
        self.state = MarketState(
            price=price,
            step_index=0,
            current_tick=current_tick,
            tick_grid=tick_grid,
            intensity=intensity,
            latent=latent,
            price_grid=grid,
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

    def step(self, n: int, *, keep_history: bool = True) -> list[StepInfo]:
        if n < 0:
            raise ValueError("n must be non-negative")

        steps = [self._step_once() for _ in range(n)]
        if keep_history:
            self.history.extend(steps)
        return steps

    def stream(self, n: int, *, keep_history: bool = False):
        if n < 0:
            raise ValueError("n must be non-negative")
        for _ in range(n):
            step = self._step_once()
            if keep_history:
                self.history.append(step)
            yield step

    def history_records(self) -> list[dict]:
        return [step.to_dict() for step in self.history]

    def _step_once(self) -> StepInfo:
        state = self.state
        step_index = state.step_index + 1
        price_before = state.price
        self._clean_orderbook()

        price_grid = self._price_grid(price_before)
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
        intensity = self._next_intensity(latent, micro)
        pre_trade_cache = _StepComputationCache(price_before, micro=micro)
        mdf = self._next_mdf(
            price_before,
            price_grid,
            latent,
            step_index=step_index,
            update_memory=True,
            cache=pre_trade_cache,
        )
        pre_trade_cache.mdf = mdf

        cancelled_volume = self._cancel_orders(price_before, latent, micro, mdf, pre_trade_cache)
        entry_flow = self._entry_flow(intensity, mdf)

        stats = _TradeStats(executed_by_price={})
        execution = self._execute_market_flows(
            entry_orders=entry_flow.orders,
            stats=stats,
        )
        self._clean_orderbook()

        price_after = self._next_price_after_trading(price_before, stats, execution, latent)
        price_after = self._snap_price(price_after)
        self._last_return_ticks = (price_after - price_before) / self.gap
        self._last_abs_return_ticks = abs(self._last_return_ticks)
        self._last_execution_volume = stats.total_volume
        self._last_executed_by_price = self._drop_zeroes(stats.executed_by_price)
        total_market_buy = execution.market_buy_volume
        total_market_sell = execution.market_sell_volume
        residual_market_buy = execution.residual_market_buy
        residual_market_sell = execution.residual_market_sell
        order_flow_imbalance = self._step_order_flow_imbalance(
            price_after,
            execution,
            cancelled_volume,
        )
        self._last_imbalance = order_flow_imbalance
        self._trim_orderbook_through_last_price(price_after)
        self._prune_orderbook_window(price_after)
        post_trade_mdf = self._reproject_mdf(price_after, mdf)
        post_trade_cache = _StepComputationCache(price_after, mdf=post_trade_mdf, micro=micro)
        self._add_post_event_quote_arrivals(
            price_after,
            latent,
            micro,
            stats,
            order_flow_imbalance,
            post_trade_mdf,
            post_trade_cache,
        )
        self._refresh_post_event_deep_quotes(
            cancelled_volume,
            price_after,
            micro,
            post_trade_mdf,
            post_trade_cache,
        )
        cancelled_volume = self._drop_zeroes(cancelled_volume)
        micro.last_cancelled_volume = sum(cancelled_volume.values())
        self._trim_orderbook_through_last_price(price_after)
        self._clean_orderbook()

        state_grid = self._price_grid(price_after)
        state_mdf = post_trade_mdf
        orderbook_after = self._snapshot_orderbook()
        best_bid_after = self._best_bid()
        best_ask_after = self._best_ask()
        spread_after = self._spread(best_bid_after, best_ask_after)

        entry_volume = self._merge_maps(
            entry_flow.buy_intent_by_price,
            entry_flow.sell_intent_by_price,
        )
        buy_volume = entry_flow.buy_intent_by_price
        sell_volume = entry_flow.sell_intent_by_price
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
            mdf_price_basis=price_before,
            relative_ticks=mdf.relative_ticks,
            buy_entry_mdf=mdf.buy_entry_mdf,
            sell_entry_mdf=mdf.sell_entry_mdf,
            buy_entry_mdf_by_price=mdf.buy_entry_mdf_by_price,
            sell_entry_mdf_by_price=mdf.sell_entry_mdf_by_price,
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
            order_flow_imbalance=order_flow_imbalance,
            orderbook_before=orderbook_before,
            orderbook_after=orderbook_after,
        )

        self.state = MarketState(
            price=price_after,
            step_index=step_index,
            current_tick=self.price_to_tick(price_after),
            tick_grid=self.relative_tick_grid(),
            intensity=intensity,
            latent=latent,
            price_grid=state_grid,
            mdf=state_mdf,
            orderbook=orderbook_after,
        )
        return step_info

    def _age_state(self) -> None:
        return None

    def _step_order_flow_imbalance(
        self,
        current_price: float,
        execution: _ExecutionResult,
        cancelled_volume: PriceMap,
    ) -> float:
        trade_delta = execution.market_buy_volume - execution.market_sell_volume
        trade_total = execution.market_buy_volume + execution.market_sell_volume
        residual_delta = execution.residual_market_buy - execution.residual_market_sell
        residual_total = execution.residual_market_buy + execution.residual_market_sell
        cancel_delta = 0.0
        cancel_total = 0.0
        for price, volume in cancelled_volume.items():
            if volume <= 1e-12:
                continue
            cancel_total += volume
            if price < current_price:
                cancel_delta -= volume
            elif price > current_price:
                cancel_delta += volume
        near_depth = self._nearby_book_depth("bid", current_price) + self._nearby_book_depth(
            "ask",
            current_price,
        )
        near_imbalance = self._near_touch_imbalance(current_price)
        context_numerator = (
            0.24 * residual_delta
            + 0.12 * cancel_delta
            + 0.10 * near_imbalance * near_depth
        )
        context_denominator = (
            0.50 * residual_total
            + 0.45 * cancel_total
            + 0.32 * near_depth
            + 2.0 * self._mean_child_order_size()
        )
        if context_denominator <= 1e-12 and trade_total <= 1e-12:
            return 0.0
        context_imbalance = (
            context_numerator / context_denominator
            if context_denominator > 1e-12
            else 0.0
        )
        if trade_total <= 1e-12:
            return self._clamp(context_imbalance, -1.0, 1.0)
        trade_imbalance = self._clamp(trade_delta / trade_total, -1.0, 1.0)
        trade_confidence = self._clamp(
            trade_total
            / (
                trade_total
                + 0.50 * residual_total
                + 0.35 * cancel_total
                + 0.22 * near_depth
                + self._mean_child_order_size()
            ),
            0.62,
            0.92,
        )
        return self._clamp(
            trade_confidence * trade_imbalance
            + (1.0 - trade_confidence) * context_imbalance,
            -1.0,
            1.0,
        )

    def _next_price_after_trading(
        self,
        price_before: float,
        stats: _TradeStats,
        execution: _ExecutionResult,
        latent: LatentState | None = None,
    ) -> float:
        if stats.total_volume <= 0 or stats.last_price is None:
            return price_before
        latent = latent or self.state.latent
        regime = self._active_settings
        vwap = stats.notional / stats.total_volume
        execution_price = 0.68 * stats.last_price + 0.32 * vwap
        execution_move_ticks = (execution_price - price_before) / self.gap
        flow_total = execution.market_buy_volume + execution.market_sell_volume
        flow_imbalance = (
            self._clamp(
                (execution.market_buy_volume - execution.market_sell_volume) / flow_total,
                -1.0,
                1.0,
            )
            if flow_total > 1e-12
            else 0.0
        )
        if abs(execution_move_ticks) <= 1e-12 and abs(flow_imbalance) <= 1e-12:
            return price_before
        volume_confidence = self._clamp(
            stats.total_volume / max(0.7, self.popularity),
            0.35,
            1.65,
        )
        volatility_response = self._clamp(
            0.86 + 0.18 * regime["volatility"] + 0.14 * latent.volatility,
            0.90,
            1.50,
        )
        # Last price shows where trades occurred; flow only nudges revealed pressure.
        price_discovery = self._clamp(abs(execution_move_ticks) / 1.35, 0.25, 1.0)
        flow_move_ticks = (
            flow_imbalance
            * (0.06 + 0.045 * volume_confidence)
            * price_discovery
            * volatility_response
        )
        proposed_ticks = (
            execution_move_ticks * (0.52 + 0.12 * volume_confidence) * volatility_response
        )
        proposed_ticks += flow_move_ticks
        max_move_ticks = self._clamp(1.0 + 0.45 * latent.volatility, 1.0, 2.0)
        proposed_ticks = self._clamp(proposed_ticks, -max_move_ticks, max_move_ticks)
        self._price_residual_ticks = self._clamp(
            self._price_residual_ticks + proposed_ticks,
            -max_move_ticks,
            max_move_ticks,
        )
        emitted_ticks = int(round(self._price_residual_ticks))
        if emitted_ticks == 0:
            return price_before
        self._price_residual_ticks -= emitted_ticks
        return max(self._min_price, price_before + emitted_ticks * self.gap)

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
