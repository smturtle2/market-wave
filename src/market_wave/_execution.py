from __future__ import annotations

from ._types import (
    _ExecutionResult,
    _IncomingOrder,
    _MarketEvent,
    _MicrostructureState,
    _ProcessedOrder,
    _StepComputationCache,
    _TradeStats,
)
from .state import OrderBookState, PriceMap, TickMap


class _MarketExecutionMixin:
    def _cancel_price_volume(
        self,
        cancelled: PriceMap,
        side: str,
        price: float,
        requested: float,
    ) -> float:
        volume_by_price = self._orderbook.volumes_for_side(side)
        removed = min(max(0.0, requested), max(0.0, volume_by_price.get(price, 0.0)))
        if removed <= 1e-12:
            return 0.0
        cancelled[price] = cancelled.get(price, 0.0) + removed
        self._orderbook.adjust_volume(side, price, -removed)
        if self._orderbook.volumes_for_side(side).get(price, 0.0) <= 1e-12:
            self._quote_age_by_side[side].pop(price, None)
        return removed

    def _stress_for_book_side(self, side: str, micro: _MicrostructureState) -> float:
        if side == "ask":
            return self._clamp(max(0.0, micro.stress_side), 0.0, 1.0)
        return self._clamp(max(0.0, -micro.stress_side), 0.0, 1.0)

    def _add_lots(self, volume_by_price: PriceMap, side: str, kind: str) -> None:
        for price, volume in volume_by_price.items():
            self._add_resting_lot(side, price, volume, kind)

    def _add_resting_lot(self, side: str, price: float, volume: float, kind: str) -> None:
        if volume <= 1e-12:
            return
        price = self._snap_price(price)
        existing_volume = max(0.0, self._orderbook.volumes_for_side(side).get(price, 0.0))
        existing_age = self._quote_age(side, price)
        self._orderbook.add_lot(price, volume, side, kind)
        total_volume = existing_volume + volume
        blended_age = (
            0
            if total_volume <= 1e-12
            else int(round(existing_age * existing_volume / total_volume))
        )
        self._set_quote_age(side, price, blended_age)

    def _execute_market_flows(
        self,
        *,
        entry_orders: list[_IncomingOrder] | None = None,
        events: list[_MarketEvent] | None = None,
        stats: _TradeStats,
    ) -> _ExecutionResult:
        event_stream = (
            list(events)
            if events is not None
            else [
                _MarketEvent(
                    event_type="buy_limit_add" if order.side == "buy" else "sell_limit_add",
                    side=order.side,
                    price=order.price,
                    volume=order.volume,
                )
                for order in (entry_orders or [])
            ]
        )
        market_buy_volume = 0.0
        market_sell_volume = 0.0
        residual_buy_by_price: PriceMap = {}
        residual_sell_by_price: PriceMap = {}
        cancelled: PriceMap = {}
        bid_cancelled: PriceMap = {}
        ask_cancelled: PriceMap = {}
        deferred_limit_adds: list[_MarketEvent] = []
        cancel_budget = {
            "bid": {
                price: volume
                for price, volume in self._orderbook.bid_volume_by_price.items()
                if volume > 1e-12
            },
            "ask": {
                price: volume
                for price, volume in self._orderbook.ask_volume_by_price.items()
                if volume > 1e-12
            },
        }
        for event in event_stream:
            if event.event_type in {"bid_cancel", "ask_cancel"}:
                book_side = "bid" if event.event_type == "bid_cancel" else "ask"
                side_cancelled = bid_cancelled if book_side == "bid" else ask_cancelled
                price = self._snap_price(event.price)
                available_from_step_start = max(0.0, cancel_budget[book_side].get(price, 0.0))
                if available_from_step_start <= 1e-12:
                    continue
                removed = self._cancel_price_volume(
                    side_cancelled,
                    book_side,
                    price,
                    min(event.volume, available_from_step_start),
                )
                if removed > 1e-12:
                    cancel_budget[book_side][price] = max(
                        0.0,
                        cancel_budget[book_side].get(price, 0.0) - removed,
                    )
                    cancelled[price] = cancelled.get(price, 0.0) + removed
                continue
            if event.side not in {"buy", "sell"}:
                continue
            if event.event_type in {"buy_limit_add", "sell_limit_add"}:
                deferred_limit_adds.append(event)
                continue
            order = _IncomingOrder(
                side=event.side,
                kind="buy_entry" if event.side == "buy" else "sell_entry",
                price=event.price,
                volume=event.volume,
            )
            executed_before = dict(stats.executed_by_price)
            result = self._process_incoming_order(
                order,
                stats=stats,
                rest_residual=False,
            )
            executed_delta = {
                price: stats.executed_by_price.get(price, 0.0)
                - executed_before.get(price, 0.0)
                for price in stats.executed_by_price
            }
            if order.side == "buy":
                market_buy_volume += result.executed
                for price, volume in executed_delta.items():
                    if volume <= 1e-12:
                        continue
                    cancel_budget["ask"][price] = max(
                        0.0,
                        cancel_budget["ask"].get(price, 0.0) - volume,
                    )
                    residual_sell_by_price[price] = max(
                        0.0,
                        residual_sell_by_price.get(price, 0.0) - volume,
                    )
                if result.rested > 1e-12:
                    price = self._snap_price(order.price)
                    residual_buy_by_price[price] = (
                        residual_buy_by_price.get(price, 0.0) + result.rested
                    )
            else:
                market_sell_volume += result.executed
                for price, volume in executed_delta.items():
                    if volume <= 1e-12:
                        continue
                    cancel_budget["bid"][price] = max(
                        0.0,
                        cancel_budget["bid"].get(price, 0.0) - volume,
                    )
                    residual_buy_by_price[price] = max(
                        0.0,
                        residual_buy_by_price.get(price, 0.0) - volume,
                    )
                if result.rested > 1e-12:
                    price = self._snap_price(order.price)
                    residual_sell_by_price[price] = (
                        residual_sell_by_price.get(price, 0.0) + result.rested
                    )
        for event in deferred_limit_adds:
            if event.volume <= 1e-12:
                continue
            price = self._snap_price(event.price)
            if event.event_type == "buy_limit_add":
                best_ask = self._best_ask()
                if best_ask is not None:
                    price = min(price, self._snap_price(best_ask - self.gap))
                if price < self._min_price:
                    continue
                self._add_resting_lot("bid", price, event.volume, "buy_entry")
                residual_buy_by_price[price] = (
                    residual_buy_by_price.get(price, 0.0) + event.volume
                )
            elif event.event_type == "sell_limit_add":
                best_bid = self._best_bid()
                if best_bid is not None:
                    price = max(price, self._snap_price(best_bid + self.gap))
                self._add_resting_lot("ask", price, event.volume, "sell_entry")
                residual_sell_by_price[price] = (
                    residual_sell_by_price.get(price, 0.0) + event.volume
                )
        return _ExecutionResult(
            residual_market_buy=sum(self._drop_zeroes(residual_buy_by_price).values()),
            residual_market_sell=sum(self._drop_zeroes(residual_sell_by_price).values()),
            crossed_market_volume=0.0,
            market_buy_volume=market_buy_volume,
            market_sell_volume=market_sell_volume,
            cancelled_volume_by_price=self._drop_zeroes(cancelled),
            bid_cancelled_volume_by_price=self._drop_zeroes(bid_cancelled),
            ask_cancelled_volume_by_price=self._drop_zeroes(ask_cancelled),
        )

    def _process_incoming_order(
        self,
        order: _IncomingOrder,
        stats: _TradeStats,
        *,
        rest_residual: bool = True,
    ) -> _ProcessedOrder:
        if order.volume <= 0:
            return _ProcessedOrder(executed=0.0, rested=0.0)
        remaining = order.volume
        executed = 0.0
        price_limit = self._snap_price(order.price)
        while remaining > 1e-12:
            price = self._best_ask() if order.side == "buy" else self._best_bid()
            if price is None:
                break
            if order.side == "buy" and price > price_limit + 1e-12:
                break
            if order.side == "sell" and price < price_limit - 1e-12:
                break

            volume_by_price = self._orderbook.volumes_for_taker_side(order.side)
            actual = min(remaining, max(0.0, volume_by_price.get(price, 0.0)))
            if actual <= 1e-12:
                self._discard_empty_head(price, "ask" if order.side == "buy" else "bid")
                continue
            resting_side = "ask" if order.side == "buy" else "bid"
            self._orderbook.adjust_volume(resting_side, price, -actual)
            remaining -= actual
            executed += actual
            stats.record(price, actual, order.side)
            self._discard_empty_head(price, "ask" if order.side == "buy" else "bid")

        restable = remaining if rest_residual else 0.0
        if restable > 1e-12:
            passive_side = "bid" if order.side == "buy" else "ask"
            self._add_resting_lot(passive_side, price_limit, restable, order.kind)
        return _ProcessedOrder(executed=executed, rested=restable)

    def _best_bid(self) -> float | None:
        return self._orderbook.best_bid()

    def _best_ask(self) -> float | None:
        return self._orderbook.best_ask()

    def _spread(self, bid: float | None, ask: float | None) -> float | None:
        if bid is None or ask is None:
            return None
        return ask - bid

    def _current_spread_ticks(self, cache: _StepComputationCache | None = None) -> float:
        if cache is not None and cache.spread_ticks is not None:
            return cache.spread_ticks
        spread = self._spread(self._best_bid(), self._best_ask())
        if spread is None:
            spread_ticks = 2.0 + self._clamp(self._microstructure.liquidity_stress, 0.0, 2.0)
        else:
            spread_ticks = max(1.0, spread / self.gap)
        if cache is not None:
            cache.spread_ticks = spread_ticks
        return spread_ticks

    def _near_touch_imbalance(self, current_price: float) -> float:
        return self._orderbook.near_touch_imbalance(current_price, self.gap)

    def _nearby_book_depth(self, side: str, current_price: float) -> float:
        volumes = (
            self._orderbook.bid_volume_by_price
            if side == "bid"
            else self._orderbook.ask_volume_by_price
        )
        depth = 0.0
        for price, volume in volumes.items():
            distance = max(1.0, abs(current_price - price) / self.gap)
            if distance <= 5:
                depth += volume / distance
        return depth

    def _expected_nearby_depth(self) -> float:
        return max(
            0.8,
            self.popularity
            * self._active_settings["liquidity"]
            * (3.6 + 0.30 * self.grid_radius),
        )

    def _discard_empty_head(self, price: float, side: str) -> None:
        self._orderbook.discard_empty_head(price, side)
        if self._orderbook.volumes_for_side(side).get(price, 0.0) <= 1e-12:
            self._quote_age_by_side[side].pop(price, None)

    def _clean_orderbook(self) -> None:
        self._orderbook.clean()
        self._sync_quote_ages()

    def _set_quote_age(self, side: str, price: float, age: int) -> None:
        self._quote_age_by_side[side][self._snap_price(price)] = max(0, int(age))

    def _quote_age(self, side: str, price: float) -> int:
        return self._quote_age_by_side[side].get(self._snap_price(price), 0)

    def _mean_quote_age(self, book: OrderBookState | None = None) -> float:
        if book is None:
            book = self._snapshot_orderbook()

        total_volume = 0.0
        weighted_age = 0.0
        for side, volumes in (
            ("bid", book.bid_volume_by_price),
            ("ask", book.ask_volume_by_price),
        ):
            for price, volume in volumes.items():
                if volume <= 1e-12:
                    continue
                total_volume += volume
                weighted_age += volume * self._quote_age(side, price)
        if total_volume <= 1e-12:
            return 0.0
        return weighted_age / total_volume

    def _quote_staleness(self, side: str, price: float) -> float:
        age = self._quote_age(side, price)
        pressure = self._clamp(
            0.45 * self._microstructure.cancel_pressure
            + 0.35 * self._microstructure.liquidity_stress
            + 0.20 * self._microstructure.cancel_burst,
            0.0,
            2.0,
        )
        return self._clamp(age / (5.0 + 3.0 * self._microstructure.resiliency), 0.0, 1.0) * (
            0.55 + 0.45 * pressure / 2.0
        )

    def _adverse_quote_pressure(self, side: str, price: float) -> float:
        side_sign = -1.0 if side == "bid" else 1.0
        return self._clamp(
            side_sign
            * (
                0.45 * self._realized_flow.flow_imbalance
                + 0.35 * self._realized_flow.return_ticks / 4.0
                + 0.20 * self._microstructure.stress_side
            ),
            0.0,
            1.0,
        )

    def _sync_quote_ages(self) -> None:
        for side in ("bid", "ask"):
            volumes = self._orderbook.volumes_for_side(side)
            ages = self._quote_age_by_side[side]
            for price in list(ages):
                if volumes.get(price, 0.0) <= 1e-12:
                    ages.pop(price, None)
            for price, volume in volumes.items():
                if volume > 1e-12:
                    ages.setdefault(price, 0)

    def _snapshot_orderbook(self) -> OrderBookState:
        return self._orderbook.snapshot()

    def _is_crossed(self) -> bool:
        best_bid = self._best_bid()
        best_ask = self._best_ask()
        return best_bid is not None and best_ask is not None and best_bid >= best_ask

    def _merge_maps(self, *maps: PriceMap) -> PriceMap:
        merged: PriceMap = {}
        for values in maps:
            for price, volume in values.items():
                merged[price] = merged.get(price, 0.0) + volume
        return self._drop_zeroes(merged)

    def _price_map_to_relative_ticks(self, basis_price: float, values: PriceMap) -> TickMap:
        basis_tick = self.price_to_tick(basis_price)
        by_tick: TickMap = {}
        min_tick = -self.grid_radius
        max_tick = self.grid_radius
        for price, value in values.items():
            relative_tick = self.price_to_tick(price) - basis_tick
            if min_tick <= relative_tick <= max_tick:
                by_tick[relative_tick] = by_tick.get(relative_tick, 0.0) + value
        return by_tick

    def _single_price_volume(self, price: float, volume: float) -> PriceMap:
        if volume <= 0:
            return {}
        return {self._snap_price(price): volume}

    def _drop_zeroes(self, values: PriceMap) -> PriceMap:
        return {price: value for price, value in sorted(values.items()) if value > 1e-12}

    def _dedupe_prices(self, prices) -> list[float]:
        return sorted(set(prices))

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
