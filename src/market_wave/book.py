from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

import numpy as np

from .theta import WorldTheta
from .units import MarketUnits

BUY = "BUY"
SELL = "SELL"


@dataclass
class Order:
    order_id: str
    side: str
    price_tick: int
    remaining_qty: int
    created_seq: int


@dataclass(frozen=True)
class MarketMessage:
    kind: str
    side: str
    price_tick: int | None
    quantity: int
    order_id: str | None = None
    new_price_tick: int | None = None
    new_quantity: int | None = None


@dataclass(frozen=True)
class Trade:
    price_tick: int
    quantity_unit: int
    side: str


@dataclass(frozen=True)
class PublicEvent:
    record_type: str
    payload: dict[str, object]


@dataclass
class BookState:
    bids: dict[int, deque[Order]]
    asks: dict[int, deque[Order]]
    orders: dict[str, Order]
    next_order_seq: int
    last_trade_tick: int | None = None

    def copy(self) -> "BookState":
        bids = {price: deque(Order(o.order_id, o.side, o.price_tick, o.remaining_qty, o.created_seq) for o in queue) for price, queue in self.bids.items()}
        asks = {price: deque(Order(o.order_id, o.side, o.price_tick, o.remaining_qty, o.created_seq) for o in queue) for price, queue in self.asks.items()}
        orders = {}
        for queues in (bids, asks):
            for queue in queues.values():
                for order in queue:
                    if order.remaining_qty > 0:
                        orders[order.order_id] = order
        return BookState(bids, asks, orders, self.next_order_seq, self.last_trade_tick)


def init_book(rng: np.random.Generator, theta: WorldTheta) -> BookState:
    book = BookState(bids={}, asks={}, orders={}, next_order_seq=0)
    for distance in range(1, theta.initial_book_levels + 1):
        bid_tick = theta.initial_mid_tick - distance
        ask_tick = theta.initial_mid_tick + distance
        for order_index in range(theta.initial_orders_per_level):
            bid_qty = _initial_level_quantity(rng, theta, distance, order_index)
            ask_qty = _initial_level_quantity(rng, theta, distance, order_index)
            _rest_order(book, BUY, bid_tick, bid_qty)
            _rest_order(book, SELL, ask_tick, ask_qty)
    assert_book_invariants(book)
    return book


def apply_message(book: BookState, message: MarketMessage, theta: WorldTheta) -> tuple[list[Trade], list[PublicEvent]]:
    if message.kind == "LIMIT":
        return _apply_limit(book, message, theta)
    if message.kind == "TAKE":
        return _apply_take(book, message, theta)
    if message.kind == "CANCEL":
        return _apply_cancel(book, message, theta)
    if message.kind == "REPLACE":
        return _apply_replace(book, message, theta)
    raise ValueError(f"unknown message kind: {message.kind}")


def public_best_bid_tick(book: BookState) -> int:
    prices = [price for price, queue in book.bids.items() if _queue_volume(queue) > 0]
    if not prices:
        raise ValueError("book has no visible bid depth")
    return max(prices)


def public_best_ask_tick(book: BookState) -> int:
    prices = [price for price, queue in book.asks.items() if _queue_volume(queue) > 0]
    if not prices:
        raise ValueError("book has no visible ask depth")
    return min(prices)


def mid_tick(book: BookState) -> int:
    return (public_best_bid_tick(book) + public_best_ask_tick(book)) // 2


def spread_tick(book: BookState) -> int:
    return public_best_ask_tick(book) - public_best_bid_tick(book)


def side_depth(book: BookState, side: str) -> int:
    queues = book.bids if side == BUY else book.asks
    return sum(_queue_volume(queue) for queue in queues.values())


def price_depth(book: BookState, side: str, price_tick: int) -> int:
    queues = book.bids if side == BUY else book.asks
    return _queue_volume(queues.get(price_tick, deque()))


def live_orders(book: BookState, side: str | None = None) -> list[Order]:
    orders = [order for order in book.orders.values() if order.remaining_qty > 0]
    if side is not None:
        orders = [order for order in orders if order.side == side]
    return sorted(orders, key=lambda order: order.created_seq)


def cancel_candidates(book: BookState, side: str) -> list[Order]:
    best = public_best_bid_tick(book) if side == BUY else public_best_ask_tick(book)
    candidates = []
    for order in live_orders(book, side):
        if order.price_tick == best and price_depth(book, side, best) - order.remaining_qty <= 0:
            continue
        candidates.append(order)
    return candidates


def replace_candidates(book: BookState, side: str) -> list[Order]:
    best = public_best_bid_tick(book) if side == BUY else public_best_ask_tick(book)
    return [order for order in live_orders(book, side) if order.price_tick != best]


def available_to_take(book: BookState, side: str, limit_tick: int) -> int:
    if side == BUY:
        return sum(volume for price, volume in _ask_levels(book) if price <= limit_tick)
    return sum(volume for price, volume in _bid_levels(book) if price >= limit_tick)


def visible_price_levels(book: BookState, side: str, levels: int) -> list[tuple[int, int]]:
    rows = _bid_levels(book) if side == BUY else _ask_levels(book)
    return rows[:levels]


def assert_book_invariants(book: BookState) -> None:
    if any(order.remaining_qty < 0 for order in book.orders.values()):
        raise AssertionError("order quantity cannot be negative")
    if any(order.side != BUY for queue in book.bids.values() for order in queue):
        raise AssertionError("bid queue contains non-bid order")
    if any(order.side != SELL for queue in book.asks.values() for order in queue):
        raise AssertionError("ask queue contains non-ask order")
    bid = public_best_bid_tick(book)
    ask = public_best_ask_tick(book)
    if bid >= ask:
        raise AssertionError("public book is crossed or locked")
    for order_id, order in book.orders.items():
        if order.order_id != order_id:
            raise AssertionError("order id index mismatch")
        if order.remaining_qty <= 0:
            raise AssertionError("dead order retained in live index")
    for side, queues in ((BUY, book.bids), (SELL, book.asks)):
        for price, queue in queues.items():
            if _queue_volume(queue) <= 0:
                raise AssertionError(f"empty {side} queue retained at {price}")


def public_book_projection(book: BookState, theta: WorldTheta, levels: int = 10) -> dict[str, object]:
    units = MarketUnits.from_strings(theta.tick_size_decimal, theta.quantity_unit_decimal)
    best_bid = public_best_bid_tick(book)
    best_ask = public_best_ask_tick(book)
    asks = [
        {"price": units.price(tick), "volume": units.quantity(price_depth(book, SELL, tick))}
        for tick in range(best_ask, best_ask + levels)
    ]
    bids = [
        {"price": units.price(tick), "volume": units.quantity(price_depth(book, BUY, tick))}
        for tick in range(best_bid, best_bid - levels, -1)
    ]
    return {
        "best_bid": units.price(best_bid),
        "best_ask": units.price(best_ask),
        "mid": units.midpoint_price(best_bid, best_ask),
        "spread": units.price(best_ask - best_bid),
        "asks": asks,
        "bids": bids,
        "total_visible_bid_depth": units.quantity(side_depth(book, BUY)),
        "total_visible_ask_depth": units.quantity(side_depth(book, SELL)),
    }


def reconstruct_projection(initial_snapshot: dict[str, object], events: Iterable[dict[str, object]], theta: WorldTheta, levels: int = 10) -> dict[str, object]:
    book = _aggregate_book_from_snapshot(initial_snapshot, theta)
    for record in events:
        if record.get("record_type") != "BOOK_DELTA":
            continue
        payload = record["payload"]
        side = str(payload["side"])
        price = _price_to_tick(str(payload["price"]), theta)
        delta = _quantity_to_units(str(payload["volume_delta"]), theta)
        levels_map = book[side]
        levels_map[price] = levels_map.get(price, 0) + delta
        if levels_map[price] < 0:
            raise AssertionError("reconstructed depth became negative")
    return _aggregate_projection(book, theta, levels)


def _apply_limit(book: BookState, message: MarketMessage, theta: WorldTheta) -> tuple[list[Trade], list[PublicEvent]]:
    if message.price_tick is None or message.quantity <= 0:
        raise ValueError("LIMIT message requires price and positive quantity")
    side = message.side
    original_qty = int(message.quantity)
    remaining = original_qty
    trades: list[Trade] = []
    events: list[PublicEvent] = [
        PublicEvent(
            "ORDER_ACCEPTED",
            {"side": side, "price_tick": message.price_tick, "quantity": original_qty},
        )
    ]
    if side == BUY:
        while remaining > 0 and _ask_levels(book) and _ask_levels(book)[0][0] <= message.price_tick:
            remaining = _match_one_price_level(book, BUY, remaining, trades, events)
    elif side == SELL:
        while remaining > 0 and _bid_levels(book) and _bid_levels(book)[0][0] >= message.price_tick:
            remaining = _match_one_price_level(book, SELL, remaining, trades, events)
    else:
        raise ValueError(f"unknown side: {side}")
    if remaining > 0 and _is_non_crossing(book, side, message.price_tick):
        order = _rest_order(book, side, message.price_tick, remaining)
        events.append(PublicEvent("BOOK_DELTA", {"side": side, "price_tick": order.price_tick, "delta": remaining, "after": price_depth(book, side, order.price_tick)}))
    assert_book_invariants(book)
    return trades, events


def _apply_take(book: BookState, message: MarketMessage, theta: WorldTheta) -> tuple[list[Trade], list[PublicEvent]]:
    if message.price_tick is None or message.quantity <= 0:
        raise ValueError("TAKE message requires price and positive quantity")
    side = message.side
    remaining = int(message.quantity)
    trades: list[Trade] = []
    events: list[PublicEvent] = [
        PublicEvent(
            "TAKE_ORDER",
            {"side": side, "price_tick": message.price_tick, "quantity": remaining},
        )
    ]
    if side == BUY:
        while remaining > 0 and _ask_levels(book) and _ask_levels(book)[0][0] <= message.price_tick:
            remaining = _match_one_price_level(book, BUY, remaining, trades, events)
    elif side == SELL:
        while remaining > 0 and _bid_levels(book) and _bid_levels(book)[0][0] >= message.price_tick:
            remaining = _match_one_price_level(book, SELL, remaining, trades, events)
    else:
        raise ValueError(f"unknown side: {side}")
    assert_book_invariants(book)
    return trades, events


def _apply_cancel(book: BookState, message: MarketMessage, theta: WorldTheta) -> tuple[list[Trade], list[PublicEvent]]:
    if message.order_id is None or message.order_id not in book.orders:
        return [], []
    order = book.orders[message.order_id]
    cancel_qty = min(order.remaining_qty, max(theta.min_quantity_unit, int(message.quantity)))
    old_price = order.price_tick
    side = order.side
    order.remaining_qty -= cancel_qty
    if order.remaining_qty == 0:
        del book.orders[order.order_id]
    _prune_queue(book, side, old_price)
    events = [
        PublicEvent("ORDER_CANCELED", {"side": side, "price_tick": old_price, "quantity": cancel_qty}),
        PublicEvent("BOOK_DELTA", {"side": side, "price_tick": old_price, "delta": -cancel_qty, "after": price_depth(book, side, old_price)}),
    ]
    assert_book_invariants(book)
    return [], events


def _apply_replace(book: BookState, message: MarketMessage, theta: WorldTheta) -> tuple[list[Trade], list[PublicEvent]]:
    if message.order_id is None or message.order_id not in book.orders:
        return [], []
    if message.new_price_tick is None or message.new_quantity is None or message.new_quantity <= 0:
        raise ValueError("REPLACE message requires new price and quantity")
    order = book.orders[message.order_id]
    old_side = order.side
    old_price = order.price_tick
    old_qty = order.remaining_qty
    del book.orders[order.order_id]
    order.remaining_qty = 0
    _prune_queue(book, old_side, old_price)
    new_order = _rest_order(book, old_side, message.new_price_tick, int(message.new_quantity))
    events = [
        PublicEvent(
            "ORDER_REPLACED",
            {
                "side": old_side,
                "old_price_tick": old_price,
                "new_price_tick": new_order.price_tick,
                "old_quantity": old_qty,
                "new_quantity": new_order.remaining_qty,
            },
        ),
        PublicEvent("BOOK_DELTA", {"side": old_side, "price_tick": old_price, "delta": -old_qty, "after": price_depth(book, old_side, old_price)}),
        PublicEvent("BOOK_DELTA", {"side": old_side, "price_tick": new_order.price_tick, "delta": new_order.remaining_qty, "after": price_depth(book, old_side, new_order.price_tick)}),
    ]
    assert_book_invariants(book)
    return [], events


def _match_one_price_level(
    book: BookState,
    aggressor_side: str,
    remaining: int,
    trades: list[Trade],
    events: list[PublicEvent],
) -> int:
    resting_side = SELL if aggressor_side == BUY else BUY
    levels = book.asks if resting_side == SELL else book.bids
    price = public_best_ask_tick(book) if resting_side == SELL else public_best_bid_tick(book)
    queue = levels[price]
    while remaining > 0 and queue:
        order = queue[0]
        fill = min(remaining, order.remaining_qty)
        order.remaining_qty -= fill
        remaining -= fill
        trades.append(Trade(price, fill, aggressor_side))
        book.last_trade_tick = price
        if order.remaining_qty == 0:
            queue.popleft()
            del book.orders[order.order_id]
    if not queue:
        del levels[price]
    after = price_depth(book, resting_side, price)
    events.append(PublicEvent("BOOK_DELTA", {"side": resting_side, "price_tick": price, "delta": -sum(t.quantity_unit for t in trades if t.price_tick == price and t.side == aggressor_side), "after": after}))
    for trade in [t for t in trades if t.price_tick == price and t.side == aggressor_side]:
        events.append(PublicEvent("TRADE", {"side": trade.side, "price_tick": trade.price_tick, "quantity": trade.quantity_unit}))
    return remaining


def _rest_order(book: BookState, side: str, price_tick: int, quantity: int) -> Order:
    order = Order(f"o{book.next_order_seq:012d}", side, int(price_tick), int(quantity), book.next_order_seq)
    book.next_order_seq += 1
    queues = book.bids if side == BUY else book.asks
    queues.setdefault(order.price_tick, deque()).append(order)
    book.orders[order.order_id] = order
    return order


def _initial_level_quantity(rng: np.random.Generator, theta: WorldTheta, distance: int, order_index: int) -> int:
    touch_factor = 0.70 + min(2.2, distance ** 0.45 * 0.24)
    queue_factor = 1.0 + 0.16 * order_index
    wall_factor = 2.8 if distance in {3, 5, 8, 13, 21} and rng.random() < 0.35 else 1.0
    emptying_noise = float(rng.lognormal(mean=-0.10, sigma=0.42))
    quantity = theta.base_order_quantity * touch_factor * queue_factor * wall_factor * emptying_noise
    return int(np.clip(quantity, theta.min_quantity_unit, theta.max_quantity_unit))


def _is_non_crossing(book: BookState, side: str, price_tick: int) -> bool:
    if side == BUY:
        return price_tick < public_best_ask_tick(book)
    return price_tick > public_best_bid_tick(book)


def _queue_volume(queue: deque[Order]) -> int:
    return sum(order.remaining_qty for order in queue if order.remaining_qty > 0)


def _bid_levels(book: BookState) -> list[tuple[int, int]]:
    return sorted(((price, _queue_volume(queue)) for price, queue in book.bids.items() if _queue_volume(queue) > 0), reverse=True)


def _ask_levels(book: BookState) -> list[tuple[int, int]]:
    return sorted((price, _queue_volume(queue)) for price, queue in book.asks.items() if _queue_volume(queue) > 0)


def _prune_queue(book: BookState, side: str, price_tick: int) -> None:
    queues = book.bids if side == BUY else book.asks
    if price_tick not in queues:
        return
    queues[price_tick] = deque(order for order in queues[price_tick] if order.remaining_qty > 0)
    if not queues[price_tick]:
        del queues[price_tick]


def _aggregate_book_from_snapshot(snapshot: dict[str, object], theta: WorldTheta) -> dict[str, dict[int, int]]:
    return {
        SELL: {_price_to_tick(str(level["price"]), theta): _quantity_to_units(str(level["volume"]), theta) for level in snapshot["asks"] if _quantity_to_units(str(level["volume"]), theta) > 0},
        BUY: {_price_to_tick(str(level["price"]), theta): _quantity_to_units(str(level["volume"]), theta) for level in snapshot["bids"] if _quantity_to_units(str(level["volume"]), theta) > 0},
    }


def _aggregate_projection(book: dict[str, dict[int, int]], theta: WorldTheta, levels: int) -> dict[str, object]:
    units = MarketUnits.from_strings(theta.tick_size_decimal, theta.quantity_unit_decimal)
    best_ask = min(price for price, depth in book[SELL].items() if depth > 0)
    best_bid = max(price for price, depth in book[BUY].items() if depth > 0)
    return {
        "best_bid": units.price(best_bid),
        "best_ask": units.price(best_ask),
        "mid": units.midpoint_price(best_bid, best_ask),
        "spread": units.price(best_ask - best_bid),
        "asks": [{"price": units.price(tick), "volume": units.quantity(book[SELL].get(tick, 0))} for tick in range(best_ask, best_ask + levels)],
        "bids": [{"price": units.price(tick), "volume": units.quantity(book[BUY].get(tick, 0))} for tick in range(best_bid, best_bid - levels, -1)],
        "total_visible_bid_depth": units.quantity(sum(book[BUY].values())),
        "total_visible_ask_depth": units.quantity(sum(book[SELL].values())),
    }


def _price_to_tick(value: str, theta: WorldTheta) -> int:
    units = MarketUnits.from_strings(theta.tick_size_decimal, theta.quantity_unit_decimal)
    return int(Decimal(value) / units.tick_size_decimal)


def _quantity_to_units(value: str, theta: WorldTheta) -> int:
    units = MarketUnits.from_strings(theta.tick_size_decimal, theta.quantity_unit_decimal)
    return int(Decimal(value) / units.quantity_unit_decimal)
