from __future__ import annotations

from .book import BUY, SELL, BookState, MarketMessage, PublicEvent, Trade, apply_message
from .theta import WorldTheta


def apply_book_physics(
    book: BookState,
    message: MarketMessage,
    theta: WorldTheta,
) -> tuple[BookState, list[Trade], list[PublicEvent]]:
    next_book = book.copy()
    trades, events = apply_message(next_book, message, theta)
    return next_book, trades, events
