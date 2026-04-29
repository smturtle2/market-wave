from __future__ import annotations

from dataclasses import dataclass, field

from .state import OrderBookState, PriceMap


@dataclass
class _OrderBook:
    bid_volume_by_price: PriceMap = field(default_factory=dict)
    ask_volume_by_price: PriceMap = field(default_factory=dict)
    _best_bid_cache: float | None = field(default=None, init=False, repr=False)
    _best_ask_cache: float | None = field(default=None, init=False, repr=False)
    _bid_best_dirty: bool = field(default=True, init=False, repr=False)
    _ask_best_dirty: bool = field(default=True, init=False, repr=False)

    def invalidate(self, side: str | None = None) -> None:
        if side in (None, "bid"):
            self._bid_best_dirty = True
        if side in (None, "ask"):
            self._ask_best_dirty = True

    def add_lot(
        self,
        price: float,
        volume: float,
        side: str,
        kind: str,
    ) -> None:
        del kind
        if volume <= 0:
            return
        volume_totals = self.volumes_for_side(side)
        volume_totals[price] = volume_totals.get(price, 0.0) + volume
        self.invalidate(side)

    def add_lots(self, volume_by_price: PriceMap, side: str, kind: str) -> None:
        for price, volume in volume_by_price.items():
            self.add_lot(price, volume, side, kind)

    def volumes_for_side(self, side: str) -> PriceMap:
        return self.bid_volume_by_price if side == "bid" else self.ask_volume_by_price

    def volumes_for_taker_side(self, side: str) -> PriceMap:
        return self.ask_volume_by_price if side == "buy" else self.bid_volume_by_price

    def adjust_volume(self, side: str, price: float, delta: float) -> None:
        if abs(delta) <= 1e-12:
            return
        volume_totals = self.volumes_for_side(side)
        next_volume = volume_totals.get(price, 0.0) + delta
        if next_volume > 1e-12:
            volume_totals[price] = next_volume
        else:
            volume_totals.pop(price, None)
        self.invalidate(side)

    def best_bid(self) -> float | None:
        if self._bid_best_dirty:
            self._best_bid_cache = max(
                (price for price, volume in self.bid_volume_by_price.items() if volume > 1e-12),
                default=None,
            )
            self._bid_best_dirty = False
        return self._best_bid_cache

    def best_ask(self) -> float | None:
        if self._ask_best_dirty:
            self._best_ask_cache = min(
                (price for price, volume in self.ask_volume_by_price.items() if volume > 1e-12),
                default=None,
            )
            self._ask_best_dirty = False
        return self._best_ask_cache

    def discard_empty_head(self, price: float, side: str) -> None:
        volume_totals = self.volumes_for_side(side)
        if volume_totals.get(price, 0.0) <= 1e-12:
            volume_totals.pop(price, None)
            self.invalidate(side)

    def clean(self) -> None:
        for side, volume_by_price in (
            ("bid", self.bid_volume_by_price),
            ("ask", self.ask_volume_by_price),
        ):
            for price in list(volume_by_price):
                if volume_by_price[price] <= 1e-12:
                    del volume_by_price[price]
                    self.invalidate(side)

    def snapshot(self) -> OrderBookState:
        return OrderBookState(
            bid_volume_by_price=self._drop_zeroes(self.bid_volume_by_price),
            ask_volume_by_price=self._drop_zeroes(self.ask_volume_by_price),
        )

    def near_touch_imbalance(self, current_price: float, gap: float) -> float:
        bid_depth = 0.0
        ask_depth = 0.0
        for price, volume in self.bid_volume_by_price.items():
            distance = max(1.0, abs(current_price - price) / gap)
            if distance <= 5:
                bid_depth += volume / distance
        for price, volume in self.ask_volume_by_price.items():
            distance = max(1.0, abs(price - current_price) / gap)
            if distance <= 5:
                ask_depth += volume / distance
        total = bid_depth + ask_depth
        if total <= 1e-12:
            return 0.0
        return max(-1.0, min(1.0, (bid_depth - ask_depth) / total))

    @staticmethod
    def _drop_zeroes(values: PriceMap) -> PriceMap:
        return {price: value for price, value in sorted(values.items()) if value > 1e-12}
