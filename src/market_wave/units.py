from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class MarketUnits:
    tick_size_decimal: Decimal
    quantity_unit_decimal: Decimal

    @classmethod
    def from_strings(cls, tick_size: str, quantity_unit: str) -> "MarketUnits":
        tick = Decimal(tick_size)
        quantity = Decimal(quantity_unit)
        if tick <= 0:
            raise ValueError("tick_size_decimal must be positive")
        if quantity <= 0:
            raise ValueError("quantity_unit_decimal must be positive")
        return cls(tick, quantity)

    def price(self, tick: int) -> str:
        return _decimal_string(Decimal(int(tick)) * self.tick_size_decimal)

    def midpoint_price(self, bid_tick: int, ask_tick: int) -> str:
        return _decimal_string((Decimal(int(bid_tick)) + Decimal(int(ask_tick))) * self.tick_size_decimal / Decimal(2))

    def quantity(self, quantity_unit: int) -> str:
        return _decimal_string(Decimal(int(quantity_unit)) * self.quantity_unit_decimal)


def _decimal_string(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f")
