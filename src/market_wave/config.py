from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 1
    world_index: int = 0
    world_id: str | None = None
    steps: int = 1_000
    currency: str = "KRW"
    tick_size_decimal: str = "100"
    quantity_unit_decimal: str = "1"
    initial_mid_tick: int = 718
    initial_book_levels: int = 80
    initial_orders_per_level: int = 3
    start_timestamp: str = "2026-06-22T09:00:00.000+09:00"
    snapshot_every: int = 1
    top_levels: int = 10
    min_quantity_unit: int = 1
    max_quantity_unit: int = 500

    def validate(self) -> None:
        if self.world_index < 0:
            raise ValueError("world_index must be non-negative")
        if self.world_id == "":
            raise ValueError("world_id must not be empty")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.initial_book_levels < self.top_levels + 2:
            raise ValueError("initial_book_levels must cover top_levels")
        if self.initial_orders_per_level < 1:
            raise ValueError("initial_orders_per_level must be positive")
        if self.snapshot_every <= 0:
            raise ValueError("snapshot_every must be positive")
        if self.top_levels <= 0:
            raise ValueError("top_levels must be positive")
        if self.min_quantity_unit < 1:
            raise ValueError("min_quantity_unit must be positive")
        if self.max_quantity_unit < self.min_quantity_unit:
            raise ValueError("max_quantity_unit must be >= min_quantity_unit")
