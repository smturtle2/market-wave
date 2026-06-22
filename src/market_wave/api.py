from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .adapter import book_events, candles as adapter_candles, current_price, orderbook, trades as adapter_trades
from .config import GenerationConfig
from .generator import run_world_streaming
from .validation import ValidationReport, validate_corpus
from .visualization import plot_all


@dataclass(frozen=True)
class Market:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def validate(self) -> ValidationReport:
        return validate_corpus(self.root)

    def plot(self, path: str | Path | None = None) -> list[Path]:
        return plot_all(self.root, Path(path) if path is not None else self.root / "plots")

    def price(self) -> dict[str, Any]:
        return _unwrap(current_price(self.root))

    def book(self, levels: int = 10) -> dict[str, Any]:
        return _unwrap(orderbook(self.root, limit=levels))

    def orderbook(self, levels: int = 10) -> dict[str, Any]:
        return self.book(levels)

    def trades(self, limit: int = 100) -> list[dict[str, Any]]:
        return _unwrap(adapter_trades(self.root, limit=limit))["trades"]

    def candles(self, interval: str | int = "1m") -> list[dict[str, Any]]:
        return _unwrap(adapter_candles(self.root, interval_seconds=_interval_seconds(interval)))["candles"]

    def events(self, limit: int = 100, cursor: int = 0) -> dict[str, Any]:
        return _unwrap(book_events(self.root, limit=limit, cursor=cursor))


def generate(
    path: str | Path,
    config: GenerationConfig | None = None,
    *,
    seed: int = 1,
    steps: int = 1_000,
    price: int | str | Decimal = 71_800,
    tick: int | str | Decimal = 100,
    depth: int = 80,
    orders_per_level: int = 3,
    world_index: int = 0,
    world_id: str | None = None,
    start_timestamp: str = "2026-06-22T09:00:00.000+09:00",
    snapshot_every: int = 1,
    force: bool = False,
) -> Market:
    root = Path(path)
    final_config = config or _config(
        seed=seed,
        steps=steps,
        price=price,
        tick=tick,
        depth=depth,
        orders_per_level=orders_per_level,
        world_index=world_index,
        world_id=world_id,
        start_timestamp=start_timestamp,
        snapshot_every=snapshot_every,
    )
    run_world_streaming(final_config, root, force=force)
    return Market(root)


def open(path: str | Path) -> Market:
    return Market(Path(path))


def replay(
    config: GenerationConfig | None = None,
    *,
    seed: int = 1,
    steps: int = 1_000,
    price: int | str | Decimal = 71_800,
    tick: int | str | Decimal = 100,
    depth: int = 80,
    orders_per_level: int = 3,
    world_index: int = 0,
    world_id: str | None = None,
    start_timestamp: str = "2026-06-22T09:00:00.000+09:00",
    snapshot_every: int = 1,
) -> dict[str, Any]:
    final_config = config or _config(
        seed=seed,
        steps=steps,
        price=price,
        tick=tick,
        depth=depth,
        orders_per_level=orders_per_level,
        world_index=world_index,
        world_id=world_id,
        start_timestamp=start_timestamp,
        snapshot_every=snapshot_every,
    )
    with tempfile.TemporaryDirectory(prefix="market-wave-replay-") as tmp:
        root = Path(tmp)
        first = root / "first"
        second = root / "second"
        run_world_streaming(final_config, first, force=True)
        run_world_streaming(final_config, second, force=True)
        first_hash = _tree_hash(first)
        second_hash = _tree_hash(second)
        if first_hash != second_hash:
            raise RuntimeError("replay mismatch")
        shutil.rmtree(first)
        shutil.rmtree(second)
        return {"byte_identical": True, "hash": first_hash}


def _config(
    *,
    seed: int,
    steps: int,
    price: int | str | Decimal,
    tick: int | str | Decimal,
    depth: int,
    orders_per_level: int,
    world_index: int,
    world_id: str | None,
    start_timestamp: str,
    snapshot_every: int,
) -> GenerationConfig:
    tick_value = Decimal(str(tick))
    price_value = Decimal(str(price))
    if tick_value <= 0:
        raise ValueError("tick must be positive")
    mid_tick = price_value / tick_value
    if mid_tick != mid_tick.to_integral_value():
        raise ValueError("price must be an exact multiple of tick")
    return GenerationConfig(
        seed=seed,
        world_index=world_index,
        world_id=world_id,
        steps=steps,
        tick_size_decimal=_decimal_string(tick_value),
        initial_mid_tick=int(mid_tick),
        initial_book_levels=depth,
        initial_orders_per_level=orders_per_level,
        start_timestamp=start_timestamp,
        snapshot_every=snapshot_every,
    )


def _unwrap(value: dict[str, Any]) -> Any:
    if "error" in value:
        error = value["error"]
        raise ValueError(str(error.get("message", "market-wave error")))
    return value["result"]


def _interval_seconds(value: str | int) -> int:
    if isinstance(value, int):
        return value
    unit = value[-1]
    amount = int(value[:-1])
    if unit == "s":
        return amount
    if unit == "m":
        return amount * 60
    if unit == "h":
        return amount * 3600
    raise ValueError("interval must use s, m, or h")


def _decimal_string(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f")


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
