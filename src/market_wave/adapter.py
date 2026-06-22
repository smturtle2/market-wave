from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from .corpus import read_public_corpus


def current_price(root: Path) -> dict[str, Any]:
    market, snapshots, _ = read_public_corpus(root)
    trades = [record for record in market if record.get("record_type") == "TRADE"]
    if trades:
        last = trades[-1]
        return {
            "result": {
                "timestamp": last["timestamp"],
                "last_price": last["payload"]["price"],
                "currency": last["payload"]["currency"],
            }
        }
    if snapshots:
        snap = snapshots[-1]
        return {
            "result": {
                "timestamp": snap["timestamp"],
                "last_price": snap["mid"],
                "currency": snap["currency"],
            }
        }
    return _error("NOT_FOUND", "no visible market data")


def orderbook(root: Path, *, limit: int = 10) -> dict[str, Any]:
    _, snapshots, _ = read_public_corpus(root)
    if not snapshots:
        return _error("NOT_FOUND", "no snapshots")
    snap = snapshots[-1]
    return {
        "result": {
            "timestamp": snap["timestamp"],
            "currency": snap["currency"],
            "asks": snap["asks"][:limit],
            "bids": snap["bids"][:limit],
        }
    }


def trades(root: Path, *, limit: int = 100) -> dict[str, Any]:
    market, _, _ = read_public_corpus(root)
    rows = []
    for record in market:
        if record.get("record_type") == "TRADE":
            rows.append(
                {
                    "price": record["payload"]["price"],
                    "volume": record["payload"]["volume"],
                    "side": record["payload"]["side"],
                    "timestamp": record["timestamp"],
                    "currency": record["payload"]["currency"],
                }
            )
    return {"result": {"trades": rows[-limit:]}}


def book_events(root: Path, *, limit: int = 100, cursor: int = 0) -> dict[str, Any]:
    market, _, _ = read_public_corpus(root)
    rows = []
    for record in market[cursor : cursor + limit]:
        rows.append({
            "event_id": record["event_id"],
            "timestamp": record["timestamp"],
            "event_type": record["record_type"],
            **record["payload"],
        })
    next_cursor = cursor + len(rows) if cursor + len(rows) < len(market) else None
    return {"result": {"events": rows, "next_cursor": next_cursor}}


def candles(root: Path, *, interval_seconds: int = 60) -> dict[str, Any]:
    market, _, _ = read_public_corpus(root)
    trades_rows = [record for record in market if record.get("record_type") == "TRADE"]
    if not trades_rows:
        return {"result": {"candles": [], "next_before": None}}
    buckets: dict[datetime, list[dict[str, Any]]] = {}
    for record in trades_rows:
        ts = datetime.fromisoformat(record["timestamp"])
        epoch = int(ts.timestamp())
        bucket_start = epoch - (epoch % interval_seconds)
        key = datetime.fromtimestamp(bucket_start, tz=ts.tzinfo)
        buckets.setdefault(key, []).append(record)
    candles_out = []
    for key in sorted(buckets):
        rows = buckets[key]
        prices = [Decimal(row["payload"]["price"]) for row in rows]
        volumes = [Decimal(row["payload"]["volume"]) for row in rows]
        currency = rows[-1]["payload"]["currency"]
        candles_out.append(
            {
                "timestamp": key.isoformat(timespec="milliseconds"),
                "open": _decimal_string(prices[0]),
                "high": _decimal_string(max(prices)),
                "low": _decimal_string(min(prices)),
                "close": _decimal_string(prices[-1]),
                "volume": _decimal_string(sum(volumes)),
                "currency": currency,
            }
        )
    return {"result": {"candles": candles_out, "next_before": None}}


def _error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details}}


def _decimal_string(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f")
