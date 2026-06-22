from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .corpus import FORBIDDEN_PUBLIC_KEYS, MANIFEST_FILE, VISIBLE_MARKET_FILE, VISIBLE_SNAPSHOT_FILE, read_jsonl


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors))


def minimal_validation_passes(counters: Any) -> bool:
    if counters.book_invariant_violations > 0:
        return False
    if counters.integer_unit_violations > 0:
        return False
    if counters.hidden_leakage_violations > 0:
        return False
    if counters.dead_world:
        return False
    if counters.exploded_world:
        return False
    return True


def validate_corpus(root: Path) -> ValidationReport:
    root = Path(root)
    corpus = root / "corpus"
    errors: list[str] = []
    counters = {
        "visible_records": 0,
        "snapshot_records": 0,
        "trade_records": 0,
        "book_update_records": 0,
        "hidden_leakage_violations": 0,
        "schema_violations": 0,
        "book_invariant_violations": 0,
    }
    manifest_path = corpus / MANIFEST_FILE
    if not manifest_path.exists():
        errors.append("missing corpus/manifest.json")
        manifest: dict[str, Any] = {}
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
            counters["schema_violations"] += 1
            errors.append("invalid corpus/manifest.json")
        hidden = _hidden_leakage(manifest)
        counters["hidden_leakage_violations"] += hidden
        if hidden:
            errors.append("hidden leakage in manifest")
    market_path = corpus / VISIBLE_MARKET_FILE
    snapshot_path = corpus / VISIBLE_SNAPSHOT_FILE
    if not market_path.exists():
        errors.append("missing visible_market_stream.jsonl")
    if not snapshot_path.exists():
        errors.append("missing visible_snapshot_stream.jsonl")
    market = read_jsonl(market_path)
    snapshots = read_jsonl(snapshot_path)
    counters["visible_records"] = len(market)
    counters["snapshot_records"] = len(snapshots)
    if not market:
        errors.append("dead world: no visible market records")
    if not snapshots:
        errors.append("dead world: no visible snapshots")

    last_event_id = ""
    for record in market:
        hidden = _hidden_leakage(record)
        counters["hidden_leakage_violations"] += hidden
        if hidden:
            errors.append(f"hidden leakage in visible record {record.get('event_id', '<missing>')}")
        required = {"event_id", "step_id", "world_id", "timestamp", "record_type", "payload"}
        if set(record) != required:
            counters["schema_violations"] += 1
            errors.append(f"invalid visible record keys: {record.get('event_id', '<missing>')}")
            continue
        if record["event_id"] <= last_event_id:
            counters["schema_violations"] += 1
            errors.append("event_id ordering violation")
        last_event_id = record["event_id"]
        if record["record_type"] == "ORDER_ACCEPTED":
            if set(record["payload"]) != {"side", "price", "volume", "currency"}:
                counters["schema_violations"] += 1
                errors.append(f"invalid ORDER_ACCEPTED payload: {record['event_id']}")
        elif record["record_type"] == "TAKE_ORDER":
            if set(record["payload"]) != {"side", "price", "volume", "currency"}:
                counters["schema_violations"] += 1
                errors.append(f"invalid TAKE_ORDER payload: {record['event_id']}")
        elif record["record_type"] == "ORDER_CANCELED":
            if set(record["payload"]) != {"side", "price", "volume", "currency"}:
                counters["schema_violations"] += 1
                errors.append(f"invalid ORDER_CANCELED payload: {record['event_id']}")
        elif record["record_type"] == "ORDER_REPLACED":
            if set(record["payload"]) != {"side", "old_price", "new_price", "old_volume", "new_volume", "currency"}:
                counters["schema_violations"] += 1
                errors.append(f"invalid ORDER_REPLACED payload: {record['event_id']}")
        elif record["record_type"] == "TRADE":
            counters["trade_records"] += 1
            if set(record["payload"]) != {"side", "price", "volume", "currency"}:
                counters["schema_violations"] += 1
                errors.append(f"invalid TRADE payload: {record['event_id']}")
        elif record["record_type"] == "BOOK_DELTA":
            counters["book_update_records"] += 1
            if set(record["payload"]) != {"side", "price", "volume_delta", "volume_after", "currency"}:
                counters["schema_violations"] += 1
                errors.append(f"invalid BOOK_DELTA payload: {record['event_id']}")
        else:
            counters["schema_violations"] += 1
            errors.append(f"unknown record_type: {record['record_type']}")
        _validate_decimal_strings(record, errors, counters)
    if counters["trade_records"] == 0:
        errors.append("dead world: no trades")

    for snapshot in snapshots:
        hidden = _hidden_leakage(snapshot)
        counters["hidden_leakage_violations"] += hidden
        if hidden:
            errors.append(f"hidden leakage in snapshot {snapshot.get('snapshot_id', '<missing>')}")
        required = {
            "snapshot_id",
            "step_id",
            "world_id",
            "timestamp",
            "currency",
            "best_bid",
            "best_ask",
            "mid",
            "spread",
            "asks",
            "bids",
            "total_visible_bid_depth",
            "total_visible_ask_depth",
        }
        if set(snapshot) != required:
            counters["schema_violations"] += 1
            errors.append(f"invalid snapshot keys: {snapshot.get('snapshot_id', '<missing>')}")
            continue
        asks = snapshot["asks"]
        bids = snapshot["bids"]
        if not asks or not bids:
            counters["schema_violations"] += 1
            errors.append("snapshot must expose visible ask and bid ladders")
        ask_prices = [_decimal_value(level.get("price")) for level in asks if isinstance(level, dict)]
        bid_prices = [_decimal_value(level.get("price")) for level in bids if isinstance(level, dict)]
        if any(price is None for price in ask_prices + bid_prices):
            counters["schema_violations"] += 1
            errors.append("snapshot level price must be decimal string")
        comparable_asks = [price for price in ask_prices if price is not None]
        comparable_bids = [price for price in bid_prices if price is not None]
        if comparable_asks != sorted(comparable_asks):
            counters["book_invariant_violations"] += 1
            errors.append("ask ordering violation")
        if comparable_bids != sorted(comparable_bids, reverse=True):
            counters["book_invariant_violations"] += 1
            errors.append("bid ordering violation")
        if comparable_bids and comparable_asks and max(comparable_bids) >= min(comparable_asks):
            counters["book_invariant_violations"] += 1
            errors.append("crossed or locked snapshot")
        _validate_decimal_strings(snapshot, errors, counters)
    return ValidationReport(ok=not errors, errors=errors, counters=counters)


def _hidden_leakage(record: Any) -> int:
    count = 0
    if isinstance(record, dict):
        for key, value in record.items():
            key_s = str(key)
            if key_s in FORBIDDEN_PUBLIC_KEYS or key_s.startswith("future_"):
                count += 1
            count += _hidden_leakage(value)
    elif isinstance(record, list):
        for value in record:
            count += _hidden_leakage(value)
    return count


def _validate_decimal_strings(record: Any, errors: list[str], counters: dict[str, int]) -> None:
    if isinstance(record, dict):
        for key, value in record.items():
            if key in {
                "price",
                "old_price",
                "new_price",
                "volume",
                "old_volume",
                "new_volume",
                "volume_delta",
                "volume_after",
                "best_bid",
                "best_ask",
                "mid",
                "spread",
            }:
                if not isinstance(value, str) or not _is_decimal_string(value):
                    counters["schema_violations"] += 1
                    errors.append(f"{key} must be decimal string")
            _validate_decimal_strings(value, errors, counters)
    elif isinstance(record, list):
        for value in record:
            _validate_decimal_strings(value, errors, counters)


def _is_decimal_string(value: str) -> bool:
    if not value:
        return False
    if value[0] == "-":
        value = value[1:]
    parts = value.split(".")
    return len(parts) <= 2 and all(part.isdigit() for part in parts if part)


def _decimal_value(value: Any) -> Decimal | None:
    if not isinstance(value, str) or not _is_decimal_string(value):
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None
