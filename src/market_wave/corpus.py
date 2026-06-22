from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import __version__

SCHEMA_VERSION = "market-wave.corpus/1.0.0"
VISIBLE_MARKET_FILE = "visible_market_stream.jsonl"
VISIBLE_SNAPSHOT_FILE = "visible_snapshot_stream.jsonl"
MANIFEST_FILE = "manifest.json"
HIDDEN_SIDECAR_FILE = "hidden_sidecar.jsonl"
THETA_MANIFEST_FILE = "theta_manifest.jsonl"
RUN_MANIFEST_FILE = "run_manifest.json"

FORBIDDEN_PUBLIC_KEYS = {
    "seed",
    "symbol",
    "rng",
    "target",
    "label",
    "return",
    "horizon",
    "regime",
    "concept",
    "z",
    "field",
    "event_field",
    "theta",
    "theta_id",
    "generator_parameter",
    "guard_depth",
    "guard_mask",
    "book_truth",
    "inventory",
    "toxicity",
}


@dataclass
class CorpusCounters:
    visible_records: int = 0
    snapshot_records: int = 0
    trade_records: int = 0
    book_update_records: int = 0
    hidden_records: int = 0
    hidden_leakage_violations: int = 0
    integer_unit_violations: int = 0
    book_invariant_violations: int = 0
    dead_world: bool = False
    exploded_world: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "visible_records": self.visible_records,
            "snapshot_records": self.snapshot_records,
            "trade_records": self.trade_records,
            "book_update_records": self.book_update_records,
        }

    def to_internal_dict(self) -> dict[str, Any]:
        return {
            **self.to_public_dict(),
            "hidden_records": self.hidden_records,
            "hidden_leakage_violations": self.hidden_leakage_violations,
            "integer_unit_violations": self.integer_unit_violations,
            "book_invariant_violations": self.book_invariant_violations,
            "dead_world": self.dead_world,
            "exploded_world": self.exploded_world,
        }


class CorpusSink:
    def __init__(self, root: Path, *, force: bool = False):
        self.root = Path(root)
        self.tmp = self.root / ".market_wave_tmp"
        self.force = force
        if self.root.exists() and force:
            for child in ("corpus", "internal", "plots", ".market_wave_tmp"):
                path = self.root / child
                if path.exists():
                    shutil.rmtree(path)
        self.tmp_corpus = self.tmp / "corpus"
        self.tmp_internal = self.tmp / "internal"
        self.tmp_corpus.mkdir(parents=True, exist_ok=True)
        self.tmp_internal.mkdir(parents=True, exist_ok=True)
        self.visible_market_path = self.tmp_corpus / VISIBLE_MARKET_FILE
        self.visible_snapshot_path = self.tmp_corpus / VISIBLE_SNAPSHOT_FILE
        self.hidden_sidecar_path = self.tmp_internal / HIDDEN_SIDECAR_FILE
        self.theta_manifest_path = self.tmp_internal / THETA_MANIFEST_FILE
        self.run_manifest_path = self.tmp_internal / RUN_MANIFEST_FILE

    def write_visible_market(self, record: dict[str, Any]) -> None:
        _assert_no_hidden_leakage(record)
        _append_jsonl(self.visible_market_path, record)

    def write_visible_snapshot(self, record: dict[str, Any]) -> None:
        _assert_no_hidden_leakage(record)
        _append_jsonl(self.visible_snapshot_path, record)

    def write_hidden(self, record: dict[str, Any]) -> None:
        _append_jsonl(self.hidden_sidecar_path, record)

    def write_theta_manifest(self, record: dict[str, Any]) -> None:
        _append_jsonl(self.theta_manifest_path, record)

    def write_run_manifest(self, record: dict[str, Any]) -> None:
        _write_json(self.run_manifest_path, record)

    def write_public_manifest(self, record: dict[str, Any]) -> None:
        _assert_no_hidden_leakage(record)
        _write_json(self.tmp_corpus / MANIFEST_FILE, record)

    def commit(self) -> None:
        corpus = self.root / "corpus"
        internal = self.root / "internal"
        if corpus.exists():
            shutil.rmtree(corpus)
        if internal.exists():
            shutil.rmtree(internal)
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.tmp_corpus), str(corpus))
        shutil.move(str(self.tmp_internal), str(internal))
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    def discard(self) -> None:
        if self.tmp.exists():
            shutil.rmtree(self.tmp)


def public_manifest(
    *,
    world_id: str,
    currency: str,
    tick_size_decimal: str,
    quantity_unit_decimal: str,
    start_timestamp: str,
    counters: CorpusCounters,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "world_id": world_id,
        "currency": currency,
        "units": {
            "tick_size_decimal": tick_size_decimal,
            "quantity_unit_decimal": quantity_unit_decimal,
        },
        "start_timestamp": start_timestamp,
        "files": {
            "visible_market_stream": VISIBLE_MARKET_FILE,
            "visible_snapshot_stream": VISIBLE_SNAPSHOT_FILE,
        },
        "record_types": ["ORDER_ACCEPTED", "TAKE_ORDER", "ORDER_CANCELED", "ORDER_REPLACED", "TRADE", "BOOK_DELTA"],
        "counters": counters.to_public_dict(),
    }


def make_visible_market_record(
    *,
    event_id: str,
    step_id: str,
    world_id: str,
    timestamp: str,
    record_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "event_id": event_id,
        "step_id": step_id,
        "world_id": world_id,
        "timestamp": timestamp,
        "record_type": record_type,
        "payload": payload,
    }
    _assert_no_hidden_leakage(record)
    return record


def make_snapshot_record(
    *,
    snapshot_id: str,
    step_id: str,
    world_id: str,
    timestamp: str,
    currency: str,
    projection: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "snapshot_id": snapshot_id,
        "step_id": step_id,
        "world_id": world_id,
        "timestamp": timestamp,
        "currency": currency,
        "best_bid": projection["best_bid"],
        "best_ask": projection["best_ask"],
        "mid": projection["mid"],
        "spread": projection["spread"],
        "asks": projection["asks"],
        "bids": projection["bids"],
        "total_visible_bid_depth": projection["total_visible_bid_depth"],
        "total_visible_ask_depth": projection["total_visible_ask_depth"],
    }
    _assert_no_hidden_leakage(record)
    return record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not Path(path).exists():
        return
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_public_corpus(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    corpus = Path(root) / "corpus"
    market = read_jsonl(corpus / VISIBLE_MARKET_FILE)
    snapshots = read_jsonl(corpus / VISIBLE_SNAPSHOT_FILE)
    manifest_path = corpus / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return market, snapshots, manifest


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_json_dumps(record))
        handle.write("\n")


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(record) + "\n", encoding="utf-8")


def _json_dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _assert_no_hidden_leakage(record: Any) -> None:
    for key, value in _walk_items(record):
        if key in FORBIDDEN_PUBLIC_KEYS or key.startswith("future_"):
            raise ValueError(f"forbidden public corpus key: {key}")
        _ = value


def _walk_items(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_items(child)
