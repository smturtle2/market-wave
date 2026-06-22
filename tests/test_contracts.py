from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from market_wave.book import (
    BUY,
    SELL,
    MarketMessage,
    assert_book_invariants,
    init_book,
    price_depth,
    public_best_ask_tick,
    public_best_bid_tick,
    reconstruct_projection,
    side_depth,
)
from market_wave.config import GenerationConfig
from market_wave.corpus import read_jsonl
from market_wave.field import encode_feedback, sample_market_message, update_latent_field
from market_wave.generator import run_world_streaming
from market_wave.physics import apply_book_physics
from market_wave.theta import sample_world_theta
from market_wave.validation import validate_corpus


def test_runtime_rng_owner_signature_contract() -> None:
    assert "rng" in inspect.signature(sample_market_message).parameters
    assert "rng" not in inspect.signature(apply_book_physics).parameters
    assert "rng" not in inspect.signature(encode_feedback).parameters
    assert "rng" not in inspect.signature(update_latent_field).parameters


def test_marketable_limit_order_matches_fifo_and_moves_depth() -> None:
    rng = np.random.default_rng(3)
    theta = sample_world_theta(rng, GenerationConfig(seed=3, steps=10))
    book = init_book(rng, theta)
    ask = public_best_ask_tick(book)
    ask_depth = price_depth(book, SELL, ask)

    next_book, trades, events = apply_book_physics(book, MarketMessage("LIMIT", BUY, ask, ask_depth), theta)

    assert sum(trade.quantity_unit for trade in trades) == ask_depth
    assert all(trade.price_tick == ask for trade in trades)
    assert price_depth(next_book, SELL, ask) == 0
    assert public_best_ask_tick(next_book) > ask
    assert any(event.record_type == "TRADE" for event in events)
    assert any(event.record_type == "BOOK_DELTA" and event.payload["after"] == 0 for event in events)
    assert_book_invariants(next_book)


def test_non_marketable_limit_order_rests_without_trade() -> None:
    rng = np.random.default_rng(5)
    theta = sample_world_theta(rng, GenerationConfig(seed=5, steps=10))
    book = init_book(rng, theta)
    bid = public_best_bid_tick(book)
    before_depth = price_depth(book, BUY, bid)

    next_book, trades, events = apply_book_physics(book, MarketMessage("LIMIT", BUY, bid, 25), theta)

    assert trades == []
    assert price_depth(next_book, BUY, bid) == before_depth + 25
    assert any(event.record_type == "ORDER_ACCEPTED" for event in events)
    assert any(event.record_type == "BOOK_DELTA" and event.payload["delta"] == 25 for event in events)
    assert_book_invariants(next_book)


def test_cancel_preserves_volume_accounting() -> None:
    rng = np.random.default_rng(7)
    theta = sample_world_theta(rng, GenerationConfig(seed=7, steps=10))
    book = init_book(rng, theta)
    order = next(iter(book.orders.values()))
    before_side_depth = side_depth(book, order.side)
    qty = max(theta.min_quantity_unit, order.remaining_qty // 2)

    next_book, trades, events = apply_book_physics(book, MarketMessage("CANCEL", order.side, order.price_tick, qty, order_id=order.order_id), theta)

    assert trades == []
    assert side_depth(next_book, order.side) == before_side_depth - qty
    assert any(event.record_type == "ORDER_CANCELED" for event in events)
    assert any(event.record_type == "BOOK_DELTA" and event.payload["delta"] == -qty for event in events)
    assert_book_invariants(next_book)


def test_public_stream_reconstructs_final_snapshot(tmp_path: Path) -> None:
    config = GenerationConfig(seed=11, world_index=11, steps=120)
    root = tmp_path / "world"
    run_world_streaming(config, root, force=True)
    rng = np.random.default_rng(config.seed)
    theta = sample_world_theta(rng, config)

    snapshots = read_jsonl(root / "corpus" / "visible_snapshot_stream.jsonl")
    market = read_jsonl(root / "corpus" / "visible_market_stream.jsonl")
    reconstructed = reconstruct_projection(snapshots[0], market, theta, levels=theta.initial_book_levels)

    for key in ("best_bid", "best_ask", "mid", "spread", "asks", "bids", "total_visible_bid_depth", "total_visible_ask_depth"):
        assert reconstructed[key] == snapshots[-1][key]


def test_validation_rejects_forbidden_public_manifest_fields(tmp_path: Path) -> None:
    root = tmp_path / "bad_manifest"
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "manifest.json").write_text('{"seed":17}\n')
    (corpus / "visible_snapshot_stream.jsonl").write_text(
        '{"snapshot_id":"s","step_id":"st","world_id":"w","timestamp":"2026-06-22T09:00:00.000+09:00","currency":"KRW","best_bid":"1","best_ask":"3","mid":"2","spread":"2","asks":[{"price":"3","volume":"1"}],"bids":[{"price":"1","volume":"1"}],"total_visible_bid_depth":"1","total_visible_ask_depth":"1"}\n'
    )
    (corpus / "visible_market_stream.jsonl").write_text(
        '{"event_id":"e","step_id":"st","world_id":"w","timestamp":"2026-06-22T09:00:00.000+09:00","record_type":"TRADE","payload":{"side":"BUY","price":"1","volume":"1","currency":"KRW"}}\n'
    )

    report = validate_corpus(root)

    assert not report.ok
    assert report.counters["hidden_leakage_violations"] > 0


def test_hidden_sidecar_is_not_read_by_public_reader(tmp_path: Path) -> None:
    root = tmp_path / "world"
    run_world_streaming(GenerationConfig(seed=23, world_index=23, steps=120), root, force=True)
    market = read_jsonl(root / "corpus" / "visible_market_stream.jsonl")
    hidden = read_jsonl(root / "internal" / "hidden_sidecar.jsonl")
    manifest = json.loads((root / "corpus" / "manifest.json").read_text())

    assert market
    assert hidden
    assert "z_summary" not in market[0]
    assert "z_summary" in hidden[0]
    assert "seed" not in manifest
