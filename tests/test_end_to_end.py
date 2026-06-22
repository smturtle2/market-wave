from __future__ import annotations

import hashlib
import json
from pathlib import Path

import market_wave as mw
from market_wave.api import replay
from market_wave.config import GenerationConfig
from market_wave.corpus import read_jsonl
from market_wave.validation import validate_corpus


def test_generate_validate_adapt_plot_and_replay(tmp_path: Path) -> None:
    config = GenerationConfig(seed=17, steps=160)
    first = tmp_path / "first"
    second = tmp_path / "second"

    generated = mw.generate(first, config, force=True)
    report = generated.validate()
    assert report.ok, report.errors

    market = read_jsonl(first / "corpus" / "visible_market_stream.jsonl")
    snapshots = read_jsonl(first / "corpus" / "visible_snapshot_stream.jsonl")
    assert market
    assert snapshots
    assert any(row["record_type"] == "TRADE" for row in market)
    assert "payload" in market[0]
    assert "result" not in market[0]
    assert not _contains_forbidden_public_key(market)
    assert not _contains_forbidden_public_key(snapshots)

    manifest = json.loads((first / "corpus" / "manifest.json").read_text())
    assert "seed" not in manifest
    assert "symbol" not in manifest
    assert "theta_id" not in json.dumps(manifest)
    assert (first / "internal" / "hidden_sidecar.jsonl").exists()
    assert (first / "internal" / "theta_manifest.jsonl").exists()

    assert "last_price" in generated.price()
    assert "asks" in generated.book()
    assert "bids" in generated.orderbook()
    assert generated.trades()
    assert generated.candles()
    events = generated.events(limit=5)
    assert len(events["events"]) == 5

    plots = generated.plot()
    assert {path.name for path in plots} == {
        "market_screen.png",
        "price_chart.png",
        "orderbook_panel.png",
        "top10_orderbook_heatmap.png",
        "event_tape.png",
        "mid_spread_depth.png",
        "depth_heatmap.png",
    }
    assert all(path.stat().st_size > 0 for path in plots)

    mw.generate(second, config, force=True)
    assert _tree_hash(first / "corpus") == _tree_hash(second / "corpus")
    assert _tree_hash(first / "internal") == _tree_hash(second / "internal")
    assert replay(config)["byte_identical"]


def _contains_forbidden_public_key(value: object) -> bool:
    forbidden = {
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
        "symbol",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden or str(key).startswith("future_"):
                return True
            if _contains_forbidden_public_key(child):
                return True
    if isinstance(value, list):
        return any(_contains_forbidden_public_key(child) for child in value)
    return False


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
