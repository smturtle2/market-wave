from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from market_wave.config import GenerationConfig
from market_wave.corpus import read_jsonl
from market_wave.generator import run_world_streaming
from market_wave.validation import validate_corpus
from market_wave.visualization import _top10_level_matrices, trade_price_bars


def test_generated_worlds_have_typed_market_messages(tmp_path: Path) -> None:
    for seed in [1, 7, 17, 23, 99]:
        root = tmp_path / f"world_{seed}"
        run_world_streaming(GenerationConfig(seed=seed, world_index=seed, steps=500), root, force=True)
        report = validate_corpus(root)
        assert report.ok, report.errors

        market = read_jsonl(root / "corpus" / "visible_market_stream.jsonl")
        snapshots = read_jsonl(root / "corpus" / "visible_snapshot_stream.jsonl")
        record_types = {row["record_type"] for row in market}
        trades = [row for row in market if row["record_type"] == "TRADE"]
        trade_prices = [Decimal(row["payload"]["price"]) for row in trades]
        mids = [Decimal(snapshot["mid"]) for snapshot in snapshots]
        spreads = [Decimal(snapshot["spread"]) for snapshot in snapshots]

        assert {"ORDER_ACCEPTED", "TAKE_ORDER", "BOOK_DELTA", "TRADE"}.issubset(record_types)
        assert any(row["record_type"] == "ORDER_CANCELED" for row in market)
        assert any(row["record_type"] == "ORDER_REPLACED" for row in market)
        assert len(snapshots) == 501
        assert all(Decimal(snapshot["best_bid"]) < Decimal(snapshot["best_ask"]) for snapshot in snapshots)
        assert len(trades) >= 200
        assert len(set(trade_prices)) >= 8
        assert max(mids) - min(mids) >= Decimal("500")
        assert len(set(spreads)) >= 3
        assert trade_price_bars(market, snapshots)

        volume, side = _top10_level_matrices(snapshots)
        assert volume.shape == (20, len(snapshots))
        assert volume.shape[1] == len(snapshots)
        assert side.min() < 0
        assert side.max() > 0
        assert any(volume[row].max() > volume[row].min() for row in range(20))
