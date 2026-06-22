# Market Wave

Market Wave is a latent-field synthetic stock market simulator backed by a
deterministic price-time matching engine. It generates visible-only public
market messages, reconstructable snapshots, internal sidecars, simple JSON
views, structural validation reports, deterministic replay checks, and
market-screen plots.

## Contract

- Runtime randomness exists only in `sample_market_message()`.
- Internal prices are integer ticks. Internal quantities are integer units.
- Crossed or locked books are structurally impossible in the matching engine.
- Public corpus never contains hidden fields, generator parameters, labels,
  targets, returns, future columns, or ticker symbols.
- Public messages are typed: `ORDER_ACCEPTED`, `TAKE_ORDER`,
  `ORDER_CANCELED`, `ORDER_REPLACED`, `TRADE`, and `BOOK_DELTA`.
- Aggressive flow is explicit. `TAKE_ORDER` walks resting liquidity and never
  rests; passive `ORDER_ACCEPTED` messages may rest or partially execute.
- Snapshots are reconstructable from the initial full visible ladder and ordered
  public `BOOK_DELTA` events.
- Public `world_id` is not derived from seed or theta. Use `--world-index` or
  `--world-id` for public identity.
- Internal artifacts are written under `internal/`, outside the trainable corpus.
- Visualization reads visible corpus files only. Price charts are built from
  trades, candles, volume, best bid/ask, and mid. Top-10 orderbook heatmaps use
  fixed quote levels on the y-axis and time on the x-axis.
- Visual output uses a white/cool-gray trading-screen theme with blue/red
  contrast for bid/ask and buy/sell. Warm-tone page backgrounds, violet accents,
  and dark terminal themes are out of contract.

## CLI

```bash
market-wave generate out --seed 7 --steps 500 --price 71800 --tick 100 --force
market-wave validate out
market-wave replay --seed 7 --steps 500
market-wave price out
market-wave book out
market-wave trades out --limit 100
market-wave candles out --interval 1m
market-wave events out --limit 100 --cursor 0
market-wave plot out
market-wave inspect out
```

## Output Layout

```text
out/
  corpus/
    manifest.json
    visible_market_stream.jsonl
    visible_snapshot_stream.jsonl
  internal/
    hidden_sidecar.jsonl
    theta_manifest.jsonl
    run_manifest.json
  plots/
    market_screen.png
    price_chart.png
    orderbook_panel.png
    event_tape.png
    mid_spread_depth.png
    depth_heatmap.png
    top10_orderbook_heatmap.png
```

Only `corpus/` is training input. `internal/` is for replay and debugging.

## Python API

The top-level Python API is intentionally small: `generate`, `open`, and
`Market`.

```python
import market_wave as mw

market = mw.generate("out", seed=7, steps=500, price=71_800, tick=100, force=True)
market.validate()
market.plot()

market.price()
market.book(levels=10)
market.trades(limit=100)
market.candles("1m")
market.events(limit=100, cursor=0)

same_market = mw.open("out")
```
