# market-wave

<p align="center">
  <img src="https://raw.githubusercontent.com/smturtle2/market-wave/main/docs/assets/market-wave-hero.png" alt="market-wave abstract market intent simulation hero" />
</p>

<p align="center">
  <strong>Fast, lightweight synthetic market data from a Dynamic Market Distribution Function.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/market-wave/"><img alt="PyPI" src="https://img.shields.io/pypi/v/market-wave"></a>
  <a href="https://pypi.org/project/market-wave/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/market-wave"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href=".github/workflows/workflow.yml"><img alt="Tests" src="https://img.shields.io/badge/tests-pytest-2563eb"></a>
</p>

<p align="center">
  English | <a href="README.ko.md">한국어</a>
</p>

`market-wave` is a Python library for generating synthetic market paths from a
market-wide entry price distribution. It does not create individual
participants. Instead, it models aggregate buy/sell entry intent, resting
order-book depth, probabilistic order cancellation, taker flow, and
execution-driven price movement from probability mass over relative ticks.

It is not a forecasting model. It is a lightweight simulation primitive for
experiments, visualization, teaching, and strategy-environment prototyping.

## Why market-wave?

- **Aggregate intent, not agents**: market participants are represented by
  probability mass over relative ticks, not by individual objects.
- **Raw-mass MDF**: buy/sell entry intent is built by summing observable
  tick-level mass, then normalizing directly.
- **Separated shape and size**: MDFs decide where intent sits; intensity decides
  how much order flow appears.
- **Execution-driven prices**: prices stay flat unless trades execute.
- **Batch generation**: generate many reproducible synthetic paths without
  keeping every path in `market.history`.
- **Inspectable state**: every step returns a `StepInfo` snapshot with MDFs,
  submitted volume, cancelled volume, executions, order book state, VWAP,
  spread, and imbalance.
- **Built-in plotting**: `matplotlib` is included, with a clean light chart style
  by default.

## Install

```bash
pip install market-wave
```

For dataframe export:

```bash
pip install "market-wave[dataframe]"
```

For local development:

```bash
git clone https://github.com/smturtle2/market-wave.git
cd market-wave
uv sync --extra dev
```

Python `>=3.10` is supported.

## Quickstart

```python
from market_wave import Market

market = Market(
    initial_price=10_000,
    gap=10,
    popularity=1.0,
    seed=42,
)
steps = market.step(500)

last = steps[-1]
print(last.price_before, "->", last.price_after)
print("entry:", round(sum(last.entry_volume_by_price.values()), 3))
print("executed:", round(last.total_executed_volume, 3))
print("resting bid/ask:", round(sum(last.orderbook_after.bid_volume_by_price.values()), 3), round(sum(last.orderbook_after.ask_volume_by_price.values()), 3))
print("imbalance:", round(last.order_flow_imbalance, 3))
```

`Market.step(n)` always returns `list[StepInfo]` and appends the same objects to
`market.history`.

For high-volume generation, skip in-memory history:

```python
steps = market.step(512, keep_history=False)

for step in market.stream(512, keep_history=False):
    consume(step)
```

For simple export workflows, use `step.to_dict()`, `step.to_json()`, or
`market.history_records()`.

Example output with `seed=42`:

```text
9930.0 -> 9930.0
entry: 1.943
executed: 0.715
resting bid/ask: 26.187 25.893
imbalance: -0.119
```

## Smoke Matrix

The simulator is deterministic for a fixed seed, so it is easy to run the same
invariants across different market conditions:

```python
from market_wave import Market

cases = [
    ("baseline", dict(initial_price=10_000, gap=10, popularity=1.0, seed=42), 500),
    ("busy", dict(initial_price=10_000, gap=10, popularity=2.5, seed=7), 500),
    ("thin", dict(initial_price=500, gap=5, popularity=0.25, seed=123), 500),
    ("low_price", dict(initial_price=1, gap=1, popularity=3.0, seed=17), 500),
    ("trend_up", dict(initial_price=10_000, gap=10, popularity=1.0, seed=42, regime="trend_up"), 500),
    ("high_vol", dict(initial_price=10_000, gap=10, popularity=1.0, seed=7, regime="high_vol"), 500),
    ("inactive", dict(initial_price=100, gap=1, popularity=0.0, seed=9), 100),
]

for name, kwargs, steps_count in cases:
    market = Market(**kwargs)
    steps = market.step(steps_count)
    prices = [step.price_after for step in steps]
    move_steps = sum(step.price_change != 0 for step in steps)
    exec_steps = sum(step.total_executed_volume > 0 for step in steps)
    print(name, min(prices), max(prices), move_steps, exec_steps, market.state.price)
```

Recent verification on the current implementation:

```text
baseline  range=  9910.0- 10030.0 unique= 13 moves=229 exec_steps=500 final=  9930.0
busy      range=  9980.0- 10080.0 unique= 11 moves=232 exec_steps=500 final= 10040.0
thin      range=   485.0-   550.0 unique= 14 moves=207 exec_steps=500 final=   485.0
low_price range=     2.0-    24.0 unique= 23 moves=221 exec_steps=500 final=    20.0
trend_up  range=  9980.0- 10330.0 unique= 36 moves=273 exec_steps=500 final= 10330.0
high_vol  range=  9990.0- 10120.0 unique= 14 moves=362 exec_steps=500 final= 10100.0
inactive  range=   100.0-   100.0 unique=  1 moves=  0 exec_steps=  0 final=   100.0
```

Those runs also checked that current-state MDF projections stay aligned with
`state.price_grid`, MDFs remain normalized, prices never fall below one tick,
order-book depth stays non-negative, and price changes only occur on steps with
executed volume. Dynamic MDF acceptance also runs seeds `10..19` and checks that
every MDF remains finite, non-negative, normalized, and broad enough not to
collapse to a single price.

Diagnostic note for the current engine: the simulator still has no anchor price or stored
target that pulls paths back to the initial price. Seeded `mood`, `trend`,
`volatility`, microstructure activity, cancellation pressure, and event pressure
evolve each step and reshape the MDFs and visible book. Prices remain
execution-driven, with volatility-aware price discovery when executed flow
reveals one-sided pressure. Treat these ranges, move counts, and
execution counts as regression diagnostics, not claims that generated paths
match any specific real market.

Entry MDF prices are treated as incoming order prices. Buy entry orders arrive as
bids, sell entry orders arrive as asks, and they execute only when they overlap
existing opposite-side quotes. Executions print at the resting quote price.
Unfilled volume remains in the book at the sampled MDF price. Resting orders
then leave through probabilistic cancellation: orders at prices that the current
same-side entry MDF no longer supports are more likely to shrink or disappear.

MDF note for the current engine: buy/sell MDFs no longer use score softmax
updates. The engine builds raw mass at each relative tick from near-market
continuity, visible book shortage, front depletion, liquidity pockets, flow
pressure, volatility, stress, and microstructure texture. The raw mass is
normalized directly, with only a small linear memory mix so the previous shape
has inertia without becoming an anchor.

Microstructure note for the current engine: order-book replenishment is shaped
by the current entry MDF, resiliency, stress-aware cancellation, event-driven
volume bursts, dry-up after cancellation pressure, trend exhaustion, and
book-pressure squeeze signals from one-sided visible liquidity. Live order-book
totals remain cached by price/side, and lots are coalesced by price/kind.

## Visualization

```python
from market_wave import Market

market = Market(
    initial_price=10_000,
    gap=10,
    popularity=1.0,
    seed=42,
)
market.step(500)

fig, ax = market.plot(last=220, orderbook_depth=12)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/smturtle2/market-wave/main/docs/assets/market-wave-plot.png" alt="market-wave light pyplot chart showing price, orderbook depth heatmaps, volume, and imbalance" />
</p>

The default `market_wave` style uses a light multi-panel chart: price/VWAP,
bid and ask orderbook depth heatmaps by simple level, executed volume, and
order-flow imbalance. To keep the legacy three-panel view, pass
`orderbook=False`.

Dark overlay mode is still available:

```python
fig, ax = market.plot(layout="overlay", style="market_wave_dark")
```

## Synthetic Data

```python
from market_wave import compute_metrics, generate_paths

paths = generate_paths(
    n_paths=100,
    horizon=512,
    config_sampler=lambda path_id: {
        "initial_price": 10_000,
        "gap": 10,
        "popularity": 1.0,
        "seed": 10_000 + path_id,
    },
)

metrics = compute_metrics(paths)
print(metrics.return_std, metrics.volume_mean, metrics.max_drawdown)
print(paths[0].metadata.config_hash)
```

`GeneratedPath.metadata` stores `seed`, `config_hash`, package `version`,
`regime`, and `augmentation_strength` so synthetic runs can be traced. Pandas is
optional: install `market-wave[dataframe]` to use `to_dataframe()`.
`ValidationMetrics.volatility_clustering_score` is computed within each generated
path and aggregated, so independent path boundaries do not affect the diagnostic.

## Core Concepts

At every step, the market builds relative ticks around the current price:

```text
relative_tick = (price - current_price) / tick_size
relative_ticks = [-grid_radius, ..., 0, ..., +grid_radius]
```

The simulator maintains two Market Distribution Functions on that relative grid:

- `buy_entry_mdf`
- `sell_entry_mdf`

Each MDF is normalized. It is built from raw tick-level mass:

```text
raw_mass(tick) =
    near_market_continuity
  + book_shortage_or_front_mass
  + local_liquidity_pockets
  + flow_pressure_tail
  + microstructure_texture

proposal = Normalize(raw_mass)
raw_MDF_next = Normalize((1 - memory_mix) * proposal + memory_mix * raw_MDF_prev + floor)
resolved_buy_MDF, resolved_sell_MDF = ResolveOverlap(raw_buy_MDF_next, raw_sell_MDF_next)
MDF_next = Mix(resolved_MDF, CenteredNormalNoise(current_tick))
```

There is no custom score model, no temperature, and no score softmax path.
`mood`, `trend`, `volatility`, visible liquidity, shortage, recent flow,
and stress reshape the raw MDF directly. Market-adjacent ticks keep nonzero
mass, while shortage/front/liquidity signals can create multiple local pockets
away from the current price.

The public MDF fields are the effective distributions used for sampling.
The raw buy/sell judgments stay internal; a small memoryless normal component
around the current price is mixed into those two MDFs after buy/sell overlap
resolution, immediately before sampling. It is not a third public MDF or a
separate noise order path. Before order, cancel, re-quote, or replenishment
sampling, buy/sell overlap is resolved with the same local competition rule on
every relative tick, so the resolver does not branch on passive/marketable
regions or special-case the current tick.

Those relative MDFs are projected onto the pre-trade grid
`price_grid = price_before +/- k * gap` for order-book formation.
`StepInfo.mdf_price_basis` records that pre-trade price basis.

MDFs generate aggregate intent. Intensity controls total size. The order book and
execution layer then turn that intent into limit flow, taker flow, cancellations,
matched volume, and price changes.

## Execution Guarantee

Price movement is execution-driven:

- If a step has no executed volume, `price_after == price_before`.
- If trades execute, `price_after` is derived from that step's execution
  statistics. The flow term is bounded and cannot move the price by itself when
  executions print at the previous price.
- `seed` makes the simulation reproducible for the same version and inputs.

This is a simulator, not a market data replay engine and not financial advice.

## API Overview

```python
from market_wave import (
    Market,
    generate_paths,
    compute_metrics,
    MarketState,
    IntensityState,
    LatentState,
    MDFState,
    OrderBookState,
    StepInfo,
)
```

Useful `StepInfo` fields include:

- `price_before`, `price_after`, `price_change`
- `tick_before`, `tick_after`, `tick_change`, `relative_ticks`
- `mdf_price_basis`, `price_grid`
- `buy_entry_mdf`, `sell_entry_mdf`
- `buy_entry_mdf_by_price`, `sell_entry_mdf_by_price`
- `entry_volume_by_price`, `cancelled_volume_by_price`
- `buy_volume_by_price`, `sell_volume_by_price`
- `executed_volume_by_price`, `total_executed_volume`, `trade_count`
- `market_buy_volume`, `market_sell_volume`
- `vwap_price`, `best_bid_before`, `best_ask_before`, `spread_after`
- `orderbook_before`, `orderbook_after`

`buy_volume_by_price` and `sell_volume_by_price` are submitted side-intent maps
keyed by sampled order price, not executed or resting liquidity. `market_*`
volume fields report the executed incoming buy/sell volume. `residual_market_*`
fields report incoming buy/sell volume that did not execute and was restable in
the book. Unfilled incoming volume rests in `orderbook_after`; `crossed_market_volume`
is kept as a compatibility diagnostic and remains zero in the current
order-book-first engine.

The `*_mdf_by_price` fields are pre-trade MDF projections keyed by
`mdf_price_basis`; current `Market.state.mdf.*_by_price` is reprojected to the
post-trade state price. Examples and public APIs use MDF names only; stale PMF
examples from earlier prototypes should be considered obsolete.

### Public Contract and Snapshot Policy

The public import surface is the package `__all__`: `Market`, `generate_paths`,
`compute_metrics`, generated-path metadata, metrics, and the state dataclasses
shown above. Custom MDF model/protocol types are no longer public API. The
entrypoints are intentionally small, but the observation contract is broad
because `StepInfo` and `MarketState` expose detailed simulator diagnostics.

During the current alpha line, existing public names and existing `StepInfo` /
state fields are kept compatible where practical. New diagnostic fields may be
added in alpha releases. MDF names are the supported public distribution names;
stale PMF names from earlier prototypes are obsolete.

Snapshot mutability: state dataclasses are `frozen=True` at the attribute level,
but nested `dict` and `list` fields are plain mutable containers so `to_dict()`
and JSON export remain simple. Treat `Market.state`, `StepInfo`, and
`GeneratedPath.hidden_states` as read-only observations. Use `Market.snapshot()`
when downstream code needs a mutation-safe deep copy of the current state.

Compatibility note: `Market.state` remains available as the live current-state
attribute for the alpha line. Future releases may add a more explicit read-model
API or deprecation path for code that mutates state containers in place.

## Development

```bash
uv sync --extra dev --extra dataframe
uv run ruff check .
uv run pytest
uv build
```

## License

MIT
