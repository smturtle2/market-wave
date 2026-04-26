# market-wave

<p align="center">
  <img src="docs/assets/market-wave-hero.png" alt="market-wave abstract market intent simulation hero" />
</p>

<p align="center">
  <strong>Fast, lightweight synthetic market data from relative-tick intent PMFs.</strong>
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

`market-wave` is a Python library for generating synthetic market paths from
market-wide entry and exit intent. It does not create individual participants.
Instead, it models aggregate buy/sell pressure, position exits, order-book depth,
cancellations, taker flow, and execution-driven price movement from probability
mass over relative ticks.

It is not a forecasting model. It is a lightweight simulation primitive for
experiments, visualization, teaching, and strategy-environment prototyping.

## Why market-wave?

- **Aggregate intent, not agents**: market participants are represented by
  probability mass over relative ticks, not by individual objects.
- **Relative tick PMFs**: entry and exit pressure are modeled as
  `P(relative_tick)`, then projected onto the current price grid.
- **Pluggable distributions**: swap only the PMF generator with
  `LaplaceMixturePMF`, `SkewedPMF`, `FatTailPMF`, `NoisyPMF`, or a custom model.
- **Separated shape and size**: PMFs decide where intent sits; intensity decides
  how much order flow appears.
- **Execution-driven prices**: prices stay flat unless trades execute.
- **Batch generation**: generate many reproducible synthetic paths without
  keeping every path in `market.history`.
- **Inspectable state**: every step returns a `StepInfo` snapshot with PMFs,
  volumes, order book state, position mass, VWAP, spread, and imbalance.
- **Built-in plotting**: `matplotlib` is included, with a clean light chart style
  by default.

## Install

```bash
pip install market-wave
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
    regime="auto",
    augmentation_strength=0.25,
)
steps = market.step(500)

last = steps[-1]
print(last.price_before, "->", last.price_after)
print("executed:", round(last.total_executed_volume, 3))
print("imbalance:", round(last.order_flow_imbalance, 3))
print("crossed flow:", round(last.crossed_market_volume, 3))
print("residual flow:", round(last.residual_market_buy_volume, 3), round(last.residual_market_sell_volume, 3))
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
10030.0 -> 10030.0
executed: 1.03
imbalance: 0.054
crossed flow: 0.693
residual flow: 0.215 0.122
```

## Smoke Matrix

The simulator is deterministic for a fixed seed, so it is easy to run the same
invariants across different market conditions:

```python
from market_wave import Market

cases = [
    ("baseline", dict(initial_price=10_000, gap=10, popularity=1.0, seed=42, grid_radius=20), 500),
    ("busy", dict(initial_price=10_000, gap=10, popularity=2.5, seed=7, grid_radius=24), 500),
    ("thin", dict(initial_price=500, gap=5, popularity=0.25, seed=123, grid_radius=12), 500),
    ("low_price", dict(initial_price=1, gap=1, popularity=3.0, seed=17, grid_radius=8), 500),
    ("inactive", dict(initial_price=100, gap=1, popularity=0.0, seed=9, grid_radius=10), 100),
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
baseline   range= 9930.0-10010.0 moves=235 exec_steps=500 final= 9950.0
busy       range= 9920.0-10010.0 moves=248 exec_steps=500 final= 9950.0
thin       range=  470.0-500.0   moves=242 exec_steps=500 final=  475.0
low_price  range=    1.0-3.0     moves=236 exec_steps=500 final=    1.0
inactive   range=  100.0-100.0   moves=  0 exec_steps=  0 final=  100.0
```

Those runs also checked that current-state PMFs stay aligned with
`state.price_grid`, PMFs remain normalized, prices never fall below one tick,
order book and position mass stay non-negative, and price changes only occur on
steps with executed volume.

## Visualization

```python
from market_wave import Market

market = Market(initial_price=10_000, gap=10, popularity=1.0, seed=42)
market.step(260)

fig, ax = market.plot(last=180)
```

<p align="center">
  <img src="docs/assets/market-wave-plot.png" alt="market-wave light pyplot chart showing price, orderbook depth heatmaps, volume, and imbalance" />
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
        "regime": "auto",
        "augmentation_strength": 0.35,
    },
)

metrics = compute_metrics(paths)
print(metrics.return_std, metrics.volume_mean, metrics.max_drawdown)
print(paths[0].metadata.config_hash)
```

`GeneratedPath.metadata` stores `seed`, `config_hash`, package `version`,
`regime`, and `augmentation_strength` so synthetic runs can be traced. Pandas is
optional: install `market-wave[dataframe]` to use `to_dataframe()`.

## Pluggable PMFs

```python
from market_wave import FatTailPMF, Market, NoisyPMF

market = Market(
    initial_price=100,
    gap=1,
    distribution_model=NoisyPMF(FatTailPMF()),
    augmentation_strength=0.5,
    seed=7,
)

step = market.step(1)[0]
print(step.relative_ticks)
print(step.buy_entry_pmf_by_tick)
```

Custom models only need one method:

```python
class MyPMF:
    def pmf(self, side, intent, relative_ticks, context):
        weights = [1.0 / (1.0 + abs(tick)) for tick in relative_ticks]
        total = sum(weights)
        return [weight / total for weight in weights]
```

## Core Concepts

At every step, the market builds relative ticks around the current price:

```text
relative_tick = (price - current_price) / tick_size
relative_ticks = [-grid_radius, ..., 0, ..., +grid_radius]
```

The simulator maintains four probability mass functions on that relative grid:

- `buy_entry_pmf`
- `sell_entry_pmf`
- `long_exit_pmf`
- `short_exit_pmf`

Each PMF is a normalized discrete mixture:

```text
pmf[tick] = sum(component_weight * kernel(tick, center_tick, spread_ticks))
kernel(tick, center, spread) proportional to exp(-abs(tick - center) / spread)
```

Those relative PMFs are projected onto `price_grid = current_price +/- k * gap`
for order-book formation. `pmf_inertia` keeps intent from jumping abruptly:

```text
pmf_t = pmf_inertia * pmf_prev + (1 - pmf_inertia) * pmf_new
```

PMFs generate aggregate intent. Intensity controls total size. The order book and
execution layer then turn that intent into limit flow, taker flow, cancellations,
exits, matched volume, and price changes.

## Execution Guarantee

Price movement is execution-driven:

- If a step has no executed volume, `price_after == price_before`.
- If trades execute, `price_after` is derived from that step's execution
  statistics. Random quote jitter is bounded and cannot move the price by itself
  when executions print at the previous price.
- `seed` makes the simulation reproducible for the same version and inputs.

This is a simulator, not a market data replay engine and not financial advice.

## API Overview

```python
from market_wave import (
    Market,
    LaplaceMixturePMF,
    SkewedPMF,
    FatTailPMF,
    NoisyPMF,
    generate_paths,
    compute_metrics,
    MarketState,
    IntensityState,
    LatentState,
    MixtureComponent,
    DiscreteMixtureDistribution,
    DistributionState,
    OrderBookState,
    PositionMassState,
    StepInfo,
)
```

Useful `StepInfo` fields include:

- `price_before`, `price_after`, `price_change`
- `tick_before`, `tick_after`, `tick_change`, `relative_ticks`
- `buy_entry_pmf`, `sell_entry_pmf`, `long_exit_pmf`, `short_exit_pmf`
- `buy_entry_pmf_by_tick`, `sell_entry_pmf_by_tick`
- `buy_volume_by_price`, `sell_volume_by_price`
- `executed_volume_by_price`, `total_executed_volume`, `trade_count`
- `market_buy_volume`, `market_sell_volume`, `crossed_market_volume`
- `residual_market_buy_volume`, `residual_market_sell_volume`
- `vwap_price`, `best_bid_before`, `best_ask_before`, `spread_after`
- `orderbook_before`, `orderbook_after`
- `position_mass_before`, `position_mass_after`

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
uv build
```

## License

MIT
