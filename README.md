# market-wave

<p align="center">
  <img src="docs/assets/market-wave-hero.png" alt="market-wave abstract market intent simulation hero" />
</p>

<p align="center">
  <strong>Aggregate market intent simulation with discrete mixture distributions.</strong>
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

`market-wave` is a Python library for simulating market-wide entry and exit intent
without creating individual participants. It models aggregate buy/sell pressure,
position exits, order-book depth, cancellations, taker flow, and execution-driven
price movement on a discrete price grid.

It is not a forecasting model. It is a lightweight simulation primitive for
experiments, visualization, teaching, and strategy-environment prototyping.

## Why market-wave?

- **Aggregate intent, not agents**: market participants are represented by
  probability mass over price, not by individual objects.
- **Discrete mixture distributions**: entry and exit pressure are PMFs on the
  current price grid.
- **Execution-driven prices**: prices stay flat unless trades execute.
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

market = Market(initial_price=10_000, gap=10, popularity=1.0, seed=42)
steps = market.step(100)

last = steps[-1]
print(last.price_before, "->", last.price_after)
print(last.total_executed_volume)
```

`Market.step(n)` always returns `list[StepInfo]` and appends the same objects to
`market.history`.

## Visualization

```python
from market_wave import Market

market = Market(initial_price=10_000, gap=10, popularity=1.0, seed=42)
market.step(260)

fig, ax = market.plot(last=180)
```

<p align="center">
  <img src="docs/assets/market-wave-plot.png" alt="market-wave light pyplot chart showing price, volume, and imbalance" />
</p>

The default `market_wave` style uses a light three-panel chart: price/VWAP,
executed volume, and order-flow imbalance. Dark overlay mode is still available:

```python
fig, ax = market.plot(layout="overlay", style="market_wave_dark")
```

## Core Concepts

At every step, the market builds a price grid around the current price:

```text
price_grid = current_price +/- k * gap
```

The simulator maintains four probability mass functions on that grid:

- `buy_entry_pmf`
- `sell_entry_pmf`
- `long_exit_pmf`
- `short_exit_pmf`

Each PMF is a normalized discrete mixture:

```text
pmf[x] = sum(component_weight * kernel(x, center_price, spread))
kernel(x, center, spread) proportional to exp(-abs(x - center) / spread)
```

The PMFs generate aggregate intent. The order book and execution layer then turn
that intent into limit flow, taker flow, cancellations, exits, matched volume, and
price changes.

## Execution Guarantee

Price movement is execution-driven:

- If a step has no executed volume, `price_after == price_before`.
- If trades execute, `price_after` is derived from that step's execution
  statistics.
- `seed` makes the simulation reproducible for the same version and inputs.

This is a simulator, not a market data replay engine and not financial advice.

## API Overview

```python
from market_wave import (
    Market,
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
- `buy_entry_pmf`, `sell_entry_pmf`, `long_exit_pmf`, `short_exit_pmf`
- `buy_volume_by_price`, `sell_volume_by_price`
- `executed_volume_by_price`, `total_executed_volume`, `trade_count`
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
