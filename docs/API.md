# market-wave API

This document describes the public API for `market-wave` 0.5.x.

The package root is intentionally small:

```python
from market_wave import (
    Market,
    MarketState,
    IntensityState,
    LatentState,
    MDFState,
    OrderBookState,
    StepInfo,
)
```

Metrics helpers live in `market_wave.metrics`.

## Market

`Market` is the simulator entry point. Configure it with constructor
parameters, call `step(n)`, and inspect returned `StepInfo` records or the
current `state`.

```python
from market_wave import Market

market = Market(initial_price=10_000, gap=10, popularity=1.6, seed=23)
steps = market.step(500)
```

### Constructor Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `initial_price` | `100.0` | Positive starting price. It is snapped to the configured tick grid. |
| `gap` | `1.0` | Positive price distance for one simulator tick. |
| `popularity` | `1.0` | Non-negative activity scale. Higher values produce more submitted flow, trades, and book replenishment. `0` creates an inactive market. |
| `seed` | `None` | Optional random seed for reproducible paths. |
| `grid_radius` | `20` | Number of gap-unit offsets on each side of the MDF grid. |
| `augmentation_strength` | `0.0` | Non-negative extra texture/noise scale. |
| `regime` | `"normal"` | Initial market condition only. Supported values are `"normal"`, `"trend_up"`, `"trend_down"`, `"high_vol"`, `"thin_liquidity"`, `"squeeze"`, and `"auto"`. |

`regime` is not a fixed scenario. It seeds the initial condition; later
`StepInfo.regime` values are active condition labels produced by simulator state
transitions.

### Methods and Attributes

| API | Returns | Notes |
| --- | --- | --- |
| `market.step(n)` | `list[StepInfo]` | Advances the market by `n` steps and appends the same records to `market.history`. `n` must be non-negative. |
| `market.snapshot()` | `MarketState` | Deep copy of the current state. Use this when downstream code may mutate containers. |
| `market.history_records()` | `list[dict]` | JSON-friendly dictionaries for all records in `market.history`. |
| `market.plot(...)` | Matplotlib `(fig, ax)` | Built-in visualization for price, orderbook heatmaps, volume, and imbalance. Requires prior history. |
| `market.state` | `MarketState` | Live current-state object. Treat it as read-only. |
| `market.history` | `list[StepInfo]` | All records returned by prior `step()` calls. |
| `market.seed` | `int | None` | Configured random seed. |
| `market.tick_size` | `float` | Alias for `gap`. |

### Plotting

```python
market.step(500)
fig, ax = market.plot(last=220, orderbook_depth=12)
```

Useful options:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `last` | `None` | Plot only the most recent `last` steps. |
| `layout` | `"panel"` | `"panel"` shows price, book heatmaps, volume, and imbalance. `"overlay"` draws a compact price/volume/imbalance view. |
| `orderbook` | `None` | In panel layout, include orderbook heatmaps by default. Set `False` for the legacy three-panel view. |
| `orderbook_snapshot` | `"after"` | Use `"before"` or `"after"` book snapshots in heatmaps. |
| `orderbook_depth` | `None` | Limit heatmap depth to the nearest N book levels. |
| `style` | `"market_wave"` | Matplotlib style name. The default is a light `market-wave` chart. |

## Observation Records

### `StepInfo`

`StepInfo` is the main per-step observation returned by `Market.step()`.

Important field groups:

| Group | Fields |
| --- | --- |
| Price and ticks | `price_before`, `price_after`, `price_change`, `tick_before`, `tick_after`, `tick_change` |
| Condition state | `intensity`, `mood`, `trend`, `volatility`, `regime`, `augmentation_strength` |
| MDF state | `mdf_price_basis`, `price_grid`, `buy_entry_mdf`, `sell_entry_mdf` |
| Submitted flow | `buy_volume_by_price`, `sell_volume_by_price`, `entry_volume_by_price` |
| Cancellation | `cancelled_volume_by_price` |
| Execution | `executed_volume_by_price`, `total_executed_volume`, `market_buy_volume`, `market_sell_volume`, `trade_count`, `vwap_price` |
| Book and quote state | `best_bid_before`, `best_ask_before`, `best_bid_after`, `best_ask_after`, `spread_before`, `spread_after`, `mean_quote_age`, `orderbook_before`, `orderbook_after` |
| Diagnostics | `crossed_market_volume`, `residual_market_buy_volume`, `residual_market_sell_volume`, `order_flow_imbalance` |

`buy_entry_mdf` and `sell_entry_mdf` are the public MDF distributions used for
sampling. Their keys are gap-unit offsets, not absolute prices.

`buy_volume_by_price` and `sell_volume_by_price` are submitted side-intent maps
keyed by order price. They are not executed volume and not the final resting
book. Use `executed_volume_by_price` for fills and `orderbook_after` for
remaining visible depth.

Serialization helpers:

```python
record = steps[-1].to_dict()
line = steps[-1].to_json()
```

### State Dataclasses

| Dataclass | Meaning |
| --- | --- |
| `MarketState` | Current market observation: price, tick index, tick grid, latent state, MDFs, and orderbook. |
| `IntensityState` | Submitted-flow intensity split: total, buy, sell, buy ratio, sell ratio. |
| `LatentState` | Latent mood, trend, and volatility values. |
| `MDFState` | Buy and sell entry MDF snapshots. |
| `OrderBookState` | Aggregated bid/ask depth keyed by price. |

State dataclasses are `frozen=True` at the attribute level, but nested `dict`
and `list` fields are plain mutable containers for JSON-friendly workflows.
Treat `Market.state` and `StepInfo` as read-only observations. Use
`Market.snapshot()` when you need a mutation-safe copy of current state.

## Metrics API

Import metrics helpers from `market_wave.metrics`:

```python
from market_wave.metrics import (
    compute_metrics,
    compute_metrics_from_records,
    compare_metrics,
    load_reference_metrics_profile,
    load_metrics_profile,
    load_metrics_profile_json,
    save_metrics_profile_json,
    validate_reference_records,
)
```

### Compute Metrics

```python
from market_wave import Market
from market_wave.metrics import compute_metrics

paths = []
for seed in range(10):
    market = Market(initial_price=10_000, gap=10, popularity=1.2, seed=seed)
    paths.append(market.step(512))

metrics = compute_metrics(paths)
print(metrics.execution_rate, metrics.cancellation_rate)
```

`compute_metrics(paths)` accepts:

- one path as an iterable of `StepInfo`
- multiple paths as an iterable of step iterables
- objects with a `steps` attribute

It returns `ValidationMetrics`, a tick-native summary record with helpers
`to_dict()`, `to_json()`, and `to_dataframe()` when `market-wave[dataframe]` is
installed.

Key metric groups:

| Group | Example fields |
| --- | --- |
| Tick returns | `tick_return_mean`, `tick_return_std`, `tick_return_tail_ratio`, `nonzero_tick_change_rate` |
| Volume and execution | `mean_executed_volume`, `execution_rate`, `zero_volume_ratio`, `mean_trade_count` |
| Cancellation | `mean_cancelled_volume`, `cancellation_rate` |
| Position and anchor movement | `mean_abs_position_change_ticks`, `position_change_rate`, `mean_abs_mdf_anchor_change_ticks`, `mdf_anchor_change_rate` |
| Book topology | `mean_spread_ticks`, `mean_near_depth_share`, `mean_far_depth_share`, `one_sided_book_rate`, `mean_quote_age` |
| Path shape | `volatility_clustering_score`, `max_drawdown_ticks`, `max_runup_ticks`, `cumulative_tick_range` |

### Records and Reference Profiles

Use `compute_metrics_from_records()` when you have exported `StepInfo.to_dict()`
rows instead of live `StepInfo` objects.

```python
from market_wave.metrics import compute_metrics_from_records

metrics = compute_metrics_from_records(records, gap=10, path_id_field="path_id")
```

Reference records must include these fields:

- `tick_change`, `tick_before`, `tick_after`
- `total_executed_volume`, `cancelled_volume_by_price`, `trade_count`
- `order_flow_imbalance`, `mdf_price_basis`, `spread_after`
- `orderbook_after`, `price_after`

`orderbook_after` must contain `bid_volume_by_price` and
`ask_volume_by_price`.

Use `validate_reference_records(records)` to check records without computing
metrics.

### Compare and Save Profiles

```python
from market_wave.metrics import (
    compare_metrics,
    load_reference_metrics_profile,
    save_metrics_profile_json,
)

save_metrics_profile_json(metrics, "synthetic-metrics.json", name="synthetic")
reference = load_reference_metrics_profile("reference-records.jsonl", gap=10)
comparison = compare_metrics(metrics, reference)
print(comparison.score)
```

`compare_metrics()` returns `MetricsComparison` with:

- `score` / `total_distance`
- `rows`, one row per compared metric
- `field_distances`, `field_deltas`, and `field_weights`
- `to_dict()` and `to_json()`

The default comparison fields are listed in
`DEFAULT_COMPARISON_FIELDS`. For custom comparisons, pass `fields=[...]` or
`weights={...}`.

## Public Contract

The supported root import surface is `market_wave.__all__`: `Market` and the
state dataclasses listed above. Simulation advances through `Market.step()`.
Market behavior is configured through constructor parameters.

Names beginning with `_` and older prototype APIs such as `generate_paths`,
scenario builders, custom MDF protocols, or price-keyed PMF examples are not
public API in 0.5.x.
