# Market Wave Matching-Engine Rewrite Plan

## 0. 목적

합성 주식 시장을 만든다.

가격 경로를 직접 만들지 않는다.
보기 좋은 world를 사후 선별하지 않는다.
public corpus에는 hidden state, seed, theta, label, target, future column을 넣지 않는다.

핵심 생성 구조는 typed public message와 deterministic matching engine이다.
시장처럼 보이는지는 사후 보정이 아니라 주문 흐름, matching, 공개 시각화 계약에서 결정한다.

```pseudo
z_t, x_t
  -> observe_book(x_t)
  -> compute_event_field(z_t, obs_t, theta)
  -> sample_market_message(field_t, x_t, rng, theta)
  -> matching_engine.apply(message_t)
  -> public ordered events, trades, book deltas, x_{t+1}
  -> encode_feedback(...)
  -> update_latent_field(...)
```

## 1. 비협상 계약

```pseudo
CONTRACTS:
  RANDOMNESS_OWNERSHIP:
    runtime rng is only allowed in sample_market_message().
    matching, feedback, adapter, validator, visualization are deterministic.

  INTEGER_MARKET_UNITS:
    internal price is integer tick.
    internal quantity is integer unit.
    public JSON converts only at boundary to decimal string.

  PRICE_TIME_BOOK:
    internal book is price-time queues with order_id, side, price_tick,
    remaining_qty, created_seq.
    aggregate depth mutation is not the market model.

  MATCHING_BY_CONSTRUCTION:
    trades occur only when a marketable limit message hits resting liquidity.
    partial fill, FIFO priority, and sweep-through execution are deterministic.
    crossed/locked public book cannot be produced.

  AGGRESSIVE_PASSIVE_SPLIT:
    passive LIMIT flow and aggressive TAKE flow are different public messages.
    TAKE never rests on the book.
    LIMIT may rest, partially execute, and rest a non-crossing remainder.

  RECONSTRUCTABLE_PUBLIC_STREAM:
    public event stream is causally ordered.
    BOOK_DELTA records reconstruct snapshots from the initial full visible ladder.
    snapshots are derived views, not the source of book truth.

  HIDDEN_VISIBLE_SPLIT:
    public records never expose seed, theta_id, latent values, generator params,
    labels, targets, future returns, inventory, toxicity, or order
    internals not present in the public feed.
    ticker symbols are not part of the public market model.

  MARKET_SCREEN_VISUALIZATION:
    visual theme is white/cool-gray.
    accent palette is blue/red for bid/ask and buy/sell contrast.
    violet accents, warm-tone backgrounds, and dark terminal themes are not allowed.
    price chart primary signal is last traded price.
    chart includes best bid/ask spread, mid, trade candles, trade dots, and volume.
    top-10 orderbook heatmap uses fixed quote level on the y-axis and market
    time on x-axis.
    y-axis order is ask10..ask1, bid1..bid10.
    heatmap rows do not move vertically when prices change.

  MINIMAL_VALIDATION:
    validator checks schema, ordering, reconstructability, hidden leakage,
    decimal units, non-crossed book, and dead/exploded worlds.
    validator does not select pretty charts.
```

## 2. 자료구조

```pseudo
WorldTheta:
  id
  currency
  tick_size_decimal
  quantity_unit_decimal
  min_quantity_unit
  max_quantity_unit
  initial_mid_tick
  initial_book_levels
  initial_orders_per_level
  base_order_quantity
  depth_slope
  top_levels
  minimum_spread_ticks
  base_limit_weight
  base_take_weight
  base_cancel_weight
  base_replace_weight
  feedback_decay
  latent_bound

LatentState:
  liquidity
  volatility
  spread_pressure
  order_flow
  activity
  trend_pressure
  sweep_intensity
  cancel_pressure
  replenishment_pressure
  shock

MarketMessage:
  kind: LIMIT | TAKE | CANCEL | REPLACE
  side: BUY | SELL
  price_tick
  quantity
  order_id internal-only
  new_price_tick optional
  new_quantity optional

PublicRecord:
  event_id
  step_id
  world_id
  timestamp
  record_type:
    ORDER_ACCEPTED | TAKE_ORDER | ORDER_CANCELED | ORDER_REPLACED | TRADE | BOOK_DELTA
  payload
```

## 3. Matching Engine

```pseudo
LIMIT:
  if non-marketable:
    append to side queue
    emit ORDER_ACCEPTED
    emit BOOK_DELTA(+qty)

  if marketable:
    emit ORDER_ACCEPTED
    walk opposite price levels in executable order
    fill FIFO orders
    emit BOOK_DELTA(-fill_qty)
    emit TRADE per resting fill
    rest non-crossing remainder if any

TAKE:
  emit TAKE_ORDER
  walk opposite price levels in executable order
  fill FIFO orders
  emit BOOK_DELTA(-fill_qty)
  emit TRADE per resting fill
  never rest remaining quantity

CANCEL:
  only sampled for live internal orders
  reduce remaining_qty
  remove dead order from queue/index
  emit ORDER_CANCELED
  emit BOOK_DELTA(-cancel_qty)

REPLACE:
  only sampled for live internal orders
  remove old queue quantity
  rest new non-crossing order
  emit ORDER_REPLACED
  emit BOOK_DELTA(old -qty)
  emit BOOK_DELTA(new +qty)
```

## 4. Public Output

```pseudo
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
  top10_orderbook_heatmap.png
  event_tape.png
  mid_spread_depth.png
  depth_heatmap.png
```

Snapshots contain the full visible ladder required for reconstruction.
API views and plots slice top levels from snapshots.
Top-10 heatmaps must keep y-axis rows fixed as quote levels.

## 4.1 Public API

```pseudo
TOP_LEVEL_API:
  market_wave.generate(path, *, seed, steps, price, tick, force) -> Market
  market_wave.open(path) -> Market
  market_wave.Market

MARKET_METHODS:
  validate() -> ValidationReport
  plot(path=None) -> list[Path]
  price() -> dict
  book(levels=10) -> dict
  trades(limit=100) -> list[dict]
  candles(interval="1m") -> list[dict]
  events(limit=100, cursor=0) -> dict

RULES:
  top-level package exports only generate, open, and Market.
  Python methods return direct values, not {"result": ...} envelopes.
  CLI keeps JSON envelopes for scriptability.
  symbol/ticker arguments are not allowed.
```

## 5. Verification

```pseudo
TESTS:
  - only sample_market_message owns runtime rng
  - non-marketable limit rests without trade
  - marketable limit trades FIFO against resting liquidity
  - take order is public and never rests
  - cancel and replace conserve volume
  - public stream reconstructs final snapshot from initial snapshot + events
  - public corpus has no hidden/generator/future keys
  - replay is byte-identical for same seed/config
  - visualization reads visible corpus only
  - generated worlds have non-degenerate trade prices, mid movement, spread
    distribution, take flow, and fixed-level heatmap intensity movement
```
