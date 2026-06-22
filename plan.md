# Latent-Field Synthetic Market Simulator 전체 계획서

## 0. 목표

합성 주식 시장을 만든다.  
가격을 직접 만들지 않는다.  
주문 타입을 먼저 만들지 않는다.  
시장을 인간 개념으로 잘라서 만들지 않는다.

핵심 구조는 하나다.

```text
시장 = 연속 잠재장 z
     + 이벤트 확률장 λ
     + 주문장 물리엔진 M
     + 피드백 업데이트 G
     + 토스증권 API 느낌의 출력 어댑터
```

내부 생성 원리는 추상적이고 연속적이어야 한다.  
외부 출력은 실제 증권 API처럼 단정해야 한다.

---

## 1. 핵심 모델

```pseudo
CORE_MODEL:
    z_t     := hidden continuous latent field
    x_t     := visible order book state
    λ_t     := event probability field
    e_t     := sampled liquidity impulse
    M       := deterministic order book physics
    G       := feedback update

    λ_t       = F(z_t, observe(x_t))
    e_t       ~ sample(λ_t)
    x_{t+1}   = M(x_t, e_t)
    z_{t+1}   = G(z_t, x_t, e_t, x_{t+1})

FINAL_LOOP:
    hidden field bends event field
    event field samples liquidity impulse
    book physics converts impulse into depth change / trades / price movement
    book deformation bends hidden field again
```

가격은 결과다.  
주문도 결과적 해석이다.  
생성 단위는 `liquidity impulse`다.

---

## 2. 설계 규칙

```pseudo
DESIGN_RULES:
    1. price_{t+1} = price_t + random_return 금지
    2. market maker / retail / panic / news 같은 인간 개념을 기본 단위로 쓰지 않음
    3. z_t는 이름 없는 연속 벡터
    4. 분포를 여러 개 만들지 말고 하나의 event field가 계속 변형되게 함
    5. limit / market / cancel은 생성 타입이 아니라 결과적 해석
    6. 주문장 반영은 결정론적 물리엔진으로 처리
    7. 확률은 event field 샘플링에서만 발생
    8. 이산화는 거래소 물리상 필요한 것만 허용
        - tick
        - side
        - queue/event order
        - price level
    9. AI 입력에는 visible data만 사용
    10. hidden data는 검증, 라벨링, 분석용으로만 보관
```

---

## 3. 전체 모듈 구조

```pseudo
MODULES:
    1. WorldParameterSampler
        - market world 하나의 수치 조건을 만든다.
        - tick_size, initial_price, latent_dim, depth_levels, field_weights 등을 샘플링한다.
        - 인간식 레짐 이름은 만들지 않는다.

    2. LatentField
        - z ∈ R^K
        - 이름 없는 연속 잠재장
        - drift + noise + jump + feedback으로 움직인다.

    3. EventField
        - z와 visible book state를 받아 λ를 만든다.
        - λ는 다음 liquidity impulse의 확률장이다.

    4. LiquidityImpulseSampler
        - λ에서 impulse 하나를 샘플링한다.
        - impulse는 유동성장의 국소 변형이다.

    5. BookPhysics
        - impulse를 depth 추가 / depth 제거 / 반대편 depth 소비로 연속 분해한다.
        - 체결, spread 변화, depth 변화, mid 변화는 여기서만 발생한다.

    6. FeedbackEncoder
        - book deformation과 trades를 벡터로 압축한다.
        - 이 벡터가 z를 다시 움직인다.

    7. OutputAdapter
        - 내부 상태를 토스증권 API 느낌의 JSON으로 변환한다.
        - public output과 hidden debug output을 분리한다.

    8. DatasetBuilder
        - visible input / visible target / hidden analysis data를 분리 저장한다.

    9. MarketValidator
        - 죽은 시장, 폭주 시장, 너무 인공적인 시장을 버린다.

    10. MarketFamilyGenerator
        - 하나의 긴 시장이 아니라 여러 market world를 생성한다.
```

---

## 4. 내부 데이터 구조

```pseudo
struct WorldTheta:
    id
    symbol
    currency
    initial_price
    tick_size
    levels
    latent_dim
    dt
    A                      # latent drift matrix
    B                      # feedback projection matrix
    W                      # event field nonlinear map weights
    noise_scale
    jump_rate
    jump_scale
    max_latent_norm
    min_quantity
    boundary_depth_strength

struct LatentField:
    z[K]

struct BookState:
    prices[L]
    bid_depth[L]
    ask_depth[L]
    tick_size
    last_trade_price
    last_trade_timestamp
    recent_trades
    recent_deformations

struct EventField:
    side_bias
    price_location
    price_scale
    size_scale
    liquidity_delta_center
    liquidity_delta_scale
    sharpness
    persistence
    tail_weight

struct LiquidityImpulse:
    side                  # +1 BUY pressure, -1 SELL pressure
    price_coordinate      # mid 기준 상대 좌표
    size                  # 유동성 충격 크기
    liquidity_delta       # +면 추가 성향, -면 제거/소비 성향
    sharpness             # 국소성
    persistence           # 흔적

struct Trade:
    price
    quantity
    timestamp
    aggressor_side

struct Feedback:
    vector[D]

struct PublicEvent:
    event_id
    timestamp
    symbol
    event_type
    side
    price
    quantity
    currency
```

---

## 5. 구현 단계

```pseudo
IMPLEMENTATION_PLAN:

    PHASE_1_ORDER_BOOK:
        depth-grid order book 구현
        개별 주문 ID 없는 L2 depth 배열로 시작

        functions:
            init_book
            best_bid
            best_ask
            mid
            spread
            depth_vector
            add_liquidity
            remove_local_liquidity
            consume_opposite_liquidity
            enforce_boundary_conditions

    PHASE_2_LATENT_FIELD:
        z ∈ R^K 구현
        z에 sentiment, volatility, panic 같은 이름 붙이지 않음

        update source:
            drift
            heavy_tail_noise
            occasional_jump
            feedback_deformation

    PHASE_3_EVENT_FIELD:
        observe(book) + z → event field λ
        λ는 impulse 샘플링을 위한 연속 확률장

    PHASE_4_IMPULSE_SAMPLING:
        limit / market / cancel을 뽑지 않음
        liquidity impulse 하나를 뽑음

    PHASE_5_BOOK_PHYSICS:
        impulse를 주문장에 반영
        add/remove/take 비율은 연속적으로 계산
        체결과 가격 변화는 여기서만 발생

    PHASE_6_FEEDBACK:
        old_book → new_book 변형을 벡터로 압축
        z 업데이트에 반영

    PHASE_7_OUTPUT_ADAPTER:
        내부 상태를 토스증권 API 느낌의 JSON으로 변환
        public endpoint와 sim/debug endpoint 분리

    PHASE_8_DATASET:
        visible input과 future visible target 생성
        hidden state는 별도 분석용 저장

    PHASE_9_VALIDATION:
        freeze / explosion / dead spread / dead depth / no clustering 제거

    PHASE_10_MARKET_FAMILY:
        seed와 theta family를 나눠 여러 시장 세계 생성
        train/valid/test는 시간 분할이 아니라 world 분할
```

---

## 6. 주문장 구현 의사코드

```pseudo
function init_book(mid, tick_size, levels, theta):
    book.prices = grid_around(mid, tick_size, levels)
    book.bid_depth = smooth_initial_depth_curve(levels, side=BUY, theta)
    book.ask_depth = smooth_initial_depth_curve(levels, side=SELL, theta)
    book.tick_size = tick_size
    book.last_trade_price = null
    book.last_trade_timestamp = null
    book.recent_trades = RingBuffer()
    book.recent_deformations = RingBuffer()
    return book

function best_bid(book):
    for price from high to low:
        if bid_depth_at(book, price) > 0:
            return price
    return null

function best_ask(book):
    for price from low to high:
        if ask_depth_at(book, price) > 0:
            return price
    return null

function mid(book):
    bid = best_bid(book)
    ask = best_ask(book)

    if bid exists and ask exists:
        return (bid + ask) / 2

    if book.last_trade_price exists:
        return book.last_trade_price

    return center_of_price_grid(book)

function spread(book):
    bid = best_bid(book)
    ask = best_ask(book)

    if bid exists and ask exists:
        return ask - bid

    return null

function local_price_scale(book):
    s = spread(book)

    if s exists and s > 0:
        return max(s, book.tick_size)

    return book.tick_size
```

---

## 7. 잠재장 구현 의사코드

```pseudo
function init_latent_field(K, rng):
    z = random_vector(K, rng)
    z = normalize(z)
    return z

function update_latent_field(z, feedback, theta, rng):
    drift = theta.A @ z * theta.dt

    noise = heavy_tail_noise(dim=len(z), rng=rng)
    noise = noise * theta.noise_scale

    if random_uniform(rng) < theta.jump_rate * theta.dt:
        jump = heavy_tail_noise(dim=len(z), rng=rng) * theta.jump_scale
    else:
        jump = zero_vector(len(z))

    deformation = theta.B @ feedback.vector

    z_next = z + drift + noise + jump + deformation
    z_next = bound_norm(z_next, theta.max_latent_norm)

    return z_next
```

---

## 8. 관측값 추출 의사코드

```pseudo
function observe_book(book):
    obs.mid = mid(book)
    obs.spread = spread(book)
    obs.depth_curve = concat(book.bid_depth, book.ask_depth)
    obs.imbalance = depth_imbalance(book)
    obs.local_slope = depth_slope(book)
    obs.recent_deformation = summarize(book.recent_deformations)
    obs.recent_trades = summarize(book.recent_trades)
    return obs

function depth_imbalance(book):
    bid = sum(book.bid_depth)
    ask = sum(book.ask_depth)
    return (bid - ask) / max(bid + ask, epsilon)

function depth_slope(book):
    return concat(
        finite_difference(book.bid_depth),
        finite_difference(book.ask_depth)
    )
```

---

## 9. 이벤트 확률장 구현 의사코드

```pseudo
function compute_event_field(z, obs, theta):
    h = concat(
        z,
        obs.spread,
        obs.depth_curve,
        obs.imbalance,
        obs.local_slope,
        obs.recent_deformation,
        obs.recent_trades
    )

    raw = smooth_nonlinear_map(h, theta.W)

    field.side_bias              = raw[0]
    field.price_location         = raw[1]
    field.price_scale            = positive(raw[2])
    field.size_scale             = positive(raw[3])
    field.liquidity_delta_center = raw[4]
    field.liquidity_delta_scale  = positive(raw[5])
    field.sharpness              = positive(raw[6])
    field.persistence            = positive(raw[7])
    field.tail_weight            = positive(raw[8])

    return field

function smooth_nonlinear_map(h, W):
    # 신경망이어도 되고 random Fourier features + low-rank map이어도 됨.
    # 중요한 건 인간 개념으로 분해하지 않는 것.
    u = nonlinear_features(h)
    raw = W @ u
    return raw
```

---

## 10. Liquidity impulse 샘플링 의사코드

```pseudo
function sample_impulse(field, rng):
    side = sample_sign(sigmoid(field.side_bias), rng)

    price_coordinate = sample_heavy_tail(
        center = field.price_location,
        scale  = field.price_scale,
        tail   = field.tail_weight,
        rng    = rng
    )

    size = sample_positive_heavy_tail(
        scale = field.size_scale,
        tail  = field.tail_weight,
        rng   = rng
    )

    liquidity_delta = sample_heavy_tail(
        center = field.liquidity_delta_center,
        scale  = field.liquidity_delta_scale,
        tail   = field.tail_weight,
        rng    = rng
    )

    impulse = LiquidityImpulse(
        side = side,
        price_coordinate = price_coordinate,
        size = size,
        liquidity_delta = liquidity_delta,
        sharpness = field.sharpness,
        persistence = field.persistence
    )

    return impulse
```

---

## 11. 주문장 물리엔진 의사코드

```pseudo
function apply_book_physics(book, impulse, theta, timestamp):
    trades = []

    current_mid = mid(book)
    scale = local_price_scale(book)

    raw_price = current_mid + impulse.side * impulse.price_coordinate * scale
    price = snap_to_tick(raw_price, book.tick_size)

    cross_score = compute_cross_score(book, impulse.side, price)

    add_weight    = sigmoid(+impulse.liquidity_delta)
    remove_weight = sigmoid(-impulse.liquidity_delta) * sigmoid(-cross_score)
    take_weight   = sigmoid(-impulse.liquidity_delta) * sigmoid(+cross_score)

    q_add    = impulse.size * add_weight
    q_remove = impulse.size * remove_weight
    q_take   = impulse.size * take_weight

    if q_add > theta.min_quantity:
        book = add_liquidity(
            book,
            side = impulse.side,
            price = price,
            quantity = q_add,
            sharpness = impulse.sharpness
        )

    if q_remove > theta.min_quantity:
        book = remove_local_liquidity(
            book,
            side = impulse.side,
            price = price,
            quantity = q_remove,
            sharpness = impulse.sharpness
        )

    if q_take > theta.min_quantity:
        book, new_trades = consume_opposite_liquidity(
            book,
            side = impulse.side,
            quantity = q_take,
            limit_price = price,
            timestamp = timestamp
        )
        trades.extend(new_trades)

    book = enforce_boundary_conditions(book, theta)

    for trade in trades:
        book.last_trade_price = trade.price
        book.last_trade_timestamp = trade.timestamp
        book.recent_trades.append(trade)

    return book, trades

function add_liquidity(book, side, price, quantity, sharpness):
    idx = price_to_index(book, price)
    kernel = local_kernel(center=idx, sharpness=sharpness, length=len(book.prices))

    if side == +1:
        book.bid_depth += quantity * kernel
    else:
        book.ask_depth += quantity * kernel

    return book

function remove_local_liquidity(book, side, price, quantity, sharpness):
    idx = price_to_index(book, price)
    kernel = local_kernel(center=idx, sharpness=sharpness, length=len(book.prices))

    if side == +1:
        book.bid_depth -= quantity * kernel
        book.bid_depth = max(book.bid_depth, 0)
    else:
        book.ask_depth -= quantity * kernel
        book.ask_depth = max(book.ask_depth, 0)

    return book

function consume_opposite_liquidity(book, side, quantity, limit_price, timestamp):
    trades = []

    if side == +1:
        # buy pressure: ask depth를 best ask부터 소비
        while quantity > 0 and best_ask(book) <= limit_price:
            p = best_ask(book)
            q = min(quantity, ask_depth_at(book, p))

            reduce_ask_depth(book, p, q)
            trades.append(Trade(
                price = p,
                quantity = q,
                timestamp = timestamp,
                aggressor_side = +1
            ))

            quantity -= q

    else:
        # sell pressure: bid depth를 best bid부터 소비
        while quantity > 0 and best_bid(book) >= limit_price:
            p = best_bid(book)
            q = min(quantity, bid_depth_at(book, p))

            reduce_bid_depth(book, p, q)
            trades.append(Trade(
                price = p,
                quantity = q,
                timestamp = timestamp,
                aggressor_side = -1
            ))

            quantity -= q

    return book, trades
```

---

## 12. 피드백 구현 의사코드

```pseudo
function encode_feedback(old_book, new_book, impulse, trades):
    old_mid = mid(old_book)
    new_mid = mid(new_book)

    old_spread = spread(old_book)
    new_spread = spread(new_book)

    old_depth = depth_vector(old_book)
    new_depth = depth_vector(new_book)

    trade_quantity = sum(trade.quantity for trade in trades)
    signed_trade_quantity = sum(trade.aggressor_side * trade.quantity for trade in trades)

    feedback_vector = concat(
        new_mid - old_mid,
        abs(new_mid - old_mid),
        nullsafe_delta(new_spread, old_spread),
        norm(new_depth - old_depth),
        signed_depth_change(old_book, new_book),
        trade_quantity,
        signed_trade_quantity,
        impulse.side * impulse.size,
        impulse.liquidity_delta,
        impulse.persistence
    )

    feedback_vector = standardize(feedback_vector)

    return Feedback(vector = feedback_vector)
```

---

## 13. 전체 시장 루프 의사코드

```pseudo
function run_market(seed, steps):
    rng = Random(seed)

    theta = sample_world_parameters(rng)

    z = init_latent_field(theta.latent_dim, rng)

    book = init_book(
        mid = theta.initial_price,
        tick_size = theta.tick_size,
        levels = theta.levels,
        theta = theta
    )

    public_event_buffer = RingBuffer()
    trade_buffer = RingBuffer()
    snapshot_buffer = RingBuffer()
    hidden_buffer = RingBuffer()
    dataset = []

    for t in range(steps):
        timestamp = simulation_time_to_kst(t, theta)
        old_book = copy(book)

        obs = observe_book(book)
        field = compute_event_field(z, obs, theta)
        impulse = sample_impulse(field, rng)

        book, trades = apply_book_physics(
            book = book,
            impulse = impulse,
            theta = theta,
            timestamp = timestamp
        )

        feedback = encode_feedback(
            old_book = old_book,
            new_book = book,
            impulse = impulse,
            trades = trades
        )

        z = update_latent_field(z, feedback, theta, rng)

        book_delta = diff_books(old_book, book)

        public_events = to_public_events(
            impulse = impulse,
            book_delta = book_delta,
            trades = trades,
            symbol = theta.symbol,
            currency = theta.currency,
            timestamp = timestamp
        )

        trade_buffer.append_all(trades)
        public_event_buffer.append_all(public_events)

        public_snapshot = to_snapshot_payload(
            book = book,
            symbol = theta.symbol,
            currency = theta.currency,
            timestamp = timestamp,
            depth = 10
        )
        snapshot_buffer.append(public_snapshot)

        hidden_record = {
            z: z,
            event_field: summarize(field),
            raw_impulse: impulse,
            feedback: feedback,
            theta_id: theta.id,
            seed: seed
        }
        hidden_buffer.append(hidden_record)

        dataset.append({
            visible: {
                t: t,
                timestamp: timestamp,
                symbol: theta.symbol,
                book: compress_book(book),
                trades: trades,
                public_events: public_events,
                mid: mid(book),
                spread: spread(book)
            },
            hidden: hidden_record
        })

    return MarketRun(
        theta = theta,
        dataset = dataset,
        book = book,
        trades = trade_buffer,
        events = public_event_buffer,
        snapshots = snapshot_buffer,
        hidden = hidden_buffer
    )
```

---

## 14. 토스증권 API 느낌의 출력 정책

외부 출력은 내부 생성 구조를 숨긴다.  
public market-data 출력은 `result` envelope를 사용한다.  
실패 출력은 `error` envelope를 사용한다.  
가격, 수량, 금액, 비율은 모두 decimal string으로 출력한다.  
timestamp는 KST offset 포함 ISO-8601 문자열로 출력한다.

```pseudo
OUTPUT_POLICY:
    success:
        { "result": T }

    failure:
        {
            "error": {
                "request_id": string | null,
                "code": string,
                "message": string,
                "data": object | null
            }
        }

    decimal:
        price, volume, quantity, amount, rate는 JSON number가 아니라 string

    timestamp:
        ISO-8601 / RFC3339 with +09:00
        example: "2026-06-22T09:01:03.124+09:00"

    public_hidden_split:
        public API에는 z, event_field, seed, raw_impulse를 노출하지 않는다.
        hidden/debug API에서만 노출한다.
```

---

## 15. Public API surface

```pseudo
PUBLIC_MARKET_DATA_ENDPOINTS:
    GET /v1/market/price
        현재가

    GET /v1/market/orderbook
        호가창

    GET /v1/market/trades
        최근 체결

    GET /v1/market/candles
        캔들

SYNTHETIC_EXTENSION_ENDPOINTS:
    GET /v1/sim/events
        합성 이벤트 스트림

    GET /v1/sim/snapshots
        학습용 order book snapshot stream

    GET /v1/sim/export
        AI 학습용 batch export

HIDDEN_DEBUG_ENDPOINTS:
    GET /v1/sim/debug
        hidden state 확인용
        기본 비활성
```

---

## 16. 공통 타입

```pseudo
type DecimalString:
    string decimal
    examples:
        "71800"
        "104.391"
        "0.128391"

type KstTimestamp:
    string ISO-8601 with +09:00 offset
    example:
        "2026-06-22T09:01:03.124+09:00"

type Symbol:
    string
    examples:
        "005930"
        "AAPL"

type Currency:
    "KRW" | "USD"

type SideName:
    "BUY" | "SELL"

type EventType:
    "BOOK_UPDATE" | "TRADE" | "PRICE_UPDATE" | "SNAPSHOT"
```

---

## 17. 현재가 출력 형태

```json
{
  "result": {
    "symbol": "005930",
    "timestamp": "2026-06-22T09:01:03.124+09:00",
    "last_price": "71800",
    "currency": "KRW"
  }
}
```

```pseudo
function to_price_response(book, symbol, currency, timestamp):
    if book.last_trade_price exists:
        last_price = book.last_trade_price
        ts = book.last_trade_timestamp
    else:
        last_price = mid(book)
        ts = timestamp

    return {
        "result": {
            "symbol": symbol,
            "timestamp": format_kst_or_null(ts),
            "last_price": dec(last_price),
            "currency": currency
        }
    }
```

---

## 18. 호가창 출력 형태

```json
{
  "result": {
    "timestamp": "2026-06-22T09:01:03.124+09:00",
    "currency": "KRW",
    "asks": [
      { "price": "71900", "volume": "820" },
      { "price": "72000", "volume": "1260" },
      { "price": "72100", "volume": "1910" }
    ],
    "bids": [
      { "price": "71800", "volume": "1040" },
      { "price": "71700", "volume": "1440" },
      { "price": "71600", "volume": "2130" }
    ]
  }
}
```

```pseudo
ORDERBOOK_RULES:
    asks는 낮은 가격부터 높은 가격 순서
    bids는 높은 가격부터 낮은 가격 순서
    각 level은 price, volume만 둔다
    내부 depth-grid가 200단이어도 외부 응답은 top N단만 자른다

function to_orderbook_response(book, currency, timestamp, depth_limit):
    asks = []
    bids = []

    for level in ask_levels_sorted_low_to_high(book):
        if level.volume > 0:
            asks.append({
                "price": dec(level.price),
                "volume": dec(level.volume)
            })
        if len(asks) == depth_limit:
            break

    for level in bid_levels_sorted_high_to_low(book):
        if level.volume > 0:
            bids.append({
                "price": dec(level.price),
                "volume": dec(level.volume)
            })
        if len(bids) == depth_limit:
            break

    return {
        "result": {
            "timestamp": format_kst_or_null(timestamp),
            "currency": currency,
            "asks": asks,
            "bids": bids
        }
    }
```

---

## 19. 최근 체결 출력 형태

```json
{
  "result": {
    "trades": [
      {
        "price": "71800",
        "volume": "50",
        "timestamp": "2026-06-22T09:01:03.124+09:00",
        "currency": "KRW"
      },
      {
        "price": "71800",
        "volume": "120",
        "timestamp": "2026-06-22T09:01:02.982+09:00",
        "currency": "KRW"
      }
    ]
  }
}
```

```pseudo
TRADES_RULES:
    public output에서는 aggressor_side를 기본 노출하지 않는다.
    학습용 export에서는 필요하면 side를 포함할 수 있다.

function to_trades_response(trade_buffer, currency, limit):
    rows = []

    for trade in latest(trade_buffer, limit):
        rows.append({
            "price": dec(trade.price),
            "volume": dec(trade.quantity),
            "timestamp": format_kst(trade.timestamp),
            "currency": currency
        })

    return {
        "result": {
            "trades": rows
        }
    }
```

---

## 20. 캔들 출력 형태

토스증권 모델 느낌에 맞춰 `open_price`, `high_price`, `low_price`, `close_price`를 쓴다.

```json
{
  "result": {
    "candles": [
      {
        "timestamp": "2026-06-22T09:01:00.000+09:00",
        "open_price": "71700",
        "high_price": "71900",
        "low_price": "71600",
        "close_price": "71800",
        "volume": "18420",
        "currency": "KRW"
      }
    ],
    "next_before": "2026-06-22T09:00:00.000+09:00"
  }
}
```

```pseudo
function build_candles(trades, interval):
    buckets = group_trades_by_time_bucket(trades, interval)
    candles = []

    for bucket in buckets:
        prices = [t.price for t in bucket.trades]
        volumes = [t.quantity for t in bucket.trades]

        if empty(prices):
            continue

        candles.append({
            "timestamp": format_kst(bucket.start_time),
            "open_price": dec(first(prices)),
            "high_price": dec(max(prices)),
            "low_price": dec(min(prices)),
            "close_price": dec(last(prices)),
            "volume": dec(sum(volumes)),
            "currency": bucket.currency
        })

    return candles

function to_candles_response(trades, interval, limit, before):
    candles = build_candles(trades, interval)
    page = page_candles_descending_time(candles, limit, before)

    return {
        "result": {
            "candles": page.items,
            "next_before": format_kst_or_null(page.next_before)
        }
    }
```

---

## 21. 합성 이벤트 출력 형태

실제 증권 API보다 학습에 유리하게 확장한 endpoint다.  
public market API와 hidden debug 사이에 위치한다.  
내부 raw impulse는 숨기고, 관측 가능한 이벤트로 바꾼다.

```json
{
  "result": {
    "events": [
      {
        "event_id": "evt_000000000913",
        "timestamp": "2026-06-22T09:01:03.124+09:00",
        "symbol": "005930",
        "event_type": "BOOK_UPDATE",
        "side": "BUY",
        "price": "71800",
        "quantity": "104",
        "currency": "KRW"
      },
      {
        "event_id": "evt_000000000914",
        "timestamp": "2026-06-22T09:01:03.129+09:00",
        "symbol": "005930",
        "event_type": "TRADE",
        "side": "BUY",
        "price": "71900",
        "quantity": "25",
        "currency": "KRW"
      }
    ],
    "next_cursor": "sim_event_cursor_000000000914"
  }
}
```

```pseudo
function to_public_events(impulse, book_delta, trades, symbol, currency, timestamp):
    events = []

    for delta in material_book_deltas(book_delta):
        events.append({
            "event_id": make_event_id(),
            "timestamp": format_kst(timestamp),
            "symbol": symbol,
            "event_type": "BOOK_UPDATE",
            "side": side_name(delta.side),
            "price": dec(delta.price),
            "quantity": dec(abs(delta.quantity)),
            "currency": currency
        })

    for trade in trades:
        events.append({
            "event_id": make_event_id(),
            "timestamp": format_kst(trade.timestamp),
            "symbol": symbol,
            "event_type": "TRADE",
            "side": side_name(trade.aggressor_side),
            "price": dec(trade.price),
            "quantity": dec(trade.quantity),
            "currency": currency
        })

    return events
```

---

## 22. Snapshot 출력 형태

```json
{
  "result": {
    "snapshot_id": "snap_000000012800",
    "timestamp": "2026-06-22T09:01:03.124+09:00",
    "symbol": "005930",
    "price": {
      "last_price": "71800",
      "mid_price": "71850",
      "currency": "KRW"
    },
    "orderbook": {
      "asks": [
        { "price": "71900", "volume": "820" }
      ],
      "bids": [
        { "price": "71800", "volume": "1040" }
      ]
    }
  }
}
```

```pseudo
function to_snapshot_payload(book, symbol, currency, timestamp, depth):
    return {
        "snapshot_id": make_snapshot_id(),
        "timestamp": format_kst(timestamp),
        "symbol": symbol,
        "price": {
            "last_price": dec(last_price_or_mid(book)),
            "mid_price": dec(mid(book)),
            "currency": currency
        },
        "orderbook": {
            "asks": top_asks(book, depth),
            "bids": top_bids(book, depth)
        }
    }

function to_snapshot_response(book, symbol, currency, timestamp, depth):
    return {
        "result": to_snapshot_payload(book, symbol, currency, timestamp, depth)
    }
```

---

## 23. AI 학습용 export 출력 형태

학습용 export는 JSONL 기본.  
input에는 visible만 둔다.  
target에는 미래 visible 결과만 둔다.  
hidden은 별도 파일로 분리한다.

```json
{
  "t": "12800",
  "timestamp": "2026-06-22T09:01:03.124+09:00",
  "symbol": "005930",
  "input": {
    "orderbook": {
      "asks": [
        { "price": "71900", "volume": "820" }
      ],
      "bids": [
        { "price": "71800", "volume": "1040" }
      ]
    },
    "trades": [
      {
        "price": "71800",
        "volume": "50",
        "timestamp": "2026-06-22T09:01:02.982+09:00",
        "currency": "KRW"
      }
    ]
  },
  "target": {
    "future_mid_path": ["71850", "71850", "71900", "71950"],
    "future_trade_flow": ["50", "0", "210", "80"],
    "future_spread_path": ["100", "100", "200", "100"]
  }
}
```

```pseudo
function export_training_jsonl(dataset, window, horizon, writer):
    for i from window to len(dataset) - horizon:
        past = dataset[i-window : i]
        now = dataset[i]
        future = dataset[i : i+horizon]

        line = {
            "t": str(i),
            "timestamp": format_kst(now.visible.timestamp),
            "symbol": now.visible.symbol,
            "input": {
                "orderbook": to_orderbook_payload(now.visible.book, depth=10),
                "trades": to_trade_payload(extract_trades(past))
            },
            "target": {
                "future_mid_path": dec_list(extract_mid_path(future)),
                "future_trade_flow": dec_list(extract_trade_flow(future)),
                "future_spread_path": dec_list(extract_spread_path(future))
            }
        }

        writer.write_jsonl(line)

function export_hidden_jsonl(dataset, window, writer):
    for i from window to len(dataset):
        row = {
            "t": str(i),
            "timestamp": format_kst(dataset[i].visible.timestamp),
            "symbol": dataset[i].visible.symbol,
            "z": dec_list(dataset[i].hidden.z),
            "event_field": dec_object(dataset[i].hidden.event_field),
            "theta_id": dataset[i].hidden.theta_id,
            "seed": str(dataset[i].hidden.seed)
        }

        writer.write_jsonl(row)
```

---

## 24. Debug 출력 형태

학습 입력으로 쓰지 않는다.  
생성기 검증용이다.

```json
{
  "result": {
    "timestamp": "2026-06-22T09:01:03.124+09:00",
    "symbol": "005930",
    "latent": {
      "z": [
        "0.128391",
        "-0.441902",
        "0.003812"
      ]
    },
    "event_field": {
      "side_bias": "0.2183",
      "price_location": "-0.0312",
      "price_scale": "0.8041",
      "size_scale": "52.1139",
      "liquidity_delta_center": "-0.3911",
      "liquidity_delta_scale": "1.2041",
      "sharpness": "3.9120",
      "persistence": "0.6621",
      "tail_weight": "2.1182"
    },
    "raw_impulse": {
      "side": "BUY",
      "price_coordinate": "0.1821",
      "size": "104.391",
      "liquidity_delta": "-0.7721",
      "sharpness": "3.9120",
      "persistence": "0.6621"
    }
  }
}
```

```pseudo
function to_debug_response(state):
    return {
        "result": {
            "timestamp": format_kst(state.timestamp),
            "symbol": state.symbol,
            "latent": {
                "z": dec_list(state.hidden.z)
            },
            "event_field": dec_object(state.hidden.event_field),
            "raw_impulse": impulse_to_debug_payload(state.hidden.raw_impulse)
        }
    }
```

---

## 25. Error 출력 형태

```json
{
  "error": {
    "request_id": "req_01JZ0000000000000000000000",
    "code": "invalid_symbol",
    "message": "symbol is invalid",
    "data": {
      "symbol": "@@@"
    }
  }
}
```

```pseudo
function error_response(code, message, data = null, request_id = null):
    return {
        "error": {
            "request_id": request_id,
            "code": code,
            "message": message,
            "data": data
        }
    }
```

---

## 26. OutputAdapter 의사코드

```pseudo
struct OutputAdapter:
    symbol
    currency
    timezone = "Asia/Seoul"
    default_depth = 10

function dec(x):
    return decimal_to_string_without_float_error(x)

function dec_list(xs):
    return [dec(x) for x in xs]

function dec_object(obj):
    out = {}
    for key, value in obj:
        out[key] = dec(value)
    return out

function format_kst(timestamp):
    return timestamp.to_timezone("Asia/Seoul").isoformat_with_offset()

function format_kst_or_null(timestamp):
    if timestamp is null:
        return null
    return format_kst(timestamp)

function side_name(side):
    if side == +1:
        return "BUY"
    else:
        return "SELL"

function top_asks(book, depth):
    rows = []

    for level in ask_levels_sorted_low_to_high(book):
        if level.volume > 0:
            rows.append({
                "price": dec(level.price),
                "volume": dec(level.volume)
            })
        if len(rows) == depth:
            break

    return rows

function top_bids(book, depth):
    rows = []

    for level in bid_levels_sorted_high_to_low(book):
        if level.volume > 0:
            rows.append({
                "price": dec(level.price),
                "volume": dec(level.volume)
            })
        if len(rows) == depth:
            break

    return rows

function route(path, query, state):
    if path == "/v1/market/price":
        return to_price_response(
            book = state.book,
            symbol = query.symbol,
            currency = state.theta.currency,
            timestamp = state.now
        )

    if path == "/v1/market/orderbook":
        return to_orderbook_response(
            book = state.book,
            currency = state.theta.currency,
            timestamp = state.now,
            depth_limit = query.depth or default_depth
        )

    if path == "/v1/market/trades":
        return to_trades_response(
            trade_buffer = state.trades,
            currency = state.theta.currency,
            limit = query.limit or 30
        )

    if path == "/v1/market/candles":
        return to_candles_response(
            trades = state.trades,
            interval = query.interval,
            limit = query.limit or 200,
            before = query.before
        )

    if path == "/v1/sim/events":
        return to_events_response(
            event_buffer = state.events,
            limit = query.limit or 100,
            cursor = query.cursor
        )

    if path == "/v1/sim/snapshots":
        return to_snapshot_response(
            book = state.book,
            symbol = query.symbol,
            currency = state.theta.currency,
            timestamp = state.now,
            depth = query.depth or default_depth
        )

    if path == "/v1/sim/export":
        return export_training_jsonl(
            dataset = state.dataset,
            window = query.window,
            horizon = query.horizon,
            writer = query.writer
        )

    if path == "/v1/sim/debug":
        return to_debug_response(state_at(query.t))

    return error_response(
        code = "not_found",
        message = "endpoint not found",
        data = { "path": path },
        request_id = make_request_id()
    )
```

---

## 27. 학습 데이터 생성

```pseudo
function build_training_examples(dataset, window, horizon):
    examples = []

    for i from window to len(dataset) - horizon:
        past = dataset[i-window : i]
        future = dataset[i : i+horizon]

        x = encode_visible_history(past)

        y = {
            future_book_delta: encode_future_book_delta(future),
            future_trade_flow: encode_future_trade_flow(future),
            future_mid_path: encode_future_mid_path(future),
            future_spread_path: encode_future_spread_path(future)
        }

        hidden_for_analysis = {
            z_now: dataset[i].hidden.z,
            theta_id: dataset[i].hidden.theta_id,
            seed: dataset[i].hidden.seed
        }

        examples.append({
            input: x,
            target: y,
            hidden_for_analysis: hidden_for_analysis
        })

    return examples

TRAINING_TASKS:
    primary:
        - next event distribution prediction
        - future book state prediction
        - future impact prediction

    secondary:
        - future mid path
        - future spread path
        - future trade flow
        - future volatility proxy
        - hidden field inference
```

---

## 28. 시장 family 생성

```pseudo
function generate_market_family(num_worlds, steps, base_seed):
    worlds = []

    for i in range(num_worlds):
        seed = hash_seed(base_seed, i)
        market_run = run_market(seed, steps)

        if accept_market(market_run.dataset):
            worlds.append(market_run)

    return worlds

function split_by_world(worlds):
    shuffle(worlds)

    train = worlds[0 : 80%]
    valid = worlds[80% : 90%]
    test  = worlds[90% : 100%]

    return train, valid, test
```

---

## 29. 검증 기준

```pseudo
function accept_market(dataset):
    stats = compute_stats(dataset)

    if stats.price_frozen:
        return false

    if stats.price_exploded:
        return false

    if stats.spread_never_changes:
        return false

    if stats.depth_never_changes:
        return false

    if stats.no_trades_for_too_long:
        return false

    if not stats.volume_clusters:
        return false

    if not stats.volatility_clusters:
        return false

    if not stats.impact_depends_on_depth:
        return false

    if not stats.event_flow_has_memory:
        return false

    if not stats.book_recovers_after_shock:
        return false

    return true

STATS_TO_COMPUTE:
    price:
        - mid path
        - return distribution
        - tail thickness
        - freeze ratio
        - explosion ratio

    spread:
        - distribution
        - time variation
        - recovery after widening

    depth:
        - bid/ask depth distribution
        - local depletion
        - recovery

    trades:
        - volume clustering
        - trade burstiness
        - signed flow persistence

    impact:
        - trade size vs price movement
        - depth low → impact high
        - order flow imbalance → short horizon movement
```

---

## 30. 최종 main

```pseudo
function main():
    config = {
        latent_dim: 16,
        levels: 200,
        tick_size: "0.01",
        steps_per_world: 200000,
        num_worlds: 1000,
        window: 128,
        horizon: 32,
        base_seed: 1234
    }

    worlds = generate_market_family(
        num_worlds = config.num_worlds,
        steps = config.steps_per_world,
        base_seed = config.base_seed
    )

    train_worlds, valid_worlds, test_worlds = split_by_world(worlds)

    train = []
    valid = []
    test = []

    for world in train_worlds:
        train += build_training_examples(world.dataset, config.window, config.horizon)

    for world in valid_worlds:
        valid += build_training_examples(world.dataset, config.window, config.horizon)

    for world in test_worlds:
        test += build_training_examples(world.dataset, config.window, config.horizon)

    save_jsonl(train, "train.synthetic.jsonl")
    save_jsonl(valid, "valid.synthetic.jsonl")
    save_jsonl(test, "test.synthetic.jsonl")

    save_hidden_jsonl(train_worlds, "train.hidden.jsonl")
    save_hidden_jsonl(valid_worlds, "valid.hidden.jsonl")
    save_hidden_jsonl(test_worlds, "test.hidden.jsonl")
```

---

## 31. 파일 구조

```text
src/
  world.py          # theta sampling, seed management
  book.py           # depth-grid order book
  latent.py         # latent field z
  event_field.py    # λ = F(z, observe(book))
  impulse.py        # liquidity impulse sampling
  physics.py        # deterministic book physics
  feedback.py       # book deformation → feedback vector
  output.py         # Toss-like JSON adapter
  dataset.py        # AI training examples / JSONL export
  validate.py       # market validator
  family.py         # multiple market worlds
  run.py            # main entry

data/
  raw/
  public_events/
  snapshots/
  train.synthetic.jsonl
  valid.synthetic.jsonl
  test.synthetic.jsonl
  train.hidden.jsonl
  valid.hidden.jsonl
  test.hidden.jsonl
```

---

## 32. 최종 불변식

```pseudo
INVARIANTS:
    price is not generated directly
    order type is not sampled directly
    latent field has no human-readable concept labels
    event field is the only stochastic source of market events
    book physics is deterministic
    trades emerge only from consuming opposite depth
    public output hides hidden state
    AI input uses visible data only
    train/valid/test split is by market world, not by time
    output shape follows Toss-like market-data JSON style
```

최종 압축:

```text
z_t
  → event field λ_t
  → liquidity impulse e_t
  → deterministic book physics
  → trades / spread / depth / price movement
  → feedback
  → z_{t+1}
  → Toss-like public JSON / AI dataset export
```
