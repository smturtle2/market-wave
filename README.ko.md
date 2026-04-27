# market-wave

<p align="center">
  <img src="https://raw.githubusercontent.com/smturtle2/market-wave/main/docs/assets/market-wave-hero.png" alt="market-wave 시장 의도 시뮬레이션 히어로 이미지" />
</p>

<p align="center">
  <strong>Dynamic Market Distribution Function으로 빠르고 가벼운 synthetic market data를 만듭니다.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/market-wave/"><img alt="PyPI" src="https://img.shields.io/pypi/v/market-wave"></a>
  <a href="https://pypi.org/project/market-wave/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/market-wave"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href=".github/workflows/workflow.yml"><img alt="Tests" src="https://img.shields.io/badge/tests-pytest-2563eb"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | 한국어
</p>

`market-wave`는 개별 참여자를 만들지 않고 시장 전체의 진입 가격과 탈출 가격
의도로 synthetic market path를 만드는 Python 라이브러리입니다. 집계된 매수/매도
진입 의도, 포지션 청산, 호가창 깊이, 주문 취소, taker flow, 체결 기반 가격 변화를
relative tick 위의 확률질량으로 다룹니다.

이 라이브러리는 가격 예측 모델이 아닙니다. 실험, 시각화, 교육, 전략 환경
프로토타이핑을 위한 가벼운 시뮬레이션 도구입니다.

## 왜 market-wave인가?

- **개별 agent가 아닌 집계 의도**: 참여자 객체를 만들지 않고 relative tick별 확률질량으로
  시장 의도를 표현합니다.
- **Dynamic MDF**: 네 개의 `MDF(relative_tick)`가 이전 step의 상태를 이어받아
  진입/탈출 의도를 표현합니다.
- **교체 가능한 score 모델**: `DynamicMDFModel` 또는 custom `MDFModel`로 MDF
  score 함수만 바꿀 수 있습니다.
- **분포와 거래강도 분리**: MDF는 의도가 어느 가격대에 있는지, intensity는 총
  주문량이 얼마나 되는지를 담당합니다.
- **체결 기반 가격**: 체결이 없으면 가격은 움직이지 않습니다.
- **batch generation**: 많은 synthetic path를 만들 때 `market.history`를 계속
  쌓지 않고 재현 가능한 경로를 생성할 수 있습니다.
- **관찰 가능한 상태**: 매 step마다 MDF, 거래량, 호가창, 포지션 mass, VWAP,
  spread, imbalance가 담긴 `StepInfo`를 반환합니다.
- **내장 시각화**: `matplotlib` 기반의 깔끔한 light chart 스타일을 제공합니다.

## 설치

```bash
pip install market-wave
```

dataframe export가 필요하면:

```bash
pip install "market-wave[dataframe]"
```

로컬 개발:

```bash
git clone https://github.com/smturtle2/market-wave.git
cd market-wave
uv sync --extra dev
```

Python `>=3.10`을 지원합니다.

## 빠른 시작

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

`Market.step(n)`은 항상 `list[StepInfo]`를 반환하고, 같은 객체들을
`market.history`에 저장합니다.

대량 생성에서는 history 저장을 끌 수 있습니다.

```python
steps = market.step(512, keep_history=False)

for step in market.stream(512, keep_history=False):
    consume(step)
```

간단한 export에는 `step.to_dict()`, `step.to_json()`,
`market.history_records()`를 사용할 수 있습니다.

`seed=42` 기준 예시 출력:

```text
9880.0 -> 9890.0
entry: 3.321
executed: 2.863
resting bid/ask: 16.4 15.632
imbalance: -0.151
```

## 스모크 매트릭스

고정 seed에서는 결정적으로 동작하므로, 서로 다른 시장 조건에 같은 invariant를
쉽게 적용할 수 있습니다.

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

현재 구현에서 최근 검증한 결과:

```text
baseline  range=  9850.0- 10050.0 unique= 21 moves=396 exec_steps=500 final=  9890.0
busy      range=  9940.0- 10260.0 unique= 32 moves=397 exec_steps=500 final= 10260.0
thin      range=   500.0-   650.0 unique= 31 moves=339 exec_steps=500 final=   630.0
low_price range=     2.0-    24.0 unique= 23 moves=393 exec_steps=500 final=    21.0
trend_up  range=  9990.0- 10420.0 unique= 44 moves=402 exec_steps=500 final= 10410.0
high_vol  range=  9980.0- 10180.0 unique= 21 moves=422 exec_steps=500 final= 10140.0
inactive  range=   100.0-   100.0 unique=  1 moves=  0 exec_steps=  0 final=   100.0
```

이 실행들은 현재 state의 MDF projection이 `state.price_grid`와 정렬되는지, MDF가 정규화되는지,
가격이 한 tick 아래로 내려가지 않는지, order book과 position mass가 음수가
아닌지, 가격 변화가 체결량이 있는 step에서만 발생하는지도 함께 확인했습니다.
Dynamic MDF acceptance는 `mdf_temperature=1.0`에서 seed `10..19`도 실행해
모든 MDF가 finite, non-negative, normalized 상태를 유지하고 한 가격으로
붕괴하지 않는지도 확인합니다.

`0.4.0` 진단 메모: 시뮬레이터는 여전히 가격을 초기값으로 되돌리는 anchor나
저장된 목표 가격을 갖지 않습니다. seed로 정해진 `mood`, `trend`, `volatility`,
microstructure activity, cancellation pressure, event pressure가 매 step 전이되며
MDF와 visible book을 바꿉니다. 가격은 체결 기반으로 움직이고, 체결 flow가
한쪽 압력을 드러낼 때 작은 price-discovery 성분을 반영합니다. 위 range,
move count, execution count는 특정 실제 시장과의 일치 주장이 아니라
regression diagnostic으로 보아야 합니다.

Entry MDF의 가격은 incoming order 가격으로 취급됩니다. 매수 entry는 bid로,
매도 entry는 ask로 들어오며, 기존 반대편 호가와 겹칠 때만 체결됩니다. 체결
가격은 resting quote 가격입니다. 체결되지 않은 물량은 sampled MDF 가격에
그대로 미체결 호가로 남습니다. Exit flow는 cohort 조건부로 생성되며, exit
주문도 원래 cohort id를 들고 visible order-book liquidity를 통해 처리됩니다.

`0.4.0` microstructure 메모: orderbook 보충은 regime별 depth shape, resiliency,
absolute tick 기준 wall memory, event-driven volume burst, cancellation pressure
이후 dry-up, trend exhaustion, short crowding과 최근 one-sided flow에서 파생한
squeeze pressure를 반영합니다. live orderbook과 position total은 계속
price/side별로 cache하고, lot은 price/kind별로 합치며, position inventory는
제한된 entry-price cohort bucket으로 유지합니다.

## 시각화

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
  <img src="https://raw.githubusercontent.com/smturtle2/market-wave/main/docs/assets/market-wave-plot.png" alt="가격, orderbook depth heatmap, 거래량, imbalance를 보여주는 market-wave light pyplot 차트" />
</p>

기본 `market_wave` 스타일은 가격/VWAP, 단순 level 기준 bid/ask orderbook depth
heatmap, 체결량, order-flow imbalance를 함께 보여주는 light multi-panel 차트입니다.
예전 3-panel 화면은 `orderbook=False`로 볼 수 있습니다.

Dark overlay 모드도 사용할 수 있습니다.

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

`GeneratedPath.metadata`에는 `seed`, `config_hash`, package `version`, `regime`,
`augmentation_strength`가 저장되어 synthetic run을 추적할 수 있습니다. Pandas는
optional입니다. `to_dataframe()`을 쓰려면 `market-wave[dataframe]`으로 설치하세요.
`ValidationMetrics.volatility_clustering_score`는 각 generated path 내부에서 계산한 뒤
집계하므로 독립 path 사이의 경계가 diagnostic에 섞이지 않습니다.

## 교체 가능한 MDF

```python
from market_wave import Market

class CenterSeekingMDF:
    def scores(self, side, intent, relative_ticks, context, signals=None):
        del side, intent, context, signals
        return [-abs(tick) for tick in relative_ticks]

market = Market(initial_price=100, gap=1, mdf_model=CenterSeekingMDF(), seed=7)

step = market.step(1)[0]
print(step.relative_ticks)
print(step.buy_entry_mdf)
```

Custom MDF model은 확률이 아니라 score를 반환합니다. 각 score는 log-growth
evidence로 해석할 수 있으며, additive score 차이는 이전 MDF에 대한
multiplicative 변화가 됩니다. `Market`은 아래의 안정화된 MDF update로 이
score를 반영합니다.

## 핵심 개념

매 step마다 현재 가격 주변의 relative tick 그리드를 만듭니다.

```text
relative_tick = (price - current_price) / tick_size
relative_ticks = [-grid_radius, ..., 0, ..., +grid_radius]
```

시뮬레이터는 이 relative grid 위에 네 개의 Market Distribution Function을 유지합니다.

- `buy_entry_mdf`
- `sell_entry_mdf`
- `long_exit_mdf`
- `short_exit_mdf`

각 MDF는 정규화됩니다. 매 step 새로 만들어지는 것이 아니라 이전 MDF에서
진화합니다.

```text
logits = persistence * log(MDF_prev(tick) + eps)
       + score(tick) / effective_temperature
proposal = softmax(clamp(logits - max(logits), -50, 0))
MDF_next = Normalize((1 - floor_mix) * Diffuse(proposal) + floor_mix * Uniform)
```

`score(tick)`에는 placement shape, trend, liquidity attraction, memory, risk,
orderbook imbalance가 반영될 수 있습니다. `mdf_temperature`는 score가 분포를
얼마나 날카롭게 바꾸는지 조절합니다. effective temperature에는 현재 volatility도
반영되므로 high-volatility regime에서 한 tick이 모든 질량을 흡수하지 않도록
score update가 완만해집니다. persistence, diffusion, uniform floor mixing은 작은
score 우위가 반복되면서 MDF가 한 tick으로 붕괴되는 일을 막습니다.

relative MDF는 호가 형성을 위해
pre-trade grid인 `price_grid = price_before +/- k * gap`에 투영됩니다.
`StepInfo.mdf_price_basis`는 이 pre-trade 가격 기준을 기록합니다.

```text
low temperature  -> 더 뾰족하고 집중된 MDF
high temperature -> 더 넓고 부드러운 MDF
```

MDF는 집계 의도를 만들고, intensity는 총 주문량을 결정합니다. 호가창/체결
레이어는 이를 limit flow, taker flow, 주문 취소, 청산, 체결량, 가격 변화로
바꿉니다.

## 체결 보장

가격 변화는 체결에 의해 발생합니다.

- 해당 step의 체결량이 없으면 `price_after == price_before`입니다.
- 체결이 있으면 `price_after`는 해당 step의 체결 통계에서 계산됩니다. bounded
  quote jitter는 보조 역할만 하며, 이전 가격에서만 체결된 경우 혼자 가격을
  움직이지 않습니다.
- 같은 버전과 같은 입력에서 `seed`는 재현 가능한 시뮬레이션을 만듭니다.

이 라이브러리는 시장 데이터 replay 엔진이 아니며, 금융 조언도 아닙니다.

## API 개요

```python
from market_wave import (
    Market,
    DynamicMDFModel,
    generate_paths,
    compute_metrics,
    MarketState,
    IntensityState,
    LatentState,
    MDFContext,
    MDFSignals,
    MDFModel,
    RelativeMDFComponent,
    MDFState,
    OrderBookState,
    PositionMassState,
    StepInfo,
)
```

자주 보는 `StepInfo` 필드:

- `price_before`, `price_after`, `price_change`
- `tick_before`, `tick_after`, `tick_change`, `relative_ticks`
- `mdf_price_basis`, `price_grid`
- `buy_entry_mdf`, `sell_entry_mdf`, `long_exit_mdf`, `short_exit_mdf`
- `buy_entry_mdf_by_price`, `sell_entry_mdf_by_price`
- `entry_volume_by_price`, `exit_volume_by_price`
- `buy_volume_by_price`, `sell_volume_by_price`
- `executed_volume_by_price`, `total_executed_volume`, `trade_count`
- `market_buy_volume`, `market_sell_volume`
- `vwap_price`, `best_bid_before`, `best_ask_before`, `spread_after`
- `orderbook_before`, `orderbook_after`
- `position_mass_before`, `position_mass_after`

`buy_volume_by_price`와 `sell_volume_by_price`는 sampled order price별 제출
의도 물량이며, 체결량이나 남은 호가 물량이 아닙니다. `market_*` volume field는
실제로 체결된 incoming buy/sell 물량을 뜻합니다. 체결되지 않은 incoming 물량은
`orderbook_after`에 남습니다. legacy `residual_market_*`와
`crossed_market_volume` field는 현재 order-book-first engine에서 호환성을 위한
0 값으로 유지됩니다.

`*_mdf_by_price` 필드는 `mdf_price_basis`를 기준으로 한 pre-trade MDF projection입니다.
현재 `Market.state.mdf.*_by_price`는 post-trade state 가격에 맞춰 다시 투영됩니다.
예시와 public API는 MDF 이름만 사용합니다. 초기 prototype의 오래된 PMF 예시는
obsolete로 보아야 합니다.

### Public Contract와 Snapshot 정책

public import surface는 package `__all__`입니다. 여기에는 `Market`,
`generate_paths`, `compute_metrics`, generated path metadata, MDF model/protocol
type, metric, 그리고 위에 보인 state dataclass들이 포함됩니다. entrypoint는
작지만, `StepInfo`와 `MarketState`가 상세 simulator diagnostic을 노출하므로
observation contract는 넓습니다.

현재 alpha line에서는 기존 public name과 기존 `StepInfo` / state field를 가능한
한 호환되게 유지합니다. alpha release 중 새 diagnostic field가 추가될 수 있습니다.
MDF 이름이 지원되는 public distribution 이름이며, 초기 prototype의 오래된 PMF
이름은 obsolete입니다.

Snapshot mutability: state dataclass는 attribute level에서 `frozen=True`이지만,
nested `dict`와 `list` field는 `to_dict()`와 JSON export를 단순하게 유지하기 위한
plain mutable container입니다. `Market.state`, `StepInfo`,
`GeneratedPath.hidden_states`는 read-only observation처럼 다루세요. downstream
code에서 현재 상태를 안전하게 수정하며 확인해야 한다면 mutation-safe deep copy를
반환하는 `Market.snapshot()`을 사용하세요.

호환성 메모: `Market.state`는 alpha line에서 live current-state attribute로
유지됩니다. 향후 release에서는 in-place state container mutation에 의존하는
코드를 위한 더 명시적인 read-model API 또는 deprecation path가 추가될 수 있습니다.

## 개발

```bash
uv sync --extra dev --extra dataframe
uv run ruff check .
uv run pytest
uv build
```

## 라이선스

MIT
