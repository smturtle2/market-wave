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

`market-wave`는 개별 참여자를 만들지 않고 시장 전체의 진입 가격 분포로
synthetic market path를 만드는 Python 라이브러리입니다. 집계된 매수/매도
진입 의도, resting order-book 깊이, 확률적 주문 취소, taker flow, 체결 기반
가격 변화를 gap 단위 offset 위의 확률질량으로 다룹니다.

이 라이브러리는 가격 예측 모델이 아닙니다. 실험, 시각화, 교육, 전략 환경
프로토타이핑을 위한 가벼운 시뮬레이션 도구입니다.

핵심 엔진 원칙은 시장 상태가 두 entry MDF를 바꾸고, incoming order offset은 그
MDF에서 직접 샘플링되며, realized sample이 다음 시장 상태에 되먹임된다는
것입니다. 샘플링 이후 원하는 path를 맞추기 위한 order 후보정이나 강제 이동은
하지 않습니다.

## 왜 market-wave인가?

- **개별 agent가 아닌 집계 의도**: 참여자 객체를 만들지 않고 gap 단위 offset별 확률질량으로
  시장 의도를 표현합니다.
- **Raw-mass MDF**: 관측 가능한 offset별 질량을 직접 더한 뒤 정규화해서
  매수/매도 진입 의도를 표현합니다.
- **분포와 거래강도 분리**: MDF는 의도가 어느 가격대에 있는지, intensity는 총
  주문량이 얼마나 되는지를 담당합니다.
- **체결 기반 가격**: 체결이 없으면 가격은 움직이지 않습니다.
- **샘플 후 후보정 없음**: sampled order price는 목표 volatility, trend, spread,
  path shape에 맞추려고 다시 쓰지 않습니다.
- **단순한 batch loop**: `Market`을 만들고 `step(n)`을 호출하는 방식으로
  재현 가능한 synthetic path를 생성합니다.
- **관찰 가능한 상태**: 매 step마다 MDF, 제출 물량, 취소 물량, 체결, 호가창,
  VWAP, spread, imbalance가 담긴 `StepInfo`를 반환합니다.
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
from market_wave.metrics import compute_metrics

market = Market(popularity=3.0, seed=42)
steps = market.step(500)
paths = [steps]
metrics = compute_metrics(paths)

last = steps[-1]
print(last.price_before, "->", last.price_after)
print("entry:", round(sum(last.entry_volume_by_price.values()), 3))
print("executed:", round(last.total_executed_volume, 3))
print("resting bid/ask:", round(sum(last.orderbook_after.bid_volume_by_price.values()), 3), round(sum(last.orderbook_after.ask_volume_by_price.values()), 3))
print("imbalance:", round(last.order_flow_imbalance, 3))
print("execution rate:", round(metrics.execution_rate, 3))
```

모델은 plain 생성자 파라미터로 직접 설정합니다.

```python
from market_wave import Market

market = Market(initial_price=10_000, gap=10, popularity=3.0, regime="normal", seed=42)
steps = market.step(500)
```

`regime`은 초기 market condition만 정의합니다. 이후 `StepInfo.regime` 값은
생성자 label을 고정해 둔 것이 아니라 simulator state transition이 만든 active
condition label입니다.

`Market.step(n)`은 항상 `list[StepInfo]`를 반환하고, 같은 객체들을
`market.history`에 저장합니다.

간단한 export에는 `step.to_dict()`, `step.to_json()`,
`market.history_records()`를 사용할 수 있습니다.

정확한 숫자는 seed와 버전에 따라 달라집니다. 출력 필드는 마지막 step의 현재
가격, sampled entry volume, executed volume, resting book depth, realized
order-flow imbalance를 확인하기 위한 예시입니다.

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
    tick_changes = [step.tick_change for step in steps]
    cumulative_ticks = [sum(tick_changes[: index + 1]) for index in range(len(tick_changes))]
    move_steps = sum(step.tick_change != 0 for step in steps)
    exec_steps = sum(step.total_executed_volume > 0 for step in steps)
    tick_range = max(cumulative_ticks, default=0) - min(cumulative_ticks, default=0)
    print(name, tick_range, move_steps, exec_steps)
```

이 matrix는 정확한 range를 보장하는 값이 아니라 plausibility check입니다.
Regression test는 고정 final price 대신 tick-native invariant를 확인합니다. MDF는
finite/normalized 상태를 유지해야 하고, 가격 좌표는 한 tick 아래로 내려가면 안 되며,
order-book depth는 음수가 될 수 없고, 체결량이 없는 step에서는 tick이 움직이지
않아야 합니다. inactive market은 flat하게 유지되어야 하고, 같은 seed에서는 busy
market이 thin market보다 대체로 더 많은 체결량과 거래를 만들어야 합니다.

현재 엔진 진단 메모: 시뮬레이터는 하나의 특정 시장을 replay하거나 calibrate하기
위한 것이 아니라 여러 synthetic market family를 표현하기 위한 도구입니다. seed로
정해진 `mood`, `trend`, `volatility`, microstructure activity, cancellation
pressure, participant pressure, visible book state가 매 step 전이되며 두 MDF를
바꿉니다. 가격은 체결 기반으로 움직이고, 체결이 발생하면 현재 quote context와
execution VWAP를 함께 사용해 thin/busy market이 같은 order flow에도 다르게
반응할 수 있습니다. range, move count, execution count는 특정 실제 시장과의
일치 주장이 아니라 regression diagnostic입니다.

Entry MDF의 key는 gap 단위 offset입니다. incoming order는 먼저 offset을 샘플링한
뒤 실행 가능한 주문 가격으로 변환됩니다. 매수 entry는 bid로, 매도 entry는 ask로
들어오며, 기존 반대편 호가와 겹칠 때만 체결됩니다. 체결 가격은 resting quote
가격입니다. 체결되지 않은 물량은 변환된 주문 가격에 그대로 미체결 호가로
남습니다. 이후 resting order는 확률적 취소로 빠져나갑니다. 현재 같은 side의 MDF
support가 약한 가격의 주문일수록 더 많이 줄어들거나 사라질 가능성이 큽니다.

현재 MDF 메모: buy/sell MDF는 더 이상 score softmax update를 사용하지 않습니다.
엔진은 near-market continuity, visible book shortage, gap/front/occupancy signal,
hidden participant pressure, volatility, stress, microstructure texture에서
gap 단위 offset별 raw mass를 만들고 직접 정규화합니다. Hidden participant pressure는
추가 public MDF나 agent 목록이 아닙니다. 시장 조건과 noise에서 도출되는 internal
state이며, upward push, downward push, upward resistance, downward resistance,
general noisy participation을 두 MDF가 샘플링되기 전에 표현합니다.

현재 엔진 microstructure 메모: orderbook 보충은 현재 entry MDF, resiliency,
stress-aware cancellation, event-driven volume burst, cancellation pressure 이후
dry-up, trend exhaustion, 한쪽 visible liquidity에서 나온 book-pressure squeeze
signal을 반영합니다. live orderbook total은 계속 price/side별로 cache하고, lot은
price/kind별로 합칩니다.

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
from market_wave import Market
from market_wave.metrics import (
    compare_metrics,
    compute_metrics,
    load_reference_metrics_profile,
    save_metrics_profile_json,
    validate_reference_records,
)

paths = []
for path_id in range(100):
    market = Market(initial_price=10_000, gap=10, popularity=1.0, seed=10_000 + path_id)
    paths.append(market.step(512))

metrics = compute_metrics(paths)
print(metrics.tick_return_std, metrics.volume_mean, metrics.max_drawdown_ticks)
print(metrics.cancellation_rate, metrics.position_change_rate)
print(metrics.mean_abs_mdf_anchor_change_ticks, metrics.mdf_anchor_event_pressure_corr)
print(metrics.mean_spread_ticks, metrics.mean_near_depth_share, metrics.one_sided_book_rate)
print(metrics.mean_quote_age)
save_metrics_profile_json(metrics, "synthetic-metrics.json", name="synthetic")
reference = load_reference_metrics_profile("real-reference-records.jsonl", name="reference", gap=10)
comparison = compare_metrics(metrics, reference)
print(comparison.score)
```

`ValidationMetrics.volatility_clustering_score`는 각 generated path 내부에서 계산한 뒤
집계하므로 독립 path 사이의 경계가 diagnostic에 섞이지 않습니다.
취소와 position-change diagnostic은 visualization 코드가 아니라 exported
`StepInfo` field에서 계산됩니다.
Anchor diagnostic은 MDF basis가 tick 단위로 얼마나 움직였는지와 그 움직임이
realized event pressure와 얼마나 연동되는지를 보여줍니다.
Book-topology diagnostic은 spread와 depth concentration을 tick-native 값으로
보여주므로 절대 가격 scale이 달라도 비교할 수 있습니다.
`mean_quote_age`는 visible resting quote의 volume-weighted lifecycle age를
보여줍니다. 기본 비교 점수에는 들어가지 않고,
`compare_metrics(fields=...)`에 명시했을 때만 비교합니다.
`compare_metrics()`는 synthetic metrics를 외부에서 준비한 reference profile과
비교하므로 calibration을 visualization 코드가 아니라 Python 안에 둘 수 있습니다.
`load_reference_metrics_profile()`은 metrics JSON 파일 또는 `StepInfo.to_dict()`
field 이름을 따르는 JSONL/CSV row를 받습니다.
실제 L2/tape 데이터는 먼저 이 step-level schema로 변환해야 합니다.
Reference record에는 `tick_change`, `tick_before`, `tick_after`, `price_after`,
`total_executed_volume`, `cancelled_volume_by_price`, `trade_count`,
`order_flow_imbalance`, `mdf_price_basis`, `spread_after`, `orderbook_after`가
필요합니다. `path_id`와 `mean_quote_age`는 optional입니다. `orderbook_after`는
`bid_volume_by_price`, `ask_volume_by_price` map을 포함해야 합니다.
이 계약은 `REFERENCE_RECORD_REQUIRED_FIELDS`,
`REFERENCE_RECORD_OPTIONAL_FIELDS`, `REFERENCE_ORDERBOOK_REQUIRED_FIELDS`로도
export됩니다.
변환한 row는 calibration 전에 `validate_reference_records(records)`로 먼저
검증할 수 있습니다.

## 핵심 개념

매 step마다 현재 가격 주변의 gap 단위 offset 그리드를 만듭니다.

```text
x = (price - current_price) / gap
gap_offsets = [-grid_radius, ..., 0, ..., +grid_radius]
```

시뮬레이터는 이 x-domain 위에 두 개의 Market Distribution Function을 유지합니다.

- `buy_entry_mdf`
- `sell_entry_mdf`

각 MDF는 정규화됩니다. MDF는 offset별 raw mass에서 만들어집니다.

```text
raw_mass(x) =
    near_market_continuity
  + book_shortage_or_front_mass
  + gap_front_occupancy_pressure
  + hidden_participant_pressure
  + microstructure_texture

proposal = Normalize(raw_mass)
MDF_next = Normalize(proposal)
```

custom score model, temperature, score softmax 경로는 없습니다. `mood`,
`trend`, `volatility`, visible liquidity, shortage, 최근 flow, stress가 raw MDF
shape를 직접 바꿉니다. 시장가 주변 tick은 0이 아닌 질량을 유지하고,
shortage/front/liquidity 신호는 현재가에서 떨어진 여러 local pocket을 만들 수
있습니다.

public MDF field는 최종 샘플링에 쓰이는 effective distribution입니다. 엔진은
원하는 path shape를 강제로 만들기 위한 별도 post-sampling correction layer를 두지
않습니다. incoming order의 가격과 taker/limit 성격은 현재 시장 상태가 바꾼 두
MDF에서 `x`를 직접 샘플링한 결과에서 나옵니다. 유일한 boundary 변환은
sampled gap offset을 `mdf_price_basis`에서 실행 가능한 tick grid로 투영하는
것입니다. resting quote의 취소는 별도 public cancellation MDF가 아니라 시장
상태에 따른 quote lifecycle hazard로 발생합니다. realized sample은 다시 book,
execution, flow memory, participant pressure, 다음 step의 market state를 바꿉니다.

MDF는 집계 의도를 만들고, intensity는 보장된 제출량이 아니라 fixed time slice의
기대 활동량을 결정합니다. 호가창/체결 레이어는 sampled arrival을 limit flow,
taker flow, 주문 취소, 체결량, 가격 변화로 바꿉니다. 체결이 없는 조용한 slice도
정상 output입니다.

## 체결 보장

가격 변화는 체결에 의해 발생합니다.

- 해당 step의 체결량이 없으면 `price_after == price_before`입니다.
- 체결이 있으면 `price_after`는 sampled execution path를 따릅니다. 모든 체결이
  이전 가격에서만 발생한 경우 flow만으로 mark를 움직일 수 없습니다.
- 같은 버전과 같은 입력에서 `seed`는 재현 가능한 시뮬레이션을 만듭니다.

이 라이브러리는 시장 데이터 replay 엔진이 아니며, 금융 조언도 아닙니다.

## API 개요

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

자주 보는 `StepInfo` 필드:

- `price_before`, `price_after`, `price_change`
- `tick_before`, `tick_after`, `tick_change`
- `mdf_price_basis`, `price_grid`
- `buy_entry_mdf`, `sell_entry_mdf`
- `entry_volume_by_price`, `cancelled_volume_by_price`
- `buy_volume_by_price`, `sell_volume_by_price`
- `executed_volume_by_price`, `total_executed_volume`, `trade_count`
- `market_buy_volume`, `market_sell_volume`
- `vwap_price`, `best_bid_before`, `best_ask_before`, `spread_after`
- `mean_quote_age`
- `orderbook_before`, `orderbook_after`

`buy_volume_by_price`와 `sell_volume_by_price`는 sampled order price별 제출
의도 물량이며, 체결량이나 남은 호가 물량이 아닙니다. `market_*` volume field는
실제로 체결된 incoming buy/sell 물량을 뜻합니다. `residual_market_*` field는
체결되지 않았고 book에 남길 수 있었던 incoming buy/sell 물량을 뜻합니다.
체결되지 않은 incoming 물량은 `orderbook_after`에 남습니다.
`crossed_market_volume` field는 현재 order-book-first engine에서 호환성을 위한
diagnostic이며 0 값으로 유지됩니다.

`buy_entry_mdf`와 `sell_entry_mdf`가 유일한 public MDF distribution입니다.
이 두 MDF의 key는 먼저 gap 단위 offset으로 샘플링됩니다. 초기 prototype의 오래된
price-keyed `*_mdf_by_price` 또는 PMF 예시는 obsolete로 보아야 합니다.

### Public Contract와 Snapshot 정책

public import surface는 package `__all__`입니다. 여기에는 `Market`, 그리고 위에
보인 state dataclass들이 포함됩니다. 시뮬레이션 진행은 `Market.step()`만
사용합니다. 시장 동작은 plain 생성자 파라미터로 설정합니다. tick-native metrics
helper는 `market_wave.metrics`에 있습니다. custom MDF model/protocol type은 더
이상 public API가 아닙니다. entrypoint는 작지만, `StepInfo`와 `MarketState`가
상세 simulator diagnostic을 노출하므로 observation contract는 넓습니다.

현재 alpha line에서는 기존 public name과 기존 `StepInfo` / state field를 가능한
한 호환되게 유지합니다. alpha release 중 새 diagnostic field가 추가될 수 있습니다.
MDF 이름이 지원되는 public distribution 이름이며, 초기 prototype의 오래된 PMF
이름은 obsolete입니다.

Snapshot mutability: state dataclass는 attribute level에서 `frozen=True`이지만,
nested `dict`와 `list` field는 `to_dict()`와 JSON export를 단순하게 유지하기 위한
plain mutable container입니다. `Market.state`와 `StepInfo`는 read-only
observation처럼 다루세요. downstream code에서 현재 상태를 안전하게 수정하며
확인해야 한다면 mutation-safe deep copy를 반환하는 `Market.snapshot()`을
사용하세요.

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
