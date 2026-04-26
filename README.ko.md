# market-wave

<p align="center">
  <img src="docs/assets/market-wave-hero.png" alt="market-wave 시장 의도 시뮬레이션 히어로 이미지" />
</p>

<p align="center">
  <strong>이산 혼합분포로 시장 전체의 진입/탈출 의도를 시뮬레이션합니다.</strong>
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
의도를 시뮬레이션하는 Python 라이브러리입니다. 집계된 매수/매도 압력, 포지션
청산, 호가창 깊이, 주문 취소, taker flow, 체결 기반 가격 변화를 이산 가격
그리드 위에서 다룹니다.

이 라이브러리는 가격 예측 모델이 아닙니다. 실험, 시각화, 교육, 전략 환경
프로토타이핑을 위한 가벼운 시뮬레이션 도구입니다.

## 왜 market-wave인가?

- **개별 agent가 아닌 집계 의도**: 참여자 객체를 만들지 않고 가격별 확률질량으로
  시장 의도를 표현합니다.
- **이산 혼합분포**: 진입/탈출 압력을 현재 `price_grid` 위의 PMF로 모델링합니다.
- **체결 기반 가격**: 체결이 없으면 가격은 움직이지 않습니다.
- **관찰 가능한 상태**: 매 step마다 PMF, 거래량, 호가창, 포지션 mass, VWAP,
  spread, imbalance가 담긴 `StepInfo`를 반환합니다.
- **내장 시각화**: `matplotlib` 기반의 깔끔한 light chart 스타일을 제공합니다.

## 설치

```bash
pip install market-wave
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

market = Market(initial_price=10_000, gap=10, popularity=1.0, seed=42)
steps = market.step(100)

last = steps[-1]
print(last.price_before, "->", last.price_after)
print(last.total_executed_volume)
```

`Market.step(n)`은 항상 `list[StepInfo]`를 반환하고, 같은 객체들을
`market.history`에 저장합니다.

## 시각화

```python
from market_wave import Market

market = Market(initial_price=10_000, gap=10, popularity=1.0, seed=42)
market.step(260)

fig, ax = market.plot(last=180)
```

<p align="center">
  <img src="docs/assets/market-wave-plot.png" alt="가격, 거래량, imbalance를 보여주는 market-wave light pyplot 차트" />
</p>

기본 `market_wave` 스타일은 가격/VWAP, 체결량, order-flow imbalance를 분리한
light 3-panel 차트입니다. Dark overlay 모드도 사용할 수 있습니다.

```python
fig, ax = market.plot(layout="overlay", style="market_wave_dark")
```

## 핵심 개념

매 step마다 현재 가격 주변에 가격 그리드를 만듭니다.

```text
price_grid = current_price +/- k * gap
```

시뮬레이터는 이 그리드 위에 네 개의 확률질량함수를 유지합니다.

- `buy_entry_pmf`
- `sell_entry_pmf`
- `long_exit_pmf`
- `short_exit_pmf`

각 PMF는 정규화된 이산 혼합분포입니다.

```text
pmf[x] = sum(component_weight * kernel(x, center_price, spread))
kernel(x, center, spread) proportional to exp(-abs(x - center) / spread)
```

PMF는 집계 의도를 만들고, 호가창/체결 레이어가 이를 limit flow, taker flow,
주문 취소, 청산, 체결량, 가격 변화로 바꿉니다.

## 체결 보장

가격 변화는 체결에 의해 발생합니다.

- 해당 step의 체결량이 없으면 `price_after == price_before`입니다.
- 체결이 있으면 `price_after`는 해당 step의 체결 통계에서 계산됩니다.
- 같은 버전과 같은 입력에서 `seed`는 재현 가능한 시뮬레이션을 만듭니다.

이 라이브러리는 시장 데이터 replay 엔진이 아니며, 금융 조언도 아닙니다.

## API 개요

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

자주 보는 `StepInfo` 필드:

- `price_before`, `price_after`, `price_change`
- `buy_entry_pmf`, `sell_entry_pmf`, `long_exit_pmf`, `short_exit_pmf`
- `buy_volume_by_price`, `sell_volume_by_price`
- `executed_volume_by_price`, `total_executed_volume`, `trade_count`
- `vwap_price`, `best_bid_before`, `best_ask_before`, `spread_after`
- `orderbook_before`, `orderbook_after`
- `position_mass_before`, `position_mass_after`

## 개발

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
uv build
```

## 라이선스

MIT
