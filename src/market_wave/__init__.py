"""Public API for the market-wave synthetic market simulator.

The package root intentionally exposes only the simulator entry point and state
snapshot dataclasses. Metrics helpers live in :mod:`market_wave.metrics`.
"""

from .market import Market
from .state import (
    IntensityState,
    LatentState,
    MarketState,
    MDFState,
    OrderBookState,
    StepInfo,
)

__all__ = [
    "IntensityState",
    "LatentState",
    "MDFState",
    "Market",
    "MarketState",
    "OrderBookState",
    "StepInfo",
]
