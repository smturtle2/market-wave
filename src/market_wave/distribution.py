from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite


@dataclass(frozen=True)
class MixtureComponent:
    weight: float
    center_price: float
    spread: float


@dataclass(frozen=True)
class DiscreteMixtureDistribution:
    components: tuple[MixtureComponent, ...]

    def pmf(self, price_grid: list[float]) -> dict[float, float]:
        if any(not isfinite(price) for price in price_grid):
            raise ValueError("price_grid must contain only finite prices")
        for component in self.components:
            if not (
                isfinite(component.weight)
                and isfinite(component.center_price)
                and isfinite(component.spread)
            ):
                raise ValueError("mixture components must contain only finite values")
            if component.spread <= 0:
                raise ValueError("mixture component spread must be positive")

        masses = {}
        for price in price_grid:
            mass = 0.0
            for component in self.components:
                if component.weight <= 0:
                    continue
                spread = max(component.spread, 1e-12)
                kernel = exp(-abs(price - component.center_price) / spread)
                mass += component.weight * kernel
            masses[price] = mass

        total = sum(masses.values())
        if total <= 0:
            if not price_grid:
                return {}
            uniform = 1.0 / len(price_grid)
            return {price: uniform for price in price_grid}

        return {price: mass / total for price, mass in masses.items()}
