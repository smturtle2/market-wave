from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class MixtureComponent:
    weight: float
    center_price: float
    spread: float


@dataclass(frozen=True)
class DiscreteMixtureDistribution:
    components: tuple[MixtureComponent, ...]

    def pmf(self, price_grid: list[float]) -> dict[float, float]:
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
