"""Latent-field synthetic stock market simulator."""

__version__ = "1.0.0"

from .api import Market, generate, open

__all__ = [
    "Market",
    "generate",
    "open",
]
