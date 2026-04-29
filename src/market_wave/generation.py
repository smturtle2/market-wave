from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .market import Market
from .state import MarketState, StepInfo

MarketConfig = Mapping[str, Any]
ConfigSampler = Callable[..., Market | MarketConfig | None]


@dataclass(frozen=True)
class GenerationMetadata:
    path_id: int
    horizon: int
    config: dict[str, Any]
    config_hash: str
    version: str
    seed: int | None
    regime: str
    augmentation_strength: float
    initial_price: float
    final_price: float
    step_count: int
    include_hidden: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass(frozen=True)
class GeneratedPath:
    path_id: int
    steps: tuple[StepInfo, ...]
    metadata: GenerationMetadata
    hidden_states: tuple[MarketState, ...] | None = None

    @property
    def prices(self) -> tuple[float, ...]:
        return tuple(step.price_after for step in self.steps)

    @property
    def returns(self) -> tuple[float, ...]:
        return tuple(step.price_change for step in self.steps)

    @property
    def final_price(self) -> float:
        return self.metadata.final_price

    def to_records(self) -> list[dict[str, Any]]:
        records = []
        for step in self.steps:
            record = step.to_dict()
            record["path_id"] = self.path_id
            records.append(record)
        return records

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "GeneratedPath.to_dataframe() requires pandas; "
                "install market-wave[dataframe] to use it"
            ) from exc
        return pd.DataFrame.from_records(self.to_records())


def generate_paths(
    n_paths: int,
    horizon: int,
    config_sampler: ConfigSampler | MarketConfig | Market | None = None,
    include_hidden: bool = False,
    as_iterator: bool = False,
) -> list[GeneratedPath] | Iterator[GeneratedPath]:
    """Generate one or more independent market paths.

    ``config_sampler`` may be ``None``, a ``dict`` of ``Market`` keyword arguments,
    a callable returning such a ``dict``, or a callable returning a prebuilt ``Market``.
    Callable samplers may accept either no arguments or the integer path id.
    """

    if n_paths < 0:
        raise ValueError("n_paths must be non-negative")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if isinstance(config_sampler, Market) and n_paths != 1:
        raise ValueError("a prebuilt Market can only be used for one generated path")

    paths = (
        _generate_path(path_id, horizon, config_sampler, include_hidden)
        for path_id in range(n_paths)
    )
    if as_iterator:
        return paths
    return list(paths)


def _generate_path(
    path_id: int,
    horizon: int,
    config_sampler: ConfigSampler | MarketConfig | None,
    include_hidden: bool,
) -> GeneratedPath:
    market, config = _market_from_sampler(path_id, config_sampler)
    initial_price = market.state.price
    hidden_states = [market.state] if include_hidden else None
    steps: list[StepInfo] = []

    for step in market.step(horizon, keep_history=False):
        steps.append(step)
        if hidden_states is not None:
            hidden_states.append(market.state)

    metadata = GenerationMetadata(
        path_id=path_id,
        horizon=horizon,
        config=_stable_payload(config),
        config_hash=_config_hash(config),
        version=_package_version(),
        seed=market.seed,
        regime=market.regime,
        augmentation_strength=market.augmentation_strength,
        initial_price=initial_price,
        final_price=market.state.price,
        step_count=len(steps),
        include_hidden=include_hidden,
    )
    return GeneratedPath(
        path_id=path_id,
        steps=tuple(steps),
        metadata=metadata,
        hidden_states=tuple(hidden_states) if hidden_states is not None else None,
    )


def _market_from_sampler(
    path_id: int,
    config_sampler: ConfigSampler | MarketConfig | Market | None,
) -> tuple[Market, dict[str, Any]]:
    sampled = _sample_config(path_id, config_sampler)
    if isinstance(sampled, Market):
        return sampled, _config_from_market(sampled)
    if sampled is None:
        config = _default_config(path_id)
    elif isinstance(sampled, Mapping):
        config = dict(sampled)
        if config_sampler is None or isinstance(config_sampler, Mapping):
            config = _path_config(config, path_id)
    else:
        raise TypeError("config_sampler must return a Market, a mapping, or None")

    try:
        market = Market(**config)
    except TypeError as exc:
        raise TypeError("config_sampler returned invalid Market keyword arguments") from exc
    return market, dict(config)


def _sample_config(
    path_id: int,
    config_sampler: ConfigSampler | MarketConfig | Market | None,
) -> Market | MarketConfig | None:
    if config_sampler is None or isinstance(config_sampler, Mapping):
        return config_sampler
    if isinstance(config_sampler, Market):
        return config_sampler
    if not callable(config_sampler):
        raise TypeError("config_sampler must be a mapping, callable, Market, or None")

    try:
        signature = inspect.signature(config_sampler)
    except (TypeError, ValueError):
        return config_sampler(path_id)

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is parameter.empty
    ]
    if positional:
        return config_sampler(path_id)
    return config_sampler()


def _default_config(path_id: int) -> dict[str, Any]:
    return {
        "initial_price": 100.0,
        "gap": 1.0,
        "popularity": 1.0,
        "seed": path_id,
        "grid_radius": 20,
    }


def _path_config(config: dict[str, Any], path_id: int) -> dict[str, Any]:
    config = dict(config)
    seed = config.get("seed")
    if seed is None:
        config["seed"] = path_id
    elif isinstance(seed, int) and not isinstance(seed, bool):
        config["seed"] = seed + path_id
    return config


def _config_from_market(market: Market) -> dict[str, Any]:
    return {
        "initial_price": market.history[0].price_before if market.history else market.state.price,
        "gap": market.gap,
        "popularity": market.popularity,
        "seed": market.seed,
        "grid_radius": market.grid_radius,
        "regime": market.regime,
        "augmentation_strength": market.augmentation_strength,
    }


def _config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(_stable_payload(config), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def _stable_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _stable_payload(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_payload(item) for item in value]
    if is_dataclass(value):
        payload = _stable_payload(asdict(value))
        payload["__class__"] = f"{type(value).__module__}.{type(value).__qualname__}"
        return payload
    if hasattr(value, "__dict__"):
        public = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
        return {
            "__class__": f"{type(value).__module__}.{type(value).__qualname__}",
            "attrs": _stable_payload(public),
        }
    slots = getattr(value, "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    if slots:
        public = {
            key: getattr(value, key)
            for key in slots
            if not key.startswith("_") and hasattr(value, key)
        }
        return {
            "__class__": f"{type(value).__module__}.{type(value).__qualname__}",
            "attrs": _stable_payload(public),
        }
    return {"__class__": f"{type(value).__module__}.{type(value).__qualname__}"}


def _package_version() -> str:
    try:
        return version("market-wave")
    except PackageNotFoundError:
        return "0.0.0+local"
