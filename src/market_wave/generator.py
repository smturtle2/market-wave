from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .book import BookState, PublicEvent, assert_book_invariants, init_book, live_orders, public_book_projection
from .config import GenerationConfig
from .corpus import (
    CorpusCounters,
    CorpusSink,
    make_snapshot_record,
    make_visible_market_record,
    public_manifest,
)
from .field import (
    RollingState,
    compute_event_field,
    encode_feedback,
    observe_book,
    sample_market_message,
    update_latent_field,
)
from .physics import apply_book_physics
from .theta import WorldTheta, init_latent_field, sample_world_theta
from .units import MarketUnits
from .validation import minimal_validation_passes


def run_world_streaming(config: GenerationConfig, output_root: Path, *, force: bool = False) -> dict[str, Any]:
    config.validate()
    rng = np.random.default_rng(config.seed)
    theta = sample_world_theta(rng, config)
    world_id = _world_id(config)
    units = MarketUnits.from_strings(theta.tick_size_decimal, theta.quantity_unit_decimal)
    z = init_latent_field(rng, theta)
    book = init_book(rng, theta)
    timestamp = datetime.fromisoformat(theta.start_timestamp)
    counters = CorpusCounters()
    rolling = RollingState()
    sink = CorpusSink(Path(output_root), force=force)
    sink.write_theta_manifest(_theta_manifest(theta))
    sink.write_run_manifest(_run_manifest(config, theta, world_id))

    try:
        timestamp_s = _format_timestamp(timestamp)
        initial_projection = public_book_projection(book, theta, levels=theta.initial_book_levels)
        sink.write_visible_snapshot(
            make_snapshot_record(
                snapshot_id=f"{world_id}_initial_snap",
                step_id=f"{world_id}_initial",
                world_id=world_id,
                timestamp=timestamp_s,
                currency=theta.currency,
                projection=initial_projection,
            )
        )
        counters.snapshot_records += 1

        for step in range(theta.steps):
            step_id = _step_id(world_id, step)
            old_book = book.copy()
            obs = observe_book(book, rolling, theta)
            field = compute_event_field(z, obs, theta)
            innovation = sample_market_message(field, book, rng, theta)
            book, trades, public_events = apply_book_physics(book, innovation.message, theta)
            feedback = encode_feedback(old_book, book, innovation.message, trades, theta)
            z = update_latent_field(z, feedback, innovation.message, innovation.latent_shock, theta)
            timestamp = timestamp + innovation.time_delta
            timestamp_s = _format_timestamp(timestamp)

            for record in _visible_records(
                events=public_events,
                step_id=step_id,
                world_id=world_id,
                timestamp=timestamp_s,
                currency=theta.currency,
                units=units,
            ):
                sink.write_visible_market(record)
                counters.visible_records += 1
                if record["record_type"] == "TRADE":
                    counters.trade_records += 1
                if record["record_type"] == "BOOK_DELTA":
                    counters.book_update_records += 1

            if (step + 1) % theta.snapshot_every == 0:
                projection = public_book_projection(book, theta, levels=theta.initial_book_levels)
                snapshot = make_snapshot_record(
                    snapshot_id=f"{step_id}_snap",
                    step_id=step_id,
                    world_id=world_id,
                    timestamp=timestamp_s,
                    currency=theta.currency,
                    projection=projection,
                )
                sink.write_visible_snapshot(snapshot)
                counters.snapshot_records += 1

            sink.write_hidden(_hidden_record(step_id, world_id, timestamp_s, z, field, innovation, theta))
            counters.hidden_records += 1
            rolling = rolling.update(trades)
            _update_health_counters(counters, book, z)

        counters.dead_world = counters.visible_records == 0 or counters.trade_records == 0 or counters.snapshot_records == 0
        if minimal_validation_passes(counters):
            sink.write_public_manifest(
                public_manifest(
                    world_id=world_id,
                    currency=theta.currency,
                    tick_size_decimal=theta.tick_size_decimal,
                    quantity_unit_decimal=theta.quantity_unit_decimal,
                    start_timestamp=theta.start_timestamp,
                    counters=counters,
                )
            )
            sink.commit()
        else:
            sink.discard()
            raise RuntimeError(f"generated world failed structural validation: {counters.to_internal_dict()}")
    except Exception:
        sink.discard()
        raise
    return {"world_id": world_id, "counters": counters.to_internal_dict()}


def _visible_records(
    *,
    events: list[PublicEvent],
    step_id: str,
    world_id: str,
    timestamp: str,
    currency: str,
    units: MarketUnits,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    local = 0
    for ev in events:
        event_id = f"{step_id}_e{local:02d}"
        records.append(
            make_visible_market_record(
                event_id=event_id,
                step_id=step_id,
                world_id=world_id,
                timestamp=timestamp,
                record_type=ev.record_type,
                payload=_public_payload(ev.payload, units, currency),
            )
        )
        local += 1
    return records


def _public_payload(payload: dict[str, object], units: MarketUnits, currency: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in payload.items():
        if key == "price_tick":
            out["price"] = units.price(int(value))
        elif key == "old_price_tick":
            out["old_price"] = units.price(int(value))
        elif key == "new_price_tick":
            out["new_price"] = units.price(int(value))
        elif key == "quantity":
            out["volume"] = units.quantity(int(value))
        elif key == "old_quantity":
            out["old_volume"] = units.quantity(int(value))
        elif key == "new_quantity":
            out["new_volume"] = units.quantity(int(value))
        elif key == "delta":
            out["volume_delta"] = units.quantity(int(value))
        elif key == "after":
            out["volume_after"] = units.quantity(int(value))
        else:
            out[key] = value
    out["currency"] = currency
    return out


def _hidden_record(step_id: str, world_id: str, timestamp: str, z: Any, field: Any, innovation: Any, theta: WorldTheta) -> dict[str, Any]:
    message = innovation.message
    return {
        "step_id": step_id,
        "world_id": world_id,
        "timestamp": timestamp,
        "z_summary": [_float_string(v) for v in z.as_tuple()],
        "event_field_summary": {
            "buy_probability": _float_string(field.buy_probability),
            "limit_weight": _float_string(field.limit_weight),
            "take_weight": _float_string(field.take_weight),
            "cancel_weight": _float_string(field.cancel_weight),
            "replace_weight": _float_string(field.replace_weight),
        },
        "innovation_summary": {
            "kind": message.kind,
            "side": message.side,
            "price_tick": message.price_tick,
            "quantity": message.quantity,
            "time_delta_ms": int(innovation.time_delta.total_seconds() * 1000),
        },
        "theta_id": theta.id,
    }


def _theta_manifest(theta: WorldTheta) -> dict[str, Any]:
    return {
        **theta.matrix_summary,
        "currency": theta.currency,
        "units": {
            "tick_size_decimal": theta.tick_size_decimal,
            "quantity_unit_decimal": theta.quantity_unit_decimal,
            "min_quantity_unit": theta.min_quantity_unit,
            "max_quantity_unit": theta.max_quantity_unit,
        },
        "book": {
            "initial_mid_tick": theta.initial_mid_tick,
            "initial_book_levels": theta.initial_book_levels,
            "initial_orders_per_level": theta.initial_orders_per_level,
            "minimum_spread_ticks": theta.minimum_spread_ticks,
        },
    }


def _run_manifest(config: GenerationConfig, theta: WorldTheta, world_id: str) -> dict[str, Any]:
    return {
        "world_id": world_id,
        "theta_id": theta.id,
        "seed": config.seed,
        "steps": config.steps,
        "rng": "numpy.PCG64.default_rng",
        "runtime_randomness_owner": "sample_market_message",
    }


def _update_health_counters(counters: CorpusCounters, book: BookState, z: Any) -> None:
    try:
        assert_book_invariants(book)
    except AssertionError:
        counters.book_invariant_violations += 1
    if not np.all(np.isfinite(z.as_tuple())):
        counters.exploded_world = True
    if sum(order.remaining_qty for order in live_orders(book)) > 100_000_000:
        counters.exploded_world = True


def _step_id(world_id: str, step: int) -> str:
    return f"{world_id}_s{step:08d}"


def _world_id(config: GenerationConfig) -> str:
    if config.world_id is not None:
        return config.world_id
    return f"w{config.world_index:016d}"


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def _float_string(value: float) -> str:
    return f"{float(value):.6f}"
