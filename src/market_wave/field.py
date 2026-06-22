from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from .book import (
    BUY,
    SELL,
    BookState,
    MarketMessage,
    Trade,
    available_to_take,
    cancel_candidates,
    live_orders,
    mid_tick,
    price_depth,
    public_best_ask_tick,
    public_best_bid_tick,
    replace_candidates,
    side_depth,
    spread_tick,
    visible_price_levels,
)
from .theta import LatentState, WorldTheta, bounded_smooth_projection


@dataclass(frozen=True)
class RollingState:
    recent_trade_quantity: int = 0
    recent_signed_trade_quantity: int = 0
    recent_trade_count: int = 0

    def update(self, trades: list[Trade]) -> "RollingState":
        quantity = sum(t.quantity_unit for t in trades)
        signed = sum(t.quantity_unit if t.side == BUY else -t.quantity_unit for t in trades)
        return RollingState(
            recent_trade_quantity=max(0, int(self.recent_trade_quantity * 0.86) + quantity),
            recent_signed_trade_quantity=int(self.recent_signed_trade_quantity * 0.86) + signed,
            recent_trade_count=max(0, int(self.recent_trade_count * 0.86) + len(trades)),
        )


@dataclass(frozen=True)
class BookObservation:
    spread_ticks: int
    mid_dislocation: float
    top_bid_depth: int
    top_ask_depth: int
    total_bid_depth: int
    total_ask_depth: int
    imbalance: float
    depth_pressure: float
    recent_signed_flow: float
    recent_activity: float


@dataclass(frozen=True)
class EventField:
    buy_probability: float
    limit_weight: float
    take_weight: float
    cancel_weight: float
    replace_weight: float
    sweep_probability: float
    inside_spread_probability: float
    join_touch_probability: float
    near_touch_cancel_probability: float
    price_depth_scale: float
    size_scale: float
    clock_scale_ms: float


@dataclass(frozen=True)
class Feedback:
    mid_move: int
    spread_move: int
    signed_trade_quantity: int
    trade_quantity: int
    depth_imbalance_after: float
    accepted_quantity: int
    canceled_quantity: int


@dataclass(frozen=True)
class MarketInnovation:
    message: MarketMessage
    latent_shock: LatentState
    time_delta: timedelta


def observe_book(book: BookState, rolling: RollingState, theta: WorldTheta) -> BookObservation:
    bid = public_best_bid_tick(book)
    ask = public_best_ask_tick(book)
    total_bid = side_depth(book, BUY)
    total_ask = side_depth(book, SELL)
    total = max(1, total_bid + total_ask)
    reference_depth = max(1, 2 * theta.initial_book_levels * theta.initial_orders_per_level * theta.base_order_quantity)
    return BookObservation(
        spread_ticks=spread_tick(book),
        mid_dislocation=(mid_tick(book) - theta.initial_mid_tick) / max(1, theta.initial_book_levels),
        top_bid_depth=price_depth(book, BUY, bid),
        top_ask_depth=price_depth(book, SELL, ask),
        total_bid_depth=total_bid,
        total_ask_depth=total_ask,
        imbalance=(total_bid - total_ask) / total,
        depth_pressure=(total_bid + total_ask) / reference_depth - 1.0,
        recent_signed_flow=rolling.recent_signed_trade_quantity / max(1, theta.max_quantity_unit * 10),
        recent_activity=min(4.0, rolling.recent_trade_count / 8.0),
    )


def compute_event_field(z: LatentState, obs: BookObservation, theta: WorldTheta) -> EventField:
    spread_gap = max(0, obs.spread_ticks - theta.minimum_spread_ticks)
    high_depth = max(0.0, obs.depth_pressure)
    low_depth = max(0.0, -obs.depth_pressure)
    shallow_top = 1.0 / max(1.0, min(obs.top_bid_depth, obs.top_ask_depth) / max(1, theta.base_order_quantity))
    flow = z.order_flow + 0.55 * z.trend_pressure - 0.50 * obs.imbalance + 0.18 * obs.recent_signed_flow
    buy_probability = _sigmoid(flow)
    limit_weight = theta.base_limit_weight * (1.0 + 0.70 * spread_gap + 0.75 * low_depth + 0.30 * shallow_top + max(0.0, z.replenishment_pressure)) / (1.0 + 1.20 * high_depth)
    take_weight = theta.base_take_weight * (1.20 + 0.70 * abs(z.trend_pressure) + 0.65 * max(0.0, z.sweep_intensity) + 0.45 * max(0.0, z.volatility) + 0.20 * obs.recent_activity)
    cancel_weight = theta.base_cancel_weight * (0.85 + 1.35 * high_depth + 0.80 * max(0.0, z.cancel_pressure) + 0.25 * max(0.0, z.volatility))
    replace_weight = theta.base_replace_weight * (0.80 + 0.30 * spread_gap + 0.30 * max(0.0, z.activity))
    sweep_probability = float(np.clip(0.07 + 0.10 * max(0.0, z.sweep_intensity) + 0.08 * abs(z.trend_pressure) + 0.05 * high_depth + 0.03 * obs.recent_activity, 0.03, 0.52))
    inside_spread_probability = float(np.clip(0.30 + 0.12 * spread_gap + 0.18 * max(0.0, z.replenishment_pressure), 0.20, 0.88))
    join_touch_probability = float(np.clip(0.58 - 0.16 * max(0.0, z.volatility) + 0.20 * low_depth, 0.25, 0.78))
    near_touch_cancel_probability = float(np.clip(0.52 + 0.28 * high_depth + 0.15 * max(0.0, z.cancel_pressure), 0.35, 0.90))
    return EventField(
        buy_probability=buy_probability,
        limit_weight=max(0.01, limit_weight),
        take_weight=max(0.01, take_weight),
        cancel_weight=max(0.01, cancel_weight),
        replace_weight=max(0.01, replace_weight),
        sweep_probability=sweep_probability,
        inside_spread_probability=inside_spread_probability,
        join_touch_probability=join_touch_probability,
        near_touch_cancel_probability=near_touch_cancel_probability,
        price_depth_scale=max(1.0, 1.0 + 1.7 * abs(z.volatility) + 0.35 * spread_gap + 0.4 * max(0.0, z.sweep_intensity)),
        size_scale=max(1.0, 1.0 + z.activity + 0.5 * max(0.0, z.volatility)),
        clock_scale_ms=max(15.0, 90.0 / max(0.2, theta.base_activity + z.activity)),
    )


def sample_market_message(
    field: EventField,
    book: BookState,
    rng: np.random.Generator,
    theta: WorldTheta,
) -> MarketInnovation:
    kind = _sample_kind(field, book, rng)
    side = BUY if rng.random() < field.buy_probability else SELL
    if kind == "LIMIT":
        message = _sample_limit_message(field, book, rng, theta, side)
    elif kind == "TAKE":
        message = _sample_take_message(field, book, rng, theta, side)
    elif kind == "CANCEL":
        message = _sample_cancel_message(book, rng, theta, side)
    elif kind == "REPLACE":
        message = _sample_replace_message(book, rng, theta, side)
    else:
        raise ValueError(f"unknown sampled kind: {kind}")
    latent_shock = LatentState(
        liquidity=float(rng.normal(0.0, 0.025)),
        volatility=float(rng.normal(0.0, 0.035)),
        spread_pressure=float(rng.normal(0.0, 0.025)),
        order_flow=float(rng.normal(0.0, 0.035)),
        activity=float(rng.normal(0.0, 0.025)),
        trend_pressure=float(rng.normal(0.0, 0.020)),
        sweep_intensity=float(rng.normal(0.0, 0.030)),
        cancel_pressure=float(rng.normal(0.0, 0.025)),
        replenishment_pressure=float(rng.normal(0.0, 0.025)),
        shock=float(rng.normal(0.0, 0.015)),
    )
    milliseconds = int(min(theta.max_clock_ms, max(1, rng.exponential(field.clock_scale_ms))))
    return MarketInnovation(message=message, latent_shock=latent_shock, time_delta=timedelta(milliseconds=milliseconds))


def encode_feedback(
    old_book: BookState,
    new_book: BookState,
    message: MarketMessage,
    trades: list[Trade],
    theta: WorldTheta,
) -> Feedback:
    accepted = message.quantity if message.kind == "LIMIT" else 0
    canceled = message.quantity if message.kind == "CANCEL" else 0
    total_bid = side_depth(new_book, BUY)
    total_ask = side_depth(new_book, SELL)
    return Feedback(
        mid_move=mid_tick(new_book) - mid_tick(old_book),
        spread_move=spread_tick(new_book) - spread_tick(old_book),
        signed_trade_quantity=sum(t.quantity_unit if t.side == BUY else -t.quantity_unit for t in trades),
        trade_quantity=sum(t.quantity_unit for t in trades),
        depth_imbalance_after=(total_bid - total_ask) / max(1, total_bid + total_ask),
        accepted_quantity=accepted,
        canceled_quantity=canceled,
    )


def update_latent_field(
    z: LatentState,
    feedback: Feedback,
    message: MarketMessage,
    latent_shock: LatentState,
    theta: WorldTheta,
) -> LatentState:
    decay = theta.feedback_decay
    trade_pressure = feedback.signed_trade_quantity / max(1, theta.max_quantity_unit)
    activity_pressure = min(2.0, (feedback.trade_quantity + feedback.accepted_quantity + feedback.canceled_quantity) / max(1, theta.max_quantity_unit))
    directional_move = float(np.clip(feedback.mid_move, -3, 3)) / 3.0
    depth_pull = feedback.canceled_quantity / max(1, theta.max_quantity_unit)
    depth_replenish = feedback.accepted_quantity / max(1, theta.max_quantity_unit)
    next_state = LatentState(
        liquidity=decay * z.liquidity + 0.14 * (depth_replenish - depth_pull) + latent_shock.liquidity,
        volatility=decay * z.volatility + 0.16 * abs(feedback.mid_move) + 0.06 * feedback.trade_quantity / max(1, theta.max_quantity_unit) + latent_shock.volatility,
        spread_pressure=decay * z.spread_pressure + 0.30 * feedback.spread_move + latent_shock.spread_pressure,
        order_flow=decay * z.order_flow + 0.35 * trade_pressure + 0.10 * feedback.depth_imbalance_after + latent_shock.order_flow,
        activity=decay * z.activity + 0.12 * activity_pressure + latent_shock.activity,
        trend_pressure=decay * z.trend_pressure + 0.22 * directional_move + 0.12 * trade_pressure + latent_shock.trend_pressure,
        sweep_intensity=decay * z.sweep_intensity + 0.10 * abs(feedback.mid_move) + 0.10 * activity_pressure + latent_shock.sweep_intensity,
        cancel_pressure=decay * z.cancel_pressure + 0.12 * depth_pull - 0.05 * depth_replenish + latent_shock.cancel_pressure,
        replenishment_pressure=decay * z.replenishment_pressure + 0.12 * max(0, feedback.spread_move) + 0.08 * depth_replenish - 0.04 * depth_pull + latent_shock.replenishment_pressure,
        shock=decay * z.shock + latent_shock.shock,
    )
    return LatentState(
        liquidity=bounded_smooth_projection(next_state.liquidity, theta.latent_bound),
        volatility=bounded_smooth_projection(next_state.volatility, theta.latent_bound),
        spread_pressure=bounded_smooth_projection(next_state.spread_pressure, theta.latent_bound),
        order_flow=bounded_smooth_projection(next_state.order_flow, theta.latent_bound),
        activity=bounded_smooth_projection(next_state.activity, theta.latent_bound),
        trend_pressure=bounded_smooth_projection(next_state.trend_pressure, theta.latent_bound),
        sweep_intensity=bounded_smooth_projection(next_state.sweep_intensity, theta.latent_bound),
        cancel_pressure=bounded_smooth_projection(next_state.cancel_pressure, theta.latent_bound),
        replenishment_pressure=bounded_smooth_projection(next_state.replenishment_pressure, theta.latent_bound),
        shock=bounded_smooth_projection(next_state.shock, theta.latent_bound),
    )


def _sample_kind(field: EventField, book: BookState, rng: np.random.Generator) -> str:
    weights = {
        "LIMIT": field.limit_weight,
        "TAKE": field.take_weight,
        "CANCEL": field.cancel_weight if live_orders(book) else 0.0,
        "REPLACE": field.replace_weight if live_orders(book) else 0.0,
    }
    total = sum(weights.values())
    if total <= 0:
        return "LIMIT"
    draw = rng.random() * total
    running = 0.0
    for kind, weight in weights.items():
        running += weight
        if draw <= running:
            return kind
    return "LIMIT"


def _sample_limit_message(field: EventField, book: BookState, rng: np.random.Generator, theta: WorldTheta, side: str) -> MarketMessage:
    spread = spread_tick(book)
    qty = _sample_quantity(field, rng, theta)
    if side == BUY:
        if spread > 1 and rng.random() < field.inside_spread_probability:
            price = public_best_ask_tick(book) - 1
        elif rng.random() < field.join_touch_probability:
            price = public_best_bid_tick(book)
        else:
            price = public_best_bid_tick(book) - _passive_offset(field, rng)
    else:
        if spread > 1 and rng.random() < field.inside_spread_probability:
            price = public_best_bid_tick(book) + 1
        elif rng.random() < field.join_touch_probability:
            price = public_best_ask_tick(book)
        else:
            price = public_best_ask_tick(book) + _passive_offset(field, rng)
    return MarketMessage("LIMIT", side, int(price), qty)


def _sample_take_message(field: EventField, book: BookState, rng: np.random.Generator, theta: WorldTheta, side: str) -> MarketMessage:
    sweep = rng.random() < field.sweep_probability
    max_extra = max(0, int(round(field.price_depth_scale)))
    extra = int(rng.integers(0, max_extra + 1)) if sweep else 0
    limit = public_best_ask_tick(book) + extra if side == BUY else public_best_bid_tick(book) - extra
    levels = visible_price_levels(book, SELL if side == BUY else BUY, extra + 1)
    available = available_to_take(book, side, limit)
    if available < theta.min_quantity_unit:
        return _sample_limit_message(field, book, rng, theta, side)
    touch_depth = levels[0][1] if levels else theta.min_quantity_unit
    if sweep:
        fill_ratio = float(rng.uniform(0.85, 1.15))
        qty = min(available, max(theta.min_quantity_unit, int(available * fill_ratio)))
    else:
        fill_ratio = float(rng.uniform(0.35, 1.05))
        qty = min(available, max(theta.min_quantity_unit, int(touch_depth * fill_ratio)))
    return MarketMessage("TAKE", side, int(limit), int(qty))


def _sample_cancel_message(book: BookState, rng: np.random.Generator, theta: WorldTheta, side: str) -> MarketMessage:
    candidates = cancel_candidates(book, side) or cancel_candidates(book, SELL if side == BUY else BUY)
    if not candidates:
        field = EventField(0.5, 1.0, 0.0, 0.0, 0.0, 0.08, 0.40, 0.60, 0.60, 1.0, 1.0, 90.0)
        return _sample_limit_message(field, book, rng, theta, side)
    order = _weighted_order_choice(candidates, book, rng)
    qty = int(rng.integers(theta.min_quantity_unit, max(theta.min_quantity_unit + 1, order.remaining_qty + 1)))
    return MarketMessage("CANCEL", order.side, order.price_tick, min(qty, order.remaining_qty), order_id=order.order_id)


def _sample_replace_message(book: BookState, rng: np.random.Generator, theta: WorldTheta, side: str) -> MarketMessage:
    candidates = replace_candidates(book, side) or replace_candidates(book, SELL if side == BUY else BUY)
    if not candidates:
        return _sample_cancel_message(book, rng, theta, side)
    order = _weighted_order_choice(candidates, book, rng)
    shift = int(rng.integers(1, max(2, int(_replace_depth(theta)) + 1)))
    if order.side == BUY:
        if rng.random() < 0.68:
            new_price = min(order.price_tick + shift, public_best_ask_tick(book) - 1)
        else:
            new_price = order.price_tick - shift
    else:
        if rng.random() < 0.68:
            new_price = max(order.price_tick - shift, public_best_bid_tick(book) + 1)
        else:
            new_price = order.price_tick + shift
    if new_price == order.price_tick:
        return _sample_cancel_message(book, rng, theta, order.side)
    new_qty = int(np.clip(order.remaining_qty + rng.integers(-theta.base_order_quantity // 2, theta.base_order_quantity // 2 + 1), theta.min_quantity_unit, theta.max_quantity_unit))
    return MarketMessage("REPLACE", order.side, order.price_tick, order.remaining_qty, order_id=order.order_id, new_price_tick=int(new_price), new_quantity=new_qty)


def _sample_quantity(field: EventField, rng: np.random.Generator, theta: WorldTheta) -> int:
    mean = max(theta.min_quantity_unit, theta.base_order_quantity * field.size_scale)
    qty = int(rng.gamma(shape=1.65, scale=mean / 1.65))
    return int(np.clip(qty, theta.min_quantity_unit, theta.max_quantity_unit))


def _passive_offset(field: EventField, rng: np.random.Generator) -> int:
    scale = max(1.0, field.price_depth_scale)
    return max(1, int(rng.geometric(1.0 / (1.0 + scale))))


def _weighted_order_choice(candidates, book: BookState, rng: np.random.Generator):
    weights = []
    for order in candidates:
        best = public_best_bid_tick(book) if order.side == BUY else public_best_ask_tick(book)
        distance = abs(order.price_tick - best)
        weights.append(1.0 / (1.0 + distance))
    probs = np.array(weights, dtype=float)
    probs = probs / probs.sum()
    return candidates[int(rng.choice(len(candidates), p=probs))]


def _replace_depth(theta: WorldTheta) -> int:
    return max(2, min(5, theta.initial_book_levels // 18))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = np.exp(-value)
        return float(1.0 / (1.0 + z))
    z = np.exp(value)
    return float(z / (1.0 + z))
