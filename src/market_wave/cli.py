from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import generate as generate_market
from .api import open as open_market
from .api import replay as replay_market
from .corpus import read_public_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="market-wave",
        description="Generate and inspect unnamed synthetic limit-order-book markets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate a market corpus")
    generate.add_argument("output", help="output directory")
    _add_generation_args(generate)
    generate.add_argument("--force", action="store_true", help="replace existing corpus/internal/plots directories")

    validate = sub.add_parser("validate", help="validate a generated corpus")
    validate.add_argument("root", help="market directory")

    replay = sub.add_parser("replay", help="prove deterministic generation for one config")
    _add_generation_args(replay)

    plot = sub.add_parser("plot", help="write market-screen plots")
    plot.add_argument("root", help="market directory")
    plot.add_argument("--output", help="plot directory; defaults to ROOT/plots")

    price = sub.add_parser("price", help="show the latest traded price")
    price.add_argument("root", help="market directory")

    book = sub.add_parser("book", help="show the latest top-of-book ladder")
    book.add_argument("root", help="market directory")
    book.add_argument("--levels", type=int, default=10, help="visible levels per side")

    trades = sub.add_parser("trades", help="show recent trades")
    trades.add_argument("root", help="market directory")
    trades.add_argument("--limit", type=int, default=100, help="maximum rows")

    candles = sub.add_parser("candles", help="show trade candles")
    candles.add_argument("root", help="market directory")
    candles.add_argument("--interval", default="1m", help="bucket size such as 1s, 1m, or 1h")

    events = sub.add_parser("events", help="show public market events")
    events.add_argument("root", help="market directory")
    events.add_argument("--limit", type=int, default=100, help="maximum rows")
    events.add_argument("--cursor", type=int, default=0, help="zero-based event offset")

    inspect = sub.add_parser("inspect", help="show corpus manifest")
    inspect.add_argument("root", help="market directory")

    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            market = generate_market(args.output, **_generation_kwargs(args), force=args.force)
            _print_json({"result": {"root": str(market.root)}})
            return 0
        if args.command == "validate":
            report = open_market(args.root).validate()
            _print_json({"result": {"ok": report.ok, "errors": report.errors, "counters": report.counters}})
            return 0 if report.ok else 1
        if args.command == "replay":
            _print_json({"result": replay_market(**_generation_kwargs(args))})
            return 0
        if args.command == "plot":
            paths = open_market(args.root).plot(args.output)
            _print_json({"result": {"plots": [str(path) for path in paths]}})
            return 0
        if args.command == "price":
            _print_json({"result": open_market(args.root).price()})
            return 0
        if args.command == "book":
            _print_json({"result": open_market(args.root).book(args.levels)})
            return 0
        if args.command == "trades":
            _print_json({"result": {"trades": open_market(args.root).trades(args.limit)}})
            return 0
        if args.command == "candles":
            _print_json({"result": {"candles": open_market(args.root).candles(args.interval)}})
            return 0
        if args.command == "events":
            _print_json({"result": open_market(args.root).events(args.limit, args.cursor)})
            return 0
        if args.command == "inspect":
            _, _, manifest = read_public_corpus(Path(args.root))
            _print_json({"result": manifest})
            return 0
    except Exception as exc:
        _print_json({"error": {"code": "MARKET_WAVE_ERROR", "message": str(exc), "details": None}})
        return 1
    return 1


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--world-index", type=int, default=0, help="public world id index")
    parser.add_argument("--world-id", default=None, help="explicit public world id")
    parser.add_argument("--steps", type=int, default=1_000, help="number of generated message steps")
    parser.add_argument("--price", default="71800", help="initial mid price")
    parser.add_argument("--tick", default="100", help="price tick size")
    parser.add_argument("--depth", type=int, default=80, help="initial visible levels per side")
    parser.add_argument("--orders-per-level", type=int, default=3, help="initial FIFO queue count per price level")
    parser.add_argument("--start-timestamp", default="2026-06-22T09:00:00.000+09:00", help="first timestamp")
    parser.add_argument("--snapshot-every", type=int, default=1, help="snapshot interval in steps")


def _generation_kwargs(args: argparse.Namespace) -> dict:
    return {
        "seed": args.seed,
        "world_index": args.world_index,
        "world_id": args.world_id,
        "steps": args.steps,
        "price": args.price,
        "tick": args.tick,
        "depth": args.depth,
        "orders_per_level": args.orders_per_level,
        "start_timestamp": args.start_timestamp,
        "snapshot_every": args.snapshot_every,
    }


def _print_json(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    sys.exit(main())
