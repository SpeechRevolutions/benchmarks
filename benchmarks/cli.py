#!/usr/bin/env python3
"""
Unified entry point for the public benchmark suite.

Examples
--------
    # List available benchmarks and providers
    python -m benchmarks.cli list

    # Run one benchmark against the local Speech Revolutions stack
    python -m benchmarks.cli run wer

    # Run the whole suite, compare to checked-in baselines, write all reports
    python -m benchmarks.cli run all

    # Freeze the current results as the regression baseline
    python -m benchmarks.cli run wer --capture-baseline

    # Choose output formats / provider
    python -m benchmarks.cli run diarization --provider speech_revolutions \\
        --formats json md csv console

Exit code is non-zero if any benchmark regresses against its baseline (unless
--capture-baseline or --no-check).
"""

from __future__ import annotations

import argparse
import sys

from .benchmarks import available_benchmarks, get_benchmark
from .providers.registry import (
    PLANNED_PROVIDERS,
    available_providers,
    get_provider,
)


def _cmd_list(_args: argparse.Namespace) -> int:
    print("Benchmarks:")
    for b in available_benchmarks():
        print(f"  - {b}")
    print("\nProviders (implemented):")
    for p in available_providers():
        print(f"  - {p}")
    print("\nProviders (planned, not yet implemented):")
    for p in PLANNED_PROVIDERS:
        print(f"  - {p}")
    return 0


def _finish_one(bench, results: dict, provider_name: str, formats: tuple[str, ...],
                capture: bool, check: bool) -> list[str]:
    """Given already-computed results, capture/compare and write reports."""
    regressions: list[str] = []

    if capture:
        path = bench.capture_baseline(results, provider_name)
        if path:
            print(f"  baseline captured -> {path}")
    elif check:
        regressions = bench.compare(results, provider_name)
    bench.generate_report(results, provider_name, formats=formats, regressions=regressions)
    return regressions


def _run_one(name: str, provider, formats: tuple[str, ...],
             capture: bool, check: bool, use_cache: bool = True) -> list[str]:
    bench = get_benchmark(name)
    print(f"\n########## {name} ##########")
    results = bench.evaluate(provider, use_cache=use_cache)
    return _finish_one(bench, results, provider.name, formats, capture, check)


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        provider = get_provider(args.provider)
    except (KeyError, NotImplementedError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.benchmark != "all" and args.benchmark not in available_benchmarks():
        print(f"ERROR: unknown benchmark '{args.benchmark}'. "
              f"Available: {available_benchmarks()} or 'all'", file=sys.stderr)
        return 2

    formats = tuple(args.formats)
    all_regressions: dict[str, list[str]] = {}

    if args.benchmark == "all":
        # Suite run with cross-benchmark transcript de-duplication (transcribe
        # each unique file once, share to all benchmarks). Halves provider cost.
        from .benchmarks import get_benchmark as _gb
        from .core.suite import run_suite
        benches = [_gb(n) for n in available_benchmarks()]
        results_map = run_suite(provider, benches, use_cache=not args.refresh)
        for b in benches:
            print(f"\n########## {b.name} ##########")
            regs = _finish_one(b, results_map[b.name], provider.name, formats,
                               args.capture_baseline, not args.no_check)
            if regs:
                all_regressions[b.name] = regs
    else:
        try:
            regs = _run_one(args.benchmark, provider, formats,
                            args.capture_baseline, not args.no_check,
                            use_cache=not args.refresh)
            if regs:
                all_regressions[args.benchmark] = regs
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR running {args.benchmark}: {e}", file=sys.stderr)
            all_regressions[args.benchmark] = [f"crashed: {e}"]

    print("\n========== SUITE SUMMARY ==========")
    if all_regressions:
        for name, regs in all_regressions.items():
            print(f"  FAIL {name}: {len(regs)} regression(s)")
        return 1
    print("  all benchmarks within baseline tolerance" if not args.capture_baseline
          else "  baselines captured")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="benchmarks", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List benchmarks and providers")

    run = sub.add_parser("run", help="Run a benchmark (or 'all')")
    run.add_argument("benchmark", help="benchmark name or 'all'")
    run.add_argument("--provider", default="speech_revolutions",
                     help="provider name (default: speech_revolutions / local)")
    run.add_argument("--capture-baseline", action="store_true",
                     help="freeze current results as the baseline instead of comparing")
    run.add_argument("--no-check", action="store_true",
                     help="skip baseline comparison (still writes reports)")
    run.add_argument("--refresh", action="store_true",
                     help="force fresh transcription, ignoring the raw-transcript "
                          "cache (external providers only; Zephyr always re-transcribes)")
    run.add_argument("--formats", nargs="+", default=["json", "md", "csv", "console"],
                     choices=["json", "md", "csv", "console"],
                     help="output formats to emit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return _cmd_list(args)
    if args.command == "run":
        return _cmd_run(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
