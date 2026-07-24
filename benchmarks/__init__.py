"""
Benchmark registry.

All eight benchmarks share the :class:`~benchmarks.core.benchmark.Benchmark`
contract: run / score / capture_baseline / compare / generate_report.
"""

from __future__ import annotations

from ..core.benchmark import Benchmark
from .diarization import DiarizationBenchmark
from .entities import EntityBenchmark
from .language_switching import LanguageSwitchingBenchmark
from .longform import LongformBenchmark
from .multilingual import MultilingualBenchmark
from .price import PriceBenchmark
from .timestamps import TimestampBenchmark
from .wer import WERBenchmark

_BENCHMARKS: dict[str, type[Benchmark]] = {
    "wer": WERBenchmark,
    "entities": EntityBenchmark,
    "diarization": DiarizationBenchmark,
    "timestamps": TimestampBenchmark,
    "multilingual": MultilingualBenchmark,
    "language_switching": LanguageSwitchingBenchmark,
    "longform": LongformBenchmark,
    "price": PriceBenchmark,
}


def get_benchmark(name: str) -> Benchmark:
    if name not in _BENCHMARKS:
        raise KeyError(f"Unknown benchmark '{name}'. Available: {available_benchmarks()}")
    return _BENCHMARKS[name]()


def available_benchmarks() -> list[str]:
    return list(_BENCHMARKS)


__all__ = ["get_benchmark", "available_benchmarks", "Benchmark"]
