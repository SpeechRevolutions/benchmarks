"""
Benchmark 8 — Price / Performance.

Reuses the per-file timing that ``Provider.transcribe_batch`` already records
(submit + processing latency) plus the wall-clock of the whole batch (captured
by the overridden ``run``) to derive throughput.

Output: rtf, hours_per_gpu_hour, hours_per_dollar, average_latency,
p95_latency, p99_latency.

Cost inputs (env, optional — local self-hosted has no per-hour list price):
  PRICE_N_GPUS            number of GPUs serving the batch (default 1)
  PRICE_GPU_USD_PER_HOUR  $/GPU-hour, used for hours_per_dollar on self-hosted
  (a provider may instead set price_per_audio_hour_usd for API pricing)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .. import config
from ..core.benchmark import Benchmark
from ..core.manifest import items_for_provider, load_manifest
from ..providers.base import Features, Provider, TranscriptionResult


@dataclass
class PriceBenchmark(Benchmark):
    name: str = "price"
    features: Features = field(default_factory=lambda: Features(
        word_timestamps=False, speaker_labels=False, punctuation=True))
    subsets: dict[str, str] = field(default_factory=lambda: {"all": "price_all"})
    regression_specs: dict = field(default_factory=lambda: {
        "rtf": ("lower", 0.05),
        "p95_latency": ("lower", 2.0),
    })
    dedup_safe: bool = False  # must transcribe independently to measure real timing

    _wall_s: float = 0.0
    _price_per_audio_hour: float | None = None

    def run(self, provider: Provider, *, progress: bool = True,
            use_cache: bool = True) -> dict[str, list[TranscriptionResult]]:
        # use_cache is ignored: price MUST transcribe for real to measure latency.
        path = config.manifest_path(self.name, self.subsets["all"])
        try:
            entries = load_manifest(path)
        except FileNotFoundError:
            if progress:
                print(f"  [skip] {self.name}: no manifest at {path}")
            return {}
        items = items_for_provider(entries)
        if not items:
            return {"all": []}
        self._price_per_audio_hour = getattr(provider, "price_per_audio_hour_usd", None)
        if progress:
            print(f"\n[{self.name}] timing {len(items)} files ...")
        t0 = time.monotonic()
        results = provider.transcribe_batch(items, self.features, progress=progress,
                                            use_cache=False)
        self._wall_s = time.monotonic() - t0
        return {"all": results}

    def score(self, runs: dict[str, list[TranscriptionResult]], *, provider_name: str) -> dict:
        results = [r for sub in runs.values() for r in sub]
        ok = [r for r in results if r.ok]
        per_file = [{
            "id": r.entry_id,
            "audio_duration_s": round(r.audio_duration_s, 2),
            "total_latency_s": round(r.total_latency_s, 3),
            "submit_latency_s": round(r.submit_latency_s, 3),
            "rtf": round(r.rtf, 4) if r.rtf is not None else None,
            "status": r.status.value,
        } for r in results]

        # Uniform per-file processing span so sync and async providers are
        # comparable: a SYNC provider (e.g. Deepgram) does all its work inside
        # submit() so its time lands in submit_latency_s with total_latency_s~0
        # (rtf would read ~0); an ASYNC provider measures it via the poll loop in
        # total_latency_s. The sum is the end-to-end per-file time for both.
        def _span(r) -> float:
            return (r.submit_latency_s or 0.0) + (r.total_latency_s or 0.0)
        rtfs = [_span(r) / r.audio_duration_s for r in ok
                if r.audio_duration_s and _span(r) > 0]
        latencies = [_span(r) for r in ok if _span(r) > 0]
        total_audio_h = sum(r.audio_duration_s for r in ok) / 3600.0
        wall_h = self._wall_s / 3600.0

        n_gpus = float(os.environ.get("PRICE_N_GPUS", "1"))
        gpu_usd_per_hour = os.environ.get("PRICE_GPU_USD_PER_HOUR")

        hours_per_gpu_hour = (
            round(total_audio_h / (wall_h * n_gpus), 4)
            if wall_h > 0 and n_gpus > 0 else None
        )

        # hours_per_dollar: API list price if provided, else self-hosted GPU cost.
        hours_per_dollar = None
        if self._price_per_audio_hour:
            hours_per_dollar = round(1.0 / self._price_per_audio_hour, 4)
        elif gpu_usd_per_hour and wall_h > 0:
            dollars = wall_h * n_gpus * float(gpu_usd_per_hour)
            hours_per_dollar = round(total_audio_h / dollars, 4) if dollars > 0 else None

        summary = {
            "rtf": round(float(np.median(rtfs)), 4) if rtfs else None,
            "hours_per_gpu_hour": hours_per_gpu_hour,
            "hours_per_dollar": hours_per_dollar,
            "average_latency": round(float(np.mean(latencies)), 3) if latencies else None,
            "p95_latency": round(float(np.percentile(latencies, 95)), 3) if latencies else None,
            "p99_latency": round(float(np.percentile(latencies, 99)), 3) if latencies else None,
            "wall_clock_s": round(self._wall_s, 2),
            "total_audio_h": round(total_audio_h, 4),
            "n_gpus": n_gpus,
            "n_files": len(ok),
        }
        return self._base_result(provider_name, summary, per_file)
