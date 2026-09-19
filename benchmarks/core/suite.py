"""
Suite runner with cross-benchmark transcript de-duplication.

Running the eight benchmarks naively re-transcribes shared source audio: the
Earnings21 calls appear in both WER and Entities, AMI in both Diarization and
Timestamps, etc. Providers bill per submission, so that roughly doubles cost.

``run_suite`` transcribes each **unique audio file once** — keyed by real path
(so symlinked copies collapse) — with the **union** of features any benchmark
requests for it, then distributes that transcript to every benchmark's
``score()``. A transcript produced with word timestamps + speakers + punctuation
is a superset that serves a benchmark needing only the text.

Benchmarks with ``dedup_safe = False`` are excluded and run independently:
  - price:        must submit for real to measure latency/throughput
  - multilingual: passes per-file language hints and overrides run()

This changes cost, not scores: each benchmark still reads exactly the fields it
needs from the shared transcript.
"""

from __future__ import annotations

import os
from collections import defaultdict

from .. import config
from ..providers.base import Features, Provider, TranscriptionResult
from .benchmark import Benchmark
from .manifest import items_for_provider, load_manifest


def _union(a: Features, b: Features) -> Features:
    return Features(
        word_timestamps=a.word_timestamps or b.word_timestamps,
        speaker_labels=a.speaker_labels or b.speaker_labels,
        punctuation=a.punctuation or b.punctuation,
        language=a.language or b.language,  # shared files are hint-free; matches in practice
    )


def run_suite(provider: Provider, benchmarks: list[Benchmark], *, progress: bool = True,
              use_cache: bool = True) -> dict:
    """Run benchmarks with dedup for the dedup-safe ones. Returns {name: results}.

    ``use_cache`` is forwarded to transcription; it only takes effect for
    cacheable providers (all external ones — never the local Zephyr model).
    """
    dedup = [b for b in benchmarks if b.dedup_safe]
    independent = [b for b in benchmarks if not b.dedup_safe]
    results: dict[str, dict] = {}

    # ── 1. Gather items + union features per unique real path ────────────────
    #   plan:   realpath -> (Features union, duration_s)
    #   layout: bench_name -> {subset -> [(item, realpath)]}
    plan: dict[str, tuple[Features, float]] = {}
    layout: dict[str, dict[str, list[tuple[dict, str]]]] = {}
    for b in dedup:
        layout[b.name] = {}
        for subset, stem in b.subsets.items():
            path = config.manifest_path(b.name, stem)
            try:
                entries = load_manifest(path)
            except FileNotFoundError:
                layout[b.name][subset] = []
                continue
            items = items_for_provider(entries)
            pairs: list[tuple[dict, str]] = []
            for it in items:
                rp = os.path.realpath(it["audio_path"])
                pairs.append((it, rp))
                dur = float(it.get("duration_s", 0.0))
                if rp in plan:
                    feat, d = plan[rp]
                    plan[rp] = (_union(feat, b.features), d or dur)
                else:
                    plan[rp] = (b.features, dur)
            layout[b.name][subset] = pairs

    total_refs = sum(len(p) for bl in layout.values() for p in bl.values())
    if progress:
        print(f"\n[suite] dedup: {total_refs} benchmark references -> "
              f"{len(plan)} unique files to transcribe "
              f"(~{sum(d for _, d in plan.values())/3600:.1f}h)")

    # ── 2. Transcribe each unique file once, grouped by feature signature ────
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rp, (feat, dur) in plan.items():
        sig = (feat.word_timestamps, feat.speaker_labels, feat.punctuation, feat.language)
        groups[sig].append({"id": rp, "audio_path": rp, "duration_s": dur})

    result_by_rp: dict[str, TranscriptionResult] = {}
    for sig, items in groups.items():
        feats = Features(word_timestamps=sig[0], speaker_labels=sig[1],
                         punctuation=sig[2], language=sig[3])
        if progress:
            print(f"\n[suite] transcribing {len(items)} unique files "
                  f"(wt={sig[0]} sl={sig[1]}) ...")
        for r in provider.transcribe_batch(items, feats, progress=progress, use_cache=use_cache):
            result_by_rp[r.entry_id] = r  # entry_id == realpath

    # ── 3. Distribute shared transcripts to each dedup benchmark, then score ─
    for b in dedup:
        runs: dict[str, list[TranscriptionResult]] = {}
        for subset, pairs in layout[b.name].items():
            out: list[TranscriptionResult] = []
            for item, rp in pairs:
                base = result_by_rp.get(rp)
                if base is None:
                    continue
                out.append(TranscriptionResult(
                    entry_id=item["id"], transcript=base.transcript, status=base.status,
                    submit_latency_s=base.submit_latency_s, total_latency_s=base.total_latency_s,
                    audio_duration_s=float(item.get("duration_s", base.audio_duration_s)),
                    error=base.error, meta=item,
                ))
            runs[subset] = out
        results[b.name] = _safe_score(b, lambda: b.score(runs, provider_name=provider.name))

    # ── 4. Run non-dedup benchmarks independently ────────────────────────────
    for b in independent:
        results[b.name] = _safe_score(
            b, lambda b=b: b.evaluate(provider, progress=progress, use_cache=use_cache))

    return results


def _safe_score(b: Benchmark, fn) -> dict:
    """Run a benchmark's scoring; on failure return an empty result so one
    benchmark crashing doesn't abort the whole suite."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR scoring {b.name}: {e}")
        return {"benchmark": b.name, "provider": "", "suite_version": config.SUITE_VERSION,
                "summary": {}, "per_file": [], "error": str(e)}
