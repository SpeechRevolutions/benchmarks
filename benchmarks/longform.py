"""
Benchmark 7 — Long-form Stability.

Datasets: Earnings21, AMI, public podcast recordings — 30 min to 4 h.
Measures production failure modes that only appear on long audio: hallucinated
content, duplicated spans, dropped audio, and timestamp drift.

Output: overall_wer, hallucination_rate, duplicate_rate, missing_audio_rate,
drift_events. (max_continuous_drift_s is also reported.)

Definitions (corpus-level proxies, computed without per-word reference timing):
  hallucination_rate  = inserted words / reference words
  missing_audio_rate  = deleted words  / reference words
  duplicate_rate      = words inside an immediately-repeated n-gram / hyp words
  drift_events        = count of large backward jumps in hyp word start times
  max_continuous_drift_s = largest backward jump observed
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.benchmark import Benchmark
from ..core.metrics import bootstrap_ci, corpus_word_error, normalize_text
from ..core.normalize import normalize_verbatim
from ..providers.base import Features, TranscriptionResult


def _wer_stat(pairs):
    if not pairs:
        return None
    return corpus_word_error([r for r, _ in pairs], [h for _, h in pairs])["wer"]

DUP_NGRAM = 5            # immediately-repeated 5-grams flag duplication
DRIFT_BACKWARD_S = 1.0   # word start jumping >1s earlier than its predecessor


def _duplicate_word_count(words: list[str], n: int = DUP_NGRAM) -> int:
    """Count words inside an immediately-repeated n-gram (looping/duplication)."""
    dup_positions: set[int] = set()
    for i in range(len(words) - 2 * n + 1):
        if words[i:i + n] == words[i + n:i + 2 * n]:
            dup_positions.update(range(i + n, i + 2 * n))
    return len(dup_positions)


def _drift(transcript) -> tuple[int, float]:
    """Count + max magnitude of backward timestamp jumps (s)."""
    starts = [w.start for w in transcript.words if w.start is not None]
    events = 0
    max_drift = 0.0
    for prev, cur in zip(starts, starts[1:]):
        back = prev - cur
        if back > DRIFT_BACKWARD_S:
            events += 1
            max_drift = max(max_drift, back)
    return events, round(max_drift, 3)


@dataclass
class LongformBenchmark(Benchmark):
    name: str = "longform"
    features: Features = field(default_factory=lambda: Features(
        word_timestamps=True, speaker_labels=False, punctuation=True))
    subsets: dict[str, str] = field(default_factory=lambda: {"all": "longform_all"})
    regression_specs: dict = field(default_factory=lambda: {
        "overall_wer": ("lower", 0.01),
        "hallucination_rate": ("lower", 0.01),
        "duplicate_rate": ("lower", 0.01),
    })

    def score(self, runs: dict[str, list[TranscriptionResult]], *, provider_name: str) -> dict:
        per_file: list[dict] = []
        all_ref: list[str] = []
        all_hyp: list[str] = []
        total_dup = 0
        total_hyp_words = 0
        total_drift_events = 0
        max_drift_overall = 0.0

        for subset, results in runs.items():
            for r in results:
                if not r.ok:
                    per_file.append({"id": r.entry_id, "error": r.error})
                    continue
                # Long-form is all Rev.com VERBATIM earnings audio: score the RAW
                # hypothesis against a disfluency-CLEANED reference, so verbatim filler
                # in the output is penalized as a cleanup failure (see wer.py rationale).
                ref = normalize_verbatim(r.meta.get("transcript", ""))
                hyp = normalize_text(r.transcript.text)
                all_ref.append(ref)
                all_hyp.append(hyp)

                hyp_tokens = hyp.split()
                dup = _duplicate_word_count(hyp_tokens)
                drift_events, max_drift = _drift(r.transcript)
                total_dup += dup
                total_hyp_words += len(hyp_tokens)
                total_drift_events += drift_events
                max_drift_overall = max(max_drift_overall, max_drift)

                file_wer = corpus_word_error([ref], [hyp]) if ref else None
                per_file.append({
                    "id": r.entry_id, "source": r.meta.get("source"),
                    "wer": file_wer["wer"] if file_wer else None,
                    "duplicate_words": dup, "hyp_words": len(hyp_tokens),
                    "drift_events": drift_events, "max_drift_s": max_drift,
                    "duration_s": r.meta.get("duration_s", 0.0),
                })

        agg = corpus_word_error(all_ref, all_hyp) if all_ref else None
        ref_words = agg["ref_words"] if agg else 0

        summary = {
            "overall_wer": agg["wer"] if agg else None,
            "overall_wer_ci95": bootstrap_ci(list(zip(all_ref, all_hyp)), _wer_stat),
            "hallucination_rate": round(agg["insertions"] / ref_words, 6) if ref_words else None,
            "missing_audio_rate": round(agg["deletions"] / ref_words, 6) if ref_words else None,
            "duplicate_rate": round(total_dup / total_hyp_words, 6) if total_hyp_words else None,
            "drift_events": total_drift_events,
            "max_continuous_drift_s": round(max_drift_overall, 3),
            "n_files": len([p for p in per_file if "error" not in p]),
        }
        return self._base_result(provider_name, summary, per_file)
