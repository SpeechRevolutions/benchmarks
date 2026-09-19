"""
Benchmark 4 — Timestamp Accuracy.

Dataset: AMI (word-level reference timings from the AMI word annotations).
Words are aligned ref<->hyp by LCS; only exact matches contribute timing errors.

A timestamp metric must measure timestamp accuracy, not transcription or matching
artifacts, so two classes of LCS pairs are excluded (identically for every provider):
  * punctuation-token matches — a ref word paired to a hyp punctuation token
    ("Mm" -> "?"); these carry no meaningful word onset.
  * wrong-instance matches — a repeated word ("the") the LCS paired across a large
    time gap (> MATCH_MAX_GAP_S); the huge "error" is a matching failure, not a
    real timing error. On clean audio a correctly-matched word is well within it.
Everything else — including genuine timing error — is kept and counted.

Output: start_mae_ms, end_mae_ms, start_p90_ms, end_p90_ms,
within_50ms, within_100ms, within_200ms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: an LCS pair whose ref/hyp onsets differ by more than this is a wrong-instance
#: match (an LCS artifact on repeated words), not a timing measurement.
MATCH_MAX_GAP_S = 2.0


def _is_punct(text: str) -> bool:
    """True if the token has no alphanumeric content (punctuation/empty)."""
    return not re.sub(r"[^\w]", "", text or "", flags=re.UNICODE).strip()


def _timing_matches(matched: list) -> list:
    """Keep only LCS pairs that are valid timestamp measurements (see module docstring)."""
    out = []
    for rw, hw in matched:
        rt = rw.get("word") or rw.get("text", "")
        ht = hw.get("word") or hw.get("text", "")
        if _is_punct(rt) or _is_punct(ht):
            continue
        if abs(float(rw["start"]) - float(hw["start"])) > MATCH_MAX_GAP_S:
            continue
        out.append((rw, hw))
    return out

from ..config import TIMESTAMP_THRESHOLDS_MS
import numpy as np

from ..core.alignment import align_word_sequences
from ..core.benchmark import Benchmark
from ..core.metrics import bootstrap_ci, timing_stats
from ..providers.base import Features, TranscriptionResult


@dataclass
class TimestampBenchmark(Benchmark):
    name: str = "timestamps"
    features: Features = field(default_factory=lambda: Features(
        word_timestamps=True, speaker_labels=False, punctuation=True))
    subsets: dict[str, str] = field(default_factory=lambda: {"ami": "timestamps_ami"})
    regression_specs: dict = field(default_factory=lambda: {
        "start_mae_ms": ("lower", 10.0),
        "end_mae_ms": ("lower", 10.0),
    })

    def score(self, runs: dict[str, list[TranscriptionResult]], *, provider_name: str) -> dict:
        per_file: list[dict] = []
        all_start: list[float] = []
        all_end: list[float] = []

        for subset, results in runs.items():
            for r in results:
                if not r.ok:
                    per_file.append({"id": r.entry_id, "n_matched": 0, "error": r.error})
                    continue
                ref_words = r.meta.get("words", [])
                hyp_words = [
                    {"word": w.text, "start": w.start, "end": w.end}
                    for w in r.transcript.words
                    if w.start is not None and w.end is not None
                ]
                matched = _timing_matches(align_word_sequences(ref_words, hyp_words))
                starts = [abs(float(rw["start"]) - float(hw["start"])) * 1000
                          for rw, hw in matched]
                ends = [abs(float(rw["end"]) - float(hw["end"])) * 1000
                        for rw, hw in matched]
                per_file.append({
                    "id": r.entry_id,
                    "start_mae_ms": round(sum(starts) / len(starts), 2) if starts else None,
                    "end_mae_ms": round(sum(ends) / len(ends), 2) if ends else None,
                    "n_matched": len(matched),
                    "n_ref_words": len(ref_words),
                    "duration_s": r.meta.get("duration_s", 0.0),
                })
                all_start.extend(starts)
                all_end.extend(ends)

        s_stats = timing_stats(all_start, TIMESTAMP_THRESHOLDS_MS)
        e_stats = timing_stats(all_end, TIMESTAMP_THRESHOLDS_MS)

        _mean = lambda xs: (float(np.mean(xs)) if xs else None)
        summary = {
            "start_median_ms": s_stats["median_ms"],
            "end_median_ms": e_stats["median_ms"],
            "start_mae_ms": s_stats["mae_ms"],
            "start_mae_ms_ci95": bootstrap_ci(all_start, _mean),
            "end_mae_ms": e_stats["mae_ms"],
            "end_mae_ms_ci95": bootstrap_ci(all_end, _mean),
            "start_p90_ms": s_stats["p90_ms"],
            "end_p90_ms": e_stats["p90_ms"],
            # within-tolerance rates, reported per boundary (start vs end) so the
            # headline isn't silently start-only.
            "start_within_50ms": s_stats["within_50ms"],
            "start_within_100ms": s_stats["within_100ms"],
            "start_within_200ms": s_stats["within_200ms"],
            "end_within_50ms": e_stats["within_50ms"],
            "end_within_100ms": e_stats["within_100ms"],
            "end_within_200ms": e_stats["within_200ms"],
            "n_matched_words": s_stats["n"],
            "n_files": len([p for p in per_file if p.get("n_matched")]),
        }
        return self._base_result(provider_name, summary, per_file)
