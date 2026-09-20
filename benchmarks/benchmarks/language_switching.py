"""
Benchmark 6 — Language Switching.

Proprietary benchmark, generated entirely from public data (FLEURS clips
concatenated with realistic pauses; see datasets/prepare_language_switching.py).
Three difficulty levels: easy (2 langs), medium (3), hard (5+).

Manifest entry::

    {"id": "...", "audio_path": "...", "duration_s": 24.5, "level": "hard",
     "segments": [{"language": "en_us", "start": 0.0, "end": 5.2,
                   "transcript": "..."}, ...]}

Metrics (per spec):
  overall_wer              — full-recording WER
  switch_boundary_wer      — WER restricted to a window around each switch
  switch_detection_accuracy— fraction of true switches the provider flagged*
  average_switch_latency_ms— mean delay between true switch and detection*
  per_language_wer         — WER per language, attributing hyp words by timing

* Requires per-word language labels from the provider. The local Speech
  Revolutions model does not emit these, so these two metrics are reported as
  ``null`` with ``switch_metrics_supported: false`` rather than fabricated. The
  scoring path activates automatically if a provider attaches ``language`` to
  its words.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import CHARACTER_LEVEL_LANGUAGES, SWITCH_BOUNDARY_WINDOW_S
from ..core.benchmark import Benchmark
from ..core.metrics import bootstrap_ci, corpus_character_error, corpus_word_error
import re as _re

from ..core.normalize import basic_normalize, normalize_for_language

# Characters from space-less scripts (CJK, kana, Thai). In a mixed-language
# recording these must be tokenized as individual characters or an entire CJK
# segment collapses to ~1 "word" and its errors become invisible in the mixed WER.
_SCRIPTLESS = _re.compile(
    r"[぀-ヿ㐀-䶿一-鿿豈-﫿฀-๿]"
)


def _mixed_tokens_str(text: str) -> str:
    """Space-separate scriptless characters so the mixed-corpus WER counts CJK/Thai
    per character (CER-equivalent) while keeping spaced languages word-level."""
    return _SCRIPTLESS.sub(lambda m: f" {m.group(0)} ", text)


def _wer_stat(pairs):
    if not pairs:
        return None
    return corpus_word_error([r for r, _ in pairs], [h for _, h in pairs])["wer"]
from ..providers.base import Features, TranscriptionResult


def _distribute_word_times(transcript: str, start: float, end: float) -> list[tuple[str, float]]:
    """Approximate per-word center times by spacing words evenly in [start, end]."""
    words = transcript.split()
    if not words:
        return []
    span = max(end - start, 1e-6)
    return [(w, start + (i + 0.5) / len(words) * span) for i, w in enumerate(words)]


def _hyp_word_centers(transcript) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for w in transcript.words:
        if w.start is not None and w.end is not None:
            out.append((w.text, (w.start + w.end) / 2.0))
    return out


@dataclass
class LanguageSwitchingBenchmark(Benchmark):
    name: str = "language_switching"
    features: Features = field(default_factory=lambda: Features(
        word_timestamps=True, speaker_labels=False, punctuation=True))
    subsets: dict[str, str] = field(default_factory=lambda: {
        "lenient": "ls_lenient",
        "easy": "ls_easy", "medium": "ls_medium", "hard": "ls_hard",
    })
    regression_specs: dict = field(default_factory=lambda: {
        "overall_wer": ("lower", 0.01),
        "switch_boundary_wer": ("lower", 0.02),
    })

    def score(self, runs: dict[str, list[TranscriptionResult]], *, provider_name: str) -> dict:
        per_file: list[dict] = []
        all_ref: list[str] = []
        all_hyp: list[str] = []
        bnd_ref: list[str] = []
        bnd_hyp: list[str] = []
        lang_ref: dict[str, list[str]] = {}
        lang_hyp: dict[str, list[str]] = {}
        level_overall: dict[str, list[str]] = {}

        switch_supported = False
        detected_correct = 0
        true_switches = 0
        latencies: list[float] = []

        for subset, results in runs.items():
            for r in results:
                if not r.ok:
                    per_file.append({"id": r.entry_id, "level": subset, "error": r.error})
                    continue
                segments = r.meta.get("segments", [])
                # overall/boundary mix languages -> script-agnostic Basic normalizer,
                # with scriptless (CJK/Thai) chars tokenized so they aren't invisible.
                ref_full = _mixed_tokens_str(basic_normalize(" ".join(s.get("transcript", "") for s in segments)))
                hyp_full = _mixed_tokens_str(basic_normalize(r.transcript.text))
                all_ref.append(ref_full)
                all_hyp.append(hyp_full)
                level_overall.setdefault(subset, []).append(ref_full)

                hyp_centers = _hyp_word_centers(r.transcript)

                # ── per-language attribution by word timing ──────────────────
                for s in segments:
                    lang = s.get("language", "unknown")
                    lang_ref.setdefault(lang, []).append(
                        normalize_for_language(s.get("transcript", ""), lang))
                    seg_words = [w for (w, t) in hyp_centers
                                 if s["start"] <= t < s["end"]]
                    lang_hyp.setdefault(lang, []).append(
                        normalize_for_language(" ".join(seg_words), lang))

                # ── boundary windows ──────────────────────────────────────────
                ref_timed = [
                    (w, t) for s in segments
                    for (w, t) in _distribute_word_times(
                        s.get("transcript", ""), s["start"], s["end"])
                ]
                boundaries = [
                    (segments[i]["end"] + segments[i + 1]["start"]) / 2.0
                    for i in range(len(segments) - 1)
                ]
                true_switches += len(boundaries)
                for b in boundaries:
                    lo, hi = b - SWITCH_BOUNDARY_WINDOW_S, b + SWITCH_BOUNDARY_WINDOW_S
                    r_words = [w for (w, t) in ref_timed if lo <= t <= hi]
                    h_words = [w for (w, t) in hyp_centers if lo <= t <= hi]
                    bnd_ref.append(_mixed_tokens_str(basic_normalize(" ".join(r_words))))
                    bnd_hyp.append(_mixed_tokens_str(basic_normalize(" ".join(h_words))))

                # ── switch detection (only if provider tags word language) ───
                word_langs = [getattr(w, "language", None) for w in r.transcript.words]
                if any(word_langs):
                    switch_supported = True
                    det, lat = _switch_detection(r.transcript, segments, boundaries)
                    detected_correct += det
                    latencies.extend(lat)

                per_file.append({
                    "id": r.entry_id, "level": subset,
                    "n_segments": len(segments),
                    "wer": corpus_word_error([ref_full], [hyp_full])["wer"] if ref_full else None,
                    "duration_s": r.meta.get("duration_s", 0.0),
                })

        overall = corpus_word_error(all_ref, all_hyp)["wer"] if all_ref else None
        boundary = corpus_word_error(bnd_ref, bnd_hyp)["wer"] if any(bnd_ref) else None
        # Per-language uses CER for languages without word spaces (zh/ja/th),
        # matching the multilingual benchmark; word-level WER there is meaningless.
        per_language_wer = {}
        for lang in sorted(lang_ref):
            if not any(lang_ref[lang]):
                continue
            scorer = (corpus_character_error if lang in CHARACTER_LEVEL_LANGUAGES
                      else corpus_word_error)
            per_language_wer[lang] = scorer(lang_ref[lang], lang_hyp.get(lang, []))["wer"]

        summary = {
            "overall_wer": overall,
            "overall_wer_ci95": bootstrap_ci(list(zip(all_ref, all_hyp)), _wer_stat),
            "switch_boundary_wer": boundary,
            "switch_detection_accuracy": (
                round(detected_correct / true_switches, 6)
                if switch_supported and true_switches else None
            ),
            "average_switch_latency_ms": (
                round(sum(latencies) / len(latencies), 2) if latencies else None
            ),
            "per_language_wer": per_language_wer,
            "switch_metrics_supported": switch_supported,
            "n_files": len([p for p in per_file if "error" not in p]),
        }
        return self._base_result(provider_name, summary, per_file)


def _switch_detection(transcript, segments, boundaries) -> tuple[int, list[float]]:
    """Count correctly detected switches and their latencies (provider-language path).

    A switch is "detected" if the provider's language label on the nearest word
    BEFORE the boundary differs from the nearest word AFTER it. This compares the
    two words that actually bracket the boundary rather than requiring words inside
    a fixed ±window — the FLEURS clips carry natural silence padding, so a boundary
    often lands in a multi-second silence where NO provider has words (a fixed
    window would wrongly score every such switch as missed for everyone). Latency
    is the gap from the boundary to the first labelled word after it — inherently
    bounded below by any silence there, which is fair across providers.

    To avoid matching across an adjacent switch, each side is bounded by the
    neighbouring boundary (else the whole recording).
    """
    timed = sorted(
        (( (w.start + w.end) / 2.0, getattr(w, "language", None))
         for w in transcript.words
         if w.start is not None and w.end is not None and getattr(w, "language", None)),
        key=lambda x: x[0],
    )
    detected = 0
    latencies: list[float] = []
    for i, b in enumerate(boundaries):
        lo = boundaries[i - 1] if i > 0 else float("-inf")
        hi = boundaries[i + 1] if i + 1 < len(boundaries) else float("inf")
        before = [(t, lang) for (t, lang) in timed if lo <= t < b]
        after = [(t, lang) for (t, lang) in timed if b <= t < hi]
        if not before or not after:
            continue
        if before[-1][1] != after[0][1]:
            detected += 1
            latencies.append(abs(after[0][0] - b) * 1000.0)
    return detected, latencies
