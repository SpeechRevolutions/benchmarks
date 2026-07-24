"""
Shared metric primitives: WER/CER and timestamp statistics.

WER aggregation is word-count weighted (jiwer over the concatenated corpus),
never a simple mean of per-file WERs — short files would otherwise dominate.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import jiwer
except ImportError:  # surfaced lazily so non-WER benchmarks still import
    jiwer = None  # type: ignore

from .normalize import normalize_text, to_characters


def _require_jiwer() -> None:
    if jiwer is None:
        raise ImportError("jiwer not installed. Run: pip install -r requirements.txt")


def word_error(reference: str, hypothesis: str) -> dict:
    """WER for one normalized ref/hyp pair with S/D/I counts."""
    _require_jiwer()
    out = jiwer.process_words(reference, hypothesis)
    return {
        "wer": round(float(out.wer), 6),
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "ref_words": len(reference.split()),
    }


_EMPTY_WER = {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 0}


def corpus_word_error(references: Sequence[str], hypotheses: Sequence[str]) -> dict:
    """Word-count-weighted WER over a corpus. Pairs with an empty reference are
    dropped — WER is undefined with no reference words (jiwer also rejects them),
    e.g. a boundary window that lands where the reference has no words."""
    _require_jiwer()
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r and r.strip()]
    if not pairs:
        return dict(_EMPTY_WER)
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    out = jiwer.process_words(refs, hyps)
    return {
        "wer": round(float(out.wer), 6),
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "ref_words": sum(len(r.split()) for r in refs),
    }


def character_error(reference: str, hypothesis: str) -> dict:
    """CER for one ref/hyp pair (for scripts without word spaces)."""
    _require_jiwer()
    ref_chars = " ".join(to_characters(reference))
    hyp_chars = " ".join(to_characters(hypothesis))
    out = jiwer.process_words(ref_chars, hyp_chars)
    return {
        "wer": round(float(out.wer), 6),  # really CER; kept key 'wer' for uniformity
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "ref_words": len(ref_chars.split()),
    }


def corpus_character_error(references: Sequence[str], hypotheses: Sequence[str]) -> dict:
    _require_jiwer()
    pairs = [(" ".join(to_characters(r)), " ".join(to_characters(h)))
             for r, h in zip(references, hypotheses)]
    pairs = [(r, h) for r, h in pairs if r.strip()]  # drop empty-reference pairs
    if not pairs:
        return dict(_EMPTY_WER)
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    out = jiwer.process_words(refs, hyps)
    return {
        "wer": round(float(out.wer), 6),
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "ref_words": sum(len(r.split()) for r in refs),
    }


def timing_stats(errors_ms: Sequence[float], thresholds_ms: Sequence[int]) -> dict:
    """MAE, P90, and fraction-within-threshold for a list of |error| values (ms)."""
    if not errors_ms:
        out = {"mae_ms": None, "median_ms": None, "p90_ms": None, "n": 0}
        for t in thresholds_ms:
            out[f"within_{t}ms"] = None
        return out
    arr = np.asarray(errors_ms, dtype=float)
    out = {
        "mae_ms": round(float(np.mean(arr)), 2),
        # Median: timestamp errors are heavy-tailed (matching residue, a few hard
        # words), so the median is the honest "typical word" accuracy; the mean is
        # kept alongside it, not replaced.
        "median_ms": round(float(np.median(arr)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "n": int(arr.size),
    }
    for t in thresholds_ms:
        out[f"within_{t}ms"] = round(float(np.mean(arr <= t)), 4)
    return out


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=float), p)), 4)


# ── Statistical confidence ────────────────────────────────────────────────────

def bootstrap_ci(items, statistic, *, n_resamples: int = 1000, seed: int = 20260601,
                 ci: float = 0.95) -> list[float] | None:
    """Bootstrap confidence interval for an aggregate statistic over items.

    Resamples ``items`` (files/utterances) with replacement ``n_resamples`` times
    and recomputes ``statistic(sample)`` each time — the standard file-level
    bootstrap for corpus metrics like WER/DER. Returns [lo, hi] at ``ci``, or
    None if there aren't enough items.

    ``statistic`` must accept a list of items and return a float (or None to
    skip that resample). Deterministic given ``seed`` so published CIs are
    reproducible.
    """
    items = list(items)
    if len(items) < 2:
        return None
    rng = np.random.default_rng(seed)
    n = len(items)
    stats: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        sample = [items[i] for i in idx]
        v = statistic(sample)
        if v is not None:
            stats.append(float(v))
    if not stats:
        return None
    lo = float(np.percentile(stats, (1 - ci) / 2 * 100))
    hi = float(np.percentile(stats, (1 + ci) / 2 * 100))
    return [round(lo, 6), round(hi, 6)]


__all__ = [
    "word_error", "corpus_word_error", "character_error", "corpus_character_error",
    "timing_stats", "percentile", "normalize_text",
]
