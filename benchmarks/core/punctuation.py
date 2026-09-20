"""Punctuation and casing accuracy, scored against a punctuated reference.

We ship a fine-tuned BERT punctuation model and have never measured whether its output is good;
the absence of support tickets is not evidence. WER cannot answer the question because the
Whisper normalisers strip punctuation and case from both sides before scoring — the entire
signal is deliberately discarded.

Scoring works on the ALIGNED word sequence rather than raw text, so a transcription error does
not silently punish punctuation: only words that both sides agree on contribute, and the number
of those is reported alongside so a low overlap is visible rather than hidden.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

#: Marks scored individually. Everything else is folded into "other".
TRACKED = (".", ",", "?", "!", ":", ";")

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+", re.UNICODE)


def _tokens(text: str) -> list[tuple[str, str, bool]]:
    """(bare word, trailing punctuation, is_capitalised) for each word in the text."""
    out: list[tuple[str, str, bool]] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        tail = text[match.end() : match.end() + 3]
        punct = ""
        for ch in tail:
            if ch in TRACKED:
                punct = ch
                break
            if not ch.isspace() and ch not in "\"')]}":
                break
        out.append((word.lower(), punct, word[:1].isupper()))
    return out


def punctuation_scores(reference: str, hypothesis: str) -> dict:
    """Per-mark precision/recall/F1 plus casing accuracy over the aligned words."""
    ref, hyp = _tokens(reference), _tokens(hypothesis)
    ref_words = [w for w, _, _ in ref]
    hyp_words = [w for w, _, _ in hyp]

    matcher = SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)
    pairs: list[tuple[tuple[str, str, bool], tuple[str, str, bool]]] = []
    for tag, i1, i2, j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend((ref[i1 + k], hyp[j1 + k]) for k in range(i2 - i1))

    stats = {mark: {"tp": 0, "fp": 0, "fn": 0} for mark in TRACKED}
    cased = correct_case = 0
    for (_, ref_punct, ref_upper), (_, hyp_punct, hyp_upper) in pairs:
        for mark in TRACKED:
            in_ref, in_hyp = ref_punct == mark, hyp_punct == mark
            if in_ref and in_hyp:
                stats[mark]["tp"] += 1
            elif in_hyp:
                stats[mark]["fp"] += 1
            elif in_ref:
                stats[mark]["fn"] += 1
        cased += 1
        correct_case += ref_upper == hyp_upper

    per_mark = {mark: _prf(counts) for mark, counts in stats.items()}
    total = {
        "tp": sum(c["tp"] for c in stats.values()),
        "fp": sum(c["fp"] for c in stats.values()),
        "fn": sum(c["fn"] for c in stats.values()),
    }
    return {
        "aligned_words": len(pairs),
        "reference_words": len(ref),
        "hypothesis_words": len(hyp),
        "alignment_rate": len(pairs) / len(ref) if ref else 0.0,
        "overall": _prf(total),
        "per_mark": per_mark,
        "casing_accuracy": correct_case / cased if cased else 0.0,
        "counts": total,
    }


def corpus_punctuation_scores(references: Sequence[str], hypotheses: Sequence[str]) -> dict:
    """Pool counts across files, so long files weigh more than short ones."""
    stats = {mark: {"tp": 0, "fp": 0, "fn": 0} for mark in TRACKED}
    aligned = ref_total = cased = correct_case = 0

    for reference, hypothesis in zip(references, hypotheses, strict=True):
        one = punctuation_scores(reference, hypothesis)
        for mark in TRACKED:
            for key in ("tp", "fp", "fn"):
                stats[mark][key] += one["per_mark"][mark]["counts"][key]
        aligned += one["aligned_words"]
        ref_total += one["reference_words"]
        cased += one["aligned_words"]
        correct_case += round(one["casing_accuracy"] * one["aligned_words"])

    total = {
        "tp": sum(c["tp"] for c in stats.values()),
        "fp": sum(c["fp"] for c in stats.values()),
        "fn": sum(c["fn"] for c in stats.values()),
    }
    return {
        "aligned_words": aligned,
        "reference_words": ref_total,
        "alignment_rate": aligned / ref_total if ref_total else 0.0,
        "overall": _prf(total),
        "per_mark": {mark: _prf(counts) for mark, counts in stats.items()},
        "casing_accuracy": correct_case / cased if cased else 0.0,
    }


def _prf(counts: dict) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "counts": dict(counts),
    }
