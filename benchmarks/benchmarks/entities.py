"""
Benchmark 2 — Entity Accuracy.

Dataset: Earnings21 (medical/legal benchmarks are future work).
Named entities are extracted from the *reference* transcript with spaCy and from
the *hypothesis* transcript the same way, then matched as a multiset keyed by
(entity_type, normalized_surface_text). This measures whether a provider
preserves the entities a downstream consumer cares about, independent of overall
WER.

HEADLINE metric is entity_content_recall — did we transcribe each gold entity's
words correctly, ignoring casing/punctuation. entity_f1 (spaCy NER on the hypothesis)
is reported too but is formatting-sensitive (it penalizes lowercase/unpunctuated
output), so it is a secondary signal, not the headline.

Output: entity_content_recall (headline), entity_precision, entity_recall, entity_f1,
missed_entity_rate, false_entity_rate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from ..config import ENTITY_TYPES
from ..core.benchmark import Benchmark
from ..core.metrics import bootstrap_ci
from ..core.normalize import normalize_text
from ..providers.base import Features, TranscriptionResult


def _f1_stat(items):
    """F1 over (tp, fp, fn) per-file items — bootstrap statistic."""
    tp = sum(i[0] for i in items)
    fp = sum(i[1] for i in items)
    fn = sum(i[2] for i in items)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0

SPACY_MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy
    except ImportError as e:
        raise ImportError(
            "spaCy not installed. Run: pip install -r requirements.txt && "
            f"python -m spacy download {SPACY_MODEL}"
        ) from e
    try:
        return spacy.load(SPACY_MODEL, disable=["lemmatizer"])
    except OSError as e:
        raise OSError(
            f"spaCy model '{SPACY_MODEL}' not found. "
            f"Run: python -m spacy download {SPACY_MODEL}"
        ) from e


def extract_entities(text: str) -> Counter:
    """Return a multiset of (entity_type, normalized_text) for tracked types.

    NOTE: entity_precision/recall/f1 below run spaCy NER on the *hypothesis* text,
    so they conflate entity recognition with output FORMATTING — spaCy relies on
    casing/punctuation to detect proper nouns, so a provider that returns lowercase
    or unpunctuated text scores low recall even when it transcribed the words
    correctly. That penalty is a real property of the provider's output (see the
    README's stance on text-only providers), not a harness bug — but it is not a
    pure recognition measure. `entity_content_recall` (below) is the
    formatting-robust companion: it asks only whether each gold entity's words are
    present in the normalized hypothesis, independent of casing/punctuation.
    """
    doc = _nlp()(text)
    counter: Counter = Counter()
    for ent in doc.ents:
        if ent.label_ in ENTITY_TYPES:
            key = (ent.label_, normalize_text(ent.text))
            if key[1]:
                counter[key] += 1
    return counter


def _contains_seq(sub: list[str], seq: list[str]) -> bool:
    """True if `sub` is a contiguous subsequence of `seq` (both token lists)."""
    if not sub:
        return False
    n = len(sub)
    return any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))


@dataclass
class EntityBenchmark(Benchmark):
    name: str = "entities"
    features: Features = field(default_factory=lambda: Features(
        word_timestamps=False, speaker_labels=False, punctuation=True))
    subsets: dict[str, str] = field(default_factory=lambda: {
        "earnings21": "entities_earnings21",
    })
    regression_specs: dict = field(default_factory=lambda: {
        # HEADLINE = entity_content_recall: did we transcribe each gold entity's WORDS
        # correctly, independent of casing/punctuation — the honest "did we get the
        # names right" measure. entity_f1 runs spaCy NER and therefore conflates
        # recognition with output FORMATTING (it penalizes lowercase/unpunctuated
        # transcripts), so it's kept as a secondary signal, not the headline.
        "entity_content_recall": ("higher", 0.02),
    })

    def score(self, runs: dict[str, list[TranscriptionResult]], *, provider_name: str) -> dict:
        per_file: list[dict] = []
        tp = fp = fn = 0  # true pos / false pos / false neg, summed over corpus
        content_matched = 0  # gold entities whose words appear in the hyp (format-robust)
        file_items: list[tuple[int, int, int]] = []  # (tp, fp, fn) per file for CI

        for subset, results in runs.items():
            for r in results:
                if not r.ok:
                    per_file.append({"id": r.entry_id, "error": r.error})
                    continue
                ref_ents = extract_entities(r.meta.get("transcript", ""))
                hyp_ents = extract_entities(r.transcript.text)

                # Format-robust recall: is each gold entity's normalized surface
                # present in the normalized hypothesis text? (independent of the
                # provider's casing/punctuation).
                hyp_tokens = normalize_text(r.transcript.text).split()
                for (_label, surface), cnt in ref_ents.items():
                    if _contains_seq(surface.split(), hyp_tokens):
                        content_matched += cnt

                # multiset intersection = matched entities
                matched = sum((ref_ents & hyp_ents).values())
                ref_total = sum(ref_ents.values())
                hyp_total = sum(hyp_ents.values())
                file_tp = matched
                file_fn = ref_total - matched
                file_fp = hyp_total - matched

                tp += file_tp
                fn += file_fn
                fp += file_fp
                file_items.append((file_tp, file_fp, file_fn))
                per_file.append({
                    "id": r.entry_id, "subset": subset,
                    "ref_entities": ref_total, "hyp_entities": hyp_total,
                    "matched": matched,
                    "recall": round(file_tp / ref_total, 4) if ref_total else None,
                })

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        summary = {
            # HEADLINE metric: did we transcribe each gold entity's WORDS correctly,
            # ignoring casing/punctuation — the honest "did we get the names right".
            "entity_content_recall": round(content_matched / (tp + fn), 6) if (tp + fn) else 0.0,
            # Secondary (spaCy-NER-based, so formatting-sensitive — see extract_entities):
            "entity_precision": round(precision, 6),
            "entity_recall": round(recall, 6),
            "entity_f1": round(f1, 6),
            "entity_f1_ci95": bootstrap_ci(file_items, _f1_stat),
            "missed_entity_rate": round(fn / (tp + fn), 6) if (tp + fn) else 0.0,
            "false_entity_rate": round(fp / (tp + fp), 6) if (tp + fp) else 0.0,
            "total_ref_entities": tp + fn,
            "n_files": len([p for p in per_file if "recall" in p]),
        }
        return self._base_result(provider_name, summary, per_file)
