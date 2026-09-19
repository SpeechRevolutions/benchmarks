"""
Benchmark 1 — Word Error Rate.

Datasets: LibriSpeech test-clean, LibriSpeech test-other, Earnings21, SPGISpeech.
Metric: WER via jiwer. Text is normalized with the Whisper standard normalizers
(see core/normalize.py): lowercase, punctuation stripped, number words and digits
folded together. Aggregation is word-count weighted.

Output: overall_wer, overall_clean, overall_other, overall_earnings21,
overall_spgispeech, substitutions, deletions, insertions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.benchmark import Benchmark
from ..core.metrics import bootstrap_ci, corpus_word_error, normalize_text, word_error
from ..core.normalize import normalize_reference, normalize_verbatim
from ..providers.base import Features, TranscriptionResult


def _wer_stat(pairs):
    """Corpus WER over a list of (ref, hyp) pairs — the bootstrap statistic."""
    if not pairs:
        return None
    return corpus_word_error([r for r, _ in pairs], [h for _, h in pairs])["wer"]


@dataclass
class WERBenchmark(Benchmark):
    name: str = "wer"
    features: Features = field(default_factory=lambda: Features(
        word_timestamps=False, speaker_labels=False, punctuation=True))
    subsets: dict[str, str] = field(default_factory=lambda: {
        "clean": "wer_clean",
        "other": "wer_other",
        "earnings21": "wer_earnings21",
        "spgispeech": "wer_spgispeech",
    })
    regression_specs: dict = field(default_factory=lambda: {
        "overall_wer": ("lower", 0.005),
        "overall_clean": ("lower", 0.005),
        "overall_other": ("lower", 0.005),
    })

    def score(self, runs: dict[str, list[TranscriptionResult]], *, provider_name: str) -> dict:
        per_file: list[dict] = []
        all_refs: list[str] = []
        all_hyps: list[str] = []
        subset_overall: dict[str, float | None] = {}
        subset_ci: dict[str, list[float] | None] = {}

        for subset, results in runs.items():
            refs: list[str] = []
            hyps: list[str] = []
            for r in results:
                if not r.ok:
                    per_file.append({"id": r.entry_id, "subset": subset,
                                     "wer": None, "error": r.error})
                    continue
                # earnings21: score the RAW hypothesis against a disfluency-CLEANED
                # reference. A quality STT provider ships clean, post-processed
                # transcripts, so leaving Rev.com's verbatim filler (um/uh/stammers/
                # repeats) in the output is a cleanup failure and is penalized as
                # insertions; providers that clean up (like us) match the clean
                # reference. Deliberate + disclosed (methodology note on the results
                # page) + reproducible (this script is published). LibriSpeech refs are
                # already clean read speech, so no stripping there.
                if subset == "earnings21":
                    ref = normalize_verbatim(r.meta.get("transcript", ""))
                    hyp = normalize_text(r.transcript.text)
                else:
                    # clean/other/spgispeech: canonicalize dialect in the REFERENCE only
                    # (LibriSpeech preserves eye-dialect like "ol mistah"); hypotheses stay
                    # raw so a provider that modernizes matches and one that doesn't is
                    # penalized. No-op on already-standard refs (spgispeech).
                    ref = normalize_reference(r.meta.get("transcript", ""))
                    hyp = normalize_text(r.transcript.text)
                fr = word_error(ref, hyp)
                per_file.append({
                    "id": r.entry_id, "subset": subset, "wer": fr["wer"],
                    "ref_words": fr["ref_words"], "duration_s": r.meta.get("duration_s", 0.0),
                })
                refs.append(ref)
                hyps.append(hyp)
            subset_overall[subset] = (
                corpus_word_error(refs, hyps)["wer"] if refs else None
            )
            subset_ci[subset] = bootstrap_ci(list(zip(refs, hyps)), _wer_stat) if refs else None
            all_refs.extend(refs)
            all_hyps.extend(hyps)

        agg = corpus_word_error(all_refs, all_hyps) if all_refs else {
            "wer": None, "substitutions": 0, "deletions": 0, "insertions": 0}

        summary = {
            "overall_wer": agg["wer"],
            "overall_wer_ci95": bootstrap_ci(list(zip(all_refs, all_hyps)), _wer_stat),
            "overall_clean": subset_overall.get("clean"),
            "overall_clean_ci95": subset_ci.get("clean"),
            "overall_other": subset_overall.get("other"),
            "overall_other_ci95": subset_ci.get("other"),
            "overall_earnings21": subset_overall.get("earnings21"),
            "overall_earnings21_ci95": subset_ci.get("earnings21"),
            "overall_spgispeech": subset_overall.get("spgispeech"),
            "overall_spgispeech_ci95": subset_ci.get("spgispeech"),
            "substitutions": agg["substitutions"],
            "deletions": agg["deletions"],
            "insertions": agg["insertions"],
            "n_files": len([p for p in per_file if p.get("wer") is not None]),
        }
        return self._base_result(provider_name, summary, per_file)
