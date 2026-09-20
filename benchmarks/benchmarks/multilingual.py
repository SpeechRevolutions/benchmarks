"""
Benchmark 5 — Multilingual.

Dataset: Google FLEURS, ~25 representative languages, roughly equal audio per
language. Metric: WER (CER for languages without word spaces — Mandarin,
Japanese, Thai). Each file is transcribed with that language as a hint so we
measure transcription quality rather than language ID.

Output: overall_multilingual_wer, language_breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import CHARACTER_LEVEL_LANGUAGES, LANGUAGE_HINTS
from ..core.benchmark import Benchmark
from ..core.manifest import items_for_provider, load_manifest
from ..core.metrics import bootstrap_ci, corpus_character_error, corpus_word_error
from ..core.normalize import normalize_for_language
from ..providers.base import Features, Provider, TranscriptionResult
from .. import config


@dataclass
class MultilingualBenchmark(Benchmark):
    name: str = "multilingual"
    features: Features = field(default_factory=lambda: Features(
        word_timestamps=False, speaker_labels=False, punctuation=True))
    subsets: dict[str, str] = field(default_factory=lambda: {"all": "multilingual_all"})
    regression_specs: dict = field(default_factory=lambda: {
        "overall_multilingual_wer": ("lower", 0.01),
    })
    dedup_safe: bool = False  # per-language hints + custom run(); files are unique anyway

    def run(self, provider: Provider, *, progress: bool = True,
            use_cache: bool = True) -> dict[str, list[TranscriptionResult]]:
        """Group the manifest by language and transcribe each group with a hint."""
        path = config.manifest_path(self.name, self.subsets["all"])
        try:
            entries = load_manifest(path)
        except FileNotFoundError:
            if progress:
                print(f"  [skip] {self.name}: no manifest at {path}")
            return {}

        by_lang: dict[str, list[dict]] = {}
        for e in entries:
            by_lang.setdefault(e.get("language", "unknown"), []).append(e)

        runs: dict[str, list[TranscriptionResult]] = {}
        for lang, lang_entries in by_lang.items():
            items = items_for_provider(lang_entries)
            if not items:
                runs[lang] = []
                continue
            feats = Features(
                word_timestamps=False, speaker_labels=False,
                punctuation=True, language=LANGUAGE_HINTS.get(lang, lang.split("_")[0]),
            )
            if progress:
                print(f"\n[{self.name}/{lang}] transcribing {len(items)} files ...")
            runs[lang] = provider.transcribe_batch(
                items, feats, progress=progress, use_cache=use_cache)
        return runs

    def score(self, runs: dict[str, list[TranscriptionResult]], *, provider_name: str) -> dict:
        per_file: list[dict] = []
        language_breakdown: dict[str, float | None] = {}
        all_refs: list[str] = []
        all_hyps: list[str] = []
        char_refs: list[str] = []
        char_hyps: list[str] = []

        for lang, results in runs.items():
            refs: list[str] = []
            hyps: list[str] = []
            for r in results:
                if not r.ok:
                    per_file.append({"id": r.entry_id, "language": lang, "error": r.error})
                    continue
                ref = normalize_for_language(r.meta.get("transcript", ""), lang)
                hyp = normalize_for_language(r.transcript.text, lang)
                refs.append(ref)
                hyps.append(hyp)
                per_file.append({
                    "id": r.entry_id, "language": lang,
                    "ref_words": len(ref.split()),
                    "duration_s": r.meta.get("duration_s", 0.0),
                })
            if not refs:
                language_breakdown[lang] = None
                continue
            if lang in CHARACTER_LEVEL_LANGUAGES:
                language_breakdown[lang] = corpus_character_error(refs, hyps)["wer"]
                char_refs.extend(refs)
                char_hyps.extend(hyps)
            else:
                language_breakdown[lang] = corpus_word_error(refs, hyps)["wer"]
                all_refs.extend(refs)
                all_hyps.extend(hyps)

        # Overall is the mean of per-language scores so each language counts
        # equally regardless of its (roughly equal) clip count.
        scored = [v for v in language_breakdown.values() if v is not None]
        overall = round(sum(scored) / len(scored), 6) if scored else None

        # CI over per-language scores — languages are the natural sampling unit
        # (each contributes ~equal audio, and overall is their mean).
        overall_ci = bootstrap_ci(scored, lambda xs: (sum(xs) / len(xs)) if xs else None)

        summary = {
            "overall_multilingual_wer": overall,
            "overall_multilingual_wer_ci95": overall_ci,
            "language_breakdown": {k: language_breakdown[k] for k in sorted(language_breakdown)},
            "n_languages": len(scored),
            "n_files": len([p for p in per_file if "error" not in p]),
        }
        return self._base_result(provider_name, summary, per_file)
