"""
Central configuration for the Speech Revolutions public benchmark suite.

Everything here is intentionally static and version-pinned. The suite is meant
to be *frozen*: a manifest committed under v1 must never change. New selections
become v2 rather than mutating v1 in place.

Path layout
-----------
    benchmarks/                  <- PACKAGE_ROOT  (this file's directory)
        config.py
        providers/  core/  benchmarks/  datasets/
        manifests/<benchmark>/<name>.jsonl   (committed, frozen)
        baselines/<benchmark>/<provider>.json (committed)
        reports/<benchmark>/<provider>.{json,md,csv} (generated, gitignored)
        _data/<benchmark>/audio/...          <- DATA_ROOT (gitignored; large)

Everything is kept inside the package so the suite is fully self-contained. The
audio store and generated reports are gitignored via benchmarks/.gitignore.

Audio is large and never committed. Manifests reference audio by a path that is
relative to DATA_ROOT, so the same manifest resolves on any machine that ran the
dataset preparation scripts.
"""

from __future__ import annotations

from pathlib import Path

# ── Roots ──────────────────────────────────────────────────────────────────────

PACKAGE_ROOT = Path(__file__).resolve().parent          # benchmarks/
REPO_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = PACKAGE_ROOT / "_data"                      # audio store (gitignored)

MANIFESTS_ROOT = PACKAGE_ROOT / "manifests"
BASELINES_ROOT = PACKAGE_ROOT / "baselines"
REPORTS_ROOT = PACKAGE_ROOT / "reports"

# ── Suite version ────────────────────────────────────────────────────────────────
# Bump this only by creating a v2; never edit a frozen v1 manifest.
SUITE_VERSION = "v1"

# ── Determinism ──────────────────────────────────────────────────────────────────
# A single master seed feeds every dataset preparation script. Each script
# derives a sub-seed from this so selections are stable and independent.
MASTER_SEED = 20260601

SUBSEEDS = {
    "wer_clean": 1001,
    "wer_other": 1002,
    "wer_earnings21": 1003,
    "wer_spgispeech": 1004,
    "entities": 2001,
    "diarization_ami": 3001,
    "diarization_earnings21": 3002,
    "diarization_notsofar": 3003,
    "diarization_dipco": 3004,
    "timestamps": 4001,
    "multilingual": 5001,
    "language_switching": 6001,
}

# ── Frozen benchmark sizes (see spec "Benchmark Size") ───────────────────────────
SIZES = {
    "wer_clean": 100,          # LibriSpeech test-clean clips
    "wer_other": 100,          # LibriSpeech test-other clips
    "wer_earnings21": 100,     # Earnings21 segments
    "wer_spgispeech": 100,     # SPGISpeech (Kensho) earnings segments — CLEAN reference
    "multilingual_languages": 14,
    "multilingual_per_language": 10,
    "diarization_notsofar": 20,   # meetings sampled from NOTSOFAR dev-set-1 (36 total)
    "diarization_dipco": 5,       # DiPCo eval sessions (S01,S03,S06,S07,S08)
    "language_switching_per_level": 20,   # generated recordings per difficulty
}

# ── Multilingual language set (FLEURS codes) ─────────────────────────────────────
# The 14 languages scored by the multilingual benchmark. LANGUAGE_HINTS below maps
# a wider set of FLEURS codes; to score additional languages, add them here and
# re-run the dataset prep.
MULTILINGUAL_LANGUAGES = [
    "en_us",  # English
    "es_419", # Spanish
    "fr_fr",  # French
    "de_de",  # German
    "it_it",  # Italian
    "pt_br",  # Portuguese
    "nl_nl",  # Dutch
    "pl_pl",  # Polish
    "ru_ru",  # Russian
    "tr_tr",  # Turkish
    "cmn_hans_cn",  # Mandarin
    "ja_jp",  # Japanese
    "id_id",  # Indonesian
    "fi_fi",  # Finnish
]

# Map FLEURS code -> ISO-ish language hint passed to providers that accept one.
LANGUAGE_HINTS = {
    "en_us": "en", "es_419": "es", "fr_fr": "fr", "de_de": "de", "it_it": "it",
    "pt_br": "pt", "nl_nl": "nl", "pl_pl": "pl", "ru_ru": "ru", "ar_eg": "ar",
    "he_il": "he", "hi_in": "hi", "ur_pk": "ur", "tr_tr": "tr", "cmn_hans_cn": "zh",
    "ja_jp": "ja", "ko_kr": "ko", "vi_vn": "vi", "th_th": "th", "id_id": "id",
    "sv_se": "sv", "nb_no": "no", "fi_fi": "fi", "el_gr": "el", "ro_ro": "ro",
}

# Languages whose orthography has no spaces — WER is computed on characters (CER)
# for these, since "word" error rate is ill-defined without word boundaries.
CHARACTER_LEVEL_LANGUAGES = {"cmn_hans_cn", "ja_jp", "th_th"}

# ── Language switching generation ────────────────────────────────────────────────
# Each level is defined by SUSTAINED per-language blocks (multiple consecutive
# same-language clips merged into one >=block_seconds turn), not one-clip micro-
# switches, reflecting how multilingual recordings tend to arrive in practice.
# Difficulty scales with language count and shorter blocks.
LANGUAGE_SWITCHING_LEVELS = {
    # A single sustained switch: ~30s+ of language 1, then ~30s+ of language 2 —
    # the realistic bilingual case.
    "lenient": {"n_languages": 2,  "block_seconds": 30},
    "easy":   {"n_languages": 3,  "block_seconds": 30},
    "medium": {"n_languages": 6,  "block_seconds": 20},
    "hard":   {"n_languages": 10, "block_seconds": 14},
}
# The switching pool is the 14-language multilingual set (includes non-Latin
# scripts: Mandarin, Japanese, Russian, alongside Latin scripts).
LANGUAGE_SWITCHING_POOL = list(MULTILINGUAL_LANGUAGES)
SWITCH_PAUSE_MS_RANGE = (300, 800)   # silence inserted between blocks
SWITCH_BOUNDARY_WINDOW_S = 3.0       # window each side of a boundary for boundary WER

# ── Long-form target lengths (seconds) ───────────────────────────────────────────
LONGFORM_TARGET_LENGTHS_S = [1800, 3600, 7200, 14400]  # 30m, 1h, 2h, 4h

# ── Scoring constants ────────────────────────────────────────────────────────────
DER_COLLAR_S = 0.25
TIMESTAMP_THRESHOLDS_MS = [50, 100, 200]
ENTITY_TYPES = ["PERSON", "ORG", "PRODUCT", "GPE", "MONEY", "DATE"]


def manifest_path(benchmark: str, name: str) -> Path:
    return MANIFESTS_ROOT / benchmark / f"{name}.jsonl"


def baseline_path(benchmark: str, provider: str) -> Path:
    return BASELINES_ROOT / benchmark / f"{provider}.json"


def report_path(benchmark: str, provider: str, ext: str) -> Path:
    return REPORTS_ROOT / benchmark / f"{provider}.{ext}"


def audio_dir(benchmark: str) -> Path:
    return DATA_ROOT / benchmark / "audio"
