#!/usr/bin/env python3
"""
Prepare Google FLEURS for the multilingual benchmark.

For each of the ~25 languages in config.MULTILINGUAL_LANGUAGES, downloads the
FLEURS test split and freezes a deterministic, seeded sample of N clips
(config.SIZES['multilingual_per_language']). Each language contributes roughly
equal audio, per the spec.

Writes:
    manifests/multilingual/multilingual_all.jsonl  (all languages, 'language' field)
    audio under data/benchmarks/multilingual/audio/<lang>/

Requires: pip install datasets soundfile

Usage:
    python -m benchmarks.datasets.prepare_fleurs
    python -m benchmarks.datasets.prepare_fleurs --languages en_us fr_fr
"""

from __future__ import annotations

import argparse

from .. import config
from . import _util


def _load_datasets():
    try:
        from datasets import Audio, load_dataset  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "FLEURS prep needs: pip install datasets soundfile"
        ) from e
    from datasets import Audio, load_dataset
    import soundfile
    return load_dataset, Audio, soundfile


def prepare(languages: list[str]) -> None:
    import io

    load_dataset, Audio, soundfile = _load_datasets()
    audio_root = config.audio_dir("multilingual")
    n_per = config.SIZES["multilingual_per_language"]
    entries: list[dict] = []

    for lang in languages:
        print(f"\n[{lang}] loading FLEURS test split ...")
        try:
            # decode=False avoids the torchcodec audio decoder (datasets>=5);
            # we decode the raw bytes ourselves with soundfile below.
            ds = load_dataset("google/fleurs", lang, split="test").cast_column(
                "audio", Audio(decode=False))
        except Exception as e:  # noqa: BLE001
            print(f"  [SKIP] {lang}: {e}")
            continue

        indices = list(range(len(ds)))
        # Per-language deterministic selection: seed the global RNG offset by a
        # stable hash of the language so each language's sample is independent.
        rng = _util.rng("multilingual")
        rng.seed(config.MASTER_SEED + config.SUBSEEDS["multilingual"] + _lang_offset(lang))
        chosen = sorted(rng.sample(indices, min(n_per, len(indices))))

        lang_dir = audio_root / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        for idx in chosen:
            ex = ds[idx]
            audio = ex["audio"]  # {'bytes': ..., 'path': ...} with decode=False
            ref = ex.get("raw_transcription") or ex.get("transcription") or ""
            raw = audio.get("bytes")
            if raw is None and audio.get("path"):
                with open(audio["path"], "rb") as fh:
                    raw = fh.read()
            arr, sr = soundfile.read(io.BytesIO(raw))
            dest = lang_dir / f"{lang}_{idx:05d}.wav"
            if not dest.exists():
                soundfile.write(str(dest), arr, sr)
            entries.append({
                "id": f"fleurs-{lang}-{idx:05d}",
                "audio_path": _util.rel_to_data(dest),
                "transcript": ref,
                "language": lang,
                "duration_s": round(len(arr) / sr, 2),
            })
        print(f"  + {lang}: {len(chosen)} clips")

    _util.write_jsonl(
        config.manifest_path("multilingual", "multilingual_all"), entries,
        header=f"Multilingual FLEURS | {len(languages)} languages x {n_per} "
               f"| frozen {config.SUITE_VERSION} | seed={config.MASTER_SEED + config.SUBSEEDS['multilingual']}")


def _lang_offset(lang: str) -> int:
    """Stable per-language seed offset (deterministic, not hash-randomized)."""
    return sum((i + 1) * ord(c) for i, c in enumerate(lang))


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare FLEURS multilingual benchmark")
    ap.add_argument("--languages", nargs="+", default=config.MULTILINGUAL_LANGUAGES)
    prepare(ap.parse_args().languages)


if __name__ == "__main__":
    main()
