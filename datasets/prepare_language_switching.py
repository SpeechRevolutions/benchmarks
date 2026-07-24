#!/usr/bin/env python3
"""
Generate the Language Switching benchmark from public FLEURS clips.

Proprietary benchmark, generated entirely automatically (no manual curation).
Reuses the multilingual clip pool (run prepare_fleurs first), then concatenates
clips from multiple languages with realistic 300-800 ms pauses to synthesize
code-switching recordings at four difficulty levels.

Everything is seeded (config.SUBSEEDS['language_switching']) so the generated
benchmark is byte-for-byte reproducible given the same FLEURS pool.

Writes:
    manifests/language_switching/ls_{lenient,easy,medium,hard}.jsonl
    audio under _data/language_switching/audio/

Requires: ffmpeg, and a prepared multilingual pool.

Usage:
    python -m benchmarks.datasets.prepare_fleurs            # build the pool
    python -m benchmarks.datasets.prepare_language_switching
"""

from __future__ import annotations

import argparse

from .. import config
from ..core.manifest import load_manifest, resolve_audio_path
from . import _util


def _load_pool() -> dict[str, list[dict]]:
    """Group the prepared multilingual clips by language."""
    path = config.manifest_path("multilingual", "multilingual_all")
    try:
        entries = load_manifest(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Multilingual pool not found. Run prepare_fleurs first:\n"
            "  python -m benchmarks.datasets.prepare_fleurs"
        ) from e
    pool: dict[str, list[dict]] = {}
    for e in entries:
        pool.setdefault(e["language"], []).append(e)
    return pool


def _generate_level(level: str, pool: dict[str, list[dict]], rng) -> list[dict]:
    spec = config.LANGUAGE_SWITCHING_LEVELS[level]
    n_langs = spec["n_languages"]
    block_seconds = spec["block_seconds"]
    n_recordings = config.SIZES["language_switching_per_level"]
    # Draw only from the strong-language pool that we actually have clips for.
    languages = [lang for lang in config.LANGUAGE_SWITCHING_POOL if pool.get(lang)]
    if len(languages) < n_langs:
        raise ValueError(f"Need >= {n_langs} strong-pool languages; have {len(languages)}")

    tmp_dir = config.DATA_ROOT / "language_switching" / "_tmp"
    out_audio = config.audio_dir("language_switching")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_audio.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for r in range(n_recordings):
        # Each chosen language appears exactly once, as a sustained block, in random
        # order (all distinct -> a guaranteed switch at every block boundary).
        chosen_langs = rng.sample(languages, n_langs)
        rng.shuffle(chosen_langs)

        parts: list = []
        segments: list[dict] = []
        cursor = 0.0
        for bi, lang in enumerate(chosen_langs):
            # Fill a >= block_seconds block from consecutive same-language clips.
            block_start = cursor
            texts: list[str] = []
            block_dur = 0.0
            k = 0
            while block_dur < block_seconds:
                clip = rng.choice(pool[lang])
                src = resolve_audio_path(clip)
                norm = _util.to_wav(src, tmp_dir / f"{level}_{r:03d}_{bi:02d}_{k:02d}.wav")
                d = _util.ffprobe_duration(norm)
                parts.append(norm)
                cursor += d
                block_dur += d
                texts.append(clip["transcript"])
                k += 1
            segments.append({
                "language": lang,
                "start": round(block_start, 3),
                "end": round(cursor, 3),
                "transcript": " ".join(texts),
            })
            if bi < len(chosen_langs) - 1:
                pause_ms = rng.randint(*config.SWITCH_PAUSE_MS_RANGE)
                sil = _util.make_silence(pause_ms / 1000.0,
                                         tmp_dir / f"{level}_{r:03d}_{bi:02d}_sil.wav")
                parts.append(sil)
                cursor += pause_ms / 1000.0

        dest = out_audio / f"ls-{level}-{r:03d}.wav"
        _util.concat_wavs(parts, dest)
        entries.append({
            "id": f"ls-{level}-{r:03d}",
            "audio_path": _util.rel_to_data(dest),
            "level": level,
            "duration_s": round(_util.ffprobe_duration(dest), 2),
            "segments": segments,
        })
        print(f"  + ls-{level}-{r:03d}  langs={chosen_langs}  segs={len(segments)}")

    return entries


def prepare(levels: list[str]) -> None:
    if not _util.have_ffmpeg():
        raise RuntimeError("ffmpeg + ffprobe are required to generate this benchmark")
    pool = _load_pool()
    rng = _util.rng("language_switching")
    for level in levels:
        entries = _generate_level(level, pool, rng)
        _util.write_jsonl(
            config.manifest_path("language_switching", f"ls_{level}"), entries,
            header=f"Language switching {level} | generated from FLEURS "
                   f"| frozen {config.SUITE_VERSION} "
                   f"| seed={config.MASTER_SEED + config.SUBSEEDS['language_switching']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate language-switching benchmark")
    ap.add_argument("--levels", nargs="+", default=list(config.LANGUAGE_SWITCHING_LEVELS))
    prepare(ap.parse_args().levels)


if __name__ == "__main__":
    main()
