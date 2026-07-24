#!/usr/bin/env python3
"""
Prepare SPGISpeech (Kensho / S&P Global) for the WER benchmark.

SPGISpeech is 5,000 h of earnings-call audio transcribed in a CLEAN, fully-
formatted, NON-VERBATIM (orthographic) style — punctuation, casing, formatted
numbers, and NO disfluencies/false-starts. It is the same domain as Earnings21
but with a *clean* reference, so it measures post-processed-transcript quality
without the verbatim-vs-clean penalty that Earnings21's Rev.com reference imposes.

We pull the frozen HuggingFace `test` split (parquet with embedded WAV bytes),
draw a seeded sample of config.SIZES['wer_spgispeech'] short segments, write each
segment's audio, and build:

    manifests/wer/wer_spgispeech.jsonl   (seeded sample; scored with normalize_text)

The reference is already clean, so the WER benchmark scores it like clean/other
(standard normalization on both sides — NOT the verbatim disfluency-stripping used
for earnings21).

Usage:
    HF_TOKEN=... python -m benchmarks.datasets.prepare_spgispeech
    python -m benchmarks.datasets.prepare_spgispeech --parquet <path-to-shard>
"""

from __future__ import annotations

import argparse
import io
import os
import wave
from pathlib import Path

import pyarrow.parquet as pq

from .. import config
from . import _util

# Default: the first test shard downloaded under _data/spgispeech_raw/test/.
_DEFAULT_SHARD = config.DATA_ROOT / "spgispeech_raw" / "test" / "test-00000-of-00003.parquet"


def _wav_duration(raw: bytes) -> float:
    with wave.open(io.BytesIO(raw), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def prepare(parquet_path: Path) -> None:
    if not parquet_path.exists():
        raise SystemExit(
            f"SPGISpeech shard not found: {parquet_path}\n"
            "Download it first, e.g.:\n"
            "  HF_TOKEN=... python -c \"from huggingface_hub import hf_hub_download; "
            "hf_hub_download('kensho/spgispeech','test/test-00000-of-00003.parquet',"
            "repo_type='dataset',local_dir='benchmarks/_data/spgispeech_raw')\""
        )

    audio_dir = config.audio_dir("wer")
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Build the pool of (row-index) candidates, then draw a frozen seeded sample.
    table = pq.read_table(parquet_path, columns=["wav_filename", "audio", "transcript"])
    n_rows = table.num_rows
    pool = list(range(n_rows))
    chosen_idx = _util.seeded_sample(pool, config.SIZES["wer_spgispeech"], "wer_spgispeech")

    wav_filename = table["wav_filename"]
    audio_col = table["audio"]
    transcript = table["transcript"]

    entries: list[dict] = []
    for rank, i in enumerate(sorted(chosen_idx)):
        raw = audio_col[i].as_py()["bytes"]
        text = transcript[i].as_py().strip()
        if not raw or not text:
            continue
        # Stable id from the source path (dir/idx.wav -> spgi-<dir8>-<idx>).
        src = wav_filename[i].as_py()
        parts = src.replace(".wav", "").split("/")
        sid = f"spgi-{parts[0][:8]}-{parts[-1]}" if len(parts) >= 2 else f"spgi-{rank:04d}"

        dest = audio_dir / f"{sid}.wav"
        dest.write_bytes(raw)
        entries.append({
            "id": sid,
            "audio_path": _util.rel_to_data(dest),
            "transcript": text,
            "duration_s": round(_wav_duration(raw), 2),
        })

    n = _util.write_jsonl(
        config.manifest_path("wer", "wer_spgispeech"),
        entries,
        header=(f"WER wer_spgispeech | SPGISpeech (Kensho) test split | CLEAN reference | "
                f"seed={config.MASTER_SEED}+{config.SUBSEEDS['wer_spgispeech']} | n={len(entries)}"),
    )
    total_s = sum(e["duration_s"] for e in entries)
    print(f"  wrote {n} segments ({total_s/60:.1f} min audio) -> {config.manifest_path('wer','wer_spgispeech')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=_DEFAULT_SHARD,
                    help="path to a SPGISpeech test/dev parquet shard")
    args = ap.parse_args()
    prepare(args.parquet)


if __name__ == "__main__":
    main()
