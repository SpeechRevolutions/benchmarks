#!/usr/bin/env python3
"""
Prepare LibriSpeech subsets for the WER benchmark (and the price subset).

Downloads test-clean and test-other (CC BY 4.0, no registration), then freezes a
deterministic, seeded sample of N clips per split. The selection is identical on
every machine because it is drawn from a fixed seed (config.SUBSEEDS).

Writes:
    manifests/wer/wer_clean.jsonl   (config.SIZES['wer_clean'] clips)
    manifests/wer/wer_other.jsonl   (config.SIZES['wer_other'] clips)
    manifests/price/price_all.jsonl (small fixed subset of clean, for timing)
    audio under data/benchmarks/wer/audio/

Usage:
    python -m benchmarks.datasets.prepare_librispeech
    python -m benchmarks.datasets.prepare_librispeech --split test-clean
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from .. import config
from . import _util

URLS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}
SPLIT_TO_SUBSET = {"test-clean": "wer_clean", "test-other": "wer_other"}
PRICE_SUBSET_SIZE = 20  # frozen subset of clean used by the price benchmark


def _parse_trans(trans_file: Path) -> dict[str, str]:
    mapping = {}
    with open(trans_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def _collect_split(split: str, tmp_dir: Path) -> list[dict]:
    """Return every utterance in a split as {utt_id, flac_src, transcript}."""
    tar_path = tmp_dir / f"{split}.tar.gz"
    _util.download(URLS[split], tar_path)
    if not (tmp_dir / "LibriSpeech" / split).exists():
        print(f"  extracting {tar_path.name} ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp_dir)

    root = tmp_dir / "LibriSpeech" / split
    utts: list[dict] = []
    for trans_file in sorted(root.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        for utt_id, text in sorted(_parse_trans(trans_file).items()):
            flac = chapter_dir / f"{utt_id}.flac"
            if flac.exists():
                utts.append({"utt_id": utt_id, "flac_src": flac, "transcript": text})
    return utts


def prepare(splits: list[str]) -> None:
    if not _util.have_ffmpeg():
        print("  [WARN] ffprobe not found; durations will be 0.0")
    tmp_dir = config.DATA_ROOT / "wer" / "_tmp_librispeech"
    audio_out = config.audio_dir("wer")
    audio_out.mkdir(parents=True, exist_ok=True)

    clean_entries: list[dict] = []
    for split in splits:
        subset = SPLIT_TO_SUBSET[split]
        all_utts = _collect_split(split, tmp_dir)
        chosen = _util.seeded_sample(all_utts, config.SIZES[subset], subset)
        chosen.sort(key=lambda u: u["utt_id"])

        entries: list[dict] = []
        for u in chosen:
            dest = audio_out / f"{u['utt_id']}.flac"
            if not dest.exists():
                shutil.copy2(u["flac_src"], dest)
            entries.append({
                "id": f"ls-{split.replace('-', '')}-{u['utt_id']}",
                "audio_path": _util.rel_to_data(dest),
                "transcript": u["transcript"],
                "duration_s": round(_util.ffprobe_duration(dest), 2),
            })
        _util.write_jsonl(
            config.manifest_path("wer", subset), entries,
            header=f"WER {subset} | LibriSpeech {split} | frozen {config.SUITE_VERSION} "
                   f"| seed={config.MASTER_SEED + config.SUBSEEDS[subset]} | n={len(entries)}",
        )
        if split == "test-clean":
            clean_entries = entries

    # Price subset: first PRICE_SUBSET_SIZE clean clips (deterministic order).
    if clean_entries:
        price_entries = clean_entries[:PRICE_SUBSET_SIZE]
        _util.write_jsonl(
            config.manifest_path("price", "price_all"), price_entries,
            header=f"Price/perf timing set | LibriSpeech test-clean subset "
                   f"| frozen {config.SUITE_VERSION} | n={len(price_entries)}",
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare LibriSpeech WER + price subsets")
    ap.add_argument("--split", nargs="+", default=["test-clean", "test-other"],
                    choices=["test-clean", "test-other"])
    prepare(ap.parse_args().split)


if __name__ == "__main__":
    main()
