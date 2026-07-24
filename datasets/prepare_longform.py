#!/usr/bin/env python3
"""
Prepare the Long-form Stability benchmark.

Long-form recordings (30 min - 4 h) from public sources:
  - AMI full meetings (Mix-Headset, ~30-45 min) — reference from word annotations
  - Earnings21 full files (~10-50 min)            — reference from .nlp tokens
  - optional: user-supplied CC-licensed podcast manifest (--podcasts file.jsonl)

A frozen seeded sample of config.SIZES['longform'] recordings is selected,
biased toward the longest available so the suite exercises real long-audio
failure modes. No manual clip curation — selection is deterministic.

Writes:
    manifests/longform/longform_all.jsonl
    audio under data/benchmarks/longform/audio/

Run prepare_ami (and optionally prepare_earnings21) first.

Usage:
    python -m benchmarks.datasets.prepare_longform
    python -m benchmarks.datasets.prepare_longform --podcasts my_podcasts.jsonl
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .. import config
from ..core.manifest import load_manifest
from . import _util
from .prepare_ami import _parse_words
from .prepare_earnings21 import _parse_nlp, _text


def _ami_candidates() -> list[dict]:
    ann_dir = config.DATA_ROOT / "diarization" / "annotations"
    audio_dir = config.audio_dir("diarization")
    out: list[dict] = []
    for wav in sorted(audio_dir.glob("*.Mix-Headset.wav")):
        meeting = wav.name.split(".")[0]
        words = _parse_words(meeting, ann_dir)
        if not words:
            continue
        transcript = " ".join(w["word"] for w in words)
        out.append({
            "src_audio": wav, "source": "ami", "id": f"longform-ami-{meeting}",
            "transcript": transcript, "duration_s": round(_util.ffprobe_duration(wav), 2),
        })
    return out


def _earnings21_candidates(repo_dir: Path) -> list[dict]:
    nlp_dir = repo_dir / "earnings21" / "transcripts" / "nlp_references"
    media_dir = repo_dir / "earnings21" / "media"
    if not nlp_dir.exists():
        return []
    out: list[dict] = []
    for nlp in sorted(nlp_dir.glob("*.nlp")):
        fid = nlp.stem
        media = next(iter(media_dir.glob(f"{fid}.*")), None)
        if not media:
            continue
        tokens = _parse_nlp(nlp)
        if not tokens:
            continue
        out.append({
            "src_audio": media, "source": "earnings21", "id": f"longform-earnings21-{fid}",
            "transcript": _text(tokens), "duration_s": round(_util.ffprobe_duration(media), 2),
        })
    return out


def prepare(repo_dir: Path, podcasts: Path | None) -> None:
    candidates = _ami_candidates() + _earnings21_candidates(repo_dir)
    if podcasts and podcasts.exists():
        for e in load_manifest(podcasts):
            candidates.append({**e, "src_audio": config.DATA_ROOT / e["audio_path"],
                               "source": e.get("source", "podcast")})

    if not candidates:
        raise RuntimeError(
            "No long-form candidates found. Run prepare_ami (and optionally "
            "prepare_earnings21) first, or pass --podcasts.")

    # Prefer the longest recordings, then take a deterministic seeded sample so
    # selection is frozen rather than purely length-ranked.
    candidates.sort(key=lambda c: c["duration_s"], reverse=True)
    pool = candidates[: max(config.SIZES["longform"] * 2, config.SIZES["longform"])]
    chosen = _util.seeded_sample(pool, config.SIZES["longform"], "longform")
    chosen.sort(key=lambda c: c["id"])

    out_audio = config.audio_dir("longform")
    out_audio.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for c in chosen:
        dest = out_audio / f"{c['id']}.wav"
        if not dest.exists():
            src = Path(c["src_audio"])
            if src.suffix.lower() == ".wav":
                try:
                    dest.symlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dest)
            elif _util.have_ffmpeg():
                _util.to_wav(src, dest)
        entries.append({
            "id": c["id"], "audio_path": _util.rel_to_data(dest),
            "transcript": c["transcript"], "source": c["source"],
            "duration_s": round(_util.ffprobe_duration(dest), 2) if dest.exists() else c["duration_s"],
        })
        print(f"  + {c['id']}  source={c['source']}  dur={c['duration_s']:.0f}s")

    _util.write_jsonl(
        config.manifest_path("longform", "longform_all"), entries,
        header=f"Long-form stability | AMI + Earnings21 (+podcasts) "
               f"| frozen {config.SUITE_VERSION} | n={len(entries)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare long-form stability benchmark")
    ap.add_argument("--repo", default=str(config.DATA_ROOT / "_earnings21_repo"))
    ap.add_argument("--podcasts", default=None, help="optional JSONL of CC podcast recordings")
    args = ap.parse_args()
    prepare(Path(args.repo), Path(args.podcasts) if args.podcasts else None)


if __name__ == "__main__":
    main()
