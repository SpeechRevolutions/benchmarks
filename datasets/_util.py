"""
Shared helpers for deterministic dataset preparation.

Every prepare script uses a fixed sub-seed (config.SUBSEEDS) so the frozen
benchmark subsets are identical on every machine. Audio is written under
config.audio_dir(<benchmark>); manifests under config.manifest_path(...).
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

from .. import config


# ── Download / media ─────────────────────────────────────────────────────────

def download(url: str, dest: Path, *, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  cached     {dest.name}")
        return dest
    print(f"  downloading {url} -> {dest.name}")
    urllib.request.urlretrieve(url, dest)
    print(f"  done        {dest.name} ({dest.stat().st_size / 1024**2:.1f} MB)")
    return dest


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def to_wav(src: Path, dest: Path, *, sample_rate: int = 16000, mono: bool = True) -> Path:
    """Transcode any audio to 16 kHz mono WAV (deterministic, no resample dither)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", str(sample_rate)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += [str(dest)]
    subprocess.run(cmd, check=True, capture_output=True)
    return dest


def make_silence(duration_s: float, dest: Path, *, sample_rate: int = 16000) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"anullsrc=r={sample_rate}:cl=mono", "-t", f"{duration_s:.3f}", str(dest)],
        check=True, capture_output=True,
    )
    return dest


def concat_wavs(parts: Sequence[Path], dest: Path) -> Path:
    """Concatenate WAVs (same format) via ffmpeg concat demuxer."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    listing = dest.with_suffix(".concat.txt")
    with open(listing, "w") as f:
        for p in parts:
            f.write(f"file '{p.resolve()}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", str(dest)],
        check=True, capture_output=True,
    )
    listing.unlink(missing_ok=True)
    return dest


# ── Determinism ───────────────────────────────────────────────────────────────

def rng(benchmark_key: str) -> random.Random:
    """A seeded RNG dedicated to one benchmark's selection."""
    return random.Random(config.MASTER_SEED + config.SUBSEEDS[benchmark_key])


def seeded_sample(items: Sequence, k: int, benchmark_key: str) -> list:
    """Deterministically sample k items (or all if fewer) using the benchmark seed."""
    items = list(items)
    if k >= len(items):
        return items
    return rng(benchmark_key).sample(items, k)


# ── Manifest writing ───────────────────────────────────────────────────────────

def write_jsonl(path: Path, entries: Iterable[dict], header: str | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        if header:
            for line in header.splitlines():
                f.write(f"# {line}\n")
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
            n += 1
    print(f"\nWrote {n} entries -> {path}")
    return n


def rel_to_data(path: Path) -> str:
    """Manifest audio_path relative to DATA_ROOT."""
    return str(Path(path).resolve().relative_to(config.DATA_ROOT.resolve()))
