#!/usr/bin/env python3
"""
Prepare DiPCo (Dinner Party Corpus, Amazon) for the diarization benchmark
(DER + cpWER).

Public, no registration. License: CDLA-Permissive-1.0 (publishing benchmark
results is permitted). Downloaded from Zenodo and MD5-verified.

IMPORTANT methodology caveat: DiPCo has **no single mixed far-field WAV**. Each
session has 5 far-field arrays (7 channels each) + per-speaker close-talk. As the
closest analog to a single uploaded recording, we use ONE far-field channel
(config: device U01, channel CH1). This is stated so the number is interpreted
correctly; it is not a beamformed/enhanced signal.

Transcription timings are per-device dicts of ``HH:MM:SS.ffffff`` strings
(synchronized across devices) and are utterance-level (no word timings).

Writes:
    manifests/diarization/diarization_dipco.jsonl (speakers + speaker_transcripts)
    audio under data/benchmarks/diarization_dipco/audio/

Frozen set: the 5 DiPCo eval sessions (S01, S03, S06, S07, S08).

Usage:
    python -m benchmarks.datasets.prepare_dipco
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile

from .. import config
from . import _util

ZENODO_URL = "https://zenodo.org/records/8122551/files/DipCo.tgz"
EXPECTED_MD5 = "2297eb9334f3b90e02b54b708e501b24"
EVAL_SESSIONS = ["S01", "S03", "S06", "S07", "S08"]
FAR_FIELD_DEVICE = "U01"   # single far-field channel used as the "uploaded file"
FAR_FIELD_CHANNEL = "CH1"


def _root() -> "object":
    from pathlib import Path
    base = config.DATA_ROOT / "diarization_dipco"
    # extracted tree is Dipco/ (capital D, lowercase ipco)
    for name in ("Dipco", "DiPCo"):
        if (base / name).is_dir():
            return base / name
    return base / "Dipco"


def _ensure_data() -> None:
    from pathlib import Path
    base = config.DATA_ROOT / "diarization_dipco"
    base.mkdir(parents=True, exist_ok=True)
    root = _root()
    if (root / "transcriptions" / "eval").is_dir():
        print(f"  using extracted DiPCo at {root}")
        return
    tgz = base / "DiPCo.tgz"
    _util.download(ZENODO_URL, tgz)
    print("  verifying MD5 ...")
    h = hashlib.md5()
    with open(tgz, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != EXPECTED_MD5:
        raise RuntimeError(f"DiPCo MD5 mismatch: {h.hexdigest()} != {EXPECTED_MD5}")
    print("  MD5 OK; extracting ...")
    with tarfile.open(tgz) as tf:
        tf.extractall(base)


def _to_seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _time(val, device: str) -> float | None:
    """start_time/end_time is a per-device dict; prefer the chosen device, else close-talk."""
    if isinstance(val, dict):
        raw = val.get(device) or val.get("close-talk") or next(iter(val.values()), None)
    else:
        raw = val
    try:
        return _to_seconds(raw) if raw else None
    except (ValueError, AttributeError):
        return None


def prepare() -> None:
    _ensure_data()
    root = _root()
    audio_src = root / "audio" / "eval"
    trans_dir = root / "transcriptions" / "eval"
    out_audio = config.audio_dir("diarization_dipco")
    out_audio.mkdir(parents=True, exist_ok=True)

    # config.SIZES limits how many eval sessions to freeze (default: all 5)
    sessions = EVAL_SESSIONS[: config.SIZES.get("diarization_dipco", len(EVAL_SESSIONS))]
    entries: list[dict] = []

    for sess in sessions:
        wav_src = audio_src / f"{sess}_{FAR_FIELD_DEVICE}.{FAR_FIELD_CHANNEL}.wav"
        gt_path = trans_dir / f"{sess}.json"
        if not wav_src.exists() or not gt_path.exists():
            print(f"  [SKIP] {sess}: missing audio or transcription")
            continue

        gt = json.loads(gt_path.read_text())
        segments: list[dict] = []
        by_spk: dict[str, list[tuple[float, str]]] = {}
        for u in gt:
            spk = u.get("speaker_id")
            start = _time(u.get("start_time"), FAR_FIELD_DEVICE)
            end = _time(u.get("end_time"), FAR_FIELD_DEVICE)
            text = u.get("words") or ""
            if spk is None or start is None or end is None:
                continue
            segments.append({"speaker": spk, "start": round(start, 3), "end": round(end, 3)})
            if text.strip():
                by_spk.setdefault(spk, []).append((start, text))
        if not segments:
            print(f"  [SKIP] {sess}: no usable GT")
            continue
        segments.sort(key=lambda s: s["start"])
        speaker_transcripts = {s: " ".join(t for _, t in sorted(w)) for s, w in by_spk.items()}

        dest = out_audio / f"{sess}_{FAR_FIELD_DEVICE}_{FAR_FIELD_CHANNEL}.wav"
        if not dest.exists():
            shutil.copy2(wav_src, dest)
        entries.append({
            "id": f"dipco-{sess}",
            "audio_path": _util.rel_to_data(dest),
            "speakers": segments,
            "speaker_transcripts": speaker_transcripts,
            "duration_s": round(_util.ffprobe_duration(dest), 2),
            "far_field_device": f"{FAR_FIELD_DEVICE}.{FAR_FIELD_CHANNEL}",
        })
        print(f"  + {sess}  speakers={len(speaker_transcripts)}  segs={len(segments)}")

    _util.write_jsonl(
        config.manifest_path("diarization", "diarization_dipco"), entries,
        header=f"Diarization DiPCo eval | single far-field channel "
               f"{FAR_FIELD_DEVICE}.{FAR_FIELD_CHANNEL} (no mixed file exists) "
               f"| DER+cpWER | frozen {config.SUITE_VERSION} | n={len(entries)} "
               f"| CDLA-Permissive-1.0")


def main() -> None:
    argparse.ArgumentParser(description="Prepare DiPCo diarization data").parse_args()
    prepare()


if __name__ == "__main__":
    main()
