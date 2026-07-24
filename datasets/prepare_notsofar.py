#!/usr/bin/env python3
"""
Prepare NOTSOFAR-1 (Microsoft) for the diarization benchmark (DER + cpWER).

NOTSOFAR-1 is now hosted on Hugging Face (public, ungated: microsoft/NOTSOFAR).
Each meeting provides a single-channel far-field stream ``sc_*/ch0.wav`` — the
closest analog to a file a user would upload — plus ``gt_transcription.json``
with per-utterance speaker_id / start_time / end_time / text / word_timing.

We download only ONE single-channel device + the GT per meeting (~12 MB each),
for a frozen seeded sample of config.SIZES['diarization_notsofar'] meetings from
dev-set-1 (240825.1_dev1, ground truth available). No HF token needed (the
dataset is public); we fetch individual files anonymously.

License: CC BY 4.0 (publishing benchmark results is permitted, with attribution).

Writes:
    manifests/diarization/diarization_notsofar.jsonl  (speakers + speaker_transcripts)
    audio under data/benchmarks/diarization_notsofar/audio/

Usage:
    python -m benchmarks.datasets.prepare_notsofar
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from .. import config
from . import _util

REPO = "microsoft/NOTSOFAR"
SUBSET = "dev_set"
VERSION = "240825.1_dev1"   # dev-set-1, ground truth available
MTG_PREFIX = f"benchmark-datasets/{SUBSET}/{VERSION}/MTG"
HF_API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
HF_RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def _api_list(path: str) -> list[dict]:
    with urllib.request.urlopen(f"{HF_API}/{path}?recursive=false", timeout=60) as r:
        return json.loads(r.read())


def _list_meetings() -> list[str]:
    return sorted(e["path"].split("/")[-1]
                  for e in _api_list(MTG_PREFIX) if e["type"] == "directory")


def _pick_sc_device(meeting: str) -> str | None:
    """Deterministically pick one single-channel device folder for a meeting."""
    entries = _api_list(f"{MTG_PREFIX}/{meeting}")
    sc = sorted(e["path"].split("/")[-1] for e in entries
                if e["type"] == "directory" and e["path"].split("/")[-1].startswith("sc_"))
    return sc[0] if sc else None


def _download(rel_path: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    urllib.request.urlretrieve(f"{HF_RESOLVE}/{rel_path}", dest)


def _parse_gt(gt: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Return (speaker segments, per-speaker concatenated text) from gt_transcription."""
    segments: list[dict] = []
    by_spk: dict[str, list[tuple[float, str]]] = {}
    for u in gt:
        spk = u.get("speaker_id") or u.get("speaker")
        start, end = u.get("start_time"), u.get("end_time")
        text = u.get("text") or u.get("words") or ""
        if spk is None or start is None or end is None:
            continue
        segments.append({"speaker": spk, "start": round(float(start), 3),
                         "end": round(float(end), 3)})
        if text.strip():
            by_spk.setdefault(spk, []).append((float(start), text))
    segments.sort(key=lambda s: s["start"])
    speaker_transcripts = {
        spk: " ".join(t for _, t in sorted(ws)) for spk, ws in by_spk.items()
    }
    return segments, speaker_transcripts


def prepare() -> None:
    meetings = _list_meetings()
    print(f"NOTSOFAR dev-set-1: {len(meetings)} meetings available")
    chosen = _util.seeded_sample(meetings, config.SIZES["diarization_notsofar"],
                                 "diarization_notsofar")
    chosen.sort()

    audio_dir = config.audio_dir("diarization_notsofar")
    entries: list[dict] = []
    for m in chosen:
        sc = _pick_sc_device(m)
        if not sc:
            print(f"  [SKIP] {m}: no single-channel device")
            continue
        wav_rel = f"{MTG_PREFIX}/{m}/{sc}/ch0.wav"
        gt_rel = f"{MTG_PREFIX}/{m}/gt_transcription.json"
        dest_wav = audio_dir / f"{m}_{sc}.wav"
        dest_gt = config.DATA_ROOT / "diarization_notsofar" / "gt" / f"{m}.json"
        try:
            _download(wav_rel, dest_wav)
            _download(gt_rel, dest_gt)
        except Exception as e:  # noqa: BLE001
            print(f"  [SKIP] {m}: download failed: {e}")
            continue
        gt = json.loads(dest_gt.read_text())
        segments, speaker_transcripts = _parse_gt(gt)
        if not segments:
            print(f"  [SKIP] {m}: no usable GT")
            continue
        entries.append({
            "id": f"notsofar-{m}",
            "audio_path": _util.rel_to_data(dest_wav),
            "speakers": segments,
            "speaker_transcripts": speaker_transcripts,
            "duration_s": round(_util.ffprobe_duration(dest_wav), 2),
            "device": sc,
        })
        print(f"  + {m} ({sc})  speakers={len(speaker_transcripts)}  segs={len(segments)}")

    _util.write_jsonl(
        config.manifest_path("diarization", "diarization_notsofar"), entries,
        header=f"Diarization NOTSOFAR-1 dev1 | single-channel (sc) | DER+cpWER "
               f"| frozen {config.SUITE_VERSION} "
               f"| seed={config.MASTER_SEED + config.SUBSEEDS['diarization_notsofar']} "
               f"| n={len(entries)} | CC BY 4.0 (microsoft/NOTSOFAR)")


def main() -> None:
    argparse.ArgumentParser(description="Prepare NOTSOFAR-1 diarization data").parse_args()
    prepare()


if __name__ == "__main__":
    main()
