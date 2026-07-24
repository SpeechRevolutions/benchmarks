#!/usr/bin/env python3
"""
Prepare Earnings21 for the WER, entity, and diarization benchmarks.

Earnings21 (Rev.com, public, Apache-2.0) is long-form earnings-call audio with
token-level NLP references. We clone the dataset repo, parse each ``.nlp``
reference (header-named columns: token, speaker, optional ts/endTs,
punctuation), and build:

    manifests/wer/wer_earnings21.jsonl          (seeded sample of segments)
    manifests/entities/entities_earnings21.jsonl (same segments, for NER)
    manifests/diarization/diarization_earnings21.jsonl (speaker turns; needs ts)

Segmentation: when token timestamps are present, each file is cut into ~30 s
windows on word boundaries to produce a pool, from which a frozen seeded sample
of config.SIZES['wer_earnings21'] segments is drawn. Without timestamps, whole
files are used and diarization is skipped (it needs timing).

Requires: git, ffmpeg. Repo is large (~10 GB of audio).

Usage:
    python -m benchmarks.datasets.prepare_earnings21
    python -m benchmarks.datasets.prepare_earnings21 --repo /path/to/speech-datasets
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .. import config
from . import _util

REPO_URL = "https://github.com/revdotcom/speech-datasets.git"
SEGMENT_TARGET_S = 30.0


def _clone(repo_dir: Path) -> Path:
    if repo_dir.exists():
        print(f"  using existing repo: {repo_dir}")
        return repo_dir
    print(f"  cloning {REPO_URL} (shallow) ...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)], check=True)
    return repo_dir


def _parse_nlp(path: Path) -> list[dict]:
    """Parse an Earnings21 .nlp reference into token dicts.

    Columns are named in the header line. We detect the delimiter and map by
    name so this is robust to column reordering. Recognized names:
    token, punctuation, speaker, ts (start), endTs (end).
    """
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not raw:
        return []
    delim = "|" if "|" in raw[0] else ("," if "," in raw[0] else "\t")
    header = [h.strip().lower() for h in raw[0].split(delim)]

    def col(name: str) -> int | None:
        return header.index(name) if name in header else None

    i_tok, i_punc = col("token"), col("punctuation")
    i_spk, i_ts, i_end = col("speaker"), col("ts"), col("endts")

    tokens: list[dict] = []
    for line in raw[1:]:
        if not line.strip():
            continue
        parts = line.split(delim)
        if i_tok is None or i_tok >= len(parts):
            continue
        tok = parts[i_tok].strip()
        if not tok:
            continue
        punc = parts[i_punc].strip() if i_punc is not None and i_punc < len(parts) else ""
        d: dict = {"token": tok, "punc": punc}
        if i_spk is not None and i_spk < len(parts):
            d["speaker"] = parts[i_spk].strip() or "spk"
        if i_ts is not None and i_ts < len(parts):
            d["start"] = _f(parts[i_ts])
        if i_end is not None and i_end < len(parts):
            d["end"] = _f(parts[i_end])
        tokens.append(d)
    return tokens


def _f(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _has_timing(tokens: list[dict]) -> bool:
    timed = [t for t in tokens if t.get("start") is not None and t.get("end") is not None]
    return len(timed) > 0.5 * max(len(tokens), 1)


def _text(tokens: list[dict]) -> str:
    return "".join(t["token"] + (t["punc"] if t.get("punc") not in (None, "") else "") + " "
                   for t in tokens).strip()


def _segments_from_timing(tokens: list[dict]) -> list[tuple[float, float, list[dict]]]:
    """Cut tokens into ~SEGMENT_TARGET_S windows on word boundaries.

    Untimed tokens that fall WITHIN a segment's timed span are kept in the
    reference: the sliced audio still contains those spoken words, so dropping
    them would make the provider's (correct) transcription of them count as
    insertions and inflate WER. Only leading/trailing untimed tokens (whose audio
    lies outside the [start, end] slice) are excluded.
    """
    out: list[tuple[float, float, list[dict]]] = []
    cur: list[dict] = []
    seg_start = None
    last_end = None
    for t in tokens:
        has_time = t.get("start") is not None
        if seg_start is None:
            if not has_time:
                continue  # leading untimed token with no audio anchor
            seg_start = t["start"]
        cur.append(t)  # include timed AND in-span untimed tokens
        if has_time and t.get("end") is not None:
            last_end = t["end"]
            if last_end - seg_start >= SEGMENT_TARGET_S:
                out.append((seg_start, last_end, cur))
                cur, seg_start, last_end = [], None, None
    if cur and seg_start is not None and last_end is not None:
        while cur and cur[-1].get("start") is None:  # drop trailing untimed (past last_end)
            cur.pop()
        if cur:
            out.append((seg_start, last_end, cur))
    return out


def _slice_audio(src: Path, start: float, end: float, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-ar", "16000", "-ac", "1", str(dest)],
        check=True, capture_output=True,
    )
    return dest


def prepare(repo_dir: Path) -> None:
    repo = _clone(repo_dir)
    base = repo / "earnings21"
    nlp_dir = base / "transcripts" / "nlp_references"
    media_dir = base / "media"
    if not nlp_dir.exists():
        raise FileNotFoundError(f"Expected Earnings21 references at {nlp_dir}")

    wer_audio = config.audio_dir("wer")
    diar_audio = config.audio_dir("diarization")
    file_ids = sorted(p.stem for p in nlp_dir.glob("*.nlp"))

    seg_pool: list[dict] = []     # candidate WER/entity segments
    diar_entries: list[dict] = []

    for fid in file_ids:
        tokens = _parse_nlp(nlp_dir / f"{fid}.nlp")
        if not tokens:
            continue
        media = next(iter(media_dir.glob(f"{fid}.*")), None)

        if media and _has_timing(tokens) and _util.have_ffmpeg():
            for si, (a, b, toks) in enumerate(_segments_from_timing(tokens)):
                dest = wer_audio / f"earnings21-{fid}-{si:03d}.wav"
                if not dest.exists():
                    _slice_audio(media, a, b, dest)
                seg_pool.append({
                    "id": f"earnings21-{fid}-{si:03d}",
                    "audio_path": _util.rel_to_data(dest),
                    "transcript": _text(toks),
                    "duration_s": round(b - a, 2),
                })
            # Diarization: speaker turns across the whole file.
            turns = _speaker_turns(tokens)
            if media and turns:
                ddest = diar_audio / f"earnings21-{fid}.wav"
                if not ddest.exists():
                    _util.to_wav(media, ddest)
                diar_entries.append({
                    "id": f"earnings21-{fid}",
                    "audio_path": _util.rel_to_data(ddest),
                    "speakers": turns,
                    "duration_s": round(_util.ffprobe_duration(ddest), 2),
                })
        elif media:
            dest = wer_audio / f"earnings21-{fid}.wav"
            if not dest.exists() and _util.have_ffmpeg():
                _util.to_wav(media, dest)
            seg_pool.append({
                "id": f"earnings21-{fid}",
                "audio_path": _util.rel_to_data(dest),
                "transcript": _text(tokens),
                "duration_s": round(_util.ffprobe_duration(dest), 2) if dest.exists() else 0.0,
            })
        print(f"  parsed {fid}: {len(tokens)} tokens")

    chosen = _util.seeded_sample(seg_pool, config.SIZES["wer_earnings21"], "wer_earnings21")
    chosen.sort(key=lambda e: e["id"])
    _util.write_jsonl(
        config.manifest_path("wer", "wer_earnings21"), chosen,
        header=f"WER Earnings21 | frozen {config.SUITE_VERSION} | n={len(chosen)}")
    _util.write_jsonl(
        config.manifest_path("entities", "entities_earnings21"), chosen,
        header=f"Entity accuracy Earnings21 | frozen {config.SUITE_VERSION} | n={len(chosen)}")
    if diar_entries:
        _util.write_jsonl(
            config.manifest_path("diarization", "diarization_earnings21"), diar_entries,
            header=f"Diarization Earnings21 | collar={config.DER_COLLAR_S}s "
                   f"| frozen {config.SUITE_VERSION} | n={len(diar_entries)}")
    else:
        print("  [note] no token timing -> diarization_earnings21 skipped "
              "(needs forced alignment)")


def _speaker_turns(tokens: list[dict]) -> list[dict]:
    """Merge consecutive same-speaker timed tokens into turns."""
    timed = [t for t in tokens if t.get("start") is not None and t.get("end") is not None
             and t.get("speaker")]
    if not timed:
        return []
    turns: list[dict] = []
    cur_spk = timed[0]["speaker"]
    a, b = timed[0]["start"], timed[0]["end"]
    for t in timed[1:]:
        if t["speaker"] == cur_spk:
            b = t["end"]
        else:
            turns.append({"speaker": cur_spk, "start": round(a, 3), "end": round(b, 3)})
            cur_spk, a, b = t["speaker"], t["start"], t["end"]
    turns.append({"speaker": cur_spk, "start": round(a, 3), "end": round(b, 3)})
    return turns


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Earnings21 WER/entity/diarization data")
    ap.add_argument("--repo", default=str(config.DATA_ROOT / "_earnings21_repo"),
                    help="path to clone (or existing) revdotcom/speech-datasets")
    prepare(Path(ap.parse_args().repo))


if __name__ == "__main__":
    main()
