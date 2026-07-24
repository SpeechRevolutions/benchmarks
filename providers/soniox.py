"""
Soniox provider (model: stt-async-v5).

Async multi-step: upload file -> create transcription job -> poll status ->
fetch transcript. Soniox returns subword *tokens* with millisecond timings; we
merge them into words (a new word begins on a leading-space token) and convert
ms -> seconds. Diarization adds a per-token speaker string.

Docs: https://soniox.com/docs/stt/async/async-transcription
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from ._common import AsyncProvider, require_key, to_float
from .base import Features, JobStatus
from .types import Transcript, Word

BASE = "https://api.soniox.com/v1"


class SonioxProvider(AsyncProvider):
    name = "soniox"
    model = "stt-async-v5"
    price_per_audio_hour_usd = 0.10  # async (verify)

    poll_interval_s = 5.0
    poll_timeout_s = 3600.0

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or require_key("SONIOX_API_KEY", self.name)
        if model:
            self.model = model
        self._auth = {"Authorization": f"Bearer {self.api_key}"}

    def _start(self, audio: bytes, audio_path: Path, features: Features) -> dict:
        up = requests.post(
            f"{BASE}/files", headers=self._auth,
            files={"file": (audio_path.name, audio, "application/octet-stream")},
            timeout=600,
        )
        if up.status_code not in (200, 201):
            raise RuntimeError(f"soniox upload HTTP {up.status_code}: {up.text[:200]}")
        file_id = up.json()["id"]

        body: dict[str, Any] = {"model": self.model, "file_id": file_id}
        if features.speaker_labels:
            body["enable_speaker_diarization"] = True
        if features.language:
            body["language_hints"] = [features.language]
        # Keyword biasing: Soniox takes a free-text `context` of domain terms.
        if features.custom_vocabulary:
            body["context"] = ", ".join(features.custom_vocabulary)
        job = requests.post(
            f"{BASE}/transcriptions",
            headers={**self._auth, "Content-Type": "application/json"},
            json=body, timeout=60,
        )
        if job.status_code not in (200, 201):
            raise RuntimeError(f"soniox create HTTP {job.status_code}: {job.text[:200]}")
        return {"id": job.json()["id"], "file_id": file_id}

    def _check(self, handle: dict) -> JobStatus:
        resp = requests.get(f"{BASE}/transcriptions/{handle['id']}", headers=self._auth, timeout=30)
        if resp.status_code != 200:
            return JobStatus.RUNNING
        status = resp.json().get("status")
        if status == "completed":
            return JobStatus.COMPLETED
        if status == "error":
            return JobStatus.FAILED
        return JobStatus.RUNNING

    def _fetch(self, handle: dict) -> Any:
        resp = requests.get(f"{BASE}/transcriptions/{handle['id']}/transcript",
                            headers=self._auth, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        words = _merge_tokens(raw.get("tokens", []))
        text = raw.get("text") or " ".join(w.text for w in words)
        return Transcript(text=text, words=words, raw=raw)


def _merge_tokens(tokens: list[dict]) -> list[Word]:
    """Merge Soniox subword tokens into words; ms -> s. New word on leading space."""
    words: list[Word] = []
    cur: Word | None = None
    for t in tokens:
        txt = t.get("text", "")
        if txt.strip() == "":  # pure spacing token closes the current word
            cur = None
            continue
        starts_new = txt[:1] in (" ", "\n", "\t")
        piece = txt.strip()
        start = _ms(t.get("start_ms"))
        end = _ms(t.get("end_ms"))
        spk = t.get("speaker")
        if cur is None or starts_new:
            cur = Word(text=piece, start=start, end=end, speaker=spk)
            words.append(cur)
        else:
            cur.text += piece
            cur.end = end
    return words


def _ms(v: Any) -> float | None:
    return to_float(v, 1000.0)
