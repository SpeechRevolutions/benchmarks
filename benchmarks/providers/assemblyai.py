"""
AssemblyAI provider.

Async raw-bytes flow: upload the file to /v2/upload (returns an upload_url),
submit a transcript job to /v2/transcript, poll /v2/transcript/{id} until
status == "completed". Word timestamps are always returned (milliseconds);
speaker_labels=true adds a per-word speaker.

Docs: https://www.assemblyai.com/docs/api-reference/transcripts/submit
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from ._common import AsyncProvider, require_key
from .base import Features, JobStatus
from .types import Transcript, Word

BASE = "https://api.assemblyai.com"


class AssemblyAIProvider(AsyncProvider):
    name = "assemblyai"
    price_per_audio_hour_usd = 0.27  # Universal pre-recorded, ~$0.0045/min (verify)

    poll_interval_s = 5.0
    poll_timeout_s = 3600.0

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or require_key("ASSEMBLYAI_API_KEY", self.name)
        self._auth = {"authorization": self.api_key}

    def _start(self, audio: bytes, audio_path: Path, features: Features) -> dict:
        up = requests.post(f"{BASE}/v2/upload", headers=self._auth, data=audio, timeout=600)
        if up.status_code != 200:
            raise RuntimeError(f"assemblyai upload HTTP {up.status_code}: {up.text[:200]}")
        upload_url = up.json()["upload_url"]

        body: dict[str, Any] = {
            "audio_url": upload_url,
            "punctuate": features.punctuation,
        }
        if features.speaker_labels:
            body["speaker_labels"] = True
        if features.language:
            body["language_code"] = features.language
        else:
            body["language_detection"] = True
        # Keyword biasing (same glossary as every keyword-capable provider).
        if features.custom_vocabulary:
            body["word_boost"] = list(features.custom_vocabulary)
            body["boost_param"] = "high"
        job = requests.post(f"{BASE}/v2/transcript",
                            headers={**self._auth, "content-type": "application/json"},
                            json=body, timeout=60)
        if job.status_code != 200:
            raise RuntimeError(f"assemblyai submit HTTP {job.status_code}: {job.text[:200]}")
        return {"id": job.json()["id"]}

    def _check(self, handle: dict) -> JobStatus:
        resp = requests.get(f"{BASE}/v2/transcript/{handle['id']}", headers=self._auth, timeout=30)
        if resp.status_code != 200:
            return JobStatus.RUNNING
        data = resp.json()
        status = data.get("status")
        if status == "completed":
            handle["_result"] = data
            return JobStatus.COMPLETED
        if status == "error":
            return JobStatus.FAILED
        return JobStatus.RUNNING

    def _fetch(self, handle: dict) -> Any:
        if "_result" in handle:
            return handle["_result"]
        return requests.get(f"{BASE}/v2/transcript/{handle['id']}",
                            headers=self._auth, timeout=60).json()

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        words = [
            Word(text=w.get("text", ""), start=_ms(w.get("start")), end=_ms(w.get("end")),
                 speaker=(f"speaker_{w['speaker']}" if w.get("speaker") is not None else None))
            for w in raw.get("words", []) or []
        ]
        return Transcript(text=raw.get("text") or "", words=words,
                          language=raw.get("language_code"), raw=raw)


def _ms(v: Any) -> float | None:
    try:
        return float(v) / 1000.0 if v is not None else None
    except (TypeError, ValueError):
        return None
