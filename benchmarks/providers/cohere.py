"""
Cohere provider (model: cohere-transcribe-03-2026).

Synchronous multipart upload to /v2/audio/transcriptions. Text-only: no word
timestamps, no diarization (the API exposes neither), so timestamp/diarization
benchmarks will see no word data — an honest reflection of the API. A language
is required (no auto-detect); defaults to English when the benchmark gives none.

Docs: https://docs.cohere.com/reference/create-audio-transcription
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from ._common import SyncProvider, require_key
from .base import Features
from .types import Transcript

ENDPOINT = "https://api.cohere.com/v2/audio/transcriptions"


class CohereProvider(SyncProvider):
    name = "cohere"
    model = "cohere-transcribe-03-2026"
    price_per_audio_hour_usd = None  # no published per-hour rate

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or require_key("CO_API_KEY", self.name)
        if model:
            self.model = model

    def _transcribe(self, audio: bytes, audio_path: Path, features: Features) -> Any:
        data = {"model": self.model, "language": features.language or "en"}
        resp = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data=data,
            files={"file": (audio_path.name, audio, "application/octet-stream")},
            timeout=self.request_timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"cohere HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        return Transcript(text=raw.get("text", ""), raw=raw)
