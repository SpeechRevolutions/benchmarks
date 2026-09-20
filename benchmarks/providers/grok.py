"""
Grok / xAI provider (POST /v1/stt).

Synchronous raw-bytes multipart upload. The STT endpoint takes no model
parameter. Word timestamps (seconds) are returned by default; diarize=true adds
an integer speaker per word.

Docs: https://docs.x.ai/developers/rest-api-reference/inference/voice
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from ._common import SyncProvider, require_key
from .base import Features
from .types import Transcript, Word

ENDPOINT = "https://api.x.ai/v1/stt"


class GrokProvider(SyncProvider):
    name = "grok"
    price_per_audio_hour_usd = None  # not published on the endpoint page

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or require_key("XAI_API_KEY", self.name)

    def _transcribe(self, audio: bytes, audio_path: Path, features: Features) -> Any:
        data: dict[str, Any] = {}
        if features.speaker_labels:
            data["diarize"] = "true"
        if features.language:
            data["language"] = features.language
        resp = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data=data,
            files={"file": (audio_path.name, audio, "application/octet-stream")},
            timeout=self.request_timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"grok HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        words = [
            Word(text=w.get("text", ""), start=_f(w.get("start")), end=_f(w.get("end")),
                 speaker=(f"speaker_{w['speaker']}" if "speaker" in w else None))
            for w in raw.get("words", [])
        ]
        return Transcript(text=raw.get("text", ""), words=words,
                          language=raw.get("language") or None, raw=raw)


def _f(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
