"""
ElevenLabs provider (model: Scribe V2).

Synchronous multipart upload to /v1/speech-to-text. Word timestamps are on by
default; diarize=true adds a speaker_id (e.g. "speaker_0") to each word. The
words array includes "spacing" tokens which we drop.

Docs: https://elevenlabs.io/docs/api-reference/speech-to-text/convert
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from ._common import SyncProvider, require_key, to_float
from .base import Features
from .types import Transcript, Word

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"


class ElevenLabsProvider(SyncProvider):
    name = "elevenlabs"
    model = "scribe_v2"
    price_per_audio_hour_usd = 0.22  # Scribe batch (verify)

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or require_key("ELEVENLABS_API_KEY", self.name)
        if model:
            self.model = model

    def _transcribe(self, audio: bytes, audio_path: Path, features: Features) -> Any:
        data = {"model_id": self.model}
        if features.speaker_labels:
            data["diarize"] = "true"
        if features.language:
            data["language_code"] = features.language
        resp = requests.post(
            ENDPOINT,
            headers={"xi-api-key": self.api_key},
            data=data,
            files={"file": (audio_path.name, audio, "application/octet-stream")},
            timeout=self.request_timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"elevenlabs HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        words = [
            Word(
                text=w.get("text", ""),
                start=_f(w.get("start")),
                end=_f(w.get("end")),
                speaker=w.get("speaker_id"),
            )
            for w in raw.get("words", [])
            if w.get("type", "word") == "word"
        ]
        return Transcript(text=raw.get("text", ""), words=words,
                          language=raw.get("language_code"), raw=raw)


_f = to_float
