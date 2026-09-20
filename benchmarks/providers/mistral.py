"""
Mistral provider (model: Voxtral Mini).

Synchronous multipart upload to /v1/audio/transcriptions. Timed output comes
back as segments[]; with timestamp_granularities=["word"] each segment is one
word. diarize=true adds segments[].speaker_id.

API quirk: timestamp_granularities is NOT compatible with the `language` param.
So we request word granularity when word timestamps are needed (and omit the
language hint), otherwise we send the language hint. This matches how the
benchmarks use features: word-level benchmarks don't set a language, the
multilingual benchmark sets a language but doesn't need words.

Docs: https://docs.mistral.ai/api/endpoint/audio/transcriptions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from ._common import SyncProvider, require_key
from .base import Features
from .types import Transcript, Word

ENDPOINT = "https://api.mistral.ai/v1/audio/transcriptions"


class MistralProvider(SyncProvider):
    name = "mistral"
    model = "voxtral-mini-latest"
    price_per_audio_hour_usd = 0.18  # ~$0.003/min (verify)

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or require_key("MISTRAL_API_KEY", self.name)
        if model:
            self.model = model

    def _transcribe(self, audio: bytes, audio_path: Path, features: Features) -> Any:
        data: dict[str, Any] = {"model": self.model}
        if features.speaker_labels:
            data["diarize"] = "true"
        if features.word_timestamps:
            # word granularity is mutually exclusive with the language param
            data["timestamp_granularities"] = "word"
        elif features.language:
            data["language"] = features.language
        resp = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data=data,
            files={"file": (audio_path.name, audio, "application/octet-stream")},
            timeout=self.request_timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"mistral HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        words = [
            Word(text=s.get("text", "").strip(), start=_f(s.get("start")),
                 end=_f(s.get("end")), speaker=s.get("speaker_id"))
            for s in raw.get("segments", [])
            if s.get("text", "").strip()
        ]
        return Transcript(text=raw.get("text", ""), words=words,
                          language=raw.get("language"), raw=raw)


def _f(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
