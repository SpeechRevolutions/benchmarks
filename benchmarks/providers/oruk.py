"""
Oruk provider (model: oruk-fourier-2b).

Synchronous multipart upload to POST https://speech-api.oruk.ai/v1/audio/transcriptions.

IMPORTANT capability scope: oruk-fourier-2b is English-only (``en``, ``en-US``,
``en-GB``) and returns TEXT ONLY. Its batch API does not expose word-level
timestamps, speaker diarization, or free-text prompt biasing (the endpoint
enforces strict request fields: ``file``, ``model``, ``response_format``, and
``language``). This adapter yields a ``Transcript`` with text and no
words/segments — matching OpenAI GPT-4o Transcribe and Cohere — so WER and
Entity Accuracy score it directly while timestamp/diarization benchmarks see no
word data (an honest reflection of the API, not a harness bug).

Docs: https://oruk.ai/docs
Pricing: https://oruk.ai/pricing ($0.0080/min = $0.48/hr, up to 30 MiB / 60 min)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from ._common import SyncProvider, require_key
from .base import Features
from .types import Transcript

ENDPOINT = "https://speech-api.oruk.ai/v1/audio/transcriptions"
MAX_BYTES = 30 * 1024 * 1024  # 30 MiB API limit
_ENGLISH_LOCALES = {
    "en": "en",
    "en-us": "en-US",
    "en-gb": "en-GB",
}


def _resolve_endpoint(base_url: str | None = None) -> str:
    raw = (base_url or os.environ.get("ORUK_BASE_URL") or "").strip()
    if not raw:
        return ENDPOINT
    cleaned = raw.rstrip("/")
    if cleaned.endswith("/v1/audio/transcriptions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/audio/transcriptions"
    return f"{cleaned}/v1/audio/transcriptions"


def _normalize_language(language: str | None) -> str:
    if not language:
        return "en"
    key = language.strip().lower().replace("_", "-")
    if key in _ENGLISH_LOCALES:
        return _ENGLISH_LOCALES[key]
    raise RuntimeError(
        f"oruk: oruk-fourier-2b supports English audio only (got language={language!r})"
    )


class OrukProvider(SyncProvider):
    name = "oruk"
    model = "oruk-fourier-2b"
    price_per_audio_hour_usd = 0.48  # $0.0080/min published at https://oruk.ai/pricing

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or require_key("ORUK_API_KEY", self.name)
        if model:
            self.model = model
        self.endpoint = _resolve_endpoint(base_url)

    def _transcribe(self, audio: bytes, audio_path: Path, features: Features) -> Any:
        if len(audio) > MAX_BYTES:
            raise RuntimeError(
                f"oruk: file {audio_path.name} is {len(audio)/1e6:.0f} MB > 30 MiB API limit"
            )
        data = {
            "model": self.model,
            "response_format": "json",
            "language": _normalize_language(features.language),
        }
        resp = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data=data,
            files={"file": (audio_path.name, audio, "application/octet-stream")},
            timeout=self.request_timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"oruk HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        # Text-only model: no words, no segments.
        return Transcript(
            text=raw.get("text", ""),
            language=raw.get("language") or "en",
            raw=raw,
        )
