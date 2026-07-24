"""
OpenAI provider (model: gpt-4o-transcribe).

Synchronous multipart upload to /v1/audio/transcriptions.

IMPORTANT capability gap: gpt-4o-transcribe returns TEXT ONLY. It does not
support word-level timestamps (timestamp_granularities requires
response_format=verbose_json, which this model does not support) nor speaker
diarization. So this adapter yields a Transcript with text but no words/segments.
The timestamp and diarization benchmarks will therefore score it as having no
word data — an honest reflection of the model's batch API, not a harness bug.

(For reference: word timestamps require whisper-1; speaker labels require the
separate gpt-4o-transcribe-diarize model. Neither is gpt-4o-transcribe.)

Docs: https://developers.openai.com/api/docs/guides/speech-to-text
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from ._common import SyncProvider, require_key
from .base import Features
from .types import Transcript

ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
MAX_BYTES = 25 * 1024 * 1024  # 25 MB API limit
_PROMPT_MAX_CHARS = 900  # ~200 tokens, under the model's ~224-token prompt ceiling


def _fit_prompt(vocab: tuple[str, ...]) -> str:
    """As many glossary terms as fit under the prompt token ceiling."""
    out: list[str] = []
    n = 0
    for term in vocab:
        n += len(term) + 2
        if n > _PROMPT_MAX_CHARS:
            break
        out.append(term)
    return ", ".join(out)


class OpenAIProvider(SyncProvider):
    name = "openai"
    model = "gpt-4o-transcribe"
    price_per_audio_hour_usd = 6.0  # ~$0.006/min audio-equivalent (verify)

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or require_key("OPENAI_API_KEY", self.name)
        if model:
            self.model = model

    def _transcribe(self, audio: bytes, audio_path: Path, features: Features) -> Any:
        if len(audio) > MAX_BYTES:
            raise RuntimeError(
                f"openai: file {audio_path.name} is {len(audio)/1e6:.0f} MB > 25 MB API limit"
            )
        data = {"model": self.model, "response_format": "json"}
        if features.language:
            data["language"] = features.language  # ISO-639-1
        # OpenAI's only biasing lever is the free-text `prompt`. It's capped at ~224
        # tokens, so we pass as many glossary terms as fit (honest API limitation).
        if features.custom_vocabulary:
            data["prompt"] = _fit_prompt(features.custom_vocabulary)
        resp = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data=data,
            files={"file": (audio_path.name, audio, "application/octet-stream")},
            timeout=self.request_timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"openai HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        # Text-only model: no words, no segments.
        return Transcript(text=raw.get("text", ""), raw=raw)
