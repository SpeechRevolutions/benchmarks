"""
Deepgram provider (model: Nova-3).

Synchronous pre-recorded API: POST raw audio bytes to /v1/listen, full JSON
returned in one response. Word timestamps are on by default; diarization via
diarize=true adds an integer speaker to each word.

Docs: https://developers.deepgram.com/reference/speech-to-text/listen-pre-recorded
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

import requests

from ._common import SyncProvider, require_key, to_float
from .base import Features
from .types import Transcript, Word

ENDPOINT = "https://api.deepgram.com/v1/listen"
# Deepgram caps keyterms at "500 tokens across all keyterms". Rare/proper-noun glossary
# terms tokenize into many subwords, so a token *estimate* is unreliable — empirically
# 100 keyterms stays safely under the ceiling for our glossaries while 124 does not.
# The provider gets as much of the shared glossary as its API allows (see methodology).
_MAX_KEYTERMS = 100


def _fit_keyterms(vocab: tuple[str, ...]) -> list[str]:
    return list(vocab[:_MAX_KEYTERMS])


class DeepgramProvider(SyncProvider):
    name = "deepgram"
    model = "nova-3"
    price_per_audio_hour_usd = 0.258  # Nova-3 pre-recorded, English (verify at run time)

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or require_key("DEEPGRAM_API_KEY", self.name)
        if model:
            self.model = model

    def _transcribe(self, audio: bytes, audio_path: Path, features: Features) -> Any:
        params = {
            "model": self.model,
            "smart_format": "true",
            "punctuate": "true" if features.punctuation else "false",
        }
        if features.speaker_labels:
            params["diarize"] = "true"
        if features.language:
            params["language"] = features.language  # BCP-47 hint
        else:
            params["detect_language"] = "true"  # multilingual auto-detect
        # Keyword biasing: nova-3 uses `keyterm` (repeatable query param; English-only).
        if features.custom_vocabulary:
            kt = _fit_keyterms(features.custom_vocabulary)
            if kt:
                params["keyterm"] = kt

        content_type = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
        resp = requests.post(
            ENDPOINT, params=params,
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": content_type},
            data=audio, timeout=self.request_timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"deepgram HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        try:
            alt = raw["results"]["channels"][0]["alternatives"][0]
        except (KeyError, IndexError):
            return Transcript(text="", raw=raw)
        words = [
            Word(
                text=w.get("word", ""),
                start=_f(w.get("start")),
                end=_f(w.get("end")),
                speaker=(f"speaker_{w['speaker']}" if "speaker" in w else None),
            )
            for w in alt.get("words", [])
        ]
        return Transcript(text=alt.get("transcript", ""), words=words,
                          language=raw.get("results", {}).get("channels", [{}])[0]
                          .get("detected_language"), raw=raw)


_f = to_float
