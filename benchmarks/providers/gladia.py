"""
Gladia provider (api.gladia.io v2, model: solaria-1).

Async two-step flow: upload the file to get an audio_url, POST a pre-recorded
job, then poll the result_url until status == "done". Words live under
transcription.utterances[].words[]; speaker is per-utterance.

Docs: https://docs.gladia.io/api-reference/pre-recorded-flow
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from ._common import AsyncProvider, require_key
from .base import Features, JobStatus
from .types import Transcript, Word

UPLOAD_URL = "https://api.gladia.io/v2/upload"
PRERECORDED_URL = "https://api.gladia.io/v2/pre-recorded"


class GladiaProvider(AsyncProvider):
    name = "gladia"
    model = "solaria-1"
    price_per_audio_hour_usd = 0.61  # async PAYG (verify)

    poll_interval_s = 5.0
    poll_timeout_s = 3600.0

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or require_key("GLADIA_API_KEY", self.name)
        if model:
            self.model = model
        self._headers = {"x-gladia-key": self.api_key}

    def _start(self, audio: bytes, audio_path: Path, features: Features) -> dict:
        up = requests.post(
            UPLOAD_URL, headers=self._headers,
            files={"audio": (audio_path.name, audio, "application/octet-stream")},
            timeout=600,
        )
        if up.status_code not in (200, 201):
            raise RuntimeError(f"gladia upload HTTP {up.status_code}: {up.text[:200]}")
        audio_url = up.json()["audio_url"]

        body: dict[str, Any] = {"audio_url": audio_url, "model": self.model}
        if features.speaker_labels:
            body["diarization"] = True
        if features.language:
            body["language_config"] = {"languages": [features.language]}
        # Keyword biasing (same glossary as every keyword-capable provider).
        if features.custom_vocabulary:
            body["custom_vocabulary"] = True
            body["custom_vocabulary_config"] = {"vocabulary": list(features.custom_vocabulary)}
        job = requests.post(
            PRERECORDED_URL, headers={**self._headers, "Content-Type": "application/json"},
            json=body, timeout=60,
        )
        if job.status_code not in (200, 201):
            raise RuntimeError(f"gladia create HTTP {job.status_code}: {job.text[:200]}")
        data = job.json()
        return {"id": data["id"], "result_url": data["result_url"]}

    def _check(self, handle: dict) -> JobStatus:
        resp = requests.get(handle["result_url"], headers=self._headers, timeout=30)
        if resp.status_code != 200:
            return JobStatus.RUNNING
        data = resp.json()
        status = data.get("status")
        if status == "done":
            handle["_result"] = data
            return JobStatus.COMPLETED
        if status == "error":
            return JobStatus.FAILED
        return JobStatus.RUNNING

    def _fetch(self, handle: dict) -> Any:
        if "_result" in handle:
            return handle["_result"]
        return requests.get(handle["result_url"], headers=self._headers, timeout=30).json()

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        tr = raw.get("result", {}).get("transcription", {})
        words: list[Word] = []
        for utt in tr.get("utterances", []):
            spk = utt.get("speaker")
            spk_label = f"speaker_{spk}" if spk is not None else None
            for w in utt.get("words", []):
                words.append(Word(text=w.get("word", ""), start=_f(w.get("start")),
                                  end=_f(w.get("end")), speaker=spk_label))
        return Transcript(text=tr.get("full_transcript", ""), words=words,
                          language=tr.get("languages", [None])[0]
                          if tr.get("languages") else None, raw=raw)


def _f(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
