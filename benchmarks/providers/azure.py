"""
Azure AI Speech — Batch Transcription provider.

Async, URL-only: Azure batch transcription accepts audio by URL (blob/SAS or
public), not raw bytes, so the benchmark audio must be hosted (set
AZURE_AUDIO_BASE_URL — see public_audio_url). Requires SPEECH_KEY + SPEECH_REGION.

Flow: POST :submit -> poll the returned self URI (status Succeeded) -> GET
links.files -> download the Transcription result JSON. Word/segment times come
in 100-ns ticks; diarization adds a per-phrase integer speaker.

Docs: https://learn.microsoft.com/azure/ai-services/speech-service/batch-transcription
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from ._common import AsyncProvider, public_audio_url, require_key, to_float
from .base import Features, JobStatus
from .types import Transcript, Word

API_VERSION = "2024-11-15"

# Minimal ISO-639-1 -> Azure locale map (extend as needed).
LOCALE = {
    "en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE", "it": "it-IT",
    "pt": "pt-BR", "nl": "nl-NL", "pl": "pl-PL", "ru": "ru-RU", "ar": "ar-EG",
    "he": "he-IL", "hi": "hi-IN", "ur": "ur-PK", "tr": "tr-TR", "zh": "zh-CN",
    "ja": "ja-JP", "ko": "ko-KR", "vi": "vi-VN", "th": "th-TH", "id": "id-ID",
    "sv": "sv-SE", "no": "nb-NO", "fi": "fi-FI", "el": "el-GR", "ro": "ro-RO",
}
TICKS_PER_SECOND = 10_000_000


class AzureProvider(AsyncProvider):
    name = "azure"
    price_per_audio_hour_usd = 1.0  # ~$1/hr standard STT (verify)

    poll_interval_s = 10.0
    poll_timeout_s = 3600.0

    def __init__(self, speech_key: str | None = None, region: str | None = None) -> None:
        self.key = speech_key or require_key("SPEECH_KEY", self.name)
        self.region = region or require_key("SPEECH_REGION", self.name)
        self.base = f"https://{self.region}.api.cognitive.microsoft.com/speechtotext"

    def _headers(self) -> dict:
        return {"Ocp-Apim-Subscription-Key": self.key, "Content-Type": "application/json"}

    def _start(self, audio: bytes, audio_path: Path, features: Features) -> dict:
        url = public_audio_url(audio_path, "AZURE_AUDIO_BASE_URL", self.name)
        locale = LOCALE.get(features.language, "en-US") if features.language else "en-US"
        body = {
            "contentUrls": [url],
            "locale": locale,
            "displayName": f"benchmark-{audio_path.stem}",
            "properties": {
                "wordLevelTimestampsEnabled": bool(features.word_timestamps),
                "diarizationEnabled": bool(features.speaker_labels),
            },
        }
        resp = requests.post(f"{self.base}/transcriptions:submit?api-version={API_VERSION}",
                             headers=self._headers(), json=body, timeout=60)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"azure submit HTTP {resp.status_code}: {resp.text[:300]}")
        self_uri = resp.json().get("self")
        if not self_uri:
            raise RuntimeError(f"azure submit: no self URI in {resp.text[:200]}")
        return {"self": self_uri}

    def _check(self, handle: dict) -> JobStatus:
        resp = requests.get(handle["self"], headers=self._headers(), timeout=30)
        if resp.status_code != 200:
            return JobStatus.RUNNING
        data = resp.json()
        status = data.get("status")
        if status == "Succeeded":
            handle["_files_url"] = data.get("links", {}).get("files")
            return JobStatus.COMPLETED
        if status == "Failed":
            return JobStatus.FAILED
        return JobStatus.RUNNING

    def _fetch(self, handle: dict) -> Any:
        files_url = handle.get("_files_url") or (
            requests.get(handle["self"], headers=self._headers(), timeout=30)
            .json().get("links", {}).get("files")
        )
        files = requests.get(files_url, headers=self._headers(), timeout=60).json()
        for entry in files.get("values", []):
            if entry.get("kind") == "Transcription":
                content_url = entry.get("links", {}).get("contentUrl")
                r = requests.get(content_url, timeout=60)
                r.raise_for_status()
                return r.json()
        raise RuntimeError("azure: no Transcription result file found")

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        text = " ".join(p.get("display", "")
                        for p in raw.get("combinedRecognizedPhrases", [])).strip()
        words: list[Word] = []
        for phrase in raw.get("recognizedPhrases", []):
            spk = phrase.get("speaker")
            spk_label = f"speaker_{spk}" if spk is not None else None
            nbest = phrase.get("nBest", [])
            if not nbest:
                continue
            for w in nbest[0].get("words", []):
                start = _ticks(w.get("offsetInTicks"))
                dur = _ticks(w.get("durationInTicks"))
                words.append(Word(
                    text=w.get("word", ""), start=start,
                    end=(start + dur) if (start is not None and dur is not None) else None,
                    speaker=spk_label,
                ))
        return Transcript(text=text, words=words, raw=raw)


def _ticks(v: Any) -> float | None:
    return to_float(v, TICKS_PER_SECOND)
