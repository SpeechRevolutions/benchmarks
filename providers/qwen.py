"""
Qwen3-ASR provider (Alibaba DashScope, model: qwen3-asr-flash-filetrans).

Async file transcription. DashScope's batch ASR takes audio by URL, so the
benchmark audio must be hosted publicly (set QWEN_AUDIO_BASE_URL — see
public_audio_url). Word timestamps via enable_words (milliseconds); no speaker
diarization (Qwen3-ASR does not support it).

Flow: submit (X-DashScope-Async: enable) -> poll GET /tasks/{id} -> on SUCCEEDED
download the per-file transcription_url JSON.

Docs: https://www.alibabacloud.com/help/en/model-studio/qwen-asr-api-reference
NOTE: exact DashScope field names can drift between API versions; verify against
your account's region/endpoint on first live run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from ._common import AsyncProvider, public_audio_url, require_key, to_float
from .base import Features, JobStatus
from .types import Transcript, Word

# Region host: intl by default; override with DASHSCOPE_BASE_URL if needed.
DEFAULT_HOST = "https://dashscope-intl.aliyuncs.com"
SUBMIT_PATH = "/api/v1/services/audio/asr/transcription"
TASK_PATH = "/api/v1/tasks/{task_id}"


class QwenProvider(AsyncProvider):
    name = "qwen"
    model = "qwen3-asr-flash-filetrans"
    price_per_audio_hour_usd = None  # no official published rate

    poll_interval_s = 5.0
    poll_timeout_s = 3600.0

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or require_key("DASHSCOPE_API_KEY", self.name)
        self.host = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_HOST).rstrip("/")
        if model:
            self.model = model

    def _headers(self, async_submit: bool = False) -> dict:
        h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if async_submit:
            h["X-DashScope-Async"] = "enable"
        return h

    def _start(self, audio: bytes, audio_path: Path, features: Features) -> dict:
        url = public_audio_url(audio_path, "QWEN_AUDIO_BASE_URL", self.name)
        params: dict[str, Any] = {"enable_words": bool(features.word_timestamps)}
        if features.language:
            params["language"] = features.language
        body = {
            "model": self.model,
            "input": {"file_urls": [url]},
            "parameters": params,
        }
        resp = requests.post(self.host + SUBMIT_PATH, headers=self._headers(async_submit=True),
                             json=body, timeout=60)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"qwen submit HTTP {resp.status_code}: {resp.text[:300]}")
        out = resp.json().get("output", {})
        task_id = out.get("task_id")
        if not task_id:
            raise RuntimeError(f"qwen submit: no task_id in {resp.text[:200]}")
        return {"id": task_id}

    def _check(self, handle: dict) -> JobStatus:
        resp = requests.get(self.host + TASK_PATH.format(task_id=handle["id"]),
                            headers=self._headers(), timeout=30)
        if resp.status_code != 200:
            return JobStatus.RUNNING
        out = resp.json().get("output", {})
        status = out.get("task_status")
        if status == "SUCCEEDED":
            handle["_output"] = out
            return JobStatus.COMPLETED
        if status in ("FAILED", "UNKNOWN"):
            return JobStatus.FAILED
        return JobStatus.RUNNING

    def _fetch(self, handle: dict) -> Any:
        out = handle.get("_output", {})
        # Result JSON is referenced by a URL; support the documented shapes.
        result = out.get("result") or {}
        result_url = result.get("transcription_url")
        if not result_url:
            for r in out.get("results", []) or []:
                if r.get("transcription_url"):
                    result_url = r["transcription_url"]
                    break
        if not result_url:
            raise RuntimeError("qwen: no transcription_url in task output")
        r = requests.get(result_url, timeout=60)
        r.raise_for_status()
        return r.json()

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        transcripts = raw.get("transcripts", [])
        if not transcripts:
            return Transcript(text="", raw=raw)
        t0 = transcripts[0]
        words: list[Word] = []
        for sent in t0.get("sentences", []):
            for w in sent.get("words", []):
                words.append(Word(text=w.get("text", "").strip(),
                                  start=_ms(w.get("begin_time")), end=_ms(w.get("end_time"))))
        return Transcript(text=t0.get("text", ""), words=words, raw=raw)


def _ms(v: Any) -> float | None:
    return to_float(v, 1000.0)
