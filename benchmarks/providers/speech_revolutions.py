"""
Speech Revolutions provider.

Uses the official ``speechrevolutions`` Python SDK rather than talking to the
upload/poll/download REST endpoints directly. Defaults to the production API
(``https://api.speechrevolutions.com``); set ``SR_API_URL`` to point at a
different deployment (e.g. a local stack during development).

submit/poll/download stay split (rather than using the SDK's blocking
``transcribe()``) so the harness can measure submit latency and total latency
separately, same as every other async provider in this suite.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from speechrevolutions import SpeechRevolutions
from speechrevolutions.models import TranscribeOptions

from ._common import require_key, to_float
from .base import Features, Job, JobStatus, Provider
from .types import Segment, Transcript, Word


@lru_cache(maxsize=1)
def _load_custom_vocab() -> tuple[str, ...]:
    """Glossary from the file at $SR_CUSTOM_VOCAB (one term per line); () if unset."""
    path = os.environ.get("SR_CUSTOM_VOCAB")
    if not path or not os.path.exists(path):
        return ()
    with open(path) as f:
        return tuple(w.strip() for w in f if w.strip())


class SpeechRevolutionsProvider(Provider):
    name = "speech_revolutions"

    # No published per-hour price; price/perf scoring uses RTF and throughput
    # instead of dollars unless a price is supplied via env.
    price_per_audio_hour_usd = None

    # Never cache: whichever deployment SR_API_URL points at may be under
    # active development, so every run should re-transcribe rather than serve
    # a stale cached result.
    cacheable = False

    poll_interval_s = 5.0
    poll_timeout_s = 3600.0
    initial_wait_s = 0.0
    submission_workers = 8

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        *,
        price_per_audio_hour_usd: float | None = None,
    ) -> None:
        key = api_key or require_key("SPEECHREVOLUTIONS_API_KEY", self.name)
        base_url = api_url or os.environ.get("SR_API_URL")
        client_kwargs: dict[str, Any] = {"api_key": key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = SpeechRevolutions(**client_kwargs)
        if price_per_audio_hour_usd is not None:
            self.price_per_audio_hour_usd = price_per_audio_hour_usd

    # ── submit ───────────────────────────────────────────────────────────────

    def submit(self, audio_path: Path, features: Features, *, entry_id: str = "",
               meta: dict | None = None) -> Job:
        audio_data = Path(audio_path).read_bytes()
        # Custom-vocabulary feature: the framework supplies the per-dataset glossary via
        # Features.custom_vocabulary (identical across providers). SR_CUSTOM_VOCAB is a
        # legacy env override for ad-hoc runs.
        vocab = tuple(features.custom_vocabulary) or _load_custom_vocab()
        options = TranscribeOptions(
            output_type="json",
            word_timestamps=features.word_timestamps,
            speaker_labels=features.speaker_labels,
            nltk=features.punctuation,
            custom_vocabulary=list(vocab) or None,
        )
        upload_job = self._client.create_upload_job(len(audio_data), options)
        # From here the job EXISTS server-side. If the upload or the completion call
        # fails, the job is registered, will never complete and will never fail --
        # and the platform sheds customer intake when a small number of those sit
        # past their deadline. So a failure here must be turned into a terminal
        # state before it is raised, or a bad benchmark run takes production down
        # for everyone. Cancelling writes a failed_jobs row, which is exactly what
        # the stranded-job query looks for.
        try:
            self._client.upload_audio(upload_job.upload_url, audio_data,
                                      job_id=upload_job.job_id)
            self._client.complete_upload(upload_job.job_id)
        except BaseException:
            try:
                self._client.cancel_job(upload_job.job_id)
            except Exception:
                # Best effort: the original failure is the one worth reporting, and
                # a job we could not cancel ages out of the breaker's cohort anyway.
                pass
            raise
        return Job(
            entry_id=entry_id or upload_job.job_id,
            provider=self.name,
            native_id=upload_job.job_id,
            audio_path=Path(audio_path),
            features=features,
            handle={"download_url": upload_job.download_url},
            meta=meta or {},
        )

    # ── poll ─────────────────────────────────────────────────────────────────

    def poll(self, job: Job) -> JobStatus:
        status = self._client.get_job_status(job.native_id)
        if status.is_completed:
            download_url = status.download_url or job.handle.get("download_url")
            job.handle["_content"] = self._client.download_result(download_url)
            return JobStatus.COMPLETED
        if status.is_failed:
            return JobStatus.FAILED
        return JobStatus.RUNNING

    # ── download ──────────────────────────────────────────────────────────────

    def download(self, job: Job) -> Any:
        if "_content" in job.handle:
            return job.handle["_content"]
        return self._client.download_result(job.handle["download_url"])

    # ── normalize ──────────────────────────────────────────────────────────────

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        data = json.loads(raw) if isinstance(raw, (bytes, str, bytearray)) else raw

        words: list[Word] = []
        for w in data.get("words", []) or []:
            words.append(Word(
                text=w.get("word", w.get("text", "")),
                start=_to_float(w.get("start")),
                end=_to_float(w.get("end")),
                speaker=w.get("speaker"),
                language=w.get("language"),
                prob=_to_float(w.get("prob")),
            ))

        text = data.get("text") or " ".join(w.text for w in words)

        # The API emits the speaker-diarization timeline under "diarization"
        # (aggregation_cluster emits proper speaker turns there). Scoring THAT is
        # correct; reconstructing turns from per-word labels is the known-bad path
        # (single-stream words can't represent overlap). Fall back to "segments"
        # only for older payloads.
        segments: list[Segment] = []
        for s in data.get("diarization") or data.get("segments") or []:
            if {"start", "end", "speaker"} <= set(s):
                segments.append(Segment(float(s["start"]), float(s["end"]), str(s["speaker"])))

        return Transcript(
            text=text,
            words=words,
            segments=segments,
            language=data.get("language"),
            raw=data,
        )


_to_float = to_float
