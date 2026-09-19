"""
Shared base classes for HTTP-based commercial STT providers.

Two lifecycle shapes cover every vendor:

  SyncProvider  — one blocking HTTP request returns the transcript
                  (Deepgram, OpenAI, ElevenLabs, Mistral, Cohere, ...).
                  submit() performs the request and stashes the raw payload;
                  poll() reports COMPLETED immediately; download() returns it.

  AsyncProvider — upload/create a job, poll for status, then fetch the result
                  (Gladia, Soniox, Azure Batch, ...). Subclass implements
                  _start(), _check(), and _fetch().

Concrete providers implement only the request/parse specifics plus normalize().
API keys are read from environment variables (see require_key()).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from .base import Features, Job, JobStatus, Provider


class MissingKeyError(RuntimeError):
    pass


def require_key(env_var: str, provider: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise MissingKeyError(
            f"{provider}: set the {env_var} environment variable with your API key."
        )
    return key


def to_float(v: Any, scale: float = 1.0) -> float | None:
    """Coerce a value to float, dividing by ``scale`` (e.g. 1000 for ms, ticks for
    100-ns units). Returns None for missing or non-numeric values."""
    try:
        return float(v) / scale if v is not None else None
    except (TypeError, ValueError):
        return None


def public_audio_url(audio_path: Path, base_env: str, provider: str) -> str:
    """Map a local _data audio file to a public URL.

    Providers that only accept a URL (Azure Batch, Qwen3 file-transcription)
    require the benchmark audio to be reachable over HTTP. Host the
    benchmarks/_data directory (S3, blob+SAS, static server, ...) and set
    ``base_env`` to its base URL; the file's path relative to DATA_ROOT is
    appended.
    """
    from ..config import DATA_ROOT

    base = os.environ.get(base_env)
    if not base:
        raise MissingKeyError(
            f"{provider}: audio must be hosted at a public URL. Set {base_env} to the "
            f"base URL serving benchmarks/_data (e.g. an S3 or blob-SAS base), "
            f"so a local file resolves to {base_env}/<path-relative-to-_data>."
        )
    rel = Path(audio_path).resolve().relative_to(DATA_ROOT.resolve())
    return base.rstrip("/") + "/" + str(rel).replace("\\", "/")


# ── Synchronous providers ───────────────────────────────────────────────────

class SyncProvider(Provider):
    """One request returns everything. Subclass implements _transcribe()."""

    request_timeout_s: int = 600

    def _transcribe(self, audio: bytes, audio_path: Path, features: Features) -> Any:
        """Perform the blocking HTTP request; return the raw payload (dict/bytes)."""
        raise NotImplementedError

    def submit(self, audio_path: Path, features: Features, *, entry_id: str = "",
               meta: dict | None = None) -> Job:
        audio = Path(audio_path).read_bytes()
        raw = self._transcribe(audio, Path(audio_path), features)
        return Job(
            entry_id=entry_id or str(audio_path),
            provider=self.name,
            native_id=entry_id or str(audio_path),
            audio_path=Path(audio_path),
            features=features,
            handle={"_raw": raw},
            meta=meta or {},
        )

    def poll(self, job: Job) -> JobStatus:
        return JobStatus.COMPLETED

    def download(self, job: Job) -> Any:
        return job.handle["_raw"]


# ── Asynchronous providers ───────────────────────────────────────────────────

class AsyncProvider(Provider):
    """Upload/create -> poll -> fetch. Subclass implements _start/_check/_fetch."""

    def _start(self, audio: bytes, audio_path: Path, features: Features) -> dict:
        """Kick off the job. Return a handle dict (e.g. {'id': ...} or {'result_url': ...})."""
        raise NotImplementedError

    def _check(self, handle: dict) -> JobStatus:
        """Return the job's current status from the provider."""
        raise NotImplementedError

    def _fetch(self, handle: dict) -> Any:
        """Fetch the raw result payload for a completed job."""
        raise NotImplementedError

    def submit(self, audio_path: Path, features: Features, *, entry_id: str = "",
               meta: dict | None = None) -> Job:
        audio = Path(audio_path).read_bytes()
        handle = self._start(audio, Path(audio_path), features)
        return Job(
            entry_id=entry_id or str(audio_path),
            provider=self.name,
            native_id=str(handle.get("id", entry_id or audio_path)),
            audio_path=Path(audio_path),
            features=features,
            handle=handle,
            meta=meta or {},
        )

    def poll(self, job: Job) -> JobStatus:
        return self._check(job.handle)

    def download(self, job: Job) -> Any:
        return self._fetch(job.handle)


# ── small HTTP helpers ────────────────────────────────────────────────────────

def post_multipart(url: str, headers: dict, files: dict, data: dict | None = None,
                   timeout: int = 600) -> requests.Response:
    return requests.post(url, headers=headers, files=files, data=data or {}, timeout=timeout)


def get_json(url: str, headers: dict, timeout: int = 60) -> Any:
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
