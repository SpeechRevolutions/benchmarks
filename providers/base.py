"""
Provider interface.

Every Speech-to-Text vendor is wrapped in a :class:`Provider` subclass that
implements the four-method contract from the spec::

    submit(audio)   -> job handle
    poll(job)       -> status
    download(job)   -> raw provider payload (bytes/dict)
    normalize(raw)  -> Transcript

The base class adds a :meth:`transcribe_batch` convenience that drives the
submit -> poll -> download -> normalize lifecycle for many files in parallel and
records per-file wall-clock timing (consumed by the price/performance benchmark).
Subclasses only implement the four primitives.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .types import Transcript


@dataclass
class Features:
    """Transcription features requested for a job.

    Providers map these onto their own API parameters in ``submit``. Not every
    provider supports every feature; unsupported requests should degrade
    gracefully (e.g. return no word timestamps) rather than raise.
    """
    word_timestamps: bool = False
    speaker_labels: bool = False
    punctuation: bool = True
    language: str | None = None   # hint; None = auto-detect / multi
    #: Domain glossary for keyword/custom-vocabulary biasing. Empty = feature off
    #: (providers must return an untouched transcript). Applied identically across
    #: every provider that supports it — see ``Benchmark.run`` + docs/methodology.
    custom_vocabulary: tuple[str, ...] = ()


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Opaque-ish handle for an in-flight transcription job."""
    entry_id: str
    provider: str
    native_id: str                       # provider-side job/request id
    audio_path: Path
    features: Features
    handle: dict = field(default_factory=dict)  # provider-specific state
    submitted_at: float = 0.0
    meta: dict = field(default_factory=dict)     # manifest entry, carried through


@dataclass
class TranscriptionResult:
    """One file's end-to-end outcome, including timing for price/perf scoring."""
    entry_id: str
    transcript: Transcript | None
    status: JobStatus
    submit_latency_s: float = 0.0   # submit() wall time
    total_latency_s: float = 0.0    # submit -> completed wall time
    audio_duration_s: float = 0.0
    error: str | None = None
    meta: dict = field(default_factory=dict)
    from_cache: bool = False   # served from the raw-transcript cache (no API call)

    @property
    def ok(self) -> bool:
        return self.status == JobStatus.COMPLETED and self.transcript is not None

    @property
    def rtf(self) -> float | None:
        """Real-time factor: processing time / audio duration."""
        if self.audio_duration_s > 0 and self.total_latency_s > 0:
            return self.total_latency_s / self.audio_duration_s
        return None


class Provider:
    """Abstract base. Subclasses implement submit/poll/download/normalize."""

    name: str = "base"

    # Pricing metadata (USD per hour of audio). Override per provider. ``None``
    # means "not published / not applicable" and price scoring skips $ metrics.
    price_per_audio_hour_usd: float | None = None

    # Whether this provider's raw transcripts may be cached to disk and reused
    # on re-runs. True for external providers (fixed model versions, paid, so
    # output is immutable). MUST be False for the local Speech Revolutions
    # (Zephyr) model — it is actively being improved, so every run has to
    # re-transcribe to reflect local code changes; caching would mask gains.
    cacheable: bool = True

    # Tunables for the polling loop.
    poll_interval_s: float = 5.0
    poll_timeout_s: float = 3600.0
    initial_wait_s: float = 0.0
    submission_workers: int = 8
    # A job whose poll() raises this many times in a row is failed early with the
    # last error, rather than silently retrying until poll_timeout_s.
    poll_max_consecutive_failures: int = 5

    def _cache_salt(self) -> str:
        """Identity folded into the transcript cache key so a provider's cached
        output is invalidated when its model or the suite version changes."""
        from ..config import SUITE_VERSION
        return f"{getattr(self, 'model', '')}|{SUITE_VERSION}"

    # ── The four-method contract ─────────────────────────────────────────────

    def submit(self, audio_path: Path, features: Features, *, entry_id: str = "",
               meta: dict | None = None) -> Job:
        """Upload/submit one audio file. Returns a :class:`Job` handle."""
        raise NotImplementedError

    def poll(self, job: Job) -> JobStatus:
        """Return the current status of ``job`` without blocking."""
        raise NotImplementedError

    def download(self, job: Job) -> Any:
        """Fetch the raw provider payload for a completed ``job``."""
        raise NotImplementedError

    def normalize(self, raw: Any, *, features: Features | None = None) -> Transcript:
        """Convert a raw provider payload into a :class:`Transcript`."""
        raise NotImplementedError

    # ── Orchestration (shared) ───────────────────────────────────────────────

    def transcribe_batch(
        self,
        items: list[dict],
        features: Features,
        *,
        progress: bool = True,
        use_cache: bool = True,
    ) -> list[TranscriptionResult]:
        """Submit, poll, download and normalize a batch of files.

        Each item is a dict with at least ``id`` and ``audio_path`` (absolute
        Path) plus an optional ``duration_s`` and arbitrary manifest metadata
        carried through to the result.

        Raw transcripts are cached to disk and reused on re-runs — but only when
        ``use_cache`` AND ``self.cacheable`` are both true. Cacheable is False
        for the local Zephyr model (always re-transcribes); ``use_cache`` is set
        False by the price benchmark (needs a real API call to measure latency)
        or the CLI ``--refresh`` flag (force fresh transcription).
        """
        from ..core import transcript_cache as tc

        caching = use_cache and self.cacheable
        results: list[TranscriptionResult] = []
        to_fetch: list[dict] = []

        # ── Cache pass: serve immutable transcripts without touching the API ──
        if caching:
            for item in items:
                hit = tc.load(self.name, item["audio_path"], features, salt=self._cache_salt())
                if hit is None:
                    to_fetch.append(item)
                    continue
                try:
                    transcript = self.normalize(hit["raw"], features=features)
                except Exception:  # noqa: BLE001 - corrupt cache entry -> refetch
                    to_fetch.append(item)
                    continue
                results.append(TranscriptionResult(
                    entry_id=item["id"], transcript=transcript, status=JobStatus.COMPLETED,
                    submit_latency_s=hit.get("submit_latency_s", 0.0),
                    total_latency_s=hit.get("total_latency_s", 0.0),
                    audio_duration_s=float(item.get("duration_s", hit.get("audio_duration_s", 0.0))),
                    meta=item, from_cache=True,
                ))
                if progress:
                    print(f"  cached     {item['id']}")
        else:
            to_fetch = list(items)

        # ── Submit in parallel (cache misses only) ───────────────────────────
        jobs: dict[str, Job] = {}
        submit_latency: dict[str, float] = {}

        def _submit(item: dict) -> tuple[dict, Job | None, float, str | None]:
            t0 = time.monotonic()
            try:
                job = self.submit(
                    Path(item["audio_path"]),
                    features,
                    entry_id=item["id"],
                    meta=item,
                )
                return item, job, time.monotonic() - t0, None
            except Exception as e:  # noqa: BLE001 - report, don't crash the batch
                return item, None, time.monotonic() - t0, str(e)

        with ThreadPoolExecutor(max_workers=self.submission_workers) as pool:
            futures = [pool.submit(_submit, it) for it in to_fetch]
            for fut in as_completed(futures):
                item, job, lat, err = fut.result()
                if job is None:
                    results.append(TranscriptionResult(
                        entry_id=item["id"], transcript=None,
                        status=JobStatus.FAILED, submit_latency_s=lat,
                        audio_duration_s=float(item.get("duration_s", 0.0)),
                        error=f"submit failed: {err}", meta=item,
                    ))
                    if progress:
                        print(f"  FAILED submit  {item['id']}: {err}")
                else:
                    job.submitted_at = time.monotonic()
                    jobs[job.entry_id] = job
                    submit_latency[job.entry_id] = lat
                    if progress:
                        print(f"  submitted  {item['id']}")

        # ── Poll until all terminal or timeout ───────────────────────────────
        if self.initial_wait_s > 0 and jobs:
            if progress:
                print(f"  waiting {self.initial_wait_s:.0f}s before first poll ...")
            time.sleep(self.initial_wait_s)

        pending = dict(jobs)
        finished: dict[str, tuple[JobStatus, float]] = {}
        last_poll_error: dict[str, str] = {}
        fail_streak: dict[str, int] = {}
        poll_start = time.monotonic()

        while pending and (time.monotonic() - poll_start) < self.poll_timeout_s:
            done_ids: list[str] = []
            for eid, job in pending.items():
                try:
                    status = self.poll(job)
                except Exception as e:  # noqa: BLE001
                    # One-off blips are retried; a job that keeps erroring is failed
                    # early with the last error instead of hanging until timeout.
                    last_poll_error[eid] = str(e)
                    fail_streak[eid] = fail_streak.get(eid, 0) + 1
                    if fail_streak[eid] >= self.poll_max_consecutive_failures:
                        finished[eid] = (JobStatus.FAILED, time.monotonic() - job.submitted_at)
                        done_ids.append(eid)
                        if progress:
                            print(f"  FAILED    {eid}: {e}")
                    continue
                fail_streak.pop(eid, None)
                if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    finished[eid] = (status, time.monotonic() - job.submitted_at)
                    done_ids.append(eid)
                    if progress:
                        print(f"  {status.value:9} {eid}")
            for eid in done_ids:
                del pending[eid]
            if pending:
                time.sleep(self.poll_interval_s)

        for eid, job in pending.items():  # timed out
            finished[eid] = (JobStatus.FAILED, time.monotonic() - job.submitted_at)
            if progress:
                print(f"  TIMEOUT   {eid}")

        # ── Download + normalize ──────────────────────────────────────────────
        for eid, job in jobs.items():
            status, total_lat = finished.get(eid, (JobStatus.FAILED, 0.0))
            duration = float(job.meta.get("duration_s", 0.0))
            if status != JobStatus.COMPLETED:
                err = last_poll_error.get(eid)
                results.append(TranscriptionResult(
                    entry_id=eid, transcript=None, status=status,
                    submit_latency_s=submit_latency.get(eid, 0.0),
                    total_latency_s=total_lat, audio_duration_s=duration,
                    error=(f"job did not complete (last poll error: {err})" if err
                           else "job did not complete"),
                    meta=job.meta,
                ))
                continue
            try:
                raw = self.download(job)
                if caching:
                    tc.save(self.name, job.audio_path, job.features, raw,
                            salt=self._cache_salt(),
                            audio_duration_s=duration,
                            submit_latency_s=submit_latency.get(eid, 0.0),
                            total_latency_s=total_lat)
                transcript = self.normalize(raw, features=job.features)
                results.append(TranscriptionResult(
                    entry_id=eid, transcript=transcript, status=JobStatus.COMPLETED,
                    submit_latency_s=submit_latency.get(eid, 0.0),
                    total_latency_s=total_lat, audio_duration_s=duration,
                    meta=job.meta,
                ))
            except Exception as e:  # noqa: BLE001
                results.append(TranscriptionResult(
                    entry_id=eid, transcript=None, status=JobStatus.FAILED,
                    submit_latency_s=submit_latency.get(eid, 0.0),
                    total_latency_s=total_lat, audio_duration_s=duration,
                    error=f"download/normalize failed: {e}", meta=job.meta,
                ))

        # Preserve input order.
        order = {it["id"]: i for i, it in enumerate(items)}
        results.sort(key=lambda r: order.get(r.entry_id, 1 << 30))
        return results
