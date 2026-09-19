"""Provider layer: vendor-agnostic STT interface + adapters."""

from .base import Features, Job, JobStatus, Provider, TranscriptionResult
from .registry import available_providers, get_provider, register
from .types import Segment, Transcript, Word

__all__ = [
    "Features", "Job", "JobStatus", "Provider", "TranscriptionResult",
    "Transcript", "Word", "Segment",
    "get_provider", "available_providers", "register",
]
