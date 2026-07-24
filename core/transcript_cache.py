"""
Raw-transcript cache.

Provider transcripts are expensive (paid API calls) but **immutable** — the raw
output for a given (provider, audio file, requested features) never changes. We
persist that raw payload to disk so re-evaluation is free: fixing a scoring
script (e.g. cpWER) or even a ``normalize()`` bug and re-running reuses the
cached raw and re-scores at zero API cost.

Layout:  benchmarks/_transcripts/<provider>/<key>.json
Key:     sha1(realpath(audio) + "|" + feature-signature)   — so different
         feature requests (e.g. with/without diarization) cache separately, and
         symlinked copies of the same file share a cache entry.

Stored raw is normalized to JSON where possible (every provider returns JSON or
JSON bytes); binary payloads fall back to base64. ``load()`` returns the raw in
the shape each provider's ``normalize()`` expects (a dict), plus the originally
measured timing (reused for reference; price/perf never uses the cache).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..config import PACKAGE_ROOT

CACHE_ROOT = PACKAGE_ROOT / "_transcripts"


def feature_signature(features) -> str:
    sig = (f"wt{int(features.word_timestamps)}_sl{int(features.speaker_labels)}"
           f"_pu{int(features.punctuation)}_la{features.language or 'auto'}")
    # Custom vocabulary changes the transcript, so it must partition the cache — a
    # glossary run must never reuse a non-glossary (or different-glossary) transcript.
    vocab = getattr(features, "custom_vocabulary", ()) or ()
    if vocab:
        import hashlib
        h = hashlib.sha1("\n".join(sorted(vocab)).encode()).hexdigest()[:10]
        sig += f"_cv{h}"
    return sig


def _key(audio_path: str | Path, features, salt: str = "") -> str:
    # `salt` folds in the provider's model + suite version so that switching a
    # provider's model (e.g. nova-3 -> nova-2) or bumping the suite invalidates
    # the cache instead of serving another model's transcript.
    rp = os.path.realpath(str(audio_path))
    return hashlib.sha1(
        f"{rp}|{feature_signature(features)}|{salt}".encode()
    ).hexdigest()[:20]


def path_for(provider: str, audio_path: str | Path, features, salt: str = "") -> Path:
    return CACHE_ROOT / provider / f"{_key(audio_path, features, salt)}.json"


def _encode_raw(raw: Any) -> tuple[str, Any]:
    if isinstance(raw, (bytes, bytearray)):
        try:
            return "json", json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return "b64", base64.b64encode(bytes(raw)).decode("ascii")
    if isinstance(raw, str):
        try:
            return "json", json.loads(raw)
        except ValueError:
            return "text", raw
    return "json", raw  # dict / list


def _decode_raw(encoding: str, value: Any) -> Any:
    if encoding == "b64":
        return base64.b64decode(value)
    return value  # "json" (dict) and "text" both pass through fine to normalize()


def load(provider: str, audio_path: str | Path, features, salt: str = "") -> dict | None:
    """Return {raw, audio_duration_s, submit_latency_s, total_latency_s} or None."""
    p = path_for(provider, audio_path, features, salt)
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
    except (ValueError, OSError):
        return None
    d["raw"] = _decode_raw(d.get("raw_encoding", "json"), d.get("raw"))
    return d


def save(provider: str, audio_path: str | Path, features, raw: Any, *,
         salt: str = "", audio_duration_s: float = 0.0, submit_latency_s: float = 0.0,
         total_latency_s: float = 0.0) -> None:
    """Persist a raw payload. Best-effort — never raises into the batch."""
    try:
        encoding, value = _encode_raw(raw)
        p = path_for(provider, audio_path, features, salt)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": provider,
            "audio_path": os.path.realpath(str(audio_path)),
            "features": feature_signature(features),
            "audio_duration_s": audio_duration_s,
            "submit_latency_s": submit_latency_s,
            "total_latency_s": total_latency_s,
            "raw_encoding": encoding,
            "raw": value,
        }
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, p)  # atomic
    except Exception:  # noqa: BLE001 - caching must never break a run
        pass


def stats(provider: str | None = None) -> dict:
    root = CACHE_ROOT / provider if provider else CACHE_ROOT
    if not root.exists():
        return {"files": 0, "bytes": 0}
    files = list(root.rglob("*.json"))
    return {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
