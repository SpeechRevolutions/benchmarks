"""Offline unit tests for the Oruk Fourier 2B provider adapter."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

# Allow running offline even if the optional SpeechRevolutions SDK isn't installed.
if "speechrevolutions" not in sys.modules:
    try:
        importlib.import_module("speechrevolutions")
    except ImportError:
        pkg = types.ModuleType("speechrevolutions")
        pkg.SpeechRevolutions = object  # type: ignore[attr-defined]
        models_mod = types.ModuleType("speechrevolutions.models")
        models_mod.TranscribeOptions = object  # type: ignore[attr-defined]
        pkg.models = models_mod  # type: ignore[attr-defined]
        sys.modules["speechrevolutions"] = pkg
        sys.modules["speechrevolutions.models"] = models_mod

from benchmarks.providers import available_providers, get_provider
from benchmarks.providers._common import MissingKeyError
from benchmarks.providers.base import Features, JobStatus
from benchmarks.providers.oruk import ENDPOINT, MAX_BYTES, OrukProvider


def _audio(tmp_path: Path, data: bytes = b"RIFF" + b"\x00" * 32) -> Path:
    path = tmp_path / "sample.wav"
    path.write_bytes(data)
    return path


def test_registered_with_expected_metadata(monkeypatch):
    monkeypatch.setenv("ORUK_API_KEY", "test-oruk-key")
    assert "oruk" in available_providers()
    provider = get_provider("oruk")
    assert isinstance(provider, OrukProvider)
    assert provider.model == "oruk-fourier-2b"
    assert provider.price_per_audio_hour_usd == pytest.approx(0.48)
    assert provider.endpoint == ENDPOINT


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ORUK_API_KEY", raising=False)
    with pytest.raises(MissingKeyError, match="ORUK_API_KEY"):
        OrukProvider()


def test_submit_poll_download_normalize_strict_contract(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_post(url, *, headers, data, files, timeout):
        captured.update(url=url, headers=headers, data=data, files=files, timeout=timeout)
        resp = Mock(status_code=200)
        resp.json.return_value = {
            "model": "oruk-fourier-2b",
            "text": "The quick brown fox.",
            "language": "en",
            "model_revision": "8c2a8871f1304aa2f634da4fb95e24438c4945b9",
        }
        return resp

    monkeypatch.setattr("benchmarks.providers.oruk.requests.post", fake_post)
    provider = OrukProvider(api_key="k-123")
    audio_path = _audio(tmp_path)

    # Custom vocabulary is intentionally not sent (Fourier2bRequest rejects extra fields).
    job = provider.submit(
        audio_path,
        Features(language="en_us", custom_vocabulary=("Fourier", "Oruk")),
        entry_id="clip-1",
    )
    assert provider.poll(job) is JobStatus.COMPLETED
    raw = provider.download(job)
    transcript = provider.normalize(raw)

    assert captured["url"] == ENDPOINT
    assert captured["headers"] == {"Authorization": "Bearer k-123"}
    assert captured["data"] == {
        "model": "oruk-fourier-2b",
        "response_format": "json",
        "language": "en-US",
    }
    assert captured["files"]["file"][0] == "sample.wav"
    assert transcript.text == "The quick brown fox."
    assert transcript.language == "en"
    assert transcript.words == []
    assert transcript.segments == []


def test_rejects_non_english_and_oversized_audio(monkeypatch, tmp_path):
    provider = OrukProvider(api_key="k-123")
    audio_path = _audio(tmp_path)

    with pytest.raises(RuntimeError, match="English audio only"):
        provider.submit(audio_path, Features(language="fr"))

    with pytest.raises(RuntimeError, match="30 MiB"):
        provider._transcribe(b"\x00" * (MAX_BYTES + 1), audio_path, Features())


def test_http_error_surfaces_status(monkeypatch, tmp_path):
    def fake_post(*args, **kwargs):
        resp = Mock(status_code=401, text='{"error":{"code":"invalid_api_key"}}')
        return resp

    monkeypatch.setattr("benchmarks.providers.oruk.requests.post", fake_post)
    provider = OrukProvider(api_key="bad-key")
    with pytest.raises(RuntimeError, match="oruk HTTP 401"):
        provider.submit(_audio(tmp_path), Features())
