"""The reference provider: what it sends, what it parses, and what it cleans up.

These run offline. The provider is the one file the public suite does not share
with our internal copy -- the internal one talks to a local stack over raw HTTP,
this one reaches production through the published SDK -- so the behaviour that
must match is pinned here rather than assumed.
"""

from pathlib import Path

import pytest

from benchmarks.providers.base import Features
from benchmarks.providers.speech_revolutions import SpeechRevolutionsProvider


class _FakeUploadJob:
    def __init__(self):
        self.job_id = "job-1"
        self.upload_url = "https://example.invalid/put"
        self.download_url = "https://example.invalid/get"


class _FakeClient:
    """Records calls; optionally fails at a chosen step."""

    def __init__(self, fail_at: str | None = None):
        self.fail_at = fail_at
        self.calls: list[str] = []
        self.options = None
        self.cancelled: list[str] = []

    def create_upload_job(self, size, options):
        self.calls.append("create")
        self.options = options
        return _FakeUploadJob()

    def upload_audio(self, url, data, job_id=None):
        self.calls.append("upload")
        if self.fail_at == "upload":
            raise RuntimeError("upload blew up")

    def complete_upload(self, job_id):
        self.calls.append("complete")
        if self.fail_at == "complete":
            raise RuntimeError("complete blew up")

    def cancel_job(self, job_id):
        self.calls.append("cancel")
        self.cancelled.append(job_id)


def _provider(client) -> SpeechRevolutionsProvider:
    p = SpeechRevolutionsProvider.__new__(SpeechRevolutionsProvider)
    p._client = client
    return p


def _audio(tmp_path: Path) -> Path:
    f = tmp_path / "clip.wav"
    f.write_bytes(b"\0" * 32)
    return f


# ── what it sends ────────────────────────────────────────────────────────────

def test_requested_options_match_the_features(tmp_path):
    client = _FakeClient()
    _provider(client).submit(_audio(tmp_path), Features(
        word_timestamps=True, speaker_labels=True, punctuation=True))
    o = client.options
    assert o.output_type == "json"
    assert o.word_timestamps is True
    assert o.speaker_labels is True
    # The API calls sentence segmentation "nltk"; the harness calls it punctuation.
    assert o.nltk is True


# ── what it cleans up ────────────────────────────────────────────────────────
#
# After create_upload_job the job exists server-side. A job that is registered and
# then neither completes nor fails is "stranded", and production sheds customer
# intake when a couple of those sit past their deadline -- so a failed submit that
# walks away leaves the platform refusing work for everyone. It has happened.

@pytest.mark.parametrize("fail_at", ["upload", "complete"])
def test_failed_submit_cancels_the_job_it_created(tmp_path, fail_at):
    client = _FakeClient(fail_at=fail_at)
    with pytest.raises(RuntimeError):
        _provider(client).submit(_audio(tmp_path), Features())
    assert client.cancelled == ["job-1"], (
        f"a submit that failed at {fail_at} left job-1 registered and unterminated"
    )
    assert client.calls[-1] == "cancel"


def test_successful_submit_cancels_nothing(tmp_path):
    client = _FakeClient()
    job = _provider(client).submit(_audio(tmp_path), Features())
    assert client.cancelled == []
    assert job.native_id == "job-1"


def test_cancel_failure_does_not_mask_the_real_error(tmp_path):
    class Stubborn(_FakeClient):
        def cancel_job(self, job_id):
            raise RuntimeError("cancel also failed")

    client = Stubborn(fail_at="upload")
    with pytest.raises(RuntimeError, match="upload blew up"):
        _provider(client).submit(_audio(tmp_path), Features())


# ── what it parses ───────────────────────────────────────────────────────────

PAYLOAD = {
    "text": "Hello there.",
    "language": "en",
    "words": [
        {"word": "Hello", "start": 0.0, "end": 0.4, "speaker": "A",
         "language": "en", "prob": 0.98},
        {"word": "there.", "start": 0.4, "end": 0.9, "speaker": "B",
         "language": "en", "prob": 0.91},
    ],
    "diarization": [
        {"start": 0.0, "end": 0.4, "speaker": "A"},
        {"start": 0.4, "end": 0.9, "speaker": "B"},
    ],
}


def test_normalize_carries_every_scored_field():
    t = _provider(_FakeClient()).normalize(PAYLOAD)
    assert t.text == "Hello there."
    assert t.language == "en"
    assert [w.text for w in t.words] == ["Hello", "there."]
    assert [w.speaker for w in t.words] == ["A", "B"]
    assert [w.language for w in t.words] == ["en", "en"]
    assert [w.prob for w in t.words] == [0.98, 0.91]
    assert [(s.start, s.end, s.speaker) for s in t.segments] == [
        (0.0, 0.4, "A"), (0.4, 0.9, "B")]


def test_diarization_is_preferred_over_segments():
    """Reconstructing turns from per-word labels cannot represent overlap, so the
    dedicated timeline wins whenever the payload has one."""
    payload = dict(PAYLOAD, segments=[{"start": 0.0, "end": 9.0, "speaker": "Z"}])
    t = _provider(_FakeClient()).normalize(payload)
    assert [s.speaker for s in t.segments] == ["A", "B"]


def test_segments_are_the_fallback_for_older_payloads():
    payload = {k: v for k, v in PAYLOAD.items() if k != "diarization"}
    payload["segments"] = [{"start": 0.0, "end": 9.0, "speaker": "Z"}]
    t = _provider(_FakeClient()).normalize(payload)
    assert [s.speaker for s in t.segments] == ["Z"]


def test_text_falls_back_to_joined_words():
    payload = {k: v for k, v in PAYLOAD.items() if k != "text"}
    assert _provider(_FakeClient()).normalize(payload).text == "Hello there."


def test_malformed_segments_are_dropped_not_guessed():
    payload = dict(PAYLOAD, diarization=[{"start": 0.0, "end": 0.4}])
    assert _provider(_FakeClient()).normalize(payload).segments == []
