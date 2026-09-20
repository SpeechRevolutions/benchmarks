"""
Normalized transcript types.

Every provider's ``normalize()`` returns a :class:`Transcript`, so all benchmark
scoring code is provider-agnostic. This is the single contract between the
"talk to a vendor API" layer and the "compute a metric" layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Word:
    """A single recognized word with optional timing and speaker attribution."""
    text: str
    start: float | None = None   # seconds
    end: float | None = None     # seconds
    speaker: str | None = None   # provider-native speaker label (e.g. "A", "spk_0")
    language: str | None = None  # per-word language tag (for language-switch scoring)
    #: Recogniser confidence for this word, 0-1, when the provider reports one.
    #: None means "not reported" and must never be read as "confident".
    prob: float | None = None


@dataclass
class Segment:
    """A contiguous speaker turn (used for diarization scoring)."""
    start: float
    end: float
    speaker: str


@dataclass
class Transcript:
    """Provider-agnostic transcription result.

    ``text`` is always populated. ``words`` is populated when word timestamps
    were requested and the provider supports them. ``segments`` is populated for
    diarization-capable providers; if absent, scoring code derives segments from
    per-word speaker labels.
    """
    text: str
    words: list[Word] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    raw: Any = None              # original provider payload, kept for debugging

    # ── Derived views ────────────────────────────────────────────────────────

    def word_texts(self) -> list[str]:
        return [w.text for w in self.words]

    def has_word_timestamps(self) -> bool:
        return bool(self.words) and all(
            w.start is not None and w.end is not None for w in self.words
        )

    def has_speakers(self) -> bool:
        return bool(self.segments) or any(w.speaker is not None for w in self.words)

    def derived_segments(self) -> list[Segment]:
        """Speaker segments, preferring explicit ``segments`` then merging words."""
        if self.segments:
            return self.segments
        return _merge_words_to_segments(self.words)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "words": [
                {"word": w.text, "start": w.start, "end": w.end, "speaker": w.speaker}
                for w in self.words
            ],
            "segments": [
                {"start": s.start, "end": s.end, "speaker": s.speaker}
                for s in self.segments
            ],
        }


def _merge_words_to_segments(
    words: list[Word], default_speaker: str = "SPEAKER_0"
) -> list[Segment]:
    """Merge consecutive same-speaker words into speaker segments."""
    timed = [w for w in words if w.start is not None and w.end is not None]
    if not timed:
        return []

    segments: list[Segment] = []
    cur_spk = timed[0].speaker or default_speaker
    cur_start = float(timed[0].start)
    cur_end = float(timed[0].end)

    for w in timed[1:]:
        spk = w.speaker or default_speaker
        if spk == cur_spk:
            cur_end = float(w.end)
        else:
            segments.append(Segment(cur_start, cur_end, cur_spk))
            cur_spk, cur_start, cur_end = spk, float(w.start), float(w.end)

    segments.append(Segment(cur_start, cur_end, cur_spk))
    return segments
