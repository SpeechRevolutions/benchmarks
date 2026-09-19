"""
cpWER — concatenated, speaker-permutation-invariant Word Error Rate.

The joint ASR+diarization metric reported by modern vendors (AssemblyAI,
NVIDIA, the CHiME/NOTSOFAR challenges). Each speaker's words are concatenated in
time order; hypothesis speakers are optimally assigned to reference speakers
(the assignment minimizing total word errors, via Hungarian matching); cpWER is
the total substitutions+deletions+insertions over that assignment divided by the
total reference words.

Unlike DER (which scores time overlap), cpWER penalizes both transcription
errors and speaker-attribution errors in a single number, so a system that
transcribes perfectly but mixes up who-said-what still scores poorly.

    cpWER = min over speaker-assignments of  (S + D + I) / N_ref_words
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .normalize import normalize_text

try:
    import jiwer
except ImportError:
    jiwer = None  # type: ignore


def _pair_errors(ref: str, hyp: str) -> tuple[int, int]:
    """Return (word_errors, ref_word_count) for one ref/hyp speaker pair."""
    rw, hw = ref.split(), hyp.split()
    if not rw and not hw:
        return 0, 0
    if not rw:
        return len(hw), 0          # all insertions
    if not hw:
        return len(rw), len(rw)    # all deletions
    if jiwer is None:
        raise ImportError("jiwer not installed. Run: pip install -r requirements.txt")
    out = jiwer.process_words(ref, hyp)
    return int(out.substitutions + out.deletions + out.insertions), len(rw)


def compute_cpwer(ref_by_speaker: dict[str, str], hyp_by_speaker: dict[str, str]) -> dict:
    """cpWER between per-speaker reference and hypothesis text.

    Args:
        ref_by_speaker: {reference_speaker: text} (any order; text time-ordered).
        hyp_by_speaker: {hypothesis_speaker: text}.

    Returns dict: cpwer, errors, ref_words, n_ref_speakers, n_hyp_speakers,
    speaker_mapping (hyp_speaker -> ref_speaker for the optimal assignment).
    """
    ref = {s: normalize_text(t) for s, t in ref_by_speaker.items()}
    hyp = {s: normalize_text(t) for s, t in hyp_by_speaker.items()}
    ref_spks = list(ref)
    hyp_spks = list(hyp)

    total_ref_words = sum(len(t.split()) for t in ref.values())
    if not ref_spks and not hyp_spks:
        return _empty()
    if total_ref_words == 0:
        # No reference words: any hypothesis word is an insertion; cpWER is 1.0
        # if the system emitted anything, else 0.0.
        emitted = sum(len(t.split()) for t in hyp.values())
        return {"cpwer": 1.0 if emitted else 0.0, "errors": emitted, "ref_words": 0,
                "n_ref_speakers": len(ref_spks), "n_hyp_speakers": len(hyp_spks),
                "speaker_mapping": {}}

    n = max(len(ref_spks), len(hyp_spks))
    # square cost matrix; pad missing speakers with empty transcripts
    padded_ref = ref_spks + [None] * (n - len(ref_spks))
    padded_hyp = hyp_spks + [None] * (n - len(hyp_spks))
    cost = np.zeros((n, n), dtype=np.int64)
    for i, rs in enumerate(padded_ref):
        for j, hs in enumerate(padded_hyp):
            r = ref.get(rs, "") if rs is not None else ""
            h = hyp.get(hs, "") if hs is not None else ""
            cost[i, j] = _pair_errors(r, h)[0]

    row_ind, col_ind = linear_sum_assignment(cost)
    total_errors = int(cost[row_ind, col_ind].sum())

    mapping: dict[str, str] = {}
    for i, j in zip(row_ind, col_ind):
        rs = padded_ref[i]
        hs = padded_hyp[j]
        if rs is not None and hs is not None:
            mapping[hs] = rs

    return {
        "cpwer": round(total_errors / total_ref_words, 6),
        "errors": total_errors,
        "ref_words": total_ref_words,
        "n_ref_speakers": len(ref_spks),
        "n_hyp_speakers": len(hyp_spks),
        "speaker_mapping": mapping,
    }


def words_to_speaker_text(words, *, text_attr: str = "text", spk_attr: str = "speaker") -> dict[str, str]:
    """Group a list of Word-like objects into {speaker: concatenated text} (time order preserved)."""
    by_spk: dict[str, list[str]] = {}
    for w in words:
        spk = getattr(w, spk_attr, None) if not isinstance(w, dict) else w.get(spk_attr)
        txt = getattr(w, text_attr, None) if not isinstance(w, dict) else w.get(text_attr)
        if not txt:
            continue
        by_spk.setdefault(spk if spk is not None else "SPEAKER_0", []).append(txt)
    return {s: " ".join(ws) for s, ws in by_spk.items()}


def _empty() -> dict:
    return {"cpwer": None, "errors": 0, "ref_words": 0,
            "n_ref_speakers": 0, "n_hyp_speakers": 0, "speaker_mapping": {}}
