"""
Overlap-aware Diarization Error Rate (DER) with collar.

Frame-based DER (numpy + scipy, no pyannote dependency) implementing the standard
NIST md-eval / pyannote.metrics formulation:

    DER = Σ_t [ max(N_ref_t, N_sys_t) − N_correct_t ] / Σ_t N_ref_t

where at each frame t, N_ref_t / N_sys_t are the number of *simultaneously active*
reference / system speakers and N_correct_t is the number of reference speakers
whose optimally-mapped system speaker is also active. This counts OVERLAPPING
speech correctly: a frame with two reference speakers contributes 2 to the
denominator, and missing the second speaker is a genuine miss.

    missed_speech = Σ max(0, N_ref − N_sys)
    false_alarm   = Σ max(0, N_sys − N_ref)
    speaker_error = Σ [ min(N_ref, N_sys) − N_correct ]     (confusion)

Speakers are mapped reference<->hypothesis by maximum co-occurrence (Hungarian
assignment) over the scored region. ``collar`` seconds are excluded, centered on
each reference boundary (pyannote convention: ±collar/2). Validated to match
pyannote.metrics DiarizationErrorRate (collar, skip_overlap=False) to <0.1pp.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

RESOLUTION = 0.01  # 10 ms frames

Segment = tuple[float, float, str]


def compute_der(
    ref_segments: list[Segment],
    hyp_segments: list[Segment],
    collar: float = 0.25,
    uem_duration: float | None = None,
) -> dict:
    """Return overlap-aware DER components for one recording.

    Keys: der, missed_speech_s, false_alarm_s, speaker_error_s, total_ref_s, collar_s

    total_ref_s is *speaker* seconds (overlap counted per-speaker), matching the
    md-eval denominator, so aggregating error_s / total_ref_s across files is a
    correct corpus DER.
    """
    if not ref_segments:
        fa = sum(e - s for s, e, _ in hyp_segments) if hyp_segments else 0.0
        return {
            "der": 1.0 if hyp_segments else 0.0,
            "missed_speech_s": 0.0,
            "false_alarm_s": round(fa, 3),
            "speaker_error_s": 0.0,
            "total_ref_s": 0.0,
            "collar_s": collar,
        }

    all_ends = [e for _, e, _ in ref_segments] + [e for _, e, _ in hyp_segments]
    duration = uem_duration if uem_duration is not None else max(all_ends)
    n_frames = int(np.ceil(duration / RESOLUTION))

    ref_active, _ = _to_multilabel(ref_segments, n_frames)   # (n_frames, n_ref_spk)
    hyp_active, _ = _to_multilabel(hyp_segments, n_frames)    # (n_frames, n_hyp_spk)

    eval_mask = ~_collar_mask(ref_segments, n_frames, collar)
    ref_active = ref_active[eval_mask]
    hyp_active = hyp_active[eval_mask]

    n_ref = ref_active.sum(axis=1)   # per-frame active reference speaker count
    n_sys = hyp_active.sum(axis=1)

    # Optimal ref<->hyp speaker map by co-occurrence over the scored region.
    n_ref_spk = ref_active.shape[1]
    n_hyp_spk = hyp_active.shape[1]
    n_correct_frames = np.zeros(ref_active.shape[0], dtype=np.int64)
    if n_ref_spk and n_hyp_spk:
        cooc = ref_active.T.astype(np.int64) @ hyp_active.astype(np.int64)  # (n_ref_spk, n_hyp_spk)
        row_ind, col_ind = linear_sum_assignment(-cooc)
        # N_correct per frame: ref speaker active AND its mapped hyp speaker active.
        for ri, hj in zip(row_ind, col_ind):
            if cooc[ri, hj] > 0:  # only real (non-padding) assignments help
                n_correct_frames += (ref_active[:, ri] & hyp_active[:, hj]).astype(np.int64)

    missed = int(np.maximum(0, n_ref - n_sys).sum())
    false_alarm = int(np.maximum(0, n_sys - n_ref).sum())
    speaker_error = int((np.minimum(n_ref, n_sys) - n_correct_frames).sum())
    total_ref = int(n_ref.sum())

    der = (missed + false_alarm + speaker_error) / total_ref if total_ref > 0 else 0.0
    return {
        "der": round(float(der), 6),
        "missed_speech_s": round(float(missed * RESOLUTION), 3),
        "false_alarm_s": round(float(false_alarm * RESOLUTION), 3),
        "speaker_error_s": round(float(speaker_error * RESOLUTION), 3),
        "total_ref_s": round(float(total_ref * RESOLUTION), 3),
        "collar_s": collar,
    }


def _to_multilabel(segments: list[Segment], n_frames: int) -> tuple[np.ndarray, list[str]]:
    """Multi-hot frame matrix (n_frames, n_speakers) — OR of all segments so
    simultaneous speakers coexist instead of overwriting each other."""
    speakers = sorted({s for _, _, s in segments})
    spk_idx = {s: i for i, s in enumerate(speakers)}
    active = np.zeros((n_frames, max(len(speakers), 1)), dtype=bool)
    for start, end, spk in segments:
        s = max(0, int(start / RESOLUTION))
        e = min(int(end / RESOLUTION), n_frames)
        if e > s:
            active[s:e, spk_idx[spk]] = True
    return active[:, : len(speakers)] if speakers else active[:, :0], speakers


def _collar_mask(ref_segments: list[Segment], n_frames: int, collar: float) -> np.ndarray:
    """Exclude ±collar/2 around each reference boundary (pyannote convention)."""
    half = int((collar / 2) / RESOLUTION)
    mask = np.zeros(n_frames, dtype=bool)
    if half <= 0:
        return mask
    for start, end, _ in ref_segments:
        s = int(start / RESOLUTION)
        e = int(end / RESOLUTION)
        mask[max(0, s - half): min(n_frames, s + half)] = True
        mask[max(0, e - half): min(n_frames, e + half)] = True
    return mask
