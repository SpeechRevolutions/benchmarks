"""
Benchmark 3 — Speaker Diarization.

Datasets: AMI, Earnings21 (+ DiPCo, NOTSOFAR when prepared).
Metrics:
  - DER at collar 0 (overlap-aware) is the headline. A lenient 0.25 s collar is
    also reported (overall AND per subset), since published DiariZen/pyannote
    numbers are reported with a collar.
  - cpWER (concatenated, speaker-permutation-invariant WER) — the joint
    ASR+diarization metric vendors like AssemblyAI report. Computed whenever the
    manifest carries per-speaker reference text (``speaker_transcripts``) and the
    provider returns speaker-labeled words.

Output: overall_der, speaker_error, false_alarm, missed_speech, overall_cpwer.
The three DER component metrics are reported as fractions of total reference
speech; per-file seconds and per-file cpWER are kept in per_file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import DER_COLLAR_S
from ..core.benchmark import Benchmark
from ..core.cpwer import compute_cpwer, words_to_speaker_text
from ..core.diarization import compute_der
from ..core.metrics import bootstrap_ci
from ..providers.base import Features, TranscriptionResult


def _ratio_stat(items):
    """sum(numerator)/sum(denominator) over (num, den) items — bootstrap statistic."""
    den = sum(d for _, d in items)
    return (sum(nu for nu, _ in items) / den) if den else None


@dataclass
class DiarizationBenchmark(Benchmark):
    name: str = "diarization"
    features: Features = field(default_factory=lambda: Features(
        word_timestamps=True, speaker_labels=True, punctuation=True))
    subsets: dict[str, str] = field(default_factory=lambda: {
        # ami_sdm (single distant mic + official RTTM) is the standard AMI condition
        # used by published DiariZen numbers. ami (Mix-Headset) is kept for continuity
        # but is OOD for segmentation models (summed close mics).
        "ami_sdm": "diarization_ami_sdm",
        "ami": "diarization_ami",
        "earnings21": "diarization_earnings21",
        "notsofar": "diarization_notsofar",
        "dipco": "diarization_dipco",
    })
    regression_specs: dict = field(default_factory=lambda: {
        "overall_der": ("lower", 0.01),
        "overall_cpwer": ("lower", 0.01),
    })

    def score(self, runs: dict[str, list[TranscriptionResult]], *, provider_name: str) -> dict:
        per_file: list[dict] = []
        total_ref_s = 0.0
        sum_missed = sum_fa = sum_spk = 0.0
        sum_err_c025 = 0.0                            # error seconds at the lenient collar
        cpwer_errors = 0
        cpwer_ref_words = 0
        der_items: list[tuple[float, float]] = []    # (error_s, ref_s) per file @ collar 0
        cpwer_items: list[tuple[int, int]] = []      # (errors, ref_words) per file

        for subset, results in runs.items():
            for r in results:
                if not r.ok:
                    per_file.append({"id": r.entry_id, "subset": subset,
                                     "der": None, "error": r.error})
                    continue
                ref_segs = [
                    (float(s["start"]), float(s["end"]), s["speaker"])
                    for s in r.meta.get("speakers", [])
                ]
                hyp_segs = [(s.start, s.end, s.speaker)
                            for s in r.transcript.derived_segments()]
                uem = r.meta.get("duration_s") or None
                # Headline DER at collar 0 (overlap-aware — matches the
                # DiariZen/pyannote model-card protocol); also the lenient 0.25 collar.
                fd = compute_der(ref_segs, hyp_segs, collar=0.0, uem_duration=uem)
                fd25 = compute_der(ref_segs, hyp_segs, collar=DER_COLLAR_S, uem_duration=uem)
                err25 = fd25["missed_speech_s"] + fd25["false_alarm_s"] + fd25["speaker_error_s"]
                row = {
                    "id": r.entry_id, "subset": subset, "der": fd["der"],
                    "der_c025": fd25["der"],
                    "err_c025_s": err25,
                    "missed_speech_s": fd["missed_speech_s"],
                    "false_alarm_s": fd["false_alarm_s"],
                    "speaker_error_s": fd["speaker_error_s"],
                    "total_ref_s": fd["total_ref_s"],
                    "duration_s": r.meta.get("duration_s", 0.0),
                }
                total_ref_s += fd["total_ref_s"]
                sum_missed += fd["missed_speech_s"]
                sum_fa += fd["false_alarm_s"]
                sum_spk += fd["speaker_error_s"]
                sum_err_c025 += err25
                der_items.append((fd["missed_speech_s"] + fd["false_alarm_s"]
                                  + fd["speaker_error_s"], fd["total_ref_s"]))

                # cpWER when the manifest has per-speaker reference text and the
                # provider labeled its words by speaker.
                ref_by_spk = r.meta.get("speaker_transcripts")
                hyp_by_spk = words_to_speaker_text(r.transcript.words)
                if ref_by_spk and any(w.speaker for w in r.transcript.words):
                    cp = compute_cpwer(ref_by_spk, hyp_by_spk)
                    row["cpwer"] = cp["cpwer"]
                    row["cpwer_errors"] = cp["errors"]
                    row["cpwer_ref_words"] = cp["ref_words"]
                    cpwer_errors += cp["errors"]
                    cpwer_ref_words += cp["ref_words"]
                    cpwer_items.append((cp["errors"], cp["ref_words"]))
                per_file.append(row)

        # Per-dataset breakout — DER/cpWER are only comparable within a dataset
        # (e.g. AssemblyAI publishes cpWER on NOTSOFAR specifically), so never
        # rely on the pooled number for cross-vendor comparison.
        subset_breakdown: dict[str, dict] = {}
        for p in per_file:
            if p.get("der") is None:
                continue
            sub = p.get("subset", "?")
            b = subset_breakdown.setdefault(sub, {"_err_s": 0.0, "_err25_s": 0.0, "_ref_s": 0.0,
                                                  "_cp_err": 0, "_cp_ref": 0, "n": 0})
            b["_err_s"] += p["missed_speech_s"] + p["false_alarm_s"] + p["speaker_error_s"]
            b["_err25_s"] += p.get("err_c025_s", 0.0)
            b["_ref_s"] += p["total_ref_s"]
            b["_cp_err"] += p.get("cpwer_errors", 0)
            b["_cp_ref"] += p.get("cpwer_ref_words", 0)
            b["n"] += 1
        for sub, b in subset_breakdown.items():
            subset_breakdown[sub] = {
                "der": round(b["_err_s"] / b["_ref_s"], 6) if b["_ref_s"] else None,
                "der_c025": round(b["_err25_s"] / b["_ref_s"], 6) if b["_ref_s"] else None,
                "cpwer": round(b["_cp_err"] / b["_cp_ref"], 6) if b["_cp_ref"] else None,
                "n_files": b["n"],
            }

        def frac(x: float) -> float | None:
            return round(x / total_ref_s, 6) if total_ref_s > 0 else None

        overall_der = frac(sum_missed + sum_fa + sum_spk)   # collar 0 (headline)
        summary = {
            "overall_der": overall_der,                     # collar 0, overlap-aware
            "overall_der_collar025": frac(sum_err_c025),    # lenient collar, for vendor comparability
            "overall_der_ci95": bootstrap_ci(der_items, _ratio_stat),
            "speaker_error": frac(sum_spk),
            "false_alarm": frac(sum_fa),
            "missed_speech": frac(sum_missed),
            "overall_cpwer": (round(cpwer_errors / cpwer_ref_words, 6)
                              if cpwer_ref_words else None),
            "overall_cpwer_ci95": bootstrap_ci(cpwer_items, _ratio_stat),
            "subset_breakdown": {k: subset_breakdown[k] for k in sorted(subset_breakdown)},
            "collar_s": 0.0,
            "collar_s_lenient": DER_COLLAR_S,
            "overlap_aware": True,
            "total_ref_s": round(total_ref_s, 3),
            "n_files": len([p for p in per_file if p.get("der") is not None]),
        }
        return self._base_result(provider_name, summary, per_file)
