# Speech Revolutions — Public Benchmark Suite (v1)

A fully reproducible, provider-agnostic evaluation suite for commercial
Speech-to-Text APIs. Every benchmark is generated from **public datasets** using
**deterministic, seeded** scripts, and every benchmark produces **identical
outputs regardless of provider**.

This package is entirely self-contained under `(repo root)/`.

---

## Status

| | |
|---|---|
| **Reference provider** | `speech_revolutions` — our own system (the "Zephyr" system), accessed via the [official Python SDK](https://github.com/SpeechRevolutions/python-sdk) against the production API. |
| **External providers** | 11 implemented: AssemblyAI, Deepgram, OpenAI, ElevenLabs, Gladia, Mistral (Voxtral), Soniox, Cohere, Grok (xAI), Qwen3-ASR, Azure Batch. Each needs its own API key (see matrix below). |
| **Benchmarks** | All 8: WER, Entity Accuracy, Diarization, Timestamps, Multilingual, Language Switching, Long-form, Price/Performance. |

## Providers

Run any benchmark against any provider with `--provider <name>`. Every provider
returns the same normalized `Transcript`, so scoring is identical.

| Provider | Model | Key(s) (env var) | Words | Diar. | Notes |
|----------|-------|------------------|:-----:|:-----:|-------|
| `speech_revolutions` | our own ("Zephyr") | `SPEECHREVOLUTIONS_API_KEY` | ✅ | ✅ | default; `SR_API_URL` optional, to target a non-production deployment |
| `assemblyai` | Universal | `ASSEMBLYAI_API_KEY` | ✅ | ✅ | async (upload→poll) |
| `deepgram` | Nova-3 | `DEEPGRAM_API_KEY` | ✅ | ✅ | sync |
| `openai` | GPT-4o Transcribe | `OPENAI_API_KEY` | ❌ | ❌ | **text only**; 25 MB file limit |
| `elevenlabs` | Scribe V2 | `ELEVENLABS_API_KEY` | ✅ | ✅ | sync |
| `gladia` | solaria-1 | `GLADIA_API_KEY` | ✅ | ✅ | async (upload→poll) |
| `mistral` | Voxtral Mini | `MISTRAL_API_KEY` | ✅ | ✅ | word-timestamps & language hint are mutually exclusive |
| `soniox` | stt-async-v5 | `SONIOX_API_KEY` | ✅ | ✅ | async; subword tokens merged to words |
| `cohere` | cohere-transcribe-03-2026 | `CO_API_KEY` | ❌ | ❌ | **text only**; language required |
| `grok` | xAI /v1/stt | `XAI_API_KEY` | ✅ | ✅ | sync |
| `qwen` | Qwen3-ASR filetrans | `DASHSCOPE_API_KEY` + `QWEN_AUDIO_BASE_URL` | ✅ | ❌ | **audio must be URL-hosted** |
| `azure` | Azure Batch | `SPEECH_KEY` + `SPEECH_REGION` + `AZURE_AUDIO_BASE_URL` | ✅ | ✅ | **audio must be URL-hosted** |

**Text-only providers** (OpenAI GPT-4o Transcribe, Cohere) can't produce word
timestamps or speaker labels — the timestamp/diarization benchmarks will show no
word data for them, which is an honest reflection of the API, not a harness bug.

**URL-hosted providers** (Qwen3, Azure Batch) accept audio only by URL. Host the
`_data/` directory (S3, blob+SAS, or any static server) and set the matching
`*_AUDIO_BASE_URL` so a local file resolves to `<base>/<path-relative-to-_data>`.

---

## The two contracts

Everything hangs off two small interfaces.

**Provider** (`providers/base.py`) — one adapter per vendor:

```python
submit(audio, features) -> Job
poll(job)               -> JobStatus
download(job)           -> raw payload
normalize(raw)          -> Transcript      # the single provider-agnostic type
```

`transcribe_batch()` drives the whole submit→poll→download→normalize lifecycle in
parallel and records per-file timing (used by price/perf). Subclasses implement
only the four primitives.

**Benchmark** (`core/benchmark.py`) — one per metric, all sharing:

```python
run()              # transcribe each subset's manifest with a provider
score()            # compute metrics from run output (provider-agnostic)
capture_baseline() # freeze the current summary
compare()          # diff vs. baseline, return regressions
generate_report()  # JSON + Markdown + CSV + console
```

Because every provider returns the same `Transcript`, scoring never depends on
which provider produced it.

---

## Quick start

```bash
git clone https://github.com/SpeechRevolutions/benchmarks
cd benchmarks

pip install -r requirements.txt
python -m spacy download en_core_web_sm     # entity benchmark

# 1. Build the frozen datasets from public sources (deterministic)
python -m benchmarks.datasets.prepare_all
#    or individually:
#    python -m benchmarks.datasets.prepare_librispeech
#    python -m benchmarks.datasets.prepare_ami
#    python -m benchmarks.datasets.prepare_earnings21
#    python -m benchmarks.datasets.prepare_fleurs
#    python -m benchmarks.datasets.prepare_language_switching
#    python -m benchmarks.datasets.prepare_longform

# 2. Set your API key for the reference provider.
#    The provider talks to production (https://api.speechrevolutions.com)
#    through the published `speechrevolutions` SDK from requirements.txt --
#    no local stack, no docker, nothing to run yourself.
export SPEECHREVOLUTIONS_API_KEY=stt_...
#    (optional) override the endpoint to test a different deployment:
#    export SR_API_URL=https://...

# 3. Run a benchmark (or all) against the reference provider
python -m benchmarks.cli run wer
python -m benchmarks.cli run all

# 4. Freeze a baseline after a known-good run
python -m benchmarks.cli run all --capture-baseline
```

Every command above runs from the repository root -- the one containing this
README. `benchmarks/` next to it is the package the `-m` flag resolves.

---

## The eight benchmarks

| # | Benchmark | Dataset(s) | Headline outputs |
|---|-----------|------------|------------------|
| 1 | **WER** | LibriSpeech clean/other, Earnings21 | `overall_wer`, `overall_clean`, `overall_other`, `overall_earnings21`, S/D/I |
| 2 | **Entity Accuracy** | Earnings21 (spaCy NER) | `entity_precision/recall/f1`, `missed_entity_rate`, `false_entity_rate` |
| 3 | **Diarization** | AMI, Earnings21 | `overall_der`, `speaker_error`, `false_alarm`, `missed_speech` (collar 0.25 s) |
| 4 | **Timestamps** | AMI (word-level refs) | `start_mae_ms`, `end_mae_ms`, `*_p90_ms`, `within_50/100/200ms` |
| 5 | **Multilingual** | FLEURS (14 langs) | `overall_multilingual_wer`, `language_breakdown` (CER for zh/ja) |
| 6 | **Language Switching** | FLEURS (auto-generated) | `overall_wer`, `switch_boundary_wer`, `switch_detection_accuracy`*, `average_switch_latency_ms`*, `per_language_wer` |
| 7 | **Long-form** | AMI, Earnings21, podcasts | `overall_wer`, `hallucination_rate`, `duplicate_rate`, `missing_audio_rate`, `drift_events` |
| 8 | **Price/Performance** | LibriSpeech subset | `rtf`, `hours_per_gpu_hour`, `hours_per_dollar`, `average/p95/p99_latency` |

\* Switch detection + latency require **per-word language labels** from the
provider. Providers that don't emit them get those two fields reported as
`null` with `switch_metrics_supported: false` rather than fabricated. The
scoring path activates automatically for any provider that does emit them.

---

## Reproducibility

- **No proprietary data, no manual curation, no hand-edited transcripts.** Every
  clip selection is a seeded sample (`config.MASTER_SEED` + per-benchmark
  sub-seed) so the frozen subsets are identical on every machine.
- **Frozen forever.** A committed v1 manifest never changes. New selections
  become **v2** (`config.SUITE_VERSION`) rather than mutating v1.
- The Language Switching benchmark is *proprietary* but generated **entirely**
  from public FLEURS clips by `prepare_language_switching.py` — concatenation
  order, chosen languages, and 300–800 ms pauses are all seeded.
- A new developer recreates the whole suite with `prepare_all`.

Benchmark sizes (see `config.SIZES`): WER 100 clean + 100 other + 100 Earnings21
+ 100 SPGISpeech; multilingual 14 langs × 10; language switching 20 recordings ×
4 levels; long-form 15 recordings.

---

## Layout

```
benchmarks/
  config.py             — paths, seeds, frozen sizes, language sets
  cli.py                — run / list entry point
  providers/
    types.py            — Transcript / Word / Segment (the normalized output)
    base.py             — Provider ABC + transcribe_batch + Features
    speech_revolutions.py — our own STT adapter (via the Python SDK)
    registry.py         — name -> provider
  core/
    benchmark.py        — Benchmark ABC (run/score/capture_baseline/compare/report)
    manifest.py normalize.py alignment.py diarization.py metrics.py report.py
  benchmarks/
    wer.py entities.py diarization.py timestamps.py
    multilingual.py language_switching.py longform.py price.py
  datasets/
    prepare_*.py        — deterministic, seeded dataset builders
  manifests/<benchmark>/<name>.jsonl   — committed, frozen
  baselines/<benchmark>/<provider>.json — committed
  reports/  _data/      — generated / large (gitignored)
```

---

## Methodology (read before publishing numbers)

Every scoring choice below is fixed so results are fair across providers and
comparable to third-party figures.

- **Text normalization — Whisper standard.** WER/CER use the Whisper
  `EnglishTextNormalizer` for English and `BasicTextNormalizer` for other
  languages (pinned `whisper_normalizer` version). This is the de-facto standard
  used by OpenAI/AssemblyAI/Whisper reporting: it folds number words↔digits,
  expands contractions, standardizes spelling, strips punctuation, lowercases.
  It **supersedes v1's "keep numbers verbatim"** rule, under which "ten" vs "10"
  wrongly counted as an error and inflated WER unevenly across providers.
- **Reference & hypothesis are normalized identically** for every provider, so
  scores reflect recognition, not formatting.
- **WER aggregation is word-count weighted** (corpus WER via jiwer), never a mean
  of per-file WERs. Languages without word spaces (zh/ja) use **CER**.
- **Confidence intervals.** Every headline metric carries a 95% bootstrap CI
  (1000 resamples, fixed seed, resampling the natural unit: files for WER/DER/
  long-form, matched words for timestamps, files for entities, languages for
  multilingual). **A difference smaller than the CI is not a claim you can
  defend** — report N and CI alongside every published number.
- **Diarization.** DER uses a **0.25 s collar, overlap included**, on AMI
  **Mix-Headset** (the single mixed file a user would upload — not the trivially
  easy per-speaker IHM channels). Numbers are only comparable across systems at
  identical collar/overlap/audio conditions. **cpWER** (concatenated,
  speaker-permutation-invariant WER, Hungarian assignment) is the joint
  ASR+diarization metric vendors report; it needs per-speaker reference text.
- **Price/Performance is indicative, not benchmarked.** RTF/throughput/latency
  swing widely with stack warmth, load, and poll granularity (observed rtf 0.57
  vs 2.02 on identical inputs). Do not publish these without a controlled
  protocol (warm-up, fixed concurrency, repeated runs, representative clip
  lengths).
- **Provider capability.** Text-only APIs (OpenAI GPT-4o Transcribe, Cohere)
  return no word timestamps or speakers; on the timestamp/diarization benchmarks
  they must be reported as **not supported**, never as a 100%-error score.
- **Raw transcripts are cached; scoring is free to re-run.** A provider's raw
  output for a given (file, features) is immutable, so it's persisted under
  `_transcripts/<provider>/` and reused on re-runs — fixing a scoring script
  (e.g. cpWER) or a `normalize()` bug and re-running costs **$0** in API calls.
  **Exception: the local Zephyr model is never cached** (`cacheable=False`) — it
  is actively improved, so every run re-transcribes to reflect current local
  code. Price/perf also bypasses the cache (it must call the API to measure
  latency). Force fresh external transcription with `--refresh`.
- **Datasets are public, permissively licensed, and frozen** (seeded selections;
  see `config.SIZES`). Datasets with licensing or reproducibility doubt
  (Meanwhile, Rev16, TEDLIUM-NC, the wiped CommonVoice HF mirror) are
  **deliberately excluded** so every published number is reproducible from a
  clean source.

## Output formats

Every benchmark emits the same four, all derived from one results dict:
**JSON** (`reports/<b>/<provider>.json`), **Markdown**, **CSV** (per-file), and a
**console** summary. Regressions vs. the committed baseline make the CLI exit
non-zero (skip with `--no-check`; refresh with `--capture-baseline`).

## Adding a provider

1. Subclass `Provider`, implement `submit/poll/download/normalize`.
2. `register("name", YourProvider)` in `providers/registry.py`.
3. `python -m benchmarks.cli run all --provider name`.

Nothing else changes — all eight benchmarks score it identically.
