"""
Uniform benchmark contract.

Every benchmark subclasses :class:`Benchmark` and implements ``score()`` plus a
small amount of metadata. The base class supplies the rest of the spec's
required surface generically::

    run()              — submit/poll/download/normalize via a Provider
    score()            — (abstract) compute metrics from run output
    capture_baseline() — freeze current summary as the regression baseline
    compare()          — diff current summary vs. baseline, return regressions
    generate_report()  — write JSON + Markdown + CSV and print a console summary

``run()`` returns ``{subset_name: list[TranscriptionResult]}``. Each result's
``.meta`` is the originating manifest entry, so ``score()`` reads references
straight off the results without re-joining.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .. import config
from ..providers.base import Features, Provider, TranscriptionResult
from . import report
from .manifest import items_for_provider, load_manifest

# direction: "lower" => higher value is a regression; "higher" => lower is a regression
RegressionSpec = tuple[str, float]  # (direction, tolerance)

#: Per-dataset domain glossaries (one term per line): glossaries/<benchmark>/<subset>.txt.
#: Opt-in via BENCH_GLOSSARY=1 so the default run stays a glossary-free honest baseline.
#: The SAME glossary is fed to every provider that supports keyword/custom-vocab biasing.
_GLOSSARY_ROOT = Path(__file__).resolve().parents[1] / "glossaries"


def _load_subset_glossary(benchmark_name: str, subset: str) -> tuple[str, ...]:
    if not os.environ.get("BENCH_GLOSSARY"):
        return ()
    p = _GLOSSARY_ROOT / benchmark_name / f"{subset}.txt"
    if not p.exists():
        return ()
    return tuple(w.strip() for w in p.read_text().splitlines() if w.strip())


@dataclass
class Benchmark:
    """Base class. Subclasses set the class attributes and implement score()."""

    #: short benchmark id, also the manifests/ + baselines/ + reports/ subdir
    name: str = "base"
    #: features requested from the provider for this benchmark
    features: Features = field(default_factory=Features)
    #: subset name -> manifest filename stem (under manifests/<name>/)
    subsets: dict[str, str] = field(default_factory=dict)
    #: summary metric key -> (direction, tolerance) for regression checks
    regression_specs: dict[str, RegressionSpec] = field(default_factory=dict)
    #: whether this benchmark's transcripts can be shared via the suite-level
    #: dedup pass. False for benchmarks that must transcribe independently:
    #: price (measures real submission timing) and multilingual (per-file
    #: language hints + a custom run()).
    dedup_safe: bool = True

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self, provider: Provider, *, progress: bool = True,
            use_cache: bool = True) -> dict[str, list[TranscriptionResult]]:
        """Transcribe every subset's manifest with ``provider``."""
        runs: dict[str, list[TranscriptionResult]] = {}
        for subset, stem in self.subsets.items():
            path = config.manifest_path(self.name, stem)
            try:
                entries = load_manifest(path)
            except FileNotFoundError:
                if progress:
                    print(f"  [skip] {self.name}/{subset}: no manifest at {path}")
                runs[subset] = []
                continue
            items = items_for_provider(entries)
            import os as _os
            _lim = _os.environ.get("BENCH_MAX_FILES")
            if _lim:
                items = items[: int(_lim)]
            if not items:
                runs[subset] = []
                continue
            feats = self.features
            gloss = _load_subset_glossary(self.name, subset)
            if gloss:
                feats = replace(feats, custom_vocabulary=gloss)
                if progress:
                    print(f"  [glossary] {self.name}/{subset}: {len(gloss)} terms")
            if progress:
                print(f"\n[{self.name}/{subset}] transcribing {len(items)} files ...")
            runs[subset] = provider.transcribe_batch(
                items, feats, progress=progress, use_cache=use_cache)
        return runs

    # ── score (abstract) ───────────────────────────────────────────────────────

    def score(self, runs: dict[str, list[TranscriptionResult]], *, provider_name: str) -> dict:
        raise NotImplementedError

    def evaluate(self, provider: Provider, *, progress: bool = True,
                 use_cache: bool = True) -> dict:
        """Convenience: run then score."""
        runs = self.run(provider, progress=progress, use_cache=use_cache)
        return self.score(runs, provider_name=provider.name)

    # ── baseline / compare ───────────────────────────────────────────────────

    def expected_file_count(self) -> int:
        """How many files this benchmark's manifests declare, across all subsets.

        0 when nothing can be read, which callers treat as "cannot verify" rather than
        "expected nothing" — a missing manifest must not silently authorise a partial
        baseline. Honours BENCH_MAX_FILES so a deliberately limited run is not misread as a
        truncated one.
        """
        limit = os.environ.get("BENCH_MAX_FILES")
        total = 0
        for stem in self.subsets.values():
            try:
                items = items_for_provider(load_manifest(config.manifest_path(self.name, stem)))
            except FileNotFoundError:
                continue
            total += min(len(items), int(limit)) if limit else len(items)
        return total

    def capture_baseline(self, results: dict, provider_name: str) -> Path | None:
        """Freeze the current summary as the regression baseline.

        Refuses to overwrite an existing baseline with a degenerate run (no
        files scored, or every tracked metric is None) — otherwise a transient
        failure like the provider being down would silently destroy a good
        baseline.
        """
        summary = results.get("summary", {})
        tracked = list(self.regression_specs) or list(summary)
        all_empty = all(summary.get(k) is None for k in tracked)
        if summary.get("n_files", 0) == 0 or all_empty:
            print(f"  [skip baseline] {self.name}/{provider_name}: run produced no "
                  f"usable data (n_files={summary.get('n_files', 0)}); baseline left unchanged.")
            return None

        # A PARTIAL run is the dangerous case, not an empty one. Empty is obvious and already
        # refused above; partial produces a clean-looking summary over whichever files
        # survived, and freezing it makes a subset the definition of correct. This is not
        # hypothetical: a 600 s job deadline dropped the long AMI meetings mid-job, and the
        # diarization baseline was captured from 49 of 61 files -- 10 of 20 AMI, 3 of 5 DiPCo.
        expected = self.expected_file_count()
        actual = summary.get("n_files", 0)
        if expected and actual < expected:
            print(f"  [skip baseline] {self.name}/{provider_name}: scored {actual} of "
                  f"{expected} manifest files. A baseline frozen from a partial run makes a "
                  f"subset the reference; fix the failures or pass BENCH_ALLOW_PARTIAL_BASELINE=1.")
            if os.environ.get("BENCH_ALLOW_PARTIAL_BASELINE") != "1":
                return None

        path = config.baseline_path(self.name, provider_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        baseline = {
            "benchmark": self.name,
            "provider": provider_name,
            "suite_version": config.SUITE_VERSION,
            "summary": results.get("summary", {}),
            "regression_specs": {k: list(v) for k, v in self.regression_specs.items()},
        }
        with open(path, "w") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        return path

    def load_baseline(self, provider_name: str) -> dict | None:
        path = config.baseline_path(self.name, provider_name)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def compare(self, results: dict, provider_name: str) -> list[str]:
        """Return regression strings (empty == pass / no baseline)."""
        cur = results.get("summary", {})

        # A run that measured NOTHING is a failure, and is checked BEFORE the baseline so it
        # fires even for a provider that has none.
        #
        # Every metric comes back None when no file completes, and the loop below skips a None
        # rather than failing on it -- so twenty files failing to submit scored as "all
        # benchmarks within baseline tolerance". A suite that cannot tell a total outage from a
        # clean pass cannot guard anything, and this is the one place whose job is that
        # distinction. Observed: the entire timestamps suite reported PASS on n_files 0 after
        # every submit returned HTTP 500.
        if not cur.get("n_files"):
            return [
                f"n_files: {cur.get('n_files')!r} — the run produced no measurements; "
                "treating as failure rather than as a pass"
            ]

        baseline = self.load_baseline(provider_name)
        if not baseline:
            return []
        bl = baseline.get("summary", {})
        regressions: list[str] = []
        for metric, (direction, tol) in self.regression_specs.items():
            base_v = bl.get(metric)
            cur_v = cur.get(metric)
            if base_v is not None and cur_v is None:
                # The baseline measured this and the run did not. Silently skipping it is how
                # a metric that stops being produced looks like a metric that stopped
                # regressing.
                regressions.append(f"{metric}: missing from this run (baseline {base_v})")
                continue
            if base_v is None or cur_v is None:
                continue
            if not isinstance(base_v, (int, float)) or not isinstance(cur_v, (int, float)):
                continue
            if direction == "lower" and cur_v > base_v + tol:
                regressions.append(
                    f"{metric}: {cur_v:.6f} > baseline {base_v:.6f} + {tol} = {base_v + tol:.6f}"
                )
            elif direction == "higher" and cur_v < base_v - tol:
                regressions.append(
                    f"{metric}: {cur_v:.6f} < baseline {base_v:.6f} - {tol} = {base_v - tol:.6f}"
                )
        return regressions

    # ── report ───────────────────────────────────────────────────────────────

    def generate_report(
        self,
        results: dict,
        provider_name: str,
        *,
        formats: tuple[str, ...] = ("json", "md", "csv", "console"),
        regressions: list[str] | None = None,
    ) -> dict[str, Path]:
        written: dict[str, Path] = {}
        if "json" in formats:
            p = config.report_path(self.name, provider_name, "json")
            report.write_json(results, p)
            written["json"] = p
        if "md" in formats:
            p = config.report_path(self.name, provider_name, "md")
            report.write_markdown(results, p)
            written["md"] = p
        if "csv" in formats:
            p = config.report_path(self.name, provider_name, "csv")
            report.write_csv(results, p)
            written["csv"] = p
        if "console" in formats:
            report.print_console(results, regressions)
        return written

    # ── helpers for subclasses ─────────────────────────────────────────────────

    def _base_result(self, provider_name: str, summary: dict, per_file: list[dict]) -> dict:
        return {
            "benchmark": self.name,
            "provider": provider_name,
            "suite_version": config.SUITE_VERSION,
            "summary": summary,
            "per_file": per_file,
        }
