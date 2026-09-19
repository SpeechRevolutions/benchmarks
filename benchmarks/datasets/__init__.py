"""
Dataset preparation scripts.

Each prepare_*.py is runnable as a module and writes frozen, seeded manifests.
Run them once to recreate the entire benchmark suite from public datasets:

    python -m benchmarks.datasets.prepare_librispeech
    python -m benchmarks.datasets.prepare_ami
    python -m benchmarks.datasets.prepare_earnings21
    python -m benchmarks.datasets.prepare_fleurs
    python -m benchmarks.datasets.prepare_language_switching
    python -m benchmarks.datasets.prepare_longform

or all at once:

    python -m benchmarks.datasets.prepare_all
"""
