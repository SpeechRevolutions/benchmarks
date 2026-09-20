#!/usr/bin/env python3
"""
Run every dataset preparation step in dependency order.

Order matters: FLEURS must be prepared before language switching (which reuses
the FLEURS clip pool); AMI/Earnings21 before long-form (which reuses them).

Usage:
    python -m benchmarks.datasets.prepare_all
    python -m benchmarks.datasets.prepare_all --skip earnings21
"""

from __future__ import annotations

import argparse
import traceback

from . import (
    prepare_ami,
    prepare_earnings21,
    prepare_fleurs,
    prepare_language_switching,
    prepare_librispeech,
)

STEPS = [
    ("librispeech", lambda: prepare_librispeech.prepare(["test-clean", "test-other"])),
    ("ami", lambda: prepare_ami.prepare(prepare_ami.AMI_SC_TEST_SET)),
    ("earnings21", lambda: prepare_earnings21.prepare(_earnings21_repo())),
    ("fleurs", lambda: prepare_fleurs.prepare(_fleurs_langs())),
    ("language_switching", lambda: prepare_language_switching.prepare(_ls_levels())),
]


def _earnings21_repo():
    from .. import config
    from pathlib import Path
    return Path(config.DATA_ROOT / "_earnings21_repo")


def _fleurs_langs():
    from .. import config
    return config.MULTILINGUAL_LANGUAGES


def _ls_levels():
    from .. import config
    return list(config.LANGUAGE_SWITCHING_LEVELS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare all benchmark datasets")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="step names to skip (e.g. earnings21 fleurs)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these steps")
    args = ap.parse_args()

    for name, fn in STEPS:
        if name in args.skip:
            print(f"\n=== SKIP {name} ===")
            continue
        if args.only and name not in args.only:
            continue
        print(f"\n=== PREPARE {name} ===")
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - keep going so one failure isn't fatal
            print(f"  [FAILED] {name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
