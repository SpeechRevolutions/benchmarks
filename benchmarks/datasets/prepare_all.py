#!/usr/bin/env python3
"""
Run every dataset preparation step in dependency order.

Order matters: FLEURS must be prepared before language switching, which reuses
the FLEURS clip pool.

Every step is public and needs no credentials, with one exception: SPGISpeech is
gated on Hugging Face, so `spgispeech` needs an accepted licence and a token. It
is still listed here rather than left out, because the WER figure published on
our site is the SPGISpeech one -- omitting the step is how a reader ends up
reproducing a different number and assuming ours is wrong. A step that cannot
run reports why and the rest continue.

Usage:
    python -m benchmarks.datasets.prepare_all
    python -m benchmarks.datasets.prepare_all --skip earnings21 spgispeech
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
    prepare_notsofar,
)

STEPS = [
    ("librispeech", lambda: prepare_librispeech.prepare(["test-clean", "test-other"])),
    # Gated on Hugging Face; backs the WER number on the site.
    ("spgispeech", lambda: _spgispeech()),
    ("ami", lambda: prepare_ami.prepare(prepare_ami.AMI_SC_TEST_SET)),
    # Public and ungated; backs the NotSoFar diarization number on the site.
    ("notsofar", lambda: prepare_notsofar.prepare()),
    ("earnings21", lambda: prepare_earnings21.prepare(_earnings21_repo())),
    ("fleurs", lambda: prepare_fleurs.prepare(_fleurs_langs())),
    ("language_switching", lambda: prepare_language_switching.prepare(_ls_levels())),
]


def _spgispeech():
    """Imported late: prepare_spgispeech needs pyarrow, and one gated dataset
    should not decide whether the other five steps can run."""
    from . import prepare_spgispeech
    prepare_spgispeech.prepare(prepare_spgispeech._DEFAULT_SHARD)


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
                    help="step names to skip (e.g. earnings21 spgispeech)")
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
        except SystemExit as e:
            # Raised by a step that cannot proceed and has already explained why
            # (e.g. a gated dataset with no token). Not a crash -- do not dump a
            # traceback over the instructions the reader needs to act on.
            print(f"  [SKIPPED] {name}: {e}")
        except Exception as e:  # noqa: BLE001 - keep going so one failure isn't fatal
            print(f"  [FAILED] {name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
