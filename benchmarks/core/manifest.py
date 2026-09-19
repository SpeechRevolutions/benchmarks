"""
JSONL manifest loading.

A manifest is one JSON object per line (``#`` lines are comments). ``audio_path``
is always relative to ``DATA_ROOT`` (the ``_data`` directory) so a committed
manifest resolves on any machine that ran the prepare scripts.

Common fields by benchmark:
    WER / multilingual:  id, audio_path, transcript, duration_s, [language]
    diarization:         id, audio_path, speakers:[{speaker,start,end}], duration_s
    timestamps:          id, audio_path, words:[{word,start,end}], duration_s
    language_switching:  id, audio_path, duration_s, segments:[{language,start,end,transcript}]
    longform:            id, audio_path, transcript, duration_s
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import DATA_ROOT


def load_manifest(path: str | Path) -> list[dict]:
    entries: list[dict] = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {e}") from e
    return entries


def write_manifest(path: str | Path, entries: list[dict], header: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if header:
            for line in header.splitlines():
                f.write(f"# {line}\n")
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def resolve_audio_path(entry: dict, data_root: Path = DATA_ROOT) -> Path:
    p = data_root / entry["audio_path"]
    if not p.exists():
        raise FileNotFoundError(
            f"Audio file not found: {p}\n"
            f"  manifest entry id: {entry.get('id', '?')}\n"
            f"  Run the matching datasets/prepare_*.py script first."
        )
    return p


def items_for_provider(entries: list[dict], data_root: Path = DATA_ROOT) -> list[dict]:
    """Resolve audio paths and return items ready for Provider.transcribe_batch.

    Entries whose audio is missing are skipped with a warning.
    """
    items: list[dict] = []
    for e in entries:
        try:
            audio = resolve_audio_path(e, data_root)
        except FileNotFoundError as exc:
            print(f"  [WARN] {exc}")
            continue
        items.append({**e, "audio_path": str(audio)})
    return items
