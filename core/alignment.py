"""
Word-sequence alignment (LCS) for timestamp evaluation.

Only exact text matches (after normalization) are aligned, so substituted words
don't contaminate timing statistics. Operates on Word objects from a Transcript
or on plain ``{"word"/"text", "start", "end"}`` dicts.
"""

from __future__ import annotations

from typing import Any

from .normalize import normalize_text


def _wtext(w: Any) -> str:
    if hasattr(w, "text"):
        return w.text
    return w.get("word", w.get("text", ""))


def align_word_sequences(ref_words: list[Any], hyp_words: list[Any]) -> list[tuple[Any, Any]]:
    """Return matched (ref_word, hyp_word) pairs via longest common subsequence."""
    ref_texts = [normalize_text(_wtext(w)) for w in ref_words]
    hyp_texts = [normalize_text(_wtext(w)) for w in hyp_words]
    m, n = len(ref_texts), len(hyp_texts)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_texts[i - 1] == hyp_texts[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    matches: list[tuple[Any, Any]] = []
    i, j = m, n
    while i > 0 and j > 0:
        if ref_texts[i - 1] == hyp_texts[j - 1]:
            matches.append((ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    matches.reverse()
    return matches
