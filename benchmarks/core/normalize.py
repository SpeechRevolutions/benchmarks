"""
Text normalization for WER/CER scoring.

Uses the **Whisper standard normalizers** (the de-facto community standard, used
by OpenAI/AssemblyAI/Whisper reporting) so our numbers are comparable to
published third-party figures and don't penalize formatting:

  - English content  -> EnglishTextNormalizer: folds number words <-> digits,
    expands contractions, standardizes spelling, strips punctuation, lowercases.
  - Other languages  -> BasicTextNormalizer: unicode-normalize, strip
    punctuation/symbols, lowercase, collapse whitespace (script-agnostic).

This deliberately supersedes the original v1 "keep numbers verbatim" rule: with
that rule "ten" vs "10" scored as an error, inflating WER and making
cross-provider comparison unfair. See README > Methodology.

The normalizers are required (listed in requirements.txt); a missing install
raises a clear error rather than silently falling back to a different metric.
"""

from __future__ import annotations

import re
from functools import lru_cache

# The Whisper normalizers delete ALL bracketed spans, including round parens. In
# FLEURS references round parens wrap SPOKEN content — transliterations,
# abbreviations, appositives, e.g. "(consulte términos)", "(FTIR)", "(Carpanedo)" —
# so deleting it drops real words and inflates WER/CER. We strip only the paren
# CHARACTERS (keeping the words), then let the Whisper normalizer still remove
# square/angle brackets, which mark NON-spoken annotations ("<inaudible>",
# "<crosstalk>"). Applied to ref and hyp identically, so it stays fair + published.
_ROUND_PARENS = re.compile(r"[()]")

# Dotted initialisms ("U.S.", "U.K.", "e.g.", "p.m.") — the Whisper normalizer strips
# the periods but leaves the inter-letter spaces, so "U.S." becomes two tokens "u s"
# while a reference that writes "US" stays one token "us". That mismatch alone was
# ~215 earnings21 hits (each an insertion + a substitution). We collapse a run of
# >=2 single-letter+period groups into the bare letters BEFORE normalization, so
# "U.S."->"US"->"us" matches "US"->"us". Applied to ref and hyp identically (fair,
# published), and a no-op on text that already lacks the dotted form.
_DOTTED_INITIALISM = re.compile(r"\b((?:[A-Za-z]\.){2,})")


def _collapse_dotted_initialisms(text: str) -> str:
    return _DOTTED_INITIALISM.sub(lambda m: m.group(1).replace(".", ""), text or "")


# The Whisper number normalizer treats the interjection "oh" and vocative "O" as the
# digit 0 (phone-number convention: "nineteen oh five" -> 1905). In prose that is a
# bug — "oh look" becomes "0 look", "O Romeo" becomes "0 romeo". We shield standalone
# "oh"/"o" (not apostrophe-attached, so "o'clock" is untouched) behind a sentinel that
# passes through normalization, then restore it. Applied to ref and hyp identically.
_OH_RE = re.compile(r"(?<![\w'])oh(?![\w'])", re.IGNORECASE)
_O_RE = re.compile(r"(?<![\w'])o(?![\w'])", re.IGNORECASE)
_OH_SENT, _O_SENT = "ohsentinelz", "osentinelz"


def _protect_oh(text: str) -> str:
    return _O_RE.sub(_O_SENT, _OH_RE.sub(_OH_SENT, text or ""))


def _restore_oh(normalized: str) -> str:
    return normalized.replace(_OH_SENT, "oh").replace(_O_SENT, "o")


def _keep_paren_content(text: str) -> str:
    return _ROUND_PARENS.sub(" ", _collapse_dotted_initialisms(_protect_oh(text)))


@lru_cache(maxsize=1)
def _english():
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "whisper_normalizer is required for scoring. "
            "Run: pip install -r requirements.txt"
        ) from e
    return EnglishTextNormalizer()


@lru_cache(maxsize=1)
def _basic():
    try:
        from whisper_normalizer.basic import BasicTextNormalizer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "whisper_normalizer is required for scoring. "
            "Run: pip install -r requirements.txt"
        ) from e
    return BasicTextNormalizer()


def english_normalize(text: str) -> str:
    return _restore_oh(_english()(_keep_paren_content(text)))


def basic_normalize(text: str) -> str:
    return _basic()(_keep_paren_content(text))


def normalize_text(text: str) -> str:
    """Default normalizer for HYPOTHESES (and any symmetric use). Raw — no dialect
    canonicalization, so a hypothesis that reproduces source dialect keeps it."""
    return english_normalize(text)


# Eye-dialect / archaic spellings that LibriSpeech (and some verbatim) REFERENCES
# preserve but that a clean, modern transcript renders in standard orthography
# ("OL MISTAH"->"old mister", "'OME"->"home", "CA'M"->"calm"). We canonicalize these in
# the REFERENCE only (normalize_reference / normalize_verbatim), leaving hypotheses raw:
# a provider that modernizes matches the canonical reference; one that reproduces the
# source dialect is penalized. We reward usable transcripts, not faithful archaic
# spelling. Same asymmetric philosophy as verbatim disfluency-stripping; published/fair.
# Applied to RAW text before normalization (so apostrophe forms match).
_DIALECT = [
    (re.compile(r"\bol\b", re.I), "old"),
    (re.compile(r"\bmistah\b", re.I), "mister"),
    (re.compile(r"'?\bome\b", re.I), "home"),
    (re.compile(r"\bca'm\b", re.I), "calm"),
    (re.compile(r"\bagone\b", re.I), "ago"),
    (re.compile(r"\b(goin|doin|comin|nothin|somethin|mornin)'", re.I),
     lambda m: {"goin": "going", "doin": "doing", "comin": "coming",
                "nothin": "nothing", "somethin": "something", "mornin": "morning"}[m.group(1).lower()]),
    (re.compile(r"'twas\b", re.I), "it was"),
    (re.compile(r"'tis\b", re.I), "it is"),
]


def _canonicalize_dialect(text: str) -> str:
    for pat, repl in _DIALECT:
        text = pat.sub(repl, text or "")
    return text


def normalize_reference(text: str) -> str:
    """Normalizer for REFERENCES on clean/read subsets (clean, other, multilingual):
    standard normalization plus dialect canonicalization (see _DIALECT)."""
    return english_normalize(_canonicalize_dialect(text))


# Filler/hesitation tokens that appear in VERBATIM references (Rev.com earnings21 and
# long-form) but not in clean ASR output. Rev transcribes every "um"/"uh"/stammer; a
# clean transcript omits them, so scoring clean output against a verbatim reference
# counts them as deletions (evidence: 66% of our earnings21 deletions were fillers/
# stammers/repeats). Stripping them measures CONTENT accuracy, not disfluency capture.
# Applied identically to ref and hyp (fair), and published/reproducible.
#
# NOTE: we deliberately do NOT extend this to discourse fillers ("you know"/"i mean"/
# "yeah"). Measured on earnings21: stripping them from the verbatim reference (the scorer
# strips ref only) HURT our WER (13.00 -> 13.27), because our own ASR emits those fillers
# (~1.24% of tokens), so removing them from the ref turns our emitted copies into
# insertions. The real fix for verbatim-vs-clean is a CLEAN (non-verbatim) reference such
# as SPGISpeech — not aggressive ref-side disfluency stripping. See research-results.
_FILLERS = {
    "um", "umm", "uh", "uhh", "uhm", "mm", "mmm", "hmm", "mhm", "mmhmm", "uhhuh",
    "er", "err", "ah", "ahh", "eh", "huh",
}


def strip_disfluencies(normalized_text: str) -> str:
    """Remove filler tokens and collapse immediate exact word repeats (stammers).

    Operates on already-normalized (lowercased, punctuation-stripped) text. Applied to
    both reference and hypothesis, so legitimate repeats collapse on both sides and stay
    matched — it only changes the score where one side has a disfluency the other lacks.
    """
    out: list[str] = []
    for tok in normalized_text.split():
        if tok in _FILLERS:
            continue
        if out and out[-1] == tok:  # collapse "the the" / "but but" stammers
            continue
        out.append(tok)
    return " ".join(out)


def normalize_verbatim(text: str) -> str:
    """Normalizer for verbatim-REFERENCE subsets (earnings21, long-form): standard
    English normalization + dialect canonicalization + disfluency stripping, so a clean
    modern transcript is scored on content rather than penalized for correctly omitting
    Rev.com's um/uh/stammers or for modernizing dialect spelling."""
    return strip_disfluencies(english_normalize(_canonicalize_dialect(text)))


def _is_english(code: str | None) -> bool:
    return bool(code) and (code == "en" or code.startswith("en"))


def normalize_for_language(text: str, language_code: str | None) -> str:
    """Language-aware: Whisper English normalizer for English, Basic otherwise."""
    return english_normalize(text) if _is_english(language_code) else basic_normalize(text)


def to_characters(text: str) -> list[str]:
    """Tokenize into non-space characters for CER (scripts without word spaces)."""
    return [c for c in text if not c.isspace()]
