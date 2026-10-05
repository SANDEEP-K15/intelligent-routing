"""Request-text cleaning, applied identically to training data, test data and live API input."""

from __future__ import annotations

import re
import unicodedata

# Byte sequences produced when UTF-8 text is decoded as Windows-1252, which is how the
# legacy Zoho export corrupted 480 training texts (e.g. "urgÃ©nt", "â€¦").
_MOJIBAKE_MARKERS = ("Ã", "â€", "Â")

# Order numbers ("KO" + digits) and product registration numbers ("SR" + digits). Discovery showed they
# carry no routing signal (they never link related requests) and they identify customers.
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z]{2}\d{4,}\b")
_PUNCT_SPACE_RE = re.compile(r"[…–—]")  # ellipsis, en dash, em dash
_WHITESPACE_RE = re.compile(r"\s+")


def repair_mojibake(text: str) -> str:
    """Undo UTF-8-read-as-cp1252 corruption. Leaves clean text untouched."""
    if not any(marker in text for marker in _MOJIBAKE_MARKERS):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def remove_identifiers(text: str) -> str:
    return _IDENTIFIER_RE.sub(" ", text)


def clean_text(text: str) -> str:
    """Canonical cleaning used everywhere a request text enters the model."""
    if not isinstance(text, str):
        raise TypeError(f"request_text must be a string, got {type(text).__name__}")
    text = repair_mojibake(text)
    text = _PUNCT_SPACE_RE.sub(" ", text)
    text = strip_accents(text)
    text = remove_identifiers(text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip(" ,.-:")
