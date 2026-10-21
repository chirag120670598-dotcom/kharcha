"""Redaction.

You cannot run a spend dashboard without touching prompts, and prompts contain
the most sensitive text in your company. This module exists so that
``prompt_key`` is a hash and nothing else, and so that if you *do* keep a
sample for debugging, it has been scrubbed first.

Rules, in order of what leaks most:

1. Indian identifiers — PAN, Aadhaar, GSTIN, UPI VPA, IFSC
2. Generic identifiers — email, phone, credit-card-shaped numbers
3. Long digit runs, which catch account and order numbers
4. Base64-ish blobs and URLs with query strings (tokens live in query strings)

This is a **best-effort defence, not a guarantee**. A regex redactor will miss
things. The real defence is not storing prompts at all — which is the default
here.

Pure standard library.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Optional

__all__ = [
    "PATTERNS",
    "redact",
    "prompt_key",
    "looks_sensitive",
    "find_sensitive",
]


def _pan(m):     return "[PAN REDACTED]"
def _aadhaar(m): return "[AADHAAR REDACTED]"
def _gstin(m):   return "[GSTIN REDACTED]"
def _vpa(m):     return "[UPI REDACTED]"
def _ifsc(m):    return "[IFSC REDACTED]"
def _email(m):   return "[EMAIL REDACTED]"
def _phone(m):   return "[PHONE REDACTED]"
def _card(m):    return "[CARD REDACTED]"
def _digits(m):  return "[NUMBER REDACTED]"
def _url(m):     return "[URL REDACTED]"
def _token(m):   return "[TOKEN REDACTED]"


#: ``(name, pattern, replacement)`` — applied in this order.
PATTERNS = (
    # PAN: 5 letters, 4 digits, 1 letter. Word-bounded so it does not eat
    # ordinary uppercase words.
    ("pan", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), _pan),
    # Aadhaar: 12 digits, first digit 2-9, optionally space/ dash separated.
    ("aadhaar", re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b"), _aadhaar),
    # GSTIN: 2-digit state code, 10-char PAN, entity code, Z, check.
    ("gstin", re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d Z?[0-9A-Z]\b".replace(" ", "")), _gstin),
    # UPI VPA
    # Email comes BEFORE the UPI VPA rule: an address like ravi@example.com
    # would otherwise match the VPA shape and be redacted as the wrong thing.
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), _email),
    ("vpa", re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9.\-_]{0,63}@[a-z]{3,20}\b"), _vpa),
    # IFSC
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), _ifsc),
    # Indian mobile, with or without the country code
    ("phone", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"), _phone),
    # 13-19 digits, optionally space-separated in 4s (card-shaped)
    ("card", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _card),
    # URLs — query strings are where API keys live
    ("url", re.compile(r"\bhttps?://\S+"), _url),
    # Anything that looks like a bearer token or key=value secret
    ("token", re.compile(
        r"(?i)\b(?:api[_-]?key|token|secret|password|passwd|authorization)"
        r"\s*[:=]\s*\S+"), _token),
)

# Applied last, because the patterns above already claimed the specific shapes.
_LONG_DIGITS = re.compile(r"(?<!\d)\d{8,}(?!\d)")


def redact(text: str, *, min_digits: int = 8) -> str:
    """Scrub identifiers out of ``text``.

    >>> redact("mail ravi@example.com pan AAAPL1234C")
    'mail [EMAIL REDACTED] pan [PAN REDACTED]'
    """
    if not text:
        return text
    out = text
    for _, pattern, repl in PATTERNS:
        out = pattern.sub(repl, out)
    out = _LONG_DIGITS.sub("[NUMBER REDACTED]", out)
    return out


def prompt_key(text: str, *, salt: str = "") -> str:
    """A stable, non-reversible hash of a prompt.

    Used to spot duplicates without ever storing the text. Salt it with a
    per-deployment secret if the prompt space is small enough to brute-force.
    """
    if not text:
        return ""
    return hashlib.sha256(f"{salt}{text}".encode("utf-8")).hexdigest()[:16]


def find_sensitive(text: str) -> List[str]:
    """Which categories of sensitive data appear in ``text``."""
    found: List[str] = []
    for name, pattern, _ in PATTERNS:
        if pattern.search(text):
            found.append(name)
    if _LONG_DIGITS.search(text):
        found.append("long_digits")
    return found


def looks_sensitive(text: str) -> bool:
    """Quick check used to decide whether to keep a sample at all."""
    return bool(find_sensitive(text))
