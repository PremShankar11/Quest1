from __future__ import annotations

import re

from rapidfuzz import fuzz

from ..models import Word, Window

_PUNCT = re.compile(r"[^a-z0-9' ]+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower().replace("\u2019", "'")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def score_contains(target: str, haystack: str) -> float:
    """How well `target` appears inside `haystack` (0..1). Tolerant of OCR noise and extra words.
    partial_ratio alone scores any short substring of the target as 1.0 (a lone OCR "R" matched a 28-char
    line), so the score is scaled by how much of the target's length the haystack can actually cover."""
    t, h = normalize(target), normalize(haystack)
    if not t or not h:
        return 0.0
    coverage = min(1.0, len(h) / len(t))
    return fuzz.partial_ratio(t, h) / 100.0 * coverage


def score_similar(a: str, b: str) -> float:
    """Order-insensitive similarity of two short phrases (0..1)."""
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def best_word_window(words: list[Word], target: str) -> Window | None:
    """Slide windows of ~len(target words) over the transcript; return the best-scoring span."""
    if not words:
        return None
    n = max(1, len(normalize(target).split()))
    best: Window | None = None
    for size in range(max(1, n - 1), n + 3):
        for i in range(0, max(1, len(words) - size + 1)):
            span = words[i:i + size]
            if not span:
                continue
            text = " ".join(w.text for w in span)
            s = score_similar(target, text)
            if best is None or s > best.score:
                best = Window(span[0].start, span[-1].end, s, text)
    return best
