from __future__ import annotations

import re
from typing import Iterator

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


def _scored_spans(words: list[Word], target: str) -> Iterator[Window]:
    """Every window of ~len(target words) sliding over `words`, scored against `target`.

    Shared by `best_word_window` (best score wins) and `all_word_windows` (all spans above
    a threshold, deduped to non-overlapping). Order is size-ascending then start-ascending,
    which callers that break ties by iteration order (e.g. `max()`) rely on.
    """
    n = max(1, len(normalize(target).split()))
    for size in range(max(1, n - 1), n + 3):
        for i in range(0, max(1, len(words) - size + 1)):
            span = words[i:i + size]
            if not span:
                continue
            text = " ".join(w.text for w in span)
            yield Window(span[0].start, span[-1].end, score_similar(target, text), text)


def all_word_windows(words: list[Word], target: str, threshold: float, cap: int) -> list[Window]:
    """Every non-overlapping span scoring >= threshold, best first (greedy: take the best, drop
    overlaps, repeat)."""
    spans = [w for w in _scored_spans(words, target) if w.score >= threshold]
    spans.sort(key=lambda w: (-w.score, w.start_s))
    chosen: list[Window] = []
    for w in spans:
        if all(w.end_s <= c.start_s or w.start_s >= c.end_s for c in chosen):
            chosen.append(w)
        if len(chosen) >= cap:
            break
    return chosen


def best_word_window(words: list[Word], target: str) -> Window | None:
    """Slide windows of ~len(target words) over the transcript; return the best-scoring span."""
    return max(_scored_spans(words, target), key=lambda w: w.score, default=None)
