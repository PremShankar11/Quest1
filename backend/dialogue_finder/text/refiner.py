from __future__ import annotations

from typing import Callable

import cv2

from ..config import Config
from .matcher import score_contains
from ..models import Candidate
from .ocr import crop_band, read_dialogue


def first_true(lo: int, hi: int, pred: Callable[[int], bool]) -> int:
    """Smallest i in (lo, hi] with pred(i) True. Contract: pred(lo) is False, pred(hi) is True."""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return hi


MAX_BACK_HOPS = 8   # module constant; each hop is one OCR call


def refine_first_frame(source, extractor, target: str, hit_index: int, prev_index: int, cfg: Config,
                       read: Callable = read_dialogue, step: int = 1) -> tuple[Candidate, bool]:
    """Binary-search OCR score between prev_index (no match) and hit_index (match) → exact first frame.

    `prev_index` is only a coarse-scan sample, not guaranteed to precede the text: if the text is
    already visible there, hop back (up to MAX_BACK_HOPS steps) looking for a real no-match anchor
    before binary-searching. Returns (Candidate, exact) — exact is False only when back-hopping ran
    out of budget while the text was still visible, so the true first frame may be earlier still.
    """
    cache: dict[int, tuple[str, float]] = {}

    def ocr_at(i: int) -> tuple[str, float]:
        if i not in cache:
            text = read(extractor, source.frame_at(i), cfg)
            cache[i] = (text, score_contains(target, text))
        return cache[i]

    hops = 0
    while prev_index >= 0 and hops < MAX_BACK_HOPS and ocr_at(prev_index)[1] >= cfg.ocr_match_threshold:
        hit_index, prev_index = prev_index, prev_index - step
        hops += 1
    exact = not (prev_index >= 0 and hops == MAX_BACK_HOPS and ocr_at(prev_index)[1] >= cfg.ocr_match_threshold)

    if exact:
        lo = max(-1, prev_index)
        first = first_true(lo, hit_index, lambda i: ocr_at(i)[1] >= cfg.ocr_match_threshold)
    else:
        # no known-False anchor within the hop budget; first_true's contract can't be satisfied,
        # so report the last confirmed-matching hop instead of pretending to refine further.
        first = hit_index
    text, score = ocr_at(first)
    return Candidate(first, source.time_for_index(first), text, score), exact


def _edge_density(frame, cfg: Config) -> float:
    band = crop_band(frame, cfg.band_fraction)
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    return float(cv2.Canny(gray, 100, 200).mean() / 255.0)


def classify_appearance(source, index: int, cfg: Config) -> str:
    """'pop-in' if the subtitle band's edge density jumps at `index`, else 'fade-in'."""
    if index <= 0:
        return "pop-in"
    before = _edge_density(source.frame_at(index - 1), cfg)
    at = _edge_density(source.frame_at(index), cfg)
    return "pop-in" if at > max(0.002, before * 2.0) else "fade-in"
