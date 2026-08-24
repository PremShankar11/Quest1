from __future__ import annotations

from typing import Callable

from ..config import Config
from .matcher import score_contains
from ..models import Candidate, StageEvent
from .ocr import read_dialogue
from ..progress import ProgressReporter


def coarse_scan(source, extractor, target: str, start_s: float, end_s: float, fps: float, cfg: Config,
                reporter: ProgressReporter, read: Callable = read_dialogue) -> list[Candidate]:
    """OCR every (source.fps / fps)-th frame in [start_s, end_s]; return one Candidate per sampled frame."""
    step = max(1, int(round(source.fps / fps)))
    a, b = source.index_for_time(start_s), source.index_for_time(end_s)
    total = max(1, (b - a) // step + 1)
    out: list[Candidate] = []
    best = 0.0
    for n, (i, frame) in enumerate(source.iter_range(a, b, step)):
        text = read(extractor, frame, cfg)
        s = score_contains(target, text)
        best = max(best, s)
        out.append(Candidate(i, source.time_for_index(i), text, s))
        if n % 10 == 0 or s >= cfg.ocr_match_threshold:
            reporter.emit(StageEvent("scan", "running", f"frame {i} score {s:.2f} (best {best:.2f})",
                                     min(1.0, (n + 1) / total),
                                     {"frame_index": i, "score": s, "text": text, "best": best}))
    return out


def group_hits(cands: list[Candidate], threshold: float, gap_s: float) -> list[list[Candidate]]:
    hits = [c for c in cands if c.score >= threshold]
    groups: list[list[Candidate]] = []
    for c in hits:
        if groups and c.timestamp_s - groups[-1][-1].timestamp_s <= gap_s:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def pick_group(groups: list[list[Candidate]], occurrence: str) -> list[list[Candidate]]:
    if not groups:
        return []
    if occurrence == "last":
        return [groups[-1]]
    if occurrence == "all":
        return groups
    return [groups[0]]
