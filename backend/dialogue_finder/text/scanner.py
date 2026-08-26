from __future__ import annotations

from typing import Callable

from ..config import Config
from .matcher import score_contains
from ..models import Candidate, CancelledError, StageEvent
from .ocr import read_dialogue
from ..progress import ProgressReporter


def coarse_scan(source, extractor, target: str, start_s: float, end_s: float, fps: float, cfg: Config,
                reporter: ProgressReporter, read: Callable = read_dialogue,
                should_cancel: Callable[[], bool] | None = None,
                existing_cands: list[Candidate] | None = None) -> list[Candidate]:
    """OCR every (source.fps / fps)-th frame in [start_s, end_s]; return one Candidate per sampled frame."""
    step = max(1, int(round(source.fps / fps)))
    start_idx, end_idx = source.index_for_time(start_s), source.index_for_time(end_s)
    total = max(1, (end_idx - start_idx) // step + 1)
    known = {c.frame_index: c for c in (existing_cands or [])}
    out: list[Candidate] = []
    best = 0.0
    for sample_num, (i, frame) in enumerate(source.iter_range(start_idx, end_idx, step)):
        if should_cancel and should_cancel():
            raise CancelledError("cancelled")
        if i in known:
            cand = known[i]
            score = cand.score
            text = cand.text
        else:
            text = read(extractor, frame, cfg)
            score = score_contains(target, text)
            cand = Candidate(i, source.time_for_index(i), text, score)
        best = max(best, score)
        out.append(cand)
        is_hit = score >= cfg.ocr_match_threshold
        if sample_num % 10 == 0 or is_hit:
            reporter.emit(StageEvent("scan", "running", f"frame {i} score {score:.2f} (best {best:.2f})",
                                     min(1.0, (sample_num + 1) / total),
                                     {"frame_index": i, "score": score, "text": text, "best": best}))
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
