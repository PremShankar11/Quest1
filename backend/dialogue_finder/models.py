from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


def format_timestamp(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Window:
    start_s: float
    end_s: float
    score: float
    matched_text: str


@dataclass
class Candidate:
    frame_index: int
    timestamp_s: float
    text: str
    score: float


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


@dataclass
class StageEvent:
    stage: str                       # download | transcribe | locate | scan | refine | done | error
    status: str                      # running | ok | skipped | fallback | error
    message: str = ""
    progress: float | None = None    # 0..1
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    t: float = 0.0


class PipelineError(Exception):
    """Fatal, user-facing. The CLI prints str(e) and exits 1.

    Defined here (not in pipeline.py) so text/scanner.py can raise CancelledError, a subclass,
    without importing pipeline.py -- pipeline.py already imports from models.py, so the reverse
    import would be circular. pipeline.py re-exports PipelineError for callers that import it
    from `dialogue_finder.pipeline` (the CLI, tests)."""


class CancelledError(PipelineError):
    """Raised when should_cancel() returns True mid-scan; message is "cancelled". A PipelineError
    subclass so it propagates through run()'s `except PipelineError: raise` unchanged, and so
    callers that catch coarse_scan's cancellation directly still see a PipelineError."""


@dataclass
class Result:
    timestamp_s: float
    frame_index: int
    text: str
    confidence: str                  # HIGH | MEDIUM | LOW
    source: str                      # ocr | audio | ocr-weak
    note: str = ""
    fps: float = 0.0
    image_path: str = ""
    prev_image_path: str = ""
    appearance: str = ""             # pop-in | fade-in | ""
    window: Window | None = None
    candidates: list[Candidate] = field(default_factory=list)
    alternatives: list[Candidate] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.timestamp_s)

    def format_block(self) -> str:
        lines = [
            f"Timestamp : {self.timestamp}",
            f"Frame     : {self.frame_index}",
            f'Text      : "{self.text}"',
            f"Confidence: {self.confidence}  (source: {self.source}{'; ' + self.note if self.note else ''})",
            f"Image     : {self.image_path}",
        ]
        if self.prev_image_path:
            lines.append(f"Previous  : {self.prev_image_path}  ({self.appearance or 'frame before'})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp
        return d
