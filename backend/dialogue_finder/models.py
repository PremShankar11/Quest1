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

    def padded(self, duration_s: float, pad_s: float) -> tuple[float, float]:
        """(start_s - pad_s, end_s + pad_s), clamped to [0, duration_s]."""
        return max(0.0, self.start_s - pad_s), min(duration_s, self.end_s + pad_s)


@dataclass
class Candidate:
    frame_index: int
    timestamp_s: float
    text: str
    score: float


@dataclass
class FaceTrack:
    """A face tracked across consecutive frames of a candidate window (visual stage, Plan 4).

    `frames[j]` is the absolute video frame index and `boxes[j]`/`scores[j]` align to it --
    see the plan's frame-alignment convention. `scores` defaults empty because a track exists
    (from the face detector) before LR-ASD has scored it."""
    track_id: int
    frames: list[int]
    boxes: list[tuple[int, int, int, int]]     # (x, y, w, h) per frame, aligned to `frames`
    scores: list[float] = field(default_factory=list)

    @property
    def start_index(self) -> int:
        return self.frames[0]

    @property
    def end_index(self) -> int:
        return self.frames[-1]

    def median_height(self) -> float:
        if not self.boxes:
            return 0.0
        heights = sorted(b[3] for b in self.boxes)
        mid = len(heights) // 2
        if len(heights) % 2:
            return float(heights[mid])
        return (heights[mid - 1] + heights[mid]) / 2.0


@dataclass
class Occurrence:
    """One candidate window classified by the visual stage (Plan 4): OCR + face tracks +
    active-speaker detection. `klass` is one of valid-text | valid-speaker | uncertain | invalid."""
    window: Window
    klass: str
    frame_index: int
    ocr_score: float
    faces: int
    asd_mean: float
    speaker_box: tuple[int, int, int, int] | None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """asdict() already recurses into `window` (a nested dataclass), turning it into a
        plain dict; tuples (e.g. `speaker_box`) survive as tuples, which json.dumps serialises
        as arrays same as a list."""
        return asdict(self)


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
    # Visual verification mode (Plan 4) fields -- populated by the hybrid mode's verify stage
    # (Task 6); empty/default for the audio, ocr and audio+ocr modes.
    occurrence_class: str = ""       # valid-text | valid-speaker | uncertain | invalid
    speaker_box: list[int] | None = None      # [x, y, w, h] on the result frame
    speaker_image_path: str = ""
    occurrences: list[dict] = field(default_factory=list)

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
        if self.occurrence_class:
            lines.append(f"Occurrence: {self.occurrence_class}")
        if self.speaker_box:
            x, y, w, h = self.speaker_box
            lines.append(f"Speaker   : {x},{y},{w},{h}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp
        return d
