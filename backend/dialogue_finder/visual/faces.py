"""Face detection (YuNet), IoU tracking, and face crops for visual verification (Plan 4).

`build_tracks` walks a candidate window frame-by-frame with a `FaceDetector`, feeds every
frame's boxes into an `IouTracker`, and returns the tracks that survive the usable-track
filter (`min_track_s` * fps frames, `min_face_px` median height -- see the plan's Global
Constraints). `crop_face` produces the square grey crop LR-ASD's preprocessing expects
(spike note `preprocessing`, replicating `Columbia_test.py`'s `crop_video` geometry).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from ..config import Config
from .model_files import fetch_verified
from ..models import CancelledError, FaceTrack, StageEvent
from ..progress import ProgressReporter

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
# raw.githubusercontent.com serves opencv_zoo's models via Git LFS and a naive `curl -L` there
# can silently return a ~134-byte pointer stub instead of the binary; media.githubusercontent.com
# serves the real blob (confirmed in docs/superpowers/spikes/2026-08-25-lrasd-spike.md).
# Manual fallback if the download below fails after retries:
#   curl -L -o cache/models/face_detection_yunet_2023mar.onnx \
#     https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
# sha256 of the file at YUNET_URL, hashed 2026-08-25. Pinned so a compromised/mirrored/corrupted
# download is rejected instead of silently loaded into cv2.FaceDetectorYN.
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

PROGRESS_EVERY = 24   # frames between "verify running" progress events in build_tracks

# crop_face geometry constants, replicated from Columbia_test.py's crop_video (see spike note
# `preprocessing`): crop_scale widens the square crop around the box, PAD_VALUE fills the
# frame's padded border (grey, matches the greyscale model input).
_CROP_SCALE = 0.40
_PAD_VALUE = 110


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]: ...


def _verify_yunet_hash(path: Path, expected: str = YUNET_SHA256) -> None:
    """Raise RuntimeError (and delete `path`) if its sha256 doesn't match `expected`."""
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"YuNet model integrity check failed for {path}: expected sha256 {expected}, "
            f"got {actual}. Deleted the bad file.\n"
            f"Manual fallback: curl -L -o {path} {YUNET_URL}"
        )


def _download_yunet(dest: Path) -> None:
    fetch_verified(YUNET_URL, dest, _verify_yunet_hash, "YuNet model")


class YuNetDetector:
    """`cv2.FaceDetectorYN` wrapper: lazy create, resizes on frame-size change, int boxes."""

    def __init__(self, models_dir: Path, score_threshold: float = 0.7) -> None:
        self.models_dir = Path(models_dir)
        self.score_threshold = score_threshold
        self._detector = None
        self._size: tuple[int, int] | None = None

    def _model_path(self) -> Path:
        path = self.models_dir / YUNET_FILENAME
        if not path.exists():
            _download_yunet(path)
        _verify_yunet_hash(path)   # re-check on every load, not just freshly-downloaded files
        return path

    def _ensure(self, w: int, h: int) -> None:
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(str(self._model_path()), "", (w, h))
            self._detector.setScoreThreshold(self.score_threshold)
            self._size = (w, h)
        elif self._size != (w, h):
            self._detector.setInputSize((w, h))
            self._size = (w, h)

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        self._ensure(w, h)
        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        return [(int(round(f[0])), int(round(f[1])), int(round(f[2])), int(round(f[3]))) for f in faces]


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class _Track:
    __slots__ = ("track_id", "frames", "boxes", "last_frame")

    def __init__(self, track_id: int, frame_index: int, box: tuple[int, int, int, int]) -> None:
        self.track_id = track_id
        self.frames = [frame_index]
        self.boxes = [box]
        self.last_frame = frame_index


class IouTracker:
    """Greedy IoU tracker: each new box joins the open track with the highest IoU above
    `iou_threshold` (each track and box used at most once per frame); unmatched boxes start
    new tracks; a track closes once it has gone unmatched for more than `max_gap` frames."""

    def __init__(self, iou_threshold: float = 0.5, max_gap: int = 3) -> None:
        self.iou_threshold = iou_threshold
        self.max_gap = max_gap
        self._tracks: list[_Track] = []
        self._next_id = 0

    def update(self, frame_index: int, boxes: list[tuple[int, int, int, int]]) -> None:
        # A track that has gone unmatched for too long can never gain a later `last_frame`,
        # so it can never re-qualify here -- no separate closed list is needed.
        open_tracks = [t for t in self._tracks if frame_index - t.last_frame <= self.max_gap]

        candidates = []
        for ti, t in enumerate(open_tracks):
            for bi, b in enumerate(boxes):
                score = _iou(t.boxes[-1], b)
                if score >= self.iou_threshold:
                    candidates.append((score, ti, bi))
        candidates.sort(key=lambda c: -c[0])

        matched_boxes: set[int] = set()
        assignment: dict[int, int] = {}
        for score, ti, bi in candidates:
            if ti in assignment or bi in matched_boxes:
                continue
            matched_boxes.add(bi)
            assignment[ti] = bi

        for ti, t in enumerate(open_tracks):
            if ti in assignment:
                box = boxes[assignment[ti]]
                t.frames.append(frame_index)
                t.boxes.append(box)
                t.last_frame = frame_index

        for bi, box in enumerate(boxes):
            if bi not in matched_boxes:
                self._tracks.append(_Track(self._next_id, frame_index, box))
                self._next_id += 1

    def tracks(self) -> list[FaceTrack]:
        ordered = sorted(self._tracks, key=lambda t: t.frames[0])
        return [FaceTrack(track_id=t.track_id, frames=list(t.frames), boxes=list(t.boxes)) for t in ordered]


def build_tracks(
    source,
    detector: FaceDetector,
    start_index: int,
    end_index: int,
    cfg: Config,
    reporter: ProgressReporter | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[FaceTrack]:
    """Track faces over every frame in `[start_index, end_index]`, then drop unusable tracks
    (too short: < `min_track_s` * fps frames, or too small: median height < `min_face_px`)."""
    tracker = IouTracker()
    span = max(1, end_index - start_index)
    for n, (frame_index, frame) in enumerate(source.iter_range(start_index, end_index, 1)):
        if should_cancel is not None and should_cancel():
            raise CancelledError("cancelled")
        tracker.update(frame_index, detector.detect(frame))
        if reporter is not None and n % PROGRESS_EVERY == 0:
            reporter.emit(StageEvent("verify", "running", f"face tracking frame {frame_index}",
                                      min(1.0, (frame_index - start_index) / span)))
    min_frames = cfg.min_track_s * source.fps
    return [t for t in tracker.tracks() if len(t.frames) >= min_frames and t.median_height() >= cfg.min_face_px]


def crop_face(frame: np.ndarray, box: tuple[int, int, int, int], size: int) -> np.ndarray:
    """Square grey crop around `box`, sized `size`x`size`, matching LR-ASD's expected input.

    Replicates `Columbia_test.py`'s `crop_video` geometry exactly (spike note `preprocessing`):
    pad the frame with grey (110), take a square region of side `2 * s * (1 + crop_scale)`
    centred on the box (`s = max(w, h) / 2`) but offset down (more chin/torso than forehead),
    resize to `2 * size` colour, convert to grey, then take the center `size`x`size` crop --
    at `size=112` this is exactly the spike's 224 -> grey -> center-112 pipeline.
    """
    x, y, w, h = box
    s = max(w, h) / 2
    cx, cy = x + w / 2, y + h / 2
    bsi = int(s * (1 + 2 * _CROP_SCALE))
    padded = np.pad(frame, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=_PAD_VALUE)
    mx, my = cx + bsi, cy + bsi
    top, bottom = int(my - s), int(my + s * (1 + 2 * _CROP_SCALE))
    left, right = int(mx - s * (1 + _CROP_SCALE)), int(mx + s * (1 + _CROP_SCALE))
    face = padded[top:bottom, left:right]
    resized = cv2.resize(face, (2 * size, 2 * size))
    grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    offset = size // 2
    return grey[offset:offset + size, offset:offset + size]
