from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

SEEK_BACK = 48   # frames to seek before the target, then decode forward (keyframe-safe)


class FrameSource:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"cannot open {self.path}")
        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._next_index = 0            # index of the frame the next read() returns

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def index_for_time(self, t: float) -> int:
        return self._clamp(int(round(t * self.fps)))

    def time_for_index(self, i: int) -> float:
        return i / self.fps

    def _clamp(self, index: int) -> int:
        return max(0, min(self.frame_count - 1, index))

    def _grab_forward(self, count: int) -> None:
        for _ in range(count):
            self._cap.grab()

    def _seek(self, index: int) -> None:
        if index == self._next_index:
            return
        if 0 <= index - self._next_index <= SEEK_BACK * 2:
            self._grab_forward(index - self._next_index)
        else:
            start = max(0, index - SEEK_BACK)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos > index:                 # backend overshot; restart from 0
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                pos = 0
            self._grab_forward(index - pos)
        self._next_index = index

    def frame_at(self, index: int) -> np.ndarray:
        index = self._clamp(index)
        self._seek(index)
        ok, frame = self._cap.read()
        self._next_index = index + 1
        if not ok or frame is None:
            raise IOError(f"failed to decode frame {index}")
        return frame

    def iter_range(self, start_index: int, end_index: int, step: int) -> Iterator[tuple[int, np.ndarray]]:
        step = max(1, step)
        i = max(0, start_index)
        end_index = min(self.frame_count - 1, end_index)
        while i <= end_index:
            yield i, self.frame_at(i)
            i += step

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
