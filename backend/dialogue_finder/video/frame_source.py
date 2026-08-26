from __future__ import annotations

import collections
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

SEEK_BACK = 48   # frames to seek before the target, then decode forward (keyframe-safe)
MAX_CACHE_FRAMES = 512  # keeps ~512 decoded frames in memory (~17 seconds of 30fps video)


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
        self._cache: collections.OrderedDict[int, np.ndarray] = collections.OrderedDict()

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def index_for_time(self, t: float) -> int:
        return self._clamp(int(round(t * self.fps)))

    def time_for_index(self, i: int) -> float:
        return i / self.fps

    def _clamp(self, index: int) -> int:
        return max(0, min(self.frame_count - 1, index))

    def _cache_frame(self, index: int, frame: np.ndarray) -> None:
        self._cache[index] = frame
        if len(self._cache) > MAX_CACHE_FRAMES:
            self._cache.popitem(last=False)

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
        if index in self._cache:
            return self._cache[index]
        self._seek(index)
        ok, frame = self._cap.read()
        self._next_index = index + 1
        if not ok or frame is None:
            raise IOError(f"failed to decode frame {index}")
        self._cache_frame(index, frame)
        return frame

    def prefetch_range(self, start_index: int, end_index: int) -> None:
        """Sequential single-pass stream decode of a contiguous window [start_index, end_index].
        Reads all frames in a fast tight loop into the memory cache with zero seek overhead."""
        start_index = self._clamp(start_index)
        end_index = self._clamp(end_index)
        if start_index > end_index:
            return
        if all(i in self._cache for i in range(start_index, end_index + 1)):
            return
        self._seek(start_index)
        for i in range(start_index, end_index + 1):
            if i in self._cache:
                self._seek(i + 1)
                continue
            ok, frame = self._cap.read()
            self._next_index = i + 1
            if not ok or frame is None:
                break
            self._cache_frame(i, frame)

    def iter_range(self, start_index: int, end_index: int, step: int) -> Iterator[tuple[int, np.ndarray]]:
        step = max(1, step)
        i = max(0, start_index)
        end_index = min(self.frame_count - 1, end_index)
        while i <= end_index:
            yield i, self.frame_at(i)
            i += step

    def close(self) -> None:
        self._cap.release()
        self._cache.clear()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

