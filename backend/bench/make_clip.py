"""Generate a synthetic video with known text appearing at a known frame. Ground truth for tests/bench."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_clip(out: Path, *, text: str, appear_s: float, duration_s: float = 15.0, fps: int = 24,
              size: tuple[int, int] = (640, 360), position: str = "bottom", fade_frames: int = 0,
              scale: float = 1.0) -> dict:
    w, h = size
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open output")
    total = int(round(duration_s * fps))
    appear_frame = int(round(appear_s * fps))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9 * scale * (w / 640)
    thickness = max(1, int(round(2 * scale * (w / 640))))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - tw) // 2
    y = {"bottom": int(h * 0.90), "top": int(h * 0.12) + th, "center": (h + th) // 2}[position]
    rng = np.random.default_rng(0)
    for i in range(total):
        # moving gradient background so frames are not identical (like real video)
        base = np.linspace(20, 90, w, dtype=np.uint8)
        frame = np.tile(base, (h, 1))
        frame = np.roll(frame, i * 2, axis=1)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.add(frame, rng.integers(0, 6, frame.shape, dtype=np.uint8))
        if i >= appear_frame:
            alpha = 1.0 if fade_frames <= 0 else min(1.0, (i - appear_frame + 1) / fade_frames)
            overlay = frame.copy()
            cv2.putText(overlay, text, (x, y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        writer.write(frame)
    writer.release()
    return {"frame": appear_frame, "timestamp": appear_frame / fps, "fps": fps, "text": text}


if __name__ == "__main__":
    import sys
    print(make_clip(Path(sys.argv[1]), text=sys.argv[2], appear_s=float(sys.argv[3])))
