from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from ..config import Config
from .matcher import normalize


class TextExtractor(Protocol):
    def read(self, image: np.ndarray) -> str: ...


def crop_band(image: np.ndarray, fraction: float) -> np.ndarray:
    h = image.shape[0]
    return image[int(h * (1 - fraction)):, :]


def prep(image: np.ndarray, upscale: float) -> np.ndarray:
    """Upscale only. Colour is kept: RapidOCR reads BGR best (grayscale made it drop word spaces in testing)."""
    if upscale and upscale != 1.0:
        return cv2.resize(image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return image


class RapidOCRExtractor:
    """RapidOCR (onnxruntime) wrapper. Engine loads lazily on first read (~0.5 s), models cached on disk."""

    def __init__(self, min_score: float = 0.2) -> None:
        self.min_score = min_score
        self._engine = None

    def _get(self):
        if self._engine is None:
            import logging
            logging.getLogger("RapidOCR").setLevel(logging.WARNING)
            from rapidocr import RapidOCR
            self._engine = RapidOCR()
        return self._engine

    def read(self, image: np.ndarray) -> str:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        out = self._get()(image)
        if out is None or not out.txts:
            return ""
        # out.boxes: (n, 4, 2) points; sort rows top-to-bottom, then left-to-right
        items = list(zip(out.boxes, out.txts, out.scores))
        items.sort(key=lambda r: (round(float(r[0][0][1]) / 20), float(r[0][0][0])))
        return " ".join(t for _, t, s in items if float(s) >= self.min_score)


def read_dialogue(extractor: TextExtractor, frame: np.ndarray, cfg: Config) -> str:
    band_text = extractor.read(prep(crop_band(frame, cfg.band_fraction), cfg.ocr_upscale))
    if normalize(band_text):
        return band_text
    return extractor.read(prep(frame, 1.0))
