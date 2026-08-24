import numpy as np
import pytest

from dialogue_finder.config import DEFAULT
from dialogue_finder.video.frame_source import FrameSource
from dialogue_finder.text.matcher import score_contains
from dialogue_finder.text.ocr import crop_band, prep, read_dialogue


def test_crop_band_takes_bottom_fraction():
    img = np.zeros((100, 50, 3), dtype=np.uint8)
    band = crop_band(img, 0.35)
    assert band.shape[0] == 35 and band.shape[1] == 50


def test_prep_upscales_to_gray():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    out = prep(img, 2.0)
    assert out.shape == (20, 40)


@pytest.mark.slow
def test_rapidocr_reads_synthetic_subtitle(synthetic_clip):
    from dialogue_finder.text.ocr import RapidOCRExtractor
    path, truth = synthetic_clip
    with FrameSource(path) as src:
        frame = src.frame_at(truth["frame"] + 5)
        blank = src.frame_at(truth["frame"] - 5)
    ex = RapidOCRExtractor()
    text = read_dialogue(ex, frame, DEFAULT)
    assert score_contains(truth["text"], text) >= 0.8, text
    assert score_contains(truth["text"], read_dialogue(ex, blank, DEFAULT)) < 0.5
