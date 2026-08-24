import numpy as np

from dialogue_finder.video.frame_source import FrameSource


def test_probe_values(synthetic_clip):
    path, truth = synthetic_clip
    with FrameSource(path) as src:
        assert abs(src.fps - 24) < 0.01
        assert src.frame_count == 24 * 15
        assert src.width == 640 and src.height == 360


def test_index_time_roundtrip(synthetic_clip):
    path, _ = synthetic_clip
    with FrameSource(path) as src:
        assert src.index_for_time(5.0) == 120
        assert abs(src.time_for_index(120) - 5.0) < 1e-9


def test_frame_before_is_blank_and_at_has_text(synthetic_clip):
    path, truth = synthetic_clip
    n = truth["frame"]
    with FrameSource(path) as src:
        before = src.frame_at(n - 1)
        at = src.frame_at(n)
    band_before = before[int(360 * 0.65):, :, :]
    band_at = at[int(360 * 0.65):, :, :]
    assert (band_before > 200).sum() == 0
    assert (band_at > 200).sum() > 500


def test_frame_at_is_exact_after_random_access(synthetic_clip):
    path, truth = synthetic_clip
    n = truth["frame"]
    with FrameSource(path) as src:
        src.frame_at(300)
        src.frame_at(10)
        at = src.frame_at(n)
        before = src.frame_at(n - 1)
    assert (at[int(360 * 0.65):] > 200).sum() > 500
    assert (before[int(360 * 0.65):] > 200).sum() == 0


def test_iter_range_yields_every_step(synthetic_clip):
    path, _ = synthetic_clip
    with FrameSource(path) as src:
        idx = [i for i, _ in src.iter_range(100, 130, 5)]
    assert idx == [100, 105, 110, 115, 120, 125, 130]
