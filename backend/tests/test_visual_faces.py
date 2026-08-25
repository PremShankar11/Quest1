import hashlib

import numpy as np
import pytest

from dialogue_finder.config import DEFAULT, REPO_ROOT
from dialogue_finder.visual.faces import IouTracker, _verify_yunet_hash, build_tracks, crop_face


class FakeSrc:
    fps = 24.0
    frame_count = 100

    def iter_range(self, a, b, step):
        for i in range(a, b + 1, step):
            yield i, i

    def frame_at(self, i):
        return i


class FakeDet:
    def __init__(self, boxes_at):
        self.boxes_at = boxes_at

    def detect(self, frame):
        return self.boxes_at(frame)


def test_tracker_links_overlapping_boxes_and_tolerates_gaps():
    t = IouTracker(iou_threshold=0.5, max_gap=3)
    for i in range(10):
        t.update(i, [] if i == 4 else [(100 + i, 100, 60, 60)])
    tracks = t.tracks()
    assert len(tracks) == 1 and tracks[0].frames[0] == 0 and tracks[0].frames[-1] == 9


def test_tracker_splits_on_long_gap_and_far_boxes():
    t = IouTracker(iou_threshold=0.5, max_gap=3)
    for i in range(0, 5):
        t.update(i, [(100, 100, 60, 60)])
    for i in range(10, 15):
        t.update(i, [(400, 100, 60, 60)])
    assert len(t.tracks()) == 2


def test_build_tracks_filters_short_and_small():
    det = FakeDet(lambda i: [(10, 10, 80, 80)] + ([(300, 300, 20, 20)] if i < 30 else []))
    tracks = build_tracks(FakeSrc(), det, 0, 47, DEFAULT)   # 2 s at 24 fps
    assert len(tracks) == 1 and tracks[0].median_height() == 80


def test_crop_face_is_square_grey_sized():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    c = crop_face(frame, (300, 100, 50, 70), 112)
    assert c.shape == (112, 112) and c.dtype == np.uint8


def test_verify_yunet_hash_rejects_wrong_bytes_and_deletes_file(tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not the real model")
    with pytest.raises(RuntimeError, match="integrity check failed"):
        _verify_yunet_hash(bad)
    assert not bad.exists()


def test_verify_yunet_hash_accepts_matching_bytes(tmp_path):
    good = tmp_path / "good.onnx"
    good.write_bytes(b"known content for hash check")
    expected = hashlib.sha256(good.read_bytes()).hexdigest()
    _verify_yunet_hash(good, expected=expected)   # no raise
    assert good.exists()


@pytest.mark.slow
def test_yunet_detects_face_on_cached_episode_frame():
    from dialogue_finder.video.frame_source import FrameSource
    from dialogue_finder.visual.faces import YuNetDetector

    video = REPO_ROOT / "cache" / "5f39d4605665a831.mp4"
    if not video.exists():
        pytest.skip(f"episode video not cached: {video}")

    detector = YuNetDetector(DEFAULT.models_dir)
    with FrameSource(video) as src:
        frame = src.frame_at(7794)
    boxes = detector.detect(frame)
    assert len(boxes) >= 1
    assert max(b[3] for b in boxes) >= 40
