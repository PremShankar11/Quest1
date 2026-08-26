import hashlib
from pathlib import Path

import numpy as np
import pytest

from dialogue_finder.config import DEFAULT, REPO_ROOT
from dialogue_finder.visual.faces import (
    FaceDetectorUnavailable,
    IouTracker,
    YuNetDetector,
    _verify_yunet_hash,
    build_tracks,
    crop_face,
)


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


# ---- I-4: missing model file + offline -> FaceDetectorUnavailable, cached ------------------

def test_ensure_caches_load_failure_and_does_not_retry(monkeypatch, tmp_path):
    """A second `_ensure()` call after a download failure must re-raise the cached error, not
    hit the network again (no repeated downloads/timeouts, I-4)."""
    import dialogue_finder.visual.faces as faces_mod

    calls = {"n": 0}

    def fake_download_yunet(dest):
        calls["n"] += 1
        raise IOError("offline: could not reach media.githubusercontent.com")

    monkeypatch.setattr(faces_mod, "_download_yunet", fake_download_yunet)
    det = YuNetDetector(models_dir=tmp_path / "models")

    with pytest.raises(FaceDetectorUnavailable):
        det._ensure(100, 100)
    with pytest.raises(FaceDetectorUnavailable):
        det._ensure(100, 100)

    assert calls["n"] == 1


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


# ---- Minor: YuNet box clipping ---------------------------------------------------------------

class _FakeCvDetector:
    def __init__(self, raw):
        self._raw = raw

    def detect(self, frame):
        return None, self._raw


def test_detect_clips_negative_origin_to_zero(monkeypatch):
    """A raw detection with x<0 or y<0 (demo repro: `218,-15,304,410`) must be clamped so the
    reported box never starts off-frame."""
    det = YuNetDetector(models_dir=Path("unused"))
    monkeypatch.setattr(det, "_ensure", lambda w, h: None)
    det._detector = _FakeCvDetector(np.array([[218.0, -15.0, 304.0, 410.0]], dtype=np.float32))
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    assert det.detect(frame) == [(218, 0, 304, 395)]


def test_detect_clips_box_extending_past_frame_edges():
    det = YuNetDetector(models_dir=Path("unused"))
    det._size = (500, 300)   # skip _ensure entirely: _size already matches, no reload needed
    det._detector = _FakeCvDetector(np.array([[450.0, 250.0, 200.0, 200.0]], dtype=np.float32))
    frame = np.zeros((300, 500, 3), dtype=np.uint8)   # H=300, W=500
    assert det.detect(frame) == [(450, 250, 50, 50)]
