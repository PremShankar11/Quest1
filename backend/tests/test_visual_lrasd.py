import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dialogue_finder.config import DEFAULT, REPO_ROOT
from dialogue_finder.visual.lrasd import (
    LrAsdDetector,
    SpeakerDetectorUnavailable,
    WeightsVerificationError,
    _verify_weights,
    asd_available,
)


def test_asd_available_false_without_torch(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "torch":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    ok, reason = asd_available()
    assert ok is False
    assert "requirements-asd.txt" in reason


def test_asd_available_true_with_torch_present():
    ok, reason = asd_available()
    assert ok is True
    assert reason == "" or "weights missing" in reason


def test_verify_weights_rejects_wrong_content(tmp_path):
    bad = tmp_path / "finetuning_TalkSet.model"
    bad.write_bytes(b"not the real weights")
    with pytest.raises(WeightsVerificationError):
        _verify_weights(bad, "finetuning_TalkSet.model")
    assert not bad.exists()  # deleted on mismatch


def test_verify_weights_accepts_unpinned_names(tmp_path):
    custom = tmp_path / "custom.model"
    custom.write_bytes(b"anything")
    _verify_weights(custom, "custom.model")  # no pinned hash -- passes through
    assert custom.exists()


def test_score_converts_logits_to_softmax_probabilities(monkeypatch):
    n = 4
    fixed_logits = np.array(
        [[1.0, 2.0], [0.0, 0.0], [-1.0, 1.0], [5.0, -5.0]], dtype=np.float32
    )

    det = LrAsdDetector(models_dir=Path("unused"))
    monkeypatch.setattr(det, "_load", lambda: None)
    monkeypatch.setattr(det, "_forward", lambda audio, video: fixed_logits)

    crops = np.zeros((n, 112, 112), dtype=np.uint8)
    mfcc = np.zeros((n * 4, 13), dtype=np.float32)
    probs = det.score(crops, mfcc)

    assert len(probs) == n
    assert all(0.0 <= p <= 1.0 for p in probs)

    m = fixed_logits.max(axis=-1, keepdims=True)
    exp = np.exp(fixed_logits - m)
    expected = (exp / exp.sum(axis=-1, keepdims=True))[:, 1]
    np.testing.assert_allclose(probs, expected, atol=1e-6)


def test_score_trims_to_shorter_of_video_and_audio(monkeypatch):
    calls = {}

    def fake_forward(audio, video):
        calls["audio_shape"] = tuple(audio.shape)
        calls["video_shape"] = tuple(video.shape)
        n = video.shape[1]
        return np.zeros((n, 2), dtype=np.float32)

    det = LrAsdDetector(models_dir=Path("unused"))
    monkeypatch.setattr(det, "_load", lambda: None)
    monkeypatch.setattr(det, "_forward", fake_forward)

    crops = np.zeros((10, 112, 112), dtype=np.uint8)     # 10 video frames
    mfcc = np.zeros((37, 13), dtype=np.float32)           # only 9 usable video frames (37 // 4)
    probs = det.score(crops, mfcc)

    assert len(probs) == 9
    assert calls["video_shape"] == (1, 9, 112, 112)
    assert calls["audio_shape"] == (1, 36, 13)


# ---- I-4: missing weights + offline -> SpeakerDetectorUnavailable, cached ------------------

def test_load_wraps_weights_download_failure(monkeypatch, tmp_path):
    import dialogue_finder.visual.lrasd as lrasd_mod

    def fake_fetch_verified(*a, **k):
        raise IOError("offline: could not reach raw.githubusercontent.com")

    monkeypatch.setattr(lrasd_mod, "fetch_verified", fake_fetch_verified)
    det = LrAsdDetector(models_dir=tmp_path / "models")
    with pytest.raises(SpeakerDetectorUnavailable):
        det._load()


def test_load_failure_is_cached_and_not_retried(monkeypatch, tmp_path):
    """A second `_load()` call after a download failure must re-raise the cached error, not
    hit the network again (no repeated downloads/timeouts, I-4)."""
    import dialogue_finder.visual.lrasd as lrasd_mod

    calls = {"n": 0}

    def fake_fetch_verified(*a, **k):
        calls["n"] += 1
        raise IOError("offline: could not reach raw.githubusercontent.com")

    monkeypatch.setattr(lrasd_mod, "fetch_verified", fake_fetch_verified)
    det = LrAsdDetector(models_dir=tmp_path / "models")

    with pytest.raises(SpeakerDetectorUnavailable):
        det._load()
    with pytest.raises(SpeakerDetectorUnavailable):
        det._load()

    assert calls["n"] == 1


@pytest.mark.slow
def test_lrasd_scores_holmes_window():
    from dialogue_finder.video.frame_source import FrameSource
    from dialogue_finder.visual.faces import YuNetDetector, build_tracks, crop_face
    from dialogue_finder.visual.audio_features import mfcc_for_video

    available, reason = asd_available()
    if not available:
        pytest.skip(f"LR-ASD not available: {reason}")

    video = REPO_ROOT / "cache" / "5f39d4605665a831.mp4"
    wav = REPO_ROOT / "cache" / "5f39d4605665a831.16k.wav"
    if not video.exists() or not wav.exists():
        pytest.skip("episode video/audio not cached")

    detector = YuNetDetector(DEFAULT.models_dir)
    with FrameSource(video) as src:
        tracks = build_tracks(src, detector, 7700, 7950, DEFAULT)
        assert tracks, "expected at least one face track over frames 7700-7950"
        track = max(tracks, key=lambda t: len(t.frames))

        crops = np.stack(
            [crop_face(src.frame_at(f), b, 112) for f, b in zip(track.frames, track.boxes)]
        )
        fps = src.fps
        start_s = track.frames[0] / fps
        end_s = (track.frames[-1] + 1) / fps

    mfcc = mfcc_for_video(wav, start_s, end_s, fps)

    lrasd = LrAsdDetector(DEFAULT.models_dir)
    probs = lrasd.score(crops, mfcc)

    frames = track.frames[: len(probs)]
    speak = [p for f, p in zip(frames, probs) if 7794 <= f <= 7900]
    quiet = [p for f, p in zip(frames, probs) if 7700 <= f <= 7780]

    assert speak, "no scored frames in the speaking span"
    assert quiet, "no scored frames in the quiet span"
    print(f"\nmean speaking (7794-7900, n={len(speak)}): {np.mean(speak):.3f}")
    print(f"mean quiet    (7700-7780, n={len(quiet)}): {np.mean(quiet):.3f}")
    assert np.mean(speak) >= 0.5
    assert np.mean(quiet) < 0.3
