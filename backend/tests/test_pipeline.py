import cv2
import numpy as np
import pytest

from dialogue_finder.config import DEFAULT
from dialogue_finder.models import Window
from dialogue_finder.pipeline import PipelineError, _write_png, confidence_for, run


def test_confidence_rules():
    assert confidence_for("ocr", 0.95, Window(1, 2, 0.9, "x")) == "HIGH"
    assert confidence_for("ocr", 0.95, None) == "HIGH"
    assert confidence_for("ocr", 0.82, None) == "MEDIUM"
    assert confidence_for("audio", 0.0, Window(1, 2, 0.9, "x")) == "MEDIUM"
    assert confidence_for("ocr-weak", 0.4, None) == "LOW"


def test_write_png_handles_non_ascii_path(tmp_path):
    frame = np.full((10, 12, 3), 200, dtype=np.uint8)
    d = tmp_path / "José"
    d.mkdir()
    p = d / "f.png"
    _write_png(p, frame)
    assert p.exists()
    decoded = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == frame.shape


def test_write_png_unwritable_parent_raises(tmp_path):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    bad_parent = tmp_path / "not_a_dir"
    bad_parent.write_text("x")   # a file, not a directory
    with pytest.raises(PipelineError):
        _write_png(bad_parent / "f.png", frame)


@pytest.mark.slow
def test_ground_truth_exact_frame(synthetic_clip, tmp_path):
    path, truth = synthetic_clip
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    res = run(str(path), truth["text"], cfg=cfg, mode="ocr", local=True)
    assert res.source == "ocr"
    assert res.frame_index == truth["frame"], (res.frame_index, truth["frame"], res.candidates[-3:])
    assert abs(res.timestamp_s - truth["timestamp"]) < 1e-6
    assert (tmp_path / "out").joinpath(f"frame_{truth['frame']}.png").exists()
    assert (tmp_path / "out").joinpath(f"frame_{truth['frame'] - 1}.png").exists()
    assert res.appearance == "pop-in"


@pytest.mark.slow
def test_ground_truth_fade_in(tmp_path):
    from bench.make_clip import make_clip
    clip = tmp_path / "fade.mp4"
    truth = make_clip(clip, text="My mind rebels at stagnation", appear_s=4.0, fade_frames=12)
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    res = run(str(clip), truth["text"], cfg=cfg, mode="ocr", local=True)
    assert res.source == "ocr"
    assert 0 <= res.frame_index - truth["frame"] <= 12     # within the fade
    assert res.appearance == "fade-in"


def test_bad_local_path_raises_pipeline_error(tmp_path):
    from dialogue_finder.pipeline import PipelineError
    with pytest.raises(PipelineError):
        run(str(tmp_path / "missing.mp4"), "x", mode="ocr", local=True)


class FakeNoMatch:
    """TextExtractor that never finds on-screen text, forcing every OCR scan to miss."""
    def read(self, image) -> str:
        return ""


class FakeLocator:
    """Locator that always returns a fixed window at 5 s."""
    def locate(self, video, target):
        return Window(5.0, 5.5, 0.95, target)


class CollectingReporter:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class FakeNoLocate:
    """Locator that never finds an audio match."""
    def locate(self, video, target):
        return None


def test_audio_mode_uses_locator(synthetic_clip, tmp_path):
    path, truth = synthetic_clip
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out_audio", cache_dir=tmp_path / "cache_audio")
    res = run(str(path), truth["text"], cfg=cfg, mode="audio", local=True, locator=FakeLocator())
    assert res.source == "audio"
    assert res.frame_index == 120


def test_audio_mode_no_match_raises_pipeline_error(synthetic_clip, tmp_path):
    from dialogue_finder.pipeline import PipelineError
    path, truth = synthetic_clip
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out_audio2", cache_dir=tmp_path / "cache_audio2")
    with pytest.raises(PipelineError):
        run(str(path), truth["text"], cfg=cfg, mode="audio", local=True, locator=FakeNoLocate())


def test_ocr_weak_fallback_text_is_explicit(synthetic_clip, tmp_path, monkeypatch):
    """When the OCR scan produces zero candidates at all (e.g. a degenerate scan range),
    the weak-fallback text must be an explicit placeholder, not an empty string."""
    import dialogue_finder.pipeline as pipeline_mod
    path, truth = synthetic_clip
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out3", cache_dir=tmp_path / "cache3")
    monkeypatch.setattr(pipeline_mod, "_scan_for_groups", lambda *a, **k: ([], []))
    res = run(str(path), truth["text"], cfg=cfg, mode="ocr", local=True, extractor=FakeNoMatch())
    assert res.source == "ocr-weak"
    assert res.text == "(no text detected)"
    assert res.note == "no text detected anywhere; frame 0 returned"


def test_hybrid_retries_widened_window_not_whole_video(synthetic_clip, tmp_path):
    path, truth = synthetic_clip
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out2", cache_dir=tmp_path / "cache2")
    reporter = CollectingReporter()
    window = Window(5.0, 5.5, 0.95, truth["text"])
    res = run(str(path), truth["text"], cfg=cfg, mode="audio+ocr", local=True,
             extractor=FakeNoMatch(), locator=FakeLocator(), reporter=reporter)
    assert res.source == "audio"
    assert res.frame_index == 120
    assert res.text == window.matched_text
    assert not any("whole video" in e.message for e in reporter.events)


def test_download_ok_event_carries_video_facts(synthetic_clip):
    path, truth = synthetic_clip
    events = []
    class Collect:
        def emit(self, e): events.append(e)
    class NoMatch:
        def read(self, image): return ""
    run(str(path), truth["text"], mode="ocr", local=True, reporter=Collect(), extractor=NoMatch())
    ok = [e for e in events if e.stage == "download" and e.status == "ok"][0]
    assert ok.payload["fps"] == 24 and ok.payload["duration_s"] == 15.0 and ok.payload["frame_count"] == 360


def test_cancel_raises_pipeline_error(synthetic_clip):
    from dialogue_finder.pipeline import PipelineError
    path, truth = synthetic_clip
    class NoMatch:
        def read(self, image): return ""
    with pytest.raises(PipelineError, match="cancelled"):
        run(str(path), truth["text"], mode="ocr", local=True, extractor=NoMatch(), should_cancel=lambda: True)


def test_run_wraps_unexpected_errors_as_pipeline_error(synthetic_clip):
    from dialogue_finder.pipeline import PipelineError
    path, truth = synthetic_clip
    class Boom:
        def read(self, image): raise ValueError("engine exploded")
    with pytest.raises(PipelineError, match="engine exploded"):
        run(str(path), truth["text"], mode="ocr", local=True, extractor=Boom())


def test_refine_fallback_events_carry_frame_index(synthetic_clip):
    from dialogue_finder.models import Window
    path, truth = synthetic_clip
    events = []
    class Collect:
        def emit(self, e): events.append(e)
    class NoMatch:
        def read(self, image): return ""
    class FakeLocator:
        def locate(self, video, target): return Window(5.0, 6.0, 0.9, "hi")
    run(str(path), truth["text"], mode="audio+ocr", local=True, reporter=Collect(), extractor=NoMatch(), locator=FakeLocator())
    fb = [e for e in events if e.stage == "refine" and e.status == "fallback"][0]
    assert fb.payload["frame_index"] == 120


def test_default_paths_are_repo_anchored():
    from dialogue_finder.config import DEFAULT, REPO_ROOT
    assert DEFAULT.cache_dir == REPO_ROOT / "cache" and DEFAULT.cache_dir.is_absolute()
    assert (REPO_ROOT / "backend").is_dir()


def test_hybrid_without_extras_matches_audio_ocr(synthetic_clip, tmp_path, monkeypatch):
    """Without the ASD extras, mode="hybrid" must degrade to exactly the audio+ocr answer and
    emit a `verify: skipped` event. `_run_hybrid` (Task 6) is what makes this true; for now
    mode="hybrid" just falls through the audio+ocr path with no verify stage at all."""
    import dialogue_finder.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "asd_available", lambda: (False, "requirements-asd.txt not installed"))
    path, truth = synthetic_clip
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out_hybrid", cache_dir=tmp_path / "cache_hybrid")
    reporter = CollectingReporter()
    res = run(str(path), truth["text"], cfg=cfg, mode="hybrid", local=True,
             extractor=FakeNoMatch(), locator=FakeLocator(), reporter=reporter)
    assert res.source == "audio"
    assert res.frame_index == 120
    assert any(e.stage == "verify" and e.status == "skipped" for e in reporter.events)


# ---- hybrid mode: fake faces + fake speaker + fake locate_all -------------------------------

class FakeLocatorAll:
    """Locator whose locate_all returns a fixed list of windows (locate() returns the first)."""
    def __init__(self, windows):
        self._windows = windows

    def locate(self, video, target):
        return self._windows[0] if self._windows else None

    def locate_all(self, video, target):
        return self._windows


class FakeBoxDetector:
    """FaceDetector giving one constant 80 px box on every frame."""
    def detect(self, frame):
        return [(50, 50, 80, 80)]


class FakeNoBoxDetector:
    """FaceDetector that never finds a face."""
    def detect(self, frame):
        return []


class FakeCountingSpeaker:
    """SpeakerDetector returning a fixed score per call (one call per window's single track,
    in window order), regardless of the crops/mfcc content."""
    def __init__(self, scores):
        self._scores = list(scores)
        self._i = 0

    def score(self, crops, mfcc):
        val = self._scores[min(self._i, len(self._scores) - 1)]
        self._i += 1
        return [val] * len(crops)


def _fake_speech_mask(wav, start_s, end_s, fps):
    return [True] * round((end_s - start_s) * fps)


def _fake_mfcc_for_video(wav, start_s, end_s, fps):
    n = round((end_s - start_s) * fps)
    return np.zeros((4 * n, 13), dtype=np.float32)


def _fake_extract_audio(video, wav):
    return wav


def _patch_hybrid_extras(monkeypatch, detector, speaker):
    import dialogue_finder.pipeline as pipeline_mod
    import dialogue_finder.visual.verifier as verifier_mod
    monkeypatch.setattr(pipeline_mod, "asd_available", lambda: (True, ""))
    monkeypatch.setattr(pipeline_mod, "YuNetDetector", lambda models_dir: detector)
    monkeypatch.setattr(pipeline_mod, "LrAsdDetector", lambda models_dir: speaker)
    monkeypatch.setattr(pipeline_mod, "extract_audio", _fake_extract_audio)
    monkeypatch.setattr(verifier_mod, "speech_mask", _fake_speech_mask)
    monkeypatch.setattr(verifier_mod, "mfcc_for_video", _fake_mfcc_for_video)


def test_hybrid_selects_valid_speaker_over_invalid(synthetic_clip, tmp_path, monkeypatch):
    _patch_hybrid_extras(monkeypatch, FakeBoxDetector(), FakeCountingSpeaker([0.1, 0.9]))
    path, truth = synthetic_clip
    windows = [Window(2.0, 2.5, 0.7, "line a"), Window(8.0, 8.5, 0.65, "line b")]
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out_h1", cache_dir=tmp_path / "cache_h1")
    reporter = CollectingReporter()
    res = run(str(path), truth["text"], cfg=cfg, mode="hybrid", local=True,
             extractor=FakeNoMatch(), locator=FakeLocatorAll(windows), reporter=reporter)

    assert res.occurrence_class == "valid-speaker"
    assert res.source == "audio+asd"
    assert res.speaker_box is not None
    lo = round((8.0 - DEFAULT.onset_lookback_s) * 24)
    hi = round(8.5 * 24)
    assert lo <= res.frame_index <= hi, (res.frame_index, lo, hi)

    occ_events = [e for e in reporter.events if e.stage == "occurrences" and e.status == "ok"]
    assert len(occ_events) == 1
    classes = [o["klass"] for o in occ_events[0].payload["occurrences"]]
    assert classes == ["invalid", "valid-speaker"]


def test_hybrid_uncertain_when_no_faces_detected(synthetic_clip, tmp_path, monkeypatch):
    _patch_hybrid_extras(monkeypatch, FakeNoBoxDetector(), FakeCountingSpeaker([0.0, 0.0]))
    path, truth = synthetic_clip
    windows = [Window(2.0, 2.5, 0.7, "line a"), Window(8.0, 8.5, 0.95, "line b")]
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out_h2", cache_dir=tmp_path / "cache_h2")
    reporter = CollectingReporter()
    res = run(str(path), truth["text"], cfg=cfg, mode="hybrid", local=True,
             extractor=FakeNoMatch(), locator=FakeLocatorAll(windows), reporter=reporter)

    assert res.occurrence_class == "uncertain"
    assert res.frame_index == round(8.0 * 24)   # first spoken word of the higher-ASR (0.95) window
    assert res.speaker_box is None
