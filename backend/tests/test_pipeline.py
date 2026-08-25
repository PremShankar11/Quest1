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
    res = run(str(path), truth["text"], cfg=cfg, mode="hybrid", local=True,
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
    run(str(path), truth["text"], mode="hybrid", local=True, reporter=Collect(), extractor=NoMatch(), locator=FakeLocator())
    fb = [e for e in events if e.stage == "refine" and e.status == "fallback"][0]
    assert fb.payload["frame_index"] == 120


def test_default_paths_are_repo_anchored():
    from dialogue_finder.config import DEFAULT, REPO_ROOT
    assert DEFAULT.cache_dir == REPO_ROOT / "cache" and DEFAULT.cache_dir.is_absolute()
    assert (REPO_ROOT / "backend").is_dir()
