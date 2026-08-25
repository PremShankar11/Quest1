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


class FakeConstantOcrExtractor:
    """TextExtractor that always reads the same (OCR-cased) text, regardless of frame content."""
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self, image) -> str:
        return self._text


def test_hybrid_valid_text_reports_ocr_text_not_asr_text(synthetic_clip, tmp_path, monkeypatch):
    """Fix round 1: `valid-text` occurrences must report the OCR-extracted text (their own
    evidence), not the ASR window's `matched_text` -- every other route already reports the text
    of its own evidence (audio -> matched_text, audio+asd -> would be visual, no text)."""
    _patch_hybrid_extras(monkeypatch, FakeBoxDetector(), FakeCountingSpeaker([0.0]))
    path, truth = synthetic_clip
    ocr_text = "MY MIND REBELS AT STAGNATION"
    window = Window(5.0, 5.5, 0.95, truth["text"])
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out_h3", cache_dir=tmp_path / "cache_h3")
    res = run(str(path), truth["text"], cfg=cfg, mode="hybrid", local=True,
             extractor=FakeConstantOcrExtractor(ocr_text), locator=FakeLocatorAll([window]))

    assert res.occurrence_class == "valid-text"
    assert res.text == ocr_text
    assert res.text != window.matched_text


def test_audio_ocr_locate_payload_has_no_windows_key(synthetic_clip, tmp_path):
    """Fix round 1: the `windows` payload key must only appear for mode="hybrid" so the old
    modes' `locate ok` events stay byte-identical."""
    path, truth = synthetic_clip
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out_locate", cache_dir=tmp_path / "cache_locate")
    reporter = CollectingReporter()
    run(str(path), truth["text"], cfg=cfg, mode="audio+ocr", local=True,
       extractor=FakeNoMatch(), locator=FakeLocator(), reporter=reporter)
    locate_ok = [e for e in reporter.events if e.stage == "locate" and e.status == "ok"]
    assert locate_ok and set(locate_ok[0].payload.keys()) == {"window"}


# ---- hybrid OCR retry: parity with audio+ocr's widened-window retry (fix round 2) -----------

class FakeEarlyFrameExtractor:
    """TextExtractor that 'matches' only on bright frames -- pairs with `_make_flip_clip`, whose
    frames are bright for index < `flip_at` and dark afterwards, so this fakes 'text visible
    only in early frames' without needing real OCR content."""
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self, image) -> str:
        return self._text if image.mean() > 128 else ""


def _make_flip_clip(path, duration_s: float = 15.0, fps: int = 24, flip_at: int = 100,
                    size: tuple[int, int] = (640, 360)) -> None:
    """A plain video, bright for frame index < `flip_at` and dark afterwards -- content a fake
    extractor (`FakeEarlyFrameExtractor`) can key off without real OCR."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open output")
    total = int(round(duration_s * fps))
    for i in range(total):
        color = 220 if i < flip_at else 20
        writer.write(np.full((h, w, 3), color, dtype=np.uint8))
    writer.release()


def test_hybrid_ocr_uses_widened_retry_like_audio_ocr(tmp_path, monkeypatch):
    """Fix round 2 (regression from validation): a subtitle just outside the padded ASD window
    (window ± `window_pad_s` = 3s) but inside the widened retry range (window ± `retry_pad_s` =
    15s) must still be found and classified `valid-text` -- `audio+ocr` already finds such a hit
    via its widened-window retry; the hybrid verify stage's `_ocr_occurrence` used to only ever
    scan the padded window, so this occurrence used to fall through to the visual stage and come
    back `invalid`/`uncertain` instead (Squid Game clip: subtitle at 9.9s, ASR window
    24.6-29.6s). Here the fake locator's window (8-9s) padded window is [5, 12]s -- entirely
    dark (frame_index >= 100) -- so the first OCR scan must miss; the widened retry range is
    [0, 15]s (clamped) -- bright from frame 0, so the retry must hit at frame 0."""
    _patch_hybrid_extras(monkeypatch, FakeBoxDetector(), FakeCountingSpeaker([0.0]))
    clip = tmp_path / "flip.mp4"
    _make_flip_clip(clip)
    target = "My mind rebels at stagnation"
    window = Window(8.0, 9.0, 0.63, target)
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out_h5", cache_dir=tmp_path / "cache_h5")
    reporter = CollectingReporter()
    res = run(str(clip), target, cfg=cfg, mode="hybrid", local=True,
             extractor=FakeEarlyFrameExtractor(target), locator=FakeLocatorAll([window]), reporter=reporter)

    assert res.occurrence_class == "valid-text"
    assert res.source == "ocr"
    assert res.frame_index == 0
    assert res.text == target
    assert any(e.stage == "scan" and e.status == "fallback" and "retrying" in e.message
              for e in reporter.events)


def test_hybrid_ocr_retry_not_run_when_first_scan_hits(synthetic_clip, tmp_path, monkeypatch):
    """Fix round 2: the widened retry must NOT run when the padded-window OCR scan already hit
    (no wasted rescan, no spurious `scan fallback` event)."""
    _patch_hybrid_extras(monkeypatch, FakeBoxDetector(), FakeCountingSpeaker([0.0]))
    path, truth = synthetic_clip
    window = Window(5.0, 5.5, 0.95, truth["text"])
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out_h6", cache_dir=tmp_path / "cache_h6")
    reporter = CollectingReporter()
    res = run(str(path), truth["text"], cfg=cfg, mode="hybrid", local=True,
             extractor=FakeConstantOcrExtractor("MY MIND REBELS AT STAGNATION"),
             locator=FakeLocatorAll([window]), reporter=reporter)

    assert res.occurrence_class == "valid-text"
    assert not any(e.stage == "scan" and e.status == "fallback" for e in reporter.events)
