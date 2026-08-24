import pytest

from dialogue_finder.config import DEFAULT
from dialogue_finder.models import Window
from dialogue_finder.pipeline import confidence_for, run


def test_confidence_rules():
    assert confidence_for("ocr", 0.95, Window(1, 2, 0.9, "x")) == "HIGH"
    assert confidence_for("ocr", 0.95, None) == "HIGH"
    assert confidence_for("ocr", 0.82, None) == "MEDIUM"
    assert confidence_for("audio", 0.0, Window(1, 2, 0.9, "x")) == "MEDIUM"
    assert confidence_for("ocr-weak", 0.4, None) == "LOW"


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
