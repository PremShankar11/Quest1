import pytest

from dialogue_finder.config import DEFAULT
from dialogue_finder.models import FaceTrack
from dialogue_finder.visual.verifier import _fill_gaps, _median_smooth_boxes, classify, find_onset


# ---- classify -------------------------------------------------------------

def test_classify_ocr_hit_is_valid_text():
    klass, track, mean = classify(0.85, [], [], DEFAULT)
    assert (klass, track, mean) == ("valid-text", None, 0.0)


def test_classify_track_active_on_enough_speech_frames_is_valid_speaker():
    # 10 speech frames; track scores 0.8 on the first 5 (50% >= asd_min_active 30%).
    speech = [True] * 10
    track = FaceTrack(track_id=0, frames=list(range(10)), boxes=[(0, 0, 80, 80)] * 10,
                      scores=[0.8] * 5 + [0.0] * 5)
    klass, chosen, mean = classify(0.0, [track], speech, DEFAULT)
    assert klass == "valid-speaker"
    assert chosen is track
    assert mean == pytest.approx(0.4)   # mean over ALL speech frames the track covers


def test_classify_tracks_present_but_below_threshold_is_invalid():
    speech = [True] * 5
    track = FaceTrack(track_id=0, frames=list(range(5)), boxes=[(0, 0, 80, 80)] * 5,
                      scores=[0.3] * 5)   # below asd_threshold (0.5): never "active"
    klass, chosen, mean = classify(0.0, [track], speech, DEFAULT)
    assert (klass, chosen, mean) == ("invalid", None, 0.0)


def test_classify_no_tracks_is_uncertain():
    klass, chosen, mean = classify(0.0, [], [True] * 5, DEFAULT)
    assert (klass, chosen, mean) == ("uncertain", None, 0.0)


def test_classify_picks_higher_mean_track_among_qualifiers():
    speech = [True] * 6
    low = FaceTrack(track_id=0, frames=list(range(6)), boxes=[(0, 0, 80, 80)] * 6,
                    scores=[0.5] * 4 + [0.5] * 2)      # active 100%, mean 0.5
    high = FaceTrack(track_id=1, frames=list(range(6)), boxes=[(0, 0, 80, 80)] * 6,
                     scores=[0.9] * 4 + [0.0] * 2)     # active 4/6 = 67% >= 30%, mean 0.6
    klass, chosen, mean = classify(0.0, [low, high], speech, DEFAULT)
    assert klass == "valid-speaker"
    assert chosen is high
    assert mean == pytest.approx(0.6)


# ---- find_onset -------------------------------------------------------------

def test_find_onset_finds_first_qualifying_run():
    # absolute frames 100..105; asd_threshold 0.5, asd_onset_frames 3 (defaults).
    frames = list(range(100, 106))
    scores = [0.1, 0.2, 0.6, 0.7, 0.8, 0.9]
    track = FaceTrack(track_id=0, frames=frames, boxes=[(0, 0, 80, 80)] * 6, scores=scores)
    speech = [True] * 6
    onset = find_onset(track, speech, first_index=100, cfg=DEFAULT)
    assert onset == 102     # first frame of the first 3-consecutive->=0.5 run (0.6, 0.7, 0.8)


def test_find_onset_skips_run_with_no_speech_to_next_qualifying():
    frames = list(range(100, 106))
    scores = [0.1, 0.2, 0.6, 0.7, 0.8, 0.9]
    track = FaceTrack(track_id=0, frames=frames, boxes=[(0, 0, 80, 80)] * 6, scores=scores)
    speech = [True, True, False, True, True, True]   # speech False at the 0.6 frame
    onset = find_onset(track, speech, first_index=100, cfg=DEFAULT)
    assert onset == 103


def test_find_onset_returns_none_when_no_run_qualifies():
    frames = list(range(100, 106))
    track = FaceTrack(track_id=0, frames=frames, boxes=[(0, 0, 80, 80)] * 6, scores=[0.1] * 6)
    speech = [True] * 6
    assert find_onset(track, speech, first_index=100, cfg=DEFAULT) is None


# ---- helpers: median smoothing / gap filling --------------------------------

def test_median_smooth_boxes_removes_single_frame_jitter():
    boxes = [(100, 100, 80, 80)] * 6 + [(400, 400, 80, 80)] + [(100, 100, 80, 80)] * 6
    smoothed = _median_smooth_boxes(boxes)
    assert smoothed[6] == (100, 100, 80, 80)   # the single-frame spike is filtered out


def test_fill_gaps_expands_to_contiguous_range():
    frames = [10, 11, 14]
    boxes = [(0, 0, 10, 10), (1, 1, 10, 10), (4, 4, 10, 10)]
    full_frames, full_boxes = _fill_gaps(frames, boxes)
    assert full_frames == [10, 11, 12, 13, 14]
    assert len(full_boxes) == 5
    assert full_boxes[2] in (boxes[1], boxes[2])   # nearest-neighbour fill for the gap
