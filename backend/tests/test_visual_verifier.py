import pytest

from dialogue_finder.config import DEFAULT
from dialogue_finder.models import FaceTrack, Occurrence, Window
from dialogue_finder.visual.verifier import (
    _fill_gaps,
    _median_smooth_boxes,
    classify,
    confidence_for_occurrence,
    find_onset,
)


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


# ---- I-3: speech denominator = the located window, not the padded region -------------------

def test_classify_line_active_only_qualifies_when_denominator_is_windowed():
    """2.6 s line (26 frames @ 10 fps) inside 9 s (90 frames) of continuous speech in the padded
    region; the track is active only during the line. Old (whole-padded-region) denominator:
    26/90 = 0.289 < asd_min_active (0.3) -> would fail and misclassify `invalid`. New (in-window)
    denominator: 26/26 = 1.0 -> qualifies `valid-speaker`."""
    speech = [True] * 90                       # continuous speech across the whole padded region
    window_start_index, window_end_index = 32, 57   # the located window: 26 frames (2.6 s @ 10 fps)
    scores = [0.0] * 32 + [0.9] * 26 + [0.0] * (90 - 58)
    track = FaceTrack(track_id=0, frames=list(range(90)), boxes=[(0, 0, 80, 80)] * 90, scores=scores)

    klass, chosen, mean = classify(0.0, [track], speech, DEFAULT, first_index=0,
                                   window_start_index=window_start_index, window_end_index=window_end_index)
    assert klass == "valid-speaker"
    assert chosen is track
    assert mean == pytest.approx(0.9)


def test_classify_track_active_only_in_padding_does_not_qualify():
    """A track active only in the ±3 s padding (outside the located window) must not count
    towards the numerator OR the denominator -- it must not qualify, even though it's clearly
    an active speaker somewhere in the padded region."""
    speech = [True] * 90
    window_start_index, window_end_index = 32, 57
    scores = [0.9] * 32 + [0.0] * 26 + [0.9] * (90 - 58)   # active before and after the window only
    track = FaceTrack(track_id=0, frames=list(range(90)), boxes=[(0, 0, 80, 80)] * 90, scores=scores)

    klass, chosen, mean = classify(0.0, [track], speech, DEFAULT, first_index=0,
                                   window_start_index=window_start_index, window_end_index=window_end_index)
    assert klass == "invalid"      # a track exists but none qualifies within the window
    assert chosen is None
    assert mean == 0.0


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


# ---- Minor: confidence parity -- inexact OCR refine must yield MEDIUM like audio+ocr ---------

def test_confidence_for_occurrence_inexact_refine_is_medium_even_at_high_score():
    """audio+ocr forces MEDIUM when refine_first_frame couldn't pin an exact frame (text already
    visible at scan start) regardless of score; hybrid's `valid-text` occurrences must match --
    carried via `Occurrence.exact`."""
    window = Window(5.0, 5.5, 0.95, "line")
    occ = Occurrence(window=window, klass="valid-text", frame_index=120, ocr_score=0.97, faces=0,
                     asd_mean=0.0, speaker_box=None, exact=False)
    assert confidence_for_occurrence(occ) == "MEDIUM"


def test_confidence_for_occurrence_exact_high_score_refine_is_high():
    window = Window(5.0, 5.5, 0.95, "line")
    occ = Occurrence(window=window, klass="valid-text", frame_index=120, ocr_score=0.97, faces=0,
                     asd_mean=0.0, speaker_box=None, exact=True)
    assert confidence_for_occurrence(occ) == "HIGH"
