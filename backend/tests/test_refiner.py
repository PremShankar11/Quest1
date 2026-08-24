import numpy as np

from dialogue_finder.config import DEFAULT
from dialogue_finder.text.refiner import classify_appearance, first_true, refine_first_frame


def test_first_true_finds_boundary():
    calls = []
    def pred(i):
        calls.append(i)
        return i >= 37
    assert first_true(30, 45, pred) == 37
    assert len(calls) <= 5          # log2(15) ≈ 4


def test_first_true_when_hi_is_first():
    assert first_true(10, 11, lambda i: i >= 11) == 11


def test_first_true_when_all_true_returns_lo_plus_one_bound():
    # lo is assumed False by contract; if pred is True right after lo we get lo+1
    assert first_true(0, 8, lambda i: True) == 1


class _FakeSource:
    """frame_at(i) returns i itself so `read=` can key off the index; time is i/24."""
    def frame_at(self, i):
        return i

    def time_for_index(self, i):
        return i / 24


def test_refine_first_frame_normal_case_exact():
    src = _FakeSource()
    read = lambda e, f, c: "my mind rebels at stagnation" if f >= 900 else ""
    cand, exact = refine_first_frame(src, None, "my mind rebels at stagnation", 1000, 985, DEFAULT,
                                     read=read, step=15)
    assert cand.frame_index == 900
    assert exact is True


def test_refine_first_frame_text_visible_at_scan_start_is_inexact():
    src = _FakeSource()
    read = lambda e, f, c: "my mind rebels at stagnation"        # visible everywhere, including frame 0
    cand, exact = refine_first_frame(src, None, "my mind rebels at stagnation", 1000, 985, DEFAULT,
                                     read=read, step=15)
    assert exact is False
    assert cand.frame_index == 1000 - 8 * 15


def test_refine_first_frame_normal_case_default_step():
    src = _FakeSource()
    read = lambda e, f, c: "my mind rebels at stagnation" if f >= 995 else ""
    cand, exact = refine_first_frame(src, None, "my mind rebels at stagnation", 1000, 985, DEFAULT,
                                     read=read)
    assert cand.frame_index == 995
    assert exact is True


def test_classify_appearance_popin_and_fade():
    h, w = 100, 200
    import cv2
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    rect = blank.copy()
    cv2.rectangle(rect, (20, 80), (180, 95), (255, 255, 255), -1)   # bright rect inside the bottom band
    faint = blank.copy()
    cv2.rectangle(faint, (20, 80), (180, 95), (120, 120, 120), -1)  # dim rect, still has edges

    class _FramesSource:
        def __init__(self, frames):
            self.frames = frames
        def frame_at(self, i):
            return self.frames[i]

    pop_src = _FramesSource([blank, rect])
    assert classify_appearance(pop_src, 1, DEFAULT) == "pop-in"

    fade_src = _FramesSource([faint, rect])
    assert classify_appearance(fade_src, 1, DEFAULT) == "fade-in"
