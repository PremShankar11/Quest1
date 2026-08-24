from dialogue_finder.text.matcher import normalize, score_contains, score_similar, best_word_window
from dialogue_finder.models import Word, format_timestamp


def test_normalize_strips_case_punctuation_and_spaces():
    assert normalize('  "My mind, REBELS at stagnation!"  ') == "my mind rebels at stagnation"


def test_score_contains_exact_is_one():
    assert score_contains("My mind rebels at stagnation", "MY MIND REBELS AT STAGNATION.") == 1.0


def test_score_contains_inside_longer_ocr_line():
    s = score_contains("My mind rebels at stagnation", "- My mind rebels at stagnation. Give me problems")
    assert s >= 0.95


def test_score_contains_ocr_noise_still_high():
    assert score_contains("My mind rebels at stagnation", "My rnind rebeIs at stagnation") >= 0.8


def test_score_contains_unrelated_is_low():
    assert score_contains("My mind rebels at stagnation", "Come along Watson") < 0.6


def test_score_contains_empty_haystack_is_zero():
    assert score_contains("anything", "") == 0.0


def test_score_similar_symmetric_ish():
    assert score_similar("my mind rebels", "mind rebels my") > 0.9


def test_best_word_window_finds_span():
    words = [Word(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(
        "come along watson my mind rebels at stagnation give me problems".split())]
    win = best_word_window(words, "My mind rebels at stagnation")
    assert win is not None
    assert win.score >= 0.9
    assert abs(win.start_s - 1.5) < 1e-6      # "my" is word index 3
    assert abs(win.end_s - 3.9) < 1e-6        # "stagnation" is index 7 → end 3.5+0.4


def test_best_word_window_none_when_no_words():
    assert best_word_window([], "x") is None


def test_best_word_window_low_score_when_absent():
    words = [Word(w, i, i + 0.5) for i, w in enumerate("the quick brown fox".split())]
    assert best_word_window(words, "my mind rebels at stagnation").score < 0.5


def test_score_contains_short_fragment_is_low():
    assert score_contains("My mind rebels at stagnation", "R") < 0.1
    assert score_contains("My mind rebels at stagnation", "mind") < 0.3


def test_score_contains_near_full_read_still_high():
    assert score_contains("My mind rebels at stagnation", "My mindrebels at stagnation") >= 0.9


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00.000"
    assert format_timestamp(3725.5) == "01:02:05.500"
    assert format_timestamp(59.9996) == "00:01:00.000"
