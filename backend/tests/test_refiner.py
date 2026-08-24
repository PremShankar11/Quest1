from dialogue_finder.text.refiner import first_true


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
