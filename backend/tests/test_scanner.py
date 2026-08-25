import pytest

from dialogue_finder.config import DEFAULT
from dialogue_finder.models import Candidate
from dialogue_finder.progress import NullReporter
from dialogue_finder.text.scanner import coarse_scan, group_hits, pick_group


class FakeSource:
    fps = 10.0
    frame_count = 100

    def index_for_time(self, t): return int(round(t * self.fps))
    def time_for_index(self, i): return i / self.fps
    def iter_range(self, a, b, step):
        for i in range(a, b + 1, step):
            yield i, i  # "frame" is just its index


class FakeExtractor:
    def __init__(self, text_at): self.text_at = text_at
    def read(self, image): return self.text_at(image)


def test_coarse_scan_samples_at_requested_fps():
    src = FakeSource()
    ex = FakeExtractor(lambda i: "my mind rebels at stagnation" if 40 <= i <= 60 else "")
    cands = coarse_scan(src, ex, "My mind rebels at stagnation", 0.0, 9.9, fps=5, cfg=DEFAULT,
                        reporter=NullReporter(), read=lambda e, f, c: e.read(f))
    idx = [c.frame_index for c in cands]
    assert idx[:3] == [0, 2, 4]
    hits = [c for c in cands if c.score >= 0.8]
    assert hits and hits[0].frame_index == 40 and hits[-1].frame_index == 60


def test_group_hits_splits_on_gap():
    cs = [Candidate(i, i / 10, "t", 0.9) for i in (10, 12, 14, 80, 82)]
    groups = group_hits(cs, threshold=0.8, gap_s=2.0)
    assert [[c.frame_index for c in g] for g in groups] == [[10, 12, 14], [80, 82]]


def test_pick_group_first_last_all():
    groups = [[Candidate(1, 0.1, "", 0.9)], [Candidate(50, 5.0, "", 0.9)]]
    assert pick_group(groups, "first") == [groups[0]]
    assert pick_group(groups, "last") == [groups[1]]
    assert pick_group(groups, "all") == groups
    assert pick_group([], "first") == []


def test_coarse_scan_stops_when_cancelled():
    from dialogue_finder.pipeline import PipelineError
    src = FakeSource()
    ex = FakeExtractor(lambda i: "")
    calls = []
    def cancel():
        calls.append(1)
        return len(calls) > 2
    with pytest.raises(PipelineError, match="cancelled"):
        coarse_scan(src, ex, "x", 0.0, 9.9, fps=5, cfg=DEFAULT, reporter=NullReporter(),
                    read=lambda e, f, c: e.read(f), should_cancel=cancel)
