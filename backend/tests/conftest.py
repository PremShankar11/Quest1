import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.make_clip import make_clip  # noqa: E402

TEXT = "My mind rebels at stagnation"


@pytest.fixture(scope="session")
def synthetic_clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("clips") / "popin.mp4"
    truth = make_clip(out, text=TEXT, appear_s=5.0)
    return out, truth
