import sys
from pathlib import Path

import pytest

from dialogue_finder.cli import main


def test_unreachable_url_is_clean(capsys):
    code = main(["--url", "https://example.invalid/video/1", "--text", "x", "--mode", "ocr"])
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error:") and "Traceback" not in err


def test_broken_pipeline_import_is_clean(capsys, monkeypatch):
    """A broken/missing dependency (e.g. opencv DLL load failure) surfacing on
    `from .pipeline import ...` must print a clean Error line, never a traceback."""
    monkeypatch.setitem(sys.modules, "dialogue_finder.pipeline", None)
    code = main(["--local", "unused.mp4", "--text", "x", "--mode", "ocr"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "Traceback" not in err


def test_corrupt_file_is_clean(capsys, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    code = main(["--local", str(bad), "--text", "x", "--mode", "ocr"])
    assert code == 1
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.slow
def test_output_write_failure_is_clean(capsys, tmp_path, synthetic_clip, monkeypatch):
    """Ruling (a): a successful run whose result.json write fails must print
    'Error: could not write output: ...' and exit 1, never a traceback."""
    path, truth = synthetic_clip
    real_write_text = Path.write_text

    def failing_write_text(self, *a, **kw):
        if self.name == "result.json":
            raise OSError("disk full (simulated)")
        return real_write_text(self, *a, **kw)

    monkeypatch.setattr(Path, "write_text", failing_write_text)
    code = main(["--local", str(path), "--text", truth["text"], "--mode", "ocr",
                "--out", str(tmp_path / "out")])
    assert code == 1
    err = capsys.readouterr().err
    assert "Error: could not write output:" in err
    assert "Traceback" not in err
