from dialogue_finder.cli import main


def test_cli_bad_url_exits_1_without_traceback(capsys, tmp_path):
    code = main(["--local", str(tmp_path / "nope.mp4"), "--text", "x", "--mode", "ocr"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "Error:" in err


def test_cli_requires_text(capsys):
    code = main(["--url", "http://x"])
    assert code == 2


def test_cli_mode_audio_ocr_accepted(tmp_path):
    code = main(["--local", str(tmp_path / "nope.mp4"), "--text", "x", "--mode", "audio+ocr"])
    assert code == 1        # gets past argument parsing; fails later on the missing file


def test_cli_bad_mode_exits_2(capsys):
    code = main(["--local", "x.mp4", "--text", "x", "--mode", "bogus"])
    assert code == 2
