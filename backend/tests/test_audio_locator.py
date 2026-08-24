import json

import pytest

from dialogue_finder.config import DEFAULT
from dialogue_finder.audio.locator import WhisperLocator, words_cache_path
from dialogue_finder.progress import NullReporter


def test_locate_uses_cached_words_and_threshold(tmp_path):
    video = tmp_path / "v.mp4"; video.write_bytes(b"x")
    cfg = DEFAULT.__class__(cache_dir=tmp_path)
    words = [{"text": w, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, w in enumerate(
        "come along watson my mind rebels at stagnation give me problems".split())]
    words_cache_path(video, cfg).write_text(json.dumps(words), encoding="utf-8")
    loc = WhisperLocator(cfg, NullReporter())
    win = loc.locate(video, "My mind rebels at stagnation")
    assert win is not None and abs(win.start_s - 1.5) < 1e-6
    assert loc.locate(video, "completely unrelated sentence here") is None


@pytest.mark.slow
def test_transcribe_real_audio_smoke(tmp_path):
    """Generates 3 s of silence and checks transcription returns a list (may be empty) without error."""
    import subprocess
    from dialogue_finder.video.downloader import ensure_ffmpeg
    from dialogue_finder.audio.locator import transcribe_words
    ensure_ffmpeg()
    wav = tmp_path / "s.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "3", str(wav)],
                   check=True, capture_output=True)
    words = transcribe_words(wav, "base", "translate", NullReporter())
    assert isinstance(words, list)
