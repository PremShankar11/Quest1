from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Protocol

from ..config import Config
from ..video.downloader import ensure_ffmpeg
from ..text.matcher import best_word_window
from ..models import StageEvent, Window, Word
from ..progress import ProgressReporter


class Locator(Protocol):
    def locate(self, video: Path, target: str) -> Window | None: ...


def words_cache_path(video: Path, cfg: Config) -> Path:
    return cfg.cache_dir / f"{video.stem}.words.json"


def extract_audio(video: Path, wav: Path) -> Path:
    ensure_ffmpeg()
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not wav.exists():
        raise RuntimeError(f"ffmpeg audio extraction failed: {r.stderr[-300:]}")
    return wav


def _load_model(name: str, device: str = "cuda"):
    from faster_whisper import WhisperModel
    if device == "cuda":
        try:
            return WhisperModel(name, device="cuda", compute_type="float16"), "cuda"
        except Exception:
            device = "cpu"
    return WhisperModel(name, device="cpu", compute_type="int8"), "cpu"


def transcribe_words(wav: Path, model_name: str, task: str, reporter: ProgressReporter) -> list[Word]:
    def run(model):
        return model.transcribe(str(wav), task=task, word_timestamps=True, vad_filter=True)

    model, device = _load_model(model_name)
    reporter.emit(StageEvent("transcribe", "running", f"whisper {model_name} on {device}", 0.0))
    try:
        segments, info = run(model)
    except RuntimeError:
        # CUDA runtime libs (e.g. cublas) can be missing even though a GPU is detected;
        # that failure only surfaces on first inference, not at WhisperModel() construction,
        # so _load_model()'s try/except can't catch it. Fall back to CPU here instead.
        if device != "cuda":
            raise
        model, device = _load_model(model_name, device="cpu")
        reporter.emit(StageEvent("transcribe", "fallback", f"cuda unavailable, retrying whisper {model_name} on {device}", 0.0))
        segments, info = run(model)
    words: list[Word] = []
    total = info.duration
    for seg in segments:
        for w in (seg.words or []):
            words.append(Word(w.word.strip(), float(w.start), float(w.end)))
        reporter.emit(StageEvent("transcribe", "running", seg.text.strip()[:80],
                                 (seg.end / total) if total else None,
                                 {"start": seg.start, "end": seg.end, "text": seg.text.strip()}))
    reporter.emit(StageEvent("transcribe", "ok", f"{len(words)} words, language {info.language}", 1.0))
    return words


class WhisperLocator:
    def __init__(self, cfg: Config, reporter: ProgressReporter) -> None:
        self.cfg, self.reporter = cfg, reporter

    def _words(self, video: Path) -> list[Word]:
        cache = words_cache_path(video, self.cfg)
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            self.reporter.emit(StageEvent("transcribe", "ok", f"transcript cache hit ({len(data)} words)", 1.0))
            return [Word(**d) for d in data]
        wav = extract_audio(video, self.cfg.cache_dir / f"{video.stem}.16k.wav")
        words = transcribe_words(wav, self.cfg.whisper_model, self.cfg.whisper_task, self.reporter)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([w.__dict__ for w in words]), encoding="utf-8")
        return words

    def locate(self, video: Path, target: str) -> Window | None:
        words = self._words(video)
        win = best_word_window(words, target)
        if win is None or win.score < self.cfg.audio_match_threshold:
            return None
        return win
