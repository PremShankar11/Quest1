from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from ..config import Config
from ..video.downloader import ensure_ffmpeg
from ..text.matcher import all_word_windows
from ..models import StageEvent, Window, Word
from ..progress import ProgressReporter


class Locator(Protocol):
    def locate(self, video: Path, target: str) -> Window | None: ...
    def locate_all(self, video: Path, target: str) -> list[Window]: ...


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


def _ensure_cuda_path() -> None:
    """Make the cuBLAS/cuDNN DLLs shipped inside the nvidia-*-cu12 pip packages
    discoverable by ctranslate2's CUDA backend.

    Those packages install their DLLs under `<site-packages>/nvidia/<component>/bin`
    rather than anywhere Windows normally searches, so without this step
    `WhisperModel(..., device="cuda")` constructs fine but `.transcribe()` fails with
    a cublas/cudnn load error (see `_looks_like_cuda_error` below, which is what
    catches that failure and falls back to CPU). Windows-only and a no-op when the
    nvidia packages aren't installed (e.g. requirements-gpu.txt was never applied,
    or this is a non-GPU machine) — CPU-only setups never call this at all.
    """
    if os.name != "nt":
        return
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    seen = set(path_entries)
    added: list[str] = []
    for entry in sys.path:
        nvidia_dir = Path(entry) / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for component in sorted(nvidia_dir.iterdir()):
            bin_dir = component / "bin"
            if not bin_dir.is_dir():
                continue
            bin_str = str(bin_dir)
            if bin_str in seen:
                continue
            seen.add(bin_str)
            added.append(bin_str)
            try:
                os.add_dll_directory(bin_str)
            except (AttributeError, OSError):
                pass
    if added:
        os.environ["PATH"] = os.pathsep.join(added + path_entries)


def _load_model(name: str, device: str = "cuda"):
    if device == "cuda":
        _ensure_cuda_path()
    from faster_whisper import WhisperModel
    if device == "cuda":
        try:
            return WhisperModel(name, device="cuda", compute_type="float16"), "cuda"
        except Exception:
            device = "cpu"
    return WhisperModel(name, device="cpu", compute_type="int8"), "cpu"


def _looks_like_cuda_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(tok in msg for tok in ("cuda", "cublas", "cudnn", "gpu"))


def _transcribe_all(model, wav: Path, task: str):
    segments, info = model.transcribe(str(wav), task=task, word_timestamps=True, vad_filter=True)
    return list(segments), info          # force the lazy generator inside the try


def transcribe_words(wav: Path, model_name: str, task: str, reporter: ProgressReporter) -> list[Word]:
    model, device = _load_model(model_name, "cuda")
    reporter.emit(StageEvent("transcribe", "running", f"whisper {model_name} on {device}", 0.0))
    try:
        segments, info = _transcribe_all(model, wav, task)
    except Exception as e:
        # CUDA runtime libs (e.g. cublas) can be missing even though a GPU is detected; that
        # failure can surface anywhere during transcription (construction, language detection,
        # or mid-generator), not just at WhisperModel() construction, so it must be caught
        # around the fully-materialised transcription, not just the constructor.
        if not _looks_like_cuda_error(e):
            raise
        reporter.emit(StageEvent("transcribe", "running", f"cuda unavailable ({type(e).__name__}); using cpu"))
        model, device = _load_model(model_name, "cpu")
        segments, info = _transcribe_all(model, wav, task)
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
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps([w.__dict__ for w in words]), encoding="utf-8")
        os.replace(tmp, cache)
        return words

    def locate_all(self, video: Path, target: str) -> list[Window]:
        words = self._words(video)
        return all_word_windows(words, target, self.cfg.audio_match_threshold, self.cfg.max_occurrences)

    def locate(self, video: Path, target: str) -> Window | None:
        wins = self.locate_all(video, target)
        return wins[0] if wins else None
