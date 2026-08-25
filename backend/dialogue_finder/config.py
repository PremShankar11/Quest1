from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    max_height: int = 480
    audio_match_threshold: float = 0.6
    ocr_match_threshold: float = 0.8
    window_pad_s: float = 3.0
    window_fps: float = 5.0
    fullscan_fps: float = 2.0
    retry_pad_s: float = 15.0            # widened window when the first OCR scan misses
    band_fraction: float = 0.35
    ocr_upscale: float = 2.0
    hit_gap_s: float = 2.0            # candidates further apart than this are separate occurrences
    whisper_model: str = "base"
    whisper_task: str = "translate"
    cache_dir: Path = field(default_factory=lambda: REPO_ROOT / "cache")
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "output")


DEFAULT = Config()
