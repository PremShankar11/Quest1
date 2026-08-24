# Dialogue Frame Finder — Plan 1: Pipeline + CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python CLI that takes a video URL (or local file) and a target dialogue, and prints the timestamp, 0-based frame number, extracted text, confidence, and saves the frame image (plus the frame before it) — never crashing.

**Architecture:** `pipeline.run()` downloads with yt-dlp, asks `audio_locator` (faster-whisper, translate task) for a ~10 s window, runs `scanner` (EasyOCR on the subtitle band at 5 fps in the window, 2 fps whole-video fallback), then `refiner` binary-searches OCR score between the last non-matching sample and the first matching sample to pin the exact first frame. Every stage reports through a `ProgressReporter`; every stage has a fallback so a `Result` is always produced.

**Tech Stack:** Python 3.14 (`py`), yt-dlp, static-ffmpeg, opencv-python, easyocr (+torch), faster-whisper, rapidfuzz, pytest. Windows 11, Git Bash for commands.

**Spec:** `C:\Users\Asus\.claude\plans\superpowers-brainstorming-c-users-asus-dapper-crystal.md` (approved 2026-08-24). Plan 2 (FastAPI + Next.js UI) and Plan 3 (docs + final validation) come after this plan.

## Global Constraints

- Repo root: `C:\Users\Asus\Quest1`. **Local git only — never push.** Commit after every task.
- Python: `py -3.14`; venv at `.venv`; all commands use `.venv/Scripts/python` (Git Bash path form).
- `prompts.txt` must contain every prompt given to any AI, verbatim, in order (Appendix A seeds it; append every new prompt).
- Output block format is fixed by the spec (see Task 7). `Text` always holds real extracted text, never a placeholder.
- Frame numbers are 0-based. Timestamp = `frame_index / fps`, formatted `HH:MM:SS.sss`.
- The CLI must never print a traceback: fatal errors → one line on stderr, exit code 1.
- Thresholds live only in `backend/dialogue_finder/config.py`: audio window ≥ 0.6, OCR match ≥ 0.8, window pad 3 s, 5 fps in-window, 2 fps whole video, band = bottom 35 %, OCR upscale 2×, Whisper `base` + `translate`, download ≤ 480p.
- Test video: `https://ok.ru/video/248244667877` (54 min, 24 fps, 960×720). Target: `My mind rebels at stagnation`.
- Don't build: multi-language OCR, scene detection, Docker, hosting, LLM-vision OCR.

---

## File structure

```
Quest1/
  .gitignore
  prompts.txt
  README.md
  requirements.txt
  docs/APPROACH.md              # phased; filled as tasks complete
  docs/DECISIONS.md             # "AI proposed / I changed / why" per phase
  docs/BENCHMARK.md             # written by bench/run_bench.py
  docs/superpowers/plans/       # this file
  backend/
    dialogue_finder/
      __init__.py
      __main__.py               # python -m dialogue_finder → cli.main()
      config.py                 # Config dataclass, DEFAULT
      models.py                 # Word, Window, Candidate, Result, StageEvent, VideoInfo
      progress.py               # ProgressReporter protocol, NullReporter, PrintReporter
      matcher.py                # normalize, score_contains, score_similar, best_word_window
      downloader.py             # ensure_ffmpeg, cache_key, fetch_video, DownloadError
      frame_source.py           # FrameSource (cv2), index<->time helpers, format_timestamp
      ocr.py                    # TextExtractor protocol, EasyOCRExtractor, crop_band, prep, read_dialogue
      scanner.py                # coarse_scan, group_hits, pick_group
      refiner.py                # first_true (binary search), refine_first_frame, classify_appearance
      audio_locator.py          # Locator protocol, WhisperLocator, extract_audio, transcribe_words
      pipeline.py               # run(): orchestration + fallbacks + confidence
      cli.py                    # argparse, output block, exit codes
    bench/
      __init__.py
      make_clip.py              # synthetic clip generator (cv2.VideoWriter) with ground truth
      run_bench.py              # variants → docs/BENCHMARK.md
    tests/
      conftest.py               # synthetic clip fixture
      test_matcher.py
      test_frame_source.py
      test_scanner.py
      test_refiner.py
      test_pipeline.py          # ground-truth end-to-end (slow)
      test_cli.py
  scripts/
    dump_frames.py              # spike: save frames at given timestamps
```

---

### Task 1: Repo scaffold, environment, prompts.txt

**Files:**
- Create: `.gitignore`, `requirements.txt`, `prompts.txt`, `README.md`, `docs/APPROACH.md`, `docs/DECISIONS.md`, `backend/dialogue_finder/__init__.py`, `backend/tests/__init__.py`, `backend/bench/__init__.py`, `scripts/.gitkeep`

**Interfaces:**
- Produces: a working venv at `.venv` with all deps importable; `git log` has the first commit.

- [ ] **Step 1: git init and .gitignore**

```bash
cd /c/Users/Asus/Quest1 && git init -b main
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.pytest_cache/
cache/
output/
bench_out/
*.mp4
*.wav
*.png
!docs/**/*.png
.easyocr/
node_modules/
.next/
EOF
```

- [ ] **Step 2: requirements.txt (exact pins verified installable on py3.14 win64 on 2026-08-24)**

```
yt-dlp==2026.8.19
static-ffmpeg==3.0
opencv-python==5.0.0.93
numpy>=2.0
rapidfuzz==3.14.5
easyocr==1.7.2
torch==2.13.0
torchvision==0.28.0
faster-whisper==1.2.1
pytest==8.4.1
```

If `pytest==8.4.1` is not found, use `pytest>=8`.

- [ ] **Step 3: venv + install (CPU torch first; GPU is an upgrade step)**

```bash
cd /c/Users/Asus/Quest1 && py -3.14 -m venv .venv && .venv/Scripts/python -m pip install --upgrade pip && .venv/Scripts/python -m pip install -r requirements.txt
```
Expected: ends with `Successfully installed ...`. Takes 5-10 min (torch ≈ 200 MB).

- [ ] **Step 4: try CUDA torch (optional; skip on failure)**

```bash
cd /c/Users/Asus/Quest1 && .venv/Scripts/python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu128 && .venv/Scripts/python -c "import torch; print('cuda', torch.cuda.is_available())"
```
Expected: `cuda True`. If pip reports no matching distribution for cp314, keep the CPU build and write in DECISIONS.md: "CUDA torch wheel unavailable for py3.14 on <date>; CPU path used; GPU noted as optional."

- [ ] **Step 5: smoke-import every dependency**

```bash
cd /c/Users/Asus/Quest1 && .venv/Scripts/python -c "import cv2, rapidfuzz, yt_dlp, static_ffmpeg, faster_whisper, easyocr, torch; print('ok', cv2.__version__, torch.__version__)"
```
Expected: `ok 5.0.0 2.13.0...`. If `easyocr` import fails on 3.14, record it in DECISIONS.md and install `rapidocr-onnxruntime` instead; Task 5 shows the alternative extractor.

- [ ] **Step 6: prompts.txt — copy Appendix A verbatim** (bottom of this file). Then `docs/DECISIONS.md`:

```markdown
# Decisions log — where the human steered the AI

Format per entry: **AI proposed** → **I changed** → **Why**.

## Phase 0 — Brainstorm (2026-08-24)
- AI's first reading (and mine) was audio-only: transcribe, return the frame at the spoken word.
  → Changed to OCR-primary hybrid. → I raised two cases audio can't handle: a dubbed track with
  English burned-in subtitles (the Godfather-Italian case) and text shown but never spoken (title cards).
  The PDF's wording ("appears", "extracts the text", "appearance") also points at on-screen text.
- AI proposed CLI-only. → Changed to CLI + live web UI. → The interviewer should paste a URL and watch the
  pipeline run; the CLI stays as the core and the fallback path.
- AI proposed a 3-day cut-down. → Kept a 10-day plan but spent the extra days on evidence (benchmark,
  decision log, proof-of-first), not features. → Company note: "don't over-engineer".
- AI proposed a pixel-change "step detector" to find the exact frame. → Replaced with binary search on
  OCR score between the last non-matching and first matching sample (3-4 OCR calls). → Same precision,
  nothing to tune, one sentence to explain. Pixel diff survives only as the pop-in/fade-in label.
```

`docs/APPROACH.md` skeleton:

```markdown
# Approach

## Phase 1 — Understand the problem
## Phase 2 — Design
## Phase 3 — Build
## Phase 4 — Test and measure
## Phase 5 — Reflect: limits and extensions
```

`README.md` (first version):

```markdown
# Dialogue Frame Finder

Finds the first video frame where a given dialogue appears on screen, from a video URL.
Status: Plan 1 (pipeline + CLI) in progress. See docs/APPROACH.md and docs/DECISIONS.md.
```

- [ ] **Step 7: package skeleton + commit**

```bash
cd /c/Users/Asus/Quest1 && mkdir -p backend/dialogue_finder backend/tests backend/bench scripts && touch backend/dialogue_finder/__init__.py backend/tests/__init__.py backend/bench/__init__.py scripts/.gitkeep && git add -A && git commit -m "chore: scaffold repo, env, prompts log, docs skeletons"
```

---

### Task 2: config, models, matcher (pure logic, TDD)

**Files:**
- Create: `backend/dialogue_finder/config.py`, `backend/dialogue_finder/models.py`, `backend/dialogue_finder/matcher.py`
- Test: `backend/tests/test_matcher.py`

**Interfaces:**
- Produces: `Config` (fields listed below), `DEFAULT: Config`; dataclasses `Word(text,start,end)`, `Window(start_s,end_s,score,matched_text)`, `Candidate(frame_index,timestamp_s,text,score)`, `VideoInfo(fps,frame_count,width,height,duration_s)`, `StageEvent(stage,status,message,progress,payload)`, `Result(...)` with `format_block()` and `to_dict()`; functions `normalize(str)->str`, `score_contains(target,haystack)->float`, `score_similar(a,b)->float`, `best_word_window(words,target)->Window|None`, `format_timestamp(seconds)->str`.

- [ ] **Step 1: write failing tests**

`backend/tests/test_matcher.py`:
```python
from dialogue_finder.matcher import normalize, score_contains, score_similar, best_word_window
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
    assert score_contains("My mind rebels at stagnation", "Come along Watson") < 0.5


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


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00.000"
    assert format_timestamp(3725.5) == "01:02:05.500"
    assert format_timestamp(59.9996) == "00:01:00.000"
```

- [ ] **Step 2: run to verify failure**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest tests/test_matcher.py -q
```
Expected: `ModuleNotFoundError: No module named 'dialogue_finder'` (or ImportError).

- [ ] **Step 3: implement**

`backend/dialogue_finder/config.py`:
```python
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    max_height: int = 480
    audio_match_threshold: float = 0.6
    ocr_match_threshold: float = 0.8
    window_pad_s: float = 3.0
    window_fps: float = 5.0
    fullscan_fps: float = 2.0
    band_fraction: float = 0.35
    ocr_upscale: float = 2.0
    hit_gap_s: float = 2.0            # candidates further apart than this are separate occurrences
    whisper_model: str = "base"
    whisper_task: str = "translate"
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    output_dir: Path = field(default_factory=lambda: Path("output"))


DEFAULT = Config()
```

`backend/dialogue_finder/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


def format_timestamp(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Window:
    start_s: float
    end_s: float
    score: float
    matched_text: str


@dataclass
class Candidate:
    frame_index: int
    timestamp_s: float
    text: str
    score: float


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


@dataclass
class StageEvent:
    stage: str                       # download | transcribe | locate | scan | refine | done | error
    status: str                      # running | ok | skipped | fallback | error
    message: str = ""
    progress: float | None = None    # 0..1
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    timestamp_s: float
    frame_index: int
    text: str
    confidence: str                  # HIGH | MEDIUM | LOW
    source: str                      # ocr | audio | ocr-weak
    note: str = ""
    fps: float = 0.0
    image_path: str = ""
    prev_image_path: str = ""
    appearance: str = ""             # pop-in | fade-in | ""
    window: Window | None = None
    candidates: list[Candidate] = field(default_factory=list)
    alternatives: list[Candidate] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.timestamp_s)

    def format_block(self) -> str:
        lines = [
            f"Timestamp : {self.timestamp}",
            f"Frame     : {self.frame_index}",
            f'Text      : "{self.text}"',
            f"Confidence: {self.confidence}  (source: {self.source}{'; ' + self.note if self.note else ''})",
            f"Image     : {self.image_path}",
        ]
        if self.prev_image_path:
            lines.append(f"Previous  : {self.prev_image_path}  ({self.appearance or 'frame before'})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp
        return d
```

`backend/dialogue_finder/matcher.py`:
```python
from __future__ import annotations

import re

from rapidfuzz import fuzz

from .models import Word, Window

_PUNCT = re.compile(r"[^a-z0-9' ]+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower().replace("'", "'")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def score_contains(target: str, haystack: str) -> float:
    """How well `target` appears inside `haystack` (0..1). Tolerant of OCR noise and extra words."""
    t, h = normalize(target), normalize(haystack)
    if not t or not h:
        return 0.0
    return fuzz.partial_ratio(t, h) / 100.0


def score_similar(a: str, b: str) -> float:
    """Order-insensitive similarity of two short phrases (0..1)."""
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def best_word_window(words: list[Word], target: str) -> Window | None:
    """Slide windows of ~len(target words) over the transcript; return the best-scoring span."""
    if not words:
        return None
    n = max(1, len(normalize(target).split()))
    best: Window | None = None
    for size in range(max(1, n - 1), n + 3):
        for i in range(0, max(1, len(words) - size + 1)):
            span = words[i:i + size]
            if not span:
                continue
            text = " ".join(w.text for w in span)
            s = score_similar(target, text)
            if best is None or s > best.score:
                best = Window(span[0].start, span[-1].end, s, text)
    return best
```

- [ ] **Step 4: run tests**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest tests/test_matcher.py -q
```
Expected: `12 passed`.

- [ ] **Step 5: commit**

```bash
cd /c/Users/Asus/Quest1 && git add -A && git commit -m "feat: config, models, fuzzy matcher with tests"
```

---

### Task 3: downloader + spike on the real video

**Files:**
- Create: `backend/dialogue_finder/downloader.py`, `backend/dialogue_finder/progress.py`, `scripts/dump_frames.py`
- Modify: `docs/APPROACH.md` (Phase 1 finding)

**Interfaces:**
- Produces: `ensure_ffmpeg()->None`, `cache_key(url)->str`, `fetch_video(url, cfg, reporter)->Path`, `probe(path)->VideoInfo`, `class DownloadError(Exception)`; `ProgressReporter` protocol with `emit(event: StageEvent)`, `NullReporter`, `PrintReporter(verbose: bool)`.

- [ ] **Step 1: progress.py**

```python
from __future__ import annotations

import sys
from typing import Protocol

from .models import StageEvent


class ProgressReporter(Protocol):
    def emit(self, event: StageEvent) -> None: ...


class NullReporter:
    def emit(self, event: StageEvent) -> None:
        return None


class PrintReporter:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._last_progress_stage = ""

    def emit(self, event: StageEvent) -> None:
        if event.progress is not None and not self.verbose:
            return
        tag = f"[{event.stage}:{event.status}]"
        extra = f" {event.progress:.0%}" if event.progress is not None else ""
        print(f"{tag}{extra} {event.message}", file=sys.stderr)
```

- [ ] **Step 2: downloader.py**

```python
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from .config import Config
from .models import StageEvent, VideoInfo
from .progress import ProgressReporter


class DownloadError(Exception):
    """Raised when the video cannot be fetched or read. Message is user-facing."""


def ensure_ffmpeg() -> None:
    import static_ffmpeg
    static_ffmpeg.add_paths()          # downloads ffmpeg+ffprobe once, then adds them to PATH


def cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def fetch_video(url: str, cfg: Config, reporter: ProgressReporter) -> Path:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    target = cfg.cache_dir / f"{cache_key(url)}.mp4"
    if target.exists() and target.stat().st_size > 0:
        reporter.emit(StageEvent("download", "ok", f"cache hit {target.name}", 1.0, {"path": str(target)}))
        return target
    ensure_ffmpeg()
    import yt_dlp

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            frac = (done / total) if total else None
            reporter.emit(StageEvent("download", "running", "downloading", frac))

    opts = {
        "format": f"bv*[height<={cfg.max_height}]+ba/b[height<={cfg.max_height}]/b",
        "outtmpl": str(target.with_suffix("")) + ".%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
    }
    reporter.emit(StageEvent("download", "running", f"fetching {url}", 0.0))
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:  # yt-dlp raises many types; all are fatal here
        raise DownloadError(f"Could not download video: {str(e).splitlines()[0][:200]}") from e
    if not target.exists():
        found = list(cfg.cache_dir.glob(f"{cache_key(url)}.*"))
        if not found:
            raise DownloadError("Download finished but no file was produced.")
        found[0].rename(target)
    reporter.emit(StageEvent("download", "ok", f"saved {target.name}", 1.0, {"path": str(target)}))
    return target


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DownloadError(f"Cannot open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or count <= 0:
        raise DownloadError(f"Video has no readable frames: {path}")
    return VideoInfo(fps=fps, frame_count=count, width=w, height=h, duration_s=count / fps)
```

- [ ] **Step 3: spike script**

`scripts/dump_frames.py`:
```python
"""Spike: download a video and save frames at the given timestamps (seconds) to bench_out/spike/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import cv2  # noqa: E402
from dialogue_finder.config import DEFAULT  # noqa: E402
from dialogue_finder.downloader import fetch_video, probe  # noqa: E402
from dialogue_finder.progress import PrintReporter  # noqa: E402

url = sys.argv[1]
times = [float(t) for t in sys.argv[2:]]
path = fetch_video(url, DEFAULT, PrintReporter(verbose=True))
info = probe(path)
print(info)
out = Path("bench_out/spike"); out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(path))
for t in times:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if ok:
        p = out / f"t{int(t):05d}.png"; cv2.imwrite(str(p), frame); print("saved", p)
```

- [ ] **Step 4: run the spike (downloads ~300-500 MB once)**

```bash
cd /c/Users/Asus/Quest1 && .venv/Scripts/python scripts/dump_frames.py "https://ok.ru/video/248244667877" 30 120 300 600 900 1200 1800 2400 3000
```
Expected: `VideoInfo(fps=24.0, frame_count≈78264, ...)` and 9 PNGs in `bench_out/spike/`. Open them (Read tool) and answer: **are subtitles burned in?** Write the answer under `## Phase 1` in `docs/APPROACH.md` with the sentence "Test video has / does not have burned-in subtitles (checked frames at t=...); therefore the expected route on this video is OCR / audio-fallback."

- [ ] **Step 5: commit**

```bash
cd /c/Users/Asus/Quest1 && git add -A && git commit -m "feat: downloader with cache and progress; spike frames recorded in APPROACH"
```

---

### Task 4: synthetic clip generator + FrameSource

**Files:**
- Create: `backend/bench/make_clip.py`, `backend/dialogue_finder/frame_source.py`, `backend/tests/conftest.py`
- Test: `backend/tests/test_frame_source.py`

**Interfaces:**
- Produces: `make_clip(out: Path, *, text: str, appear_s: float, duration_s=15.0, fps=24, size=(640,360), position="bottom", fade_frames=0, scale=1.0) -> dict` returning `{"frame": int, "timestamp": float, "fps": int}`; `class FrameSource` with `fps`, `frame_count`, `width`, `height`, `duration_s`, `index_for_time(t)->int`, `time_for_index(i)->float`, `frame_at(i)->np.ndarray`, `iter_range(start_i, end_i, step)->Iterator[tuple[int, np.ndarray]]`, `close()`; context-manager support.

- [ ] **Step 1: write failing tests**

`backend/tests/conftest.py`:
```python
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
```

`backend/tests/test_frame_source.py`:
```python
import numpy as np

from dialogue_finder.frame_source import FrameSource


def test_probe_values(synthetic_clip):
    path, truth = synthetic_clip
    with FrameSource(path) as src:
        assert abs(src.fps - 24) < 0.01
        assert src.frame_count == 24 * 15
        assert src.width == 640 and src.height == 360


def test_index_time_roundtrip(synthetic_clip):
    path, _ = synthetic_clip
    with FrameSource(path) as src:
        assert src.index_for_time(5.0) == 120
        assert abs(src.time_for_index(120) - 5.0) < 1e-9


def test_frame_before_is_blank_and_at_has_text(synthetic_clip):
    path, truth = synthetic_clip
    n = truth["frame"]
    with FrameSource(path) as src:
        before = src.frame_at(n - 1)
        at = src.frame_at(n)
    band_before = before[int(360 * 0.65):, :, :]
    band_at = at[int(360 * 0.65):, :, :]
    assert (band_before > 200).sum() == 0
    assert (band_at > 200).sum() > 500


def test_frame_at_is_exact_after_random_access(synthetic_clip):
    path, truth = synthetic_clip
    n = truth["frame"]
    with FrameSource(path) as src:
        src.frame_at(300)
        src.frame_at(10)
        at = src.frame_at(n)
        before = src.frame_at(n - 1)
    assert (at[int(360 * 0.65):] > 200).sum() > 500
    assert (before[int(360 * 0.65):] > 200).sum() == 0


def test_iter_range_yields_every_step(synthetic_clip):
    path, _ = synthetic_clip
    with FrameSource(path) as src:
        idx = [i for i, _ in src.iter_range(100, 130, 5)]
    assert idx == [100, 105, 110, 115, 120, 125, 130]
```

- [ ] **Step 2: run to verify failure**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest tests/test_frame_source.py -q
```
Expected: ImportError on `bench.make_clip` / `dialogue_finder.frame_source`.

- [ ] **Step 3: implement make_clip.py**

```python
"""Generate a synthetic video with known text appearing at a known frame. Ground truth for tests/bench."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_clip(out: Path, *, text: str, appear_s: float, duration_s: float = 15.0, fps: int = 24,
              size: tuple[int, int] = (640, 360), position: str = "bottom", fade_frames: int = 0,
              scale: float = 1.0) -> dict:
    w, h = size
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open output")
    total = int(round(duration_s * fps))
    appear_frame = int(round(appear_s * fps))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9 * scale * (w / 640)
    thickness = max(1, int(round(2 * scale * (w / 640))))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - tw) // 2
    y = {"bottom": int(h * 0.90), "top": int(h * 0.12) + th, "center": (h + th) // 2}[position]
    rng = np.random.default_rng(0)
    for i in range(total):
        # moving gradient background so frames are not identical (like real video)
        base = np.linspace(20, 90, w, dtype=np.uint8)
        frame = np.tile(base, (h, 1))
        frame = np.roll(frame, i * 2, axis=1)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.add(frame, rng.integers(0, 6, frame.shape, dtype=np.uint8))
        if i >= appear_frame:
            alpha = 1.0 if fade_frames <= 0 else min(1.0, (i - appear_frame + 1) / fade_frames)
            overlay = frame.copy()
            cv2.putText(overlay, text, (x, y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        writer.write(frame)
    writer.release()
    return {"frame": appear_frame, "timestamp": appear_frame / fps, "fps": fps, "text": text}


if __name__ == "__main__":
    import sys
    print(make_clip(Path(sys.argv[1]), text=sys.argv[2], appear_s=float(sys.argv[3])))
```

- [ ] **Step 4: implement frame_source.py**

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

SEEK_BACK = 48   # frames to seek before the target, then decode forward (keyframe-safe)


class FrameSource:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"cannot open {self.path}")
        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._next_index = 0            # index of the frame the next read() returns

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def index_for_time(self, t: float) -> int:
        return max(0, min(self.frame_count - 1, int(round(t * self.fps))))

    def time_for_index(self, i: int) -> float:
        return i / self.fps

    def _seek(self, index: int) -> None:
        if index == self._next_index:
            return
        if 0 <= index - self._next_index <= SEEK_BACK * 2:
            for _ in range(index - self._next_index):
                self._cap.grab()
            self._next_index = index
            return
        start = max(0, index - SEEK_BACK)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos > index:                     # backend overshot; restart from 0
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            pos = 0
        for _ in range(index - pos):
            self._cap.grab()
        self._next_index = index

    def frame_at(self, index: int) -> np.ndarray:
        index = max(0, min(self.frame_count - 1, index))
        self._seek(index)
        ok, frame = self._cap.read()
        self._next_index = index + 1
        if not ok or frame is None:
            raise IOError(f"failed to decode frame {index}")
        return frame

    def iter_range(self, start_index: int, end_index: int, step: int) -> Iterator[tuple[int, np.ndarray]]:
        step = max(1, step)
        i = max(0, start_index)
        end_index = min(self.frame_count - 1, end_index)
        while i <= end_index:
            yield i, self.frame_at(i)
            i += step

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

- [ ] **Step 5: run tests**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest tests/test_frame_source.py -q
```
Expected: `5 passed`. If `test_frame_at_is_exact_after_random_access` fails, raise `SEEK_BACK` to 120 and re-run (mp4v keyframe interval can be large).

- [ ] **Step 6: commit**

```bash
cd /c/Users/Asus/Quest1 && git add -A && git commit -m "feat: synthetic clip generator and FrameSource with exact random access"
```

---

### Task 5: OCR extractor

**Files:**
- Create: `backend/dialogue_finder/ocr.py`
- Test: `backend/tests/test_ocr.py`

**Interfaces:**
- Produces: `class TextExtractor(Protocol): read(image: np.ndarray) -> str`; `class EasyOCRExtractor(languages=("en",), gpu: bool | None = None)`; `crop_band(image, fraction)->np.ndarray`; `prep(image, upscale)->np.ndarray`; `read_dialogue(extractor, frame, cfg)->str` (band first, full frame if band is empty).

- [ ] **Step 1: write failing test (marked slow: first run downloads the EasyOCR model ≈ 100 MB)**

`backend/tests/test_ocr.py`:
```python
import numpy as np
import pytest

from dialogue_finder.config import DEFAULT
from dialogue_finder.frame_source import FrameSource
from dialogue_finder.matcher import score_contains
from dialogue_finder.ocr import crop_band, prep, read_dialogue


def test_crop_band_takes_bottom_fraction():
    img = np.zeros((100, 50, 3), dtype=np.uint8)
    band = crop_band(img, 0.35)
    assert band.shape[0] == 35 and band.shape[1] == 50


def test_prep_upscales_to_gray():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    out = prep(img, 2.0)
    assert out.shape == (20, 40)


@pytest.mark.slow
def test_easyocr_reads_synthetic_subtitle(synthetic_clip):
    from dialogue_finder.ocr import EasyOCRExtractor
    path, truth = synthetic_clip
    with FrameSource(path) as src:
        frame = src.frame_at(truth["frame"] + 5)
        blank = src.frame_at(truth["frame"] - 5)
    ex = EasyOCRExtractor()
    text = read_dialogue(ex, frame, DEFAULT)
    assert score_contains(truth["text"], text) >= 0.8, text
    assert score_contains(truth["text"], read_dialogue(ex, blank, DEFAULT)) < 0.5
```

Add `backend/pytest.ini`:
```ini
[pytest]
markers =
    slow: downloads models or runs the real pipeline
```

- [ ] **Step 2: run to verify failure**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest tests/test_ocr.py -q
```
Expected: ImportError `dialogue_finder.ocr`.

- [ ] **Step 3: implement ocr.py**

```python
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .config import Config
from .matcher import normalize


class TextExtractor(Protocol):
    def read(self, image: np.ndarray) -> str: ...


def crop_band(image: np.ndarray, fraction: float) -> np.ndarray:
    h = image.shape[0]
    return image[int(h * (1 - fraction)):, :]


def prep(image: np.ndarray, upscale: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if upscale and upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return gray


class EasyOCRExtractor:
    """EasyOCR wrapper. Model loads lazily on first read (slow once, then cached in memory)."""

    def __init__(self, languages: tuple[str, ...] = ("en",), gpu: bool | None = None) -> None:
        self.languages = list(languages)
        self.gpu = gpu
        self._reader = None

    def _get(self):
        if self._reader is None:
            import easyocr
            gpu = self.gpu
            if gpu is None:
                try:
                    import torch
                    gpu = bool(torch.cuda.is_available())
                except Exception:
                    gpu = False
            self._reader = easyocr.Reader(self.languages, gpu=gpu, verbose=False)
        return self._reader

    def read(self, image: np.ndarray) -> str:
        results = self._get().readtext(image, detail=1, paragraph=False)
        # results: [(bbox, text, conf), ...]; sort top-to-bottom then left-to-right
        results.sort(key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))
        return " ".join(r[1] for r in results if r[2] >= 0.2)


def read_dialogue(extractor: TextExtractor, frame: np.ndarray, cfg: Config) -> str:
    band_text = extractor.read(prep(crop_band(frame, cfg.band_fraction), cfg.ocr_upscale))
    if normalize(band_text):
        return band_text
    return extractor.read(prep(frame, 1.0))
```

If EasyOCR could not be installed (Task 1 Step 5), use this drop-in instead and name it the same in `pipeline.py`:
```python
class RapidOCRExtractor:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR
        self._ocr = RapidOCR()

    def read(self, image: np.ndarray) -> str:
        result, _ = self._ocr(image)
        if not result:
            return ""
        result.sort(key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))
        return " ".join(r[1] for r in result)
```

- [ ] **Step 4: run tests (slow one included)**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest tests/test_ocr.py -q
```
Expected: `3 passed` (first run 1-3 min for model download).

- [ ] **Step 5: commit**

```bash
cd /c/Users/Asus/Quest1 && git add -A && git commit -m "feat: OCR extractor with band-first reading"
```

---

### Task 6: scanner (coarse) + refiner (binary search)

**Files:**
- Create: `backend/dialogue_finder/scanner.py`, `backend/dialogue_finder/refiner.py`
- Test: `backend/tests/test_scanner.py`, `backend/tests/test_refiner.py`

**Interfaces:**
- Produces: `coarse_scan(source, extractor, target, start_s, end_s, fps, cfg, reporter)->list[Candidate]` (every sampled frame, ascending index); `group_hits(cands, threshold, gap_s)->list[list[Candidate]]`; `pick_group(groups, occurrence)->list[list[Candidate]]`; `first_true(lo, hi, pred)->int`; `refine_first_frame(source, extractor, target, hit_index, prev_index, cfg)->Candidate`; `classify_appearance(source, index, cfg)->str`.

- [ ] **Step 1: write failing tests**

`backend/tests/test_scanner.py`:
```python
from dialogue_finder.config import DEFAULT
from dialogue_finder.models import Candidate
from dialogue_finder.progress import NullReporter
from dialogue_finder.scanner import coarse_scan, group_hits, pick_group


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
```

`backend/tests/test_refiner.py`:
```python
from dialogue_finder.refiner import first_true


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
```

- [ ] **Step 2: run to verify failure**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest tests/test_scanner.py tests/test_refiner.py -q
```
Expected: ImportError.

- [ ] **Step 3: implement scanner.py**

```python
from __future__ import annotations

from typing import Callable

from .config import Config
from .matcher import score_contains
from .models import Candidate, StageEvent
from .ocr import read_dialogue
from .progress import ProgressReporter


def coarse_scan(source, extractor, target: str, start_s: float, end_s: float, fps: float, cfg: Config,
                reporter: ProgressReporter, read: Callable = read_dialogue) -> list[Candidate]:
    """OCR every (source.fps / fps)-th frame in [start_s, end_s]; return one Candidate per sampled frame."""
    step = max(1, int(round(source.fps / fps)))
    a, b = source.index_for_time(start_s), source.index_for_time(end_s)
    total = max(1, (b - a) // step + 1)
    out: list[Candidate] = []
    best = 0.0
    for n, (i, frame) in enumerate(source.iter_range(a, b, step)):
        text = read(extractor, frame, cfg)
        s = score_contains(target, text)
        best = max(best, s)
        out.append(Candidate(i, source.time_for_index(i), text, s))
        if n % 10 == 0 or s >= cfg.ocr_match_threshold:
            reporter.emit(StageEvent("scan", "running", f"frame {i} score {s:.2f} (best {best:.2f})",
                                     min(1.0, (n + 1) / total),
                                     {"frame_index": i, "score": s, "text": text, "best": best}))
    return out


def group_hits(cands: list[Candidate], threshold: float, gap_s: float) -> list[list[Candidate]]:
    hits = [c for c in cands if c.score >= threshold]
    groups: list[list[Candidate]] = []
    for c in hits:
        if groups and c.timestamp_s - groups[-1][-1].timestamp_s <= gap_s:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def pick_group(groups: list[list[Candidate]], occurrence: str) -> list[list[Candidate]]:
    if not groups:
        return []
    if occurrence == "last":
        return [groups[-1]]
    if occurrence == "all":
        return groups
    return [groups[0]]
```

- [ ] **Step 4: implement refiner.py**

```python
from __future__ import annotations

from typing import Callable

import cv2

from .config import Config
from .matcher import score_contains
from .models import Candidate
from .ocr import crop_band, read_dialogue


def first_true(lo: int, hi: int, pred: Callable[[int], bool]) -> int:
    """Smallest i in (lo, hi] with pred(i) True. Contract: pred(lo) is False, pred(hi) is True."""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return hi


def refine_first_frame(source, extractor, target: str, hit_index: int, prev_index: int, cfg: Config,
                       read: Callable = read_dialogue) -> Candidate:
    """Binary-search OCR score between prev_index (no match) and hit_index (match) → exact first frame."""
    cache: dict[int, tuple[str, float]] = {}

    def look(i: int) -> tuple[str, float]:
        if i not in cache:
            text = read(extractor, source.frame_at(i), cfg)
            cache[i] = (text, score_contains(target, text))
        return cache[i]

    lo = max(-1, prev_index)
    first = first_true(lo, hit_index, lambda i: look(i)[1] >= cfg.ocr_match_threshold)
    text, score = look(first)
    return Candidate(first, source.time_for_index(first), text, score)


def _edge_density(frame, cfg: Config) -> float:
    band = crop_band(frame, cfg.band_fraction)
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    return float(cv2.Canny(gray, 100, 200).mean() / 255.0)


def classify_appearance(source, index: int, cfg: Config) -> str:
    """'pop-in' if the subtitle band's edge density jumps at `index`, else 'fade-in'."""
    if index <= 0:
        return "pop-in"
    before = _edge_density(source.frame_at(index - 1), cfg)
    at = _edge_density(source.frame_at(index), cfg)
    return "pop-in" if at > max(0.002, before * 2.0) else "fade-in"
```

- [ ] **Step 5: run tests**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest tests/test_scanner.py tests/test_refiner.py -q
```
Expected: `6 passed`.

- [ ] **Step 6: commit**

```bash
cd /c/Users/Asus/Quest1 && git add -A && git commit -m "feat: coarse OCR scan, hit grouping, binary-search refiner"
```

---

### Task 7: pipeline + CLI (OCR-only path end-to-end, ground-truth test)

**Files:**
- Create: `backend/dialogue_finder/pipeline.py`, `backend/dialogue_finder/cli.py`, `backend/dialogue_finder/__main__.py`
- Test: `backend/tests/test_pipeline.py`, `backend/tests/test_cli.py`
- Modify: `docs/APPROACH.md` (Phase 3 measured timing)

**Interfaces:**
- Produces: `run(source_spec: str, target: str, *, cfg=DEFAULT, reporter=NullReporter(), mode="hybrid", occurrence="first", local=False, extractor=None, locator=None) -> Result`; `class PipelineError(Exception)`; `confidence_for(source, ocr_score, window)->str`; `cli.main(argv)->int`.
- Consumes: everything from Tasks 2-6. `locator` is any object with `locate(video_path, target) -> Window | None` (Task 8 provides `WhisperLocator`; until then `None` means "skip audio").

- [ ] **Step 1: write failing tests**

`backend/tests/test_pipeline.py`:
```python
import pytest

from dialogue_finder.config import DEFAULT
from dialogue_finder.models import Window
from dialogue_finder.pipeline import confidence_for, run


def test_confidence_rules():
    assert confidence_for("ocr", 0.95, Window(1, 2, 0.9, "x")) == "HIGH"
    assert confidence_for("ocr", 0.95, None) == "HIGH"
    assert confidence_for("ocr", 0.82, None) == "MEDIUM"
    assert confidence_for("audio", 0.0, Window(1, 2, 0.9, "x")) == "MEDIUM"
    assert confidence_for("ocr-weak", 0.4, None) == "LOW"


@pytest.mark.slow
def test_ground_truth_exact_frame(synthetic_clip, tmp_path):
    path, truth = synthetic_clip
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    res = run(str(path), truth["text"], cfg=cfg, mode="ocr", local=True)
    assert res.source == "ocr"
    assert res.frame_index == truth["frame"], (res.frame_index, truth["frame"], res.candidates[-3:])
    assert abs(res.timestamp_s - truth["timestamp"]) < 1e-6
    assert (tmp_path / "out").joinpath(f"frame_{truth['frame']}.png").exists()
    assert (tmp_path / "out").joinpath(f"frame_{truth['frame'] - 1}.png").exists()
    assert res.appearance == "pop-in"


@pytest.mark.slow
def test_ground_truth_fade_in(tmp_path):
    from bench.make_clip import make_clip
    clip = tmp_path / "fade.mp4"
    truth = make_clip(clip, text="My mind rebels at stagnation", appear_s=4.0, fade_frames=12)
    cfg = DEFAULT.__class__(output_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    res = run(str(clip), truth["text"], cfg=cfg, mode="ocr", local=True)
    assert res.source == "ocr"
    assert 0 <= res.frame_index - truth["frame"] <= 12     # within the fade
    assert res.appearance == "fade-in"


def test_bad_local_path_raises_pipeline_error(tmp_path):
    from dialogue_finder.pipeline import PipelineError
    with pytest.raises(PipelineError):
        run(str(tmp_path / "missing.mp4"), "x", mode="ocr", local=True)
```

`backend/tests/test_cli.py`:
```python
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
```

- [ ] **Step 2: run to verify failure**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest tests/test_pipeline.py tests/test_cli.py -q
```
Expected: ImportError.

- [ ] **Step 3: implement pipeline.py**

```python
from __future__ import annotations

import time
from pathlib import Path

import cv2

from .config import DEFAULT, Config
from .downloader import DownloadError, fetch_video, probe
from .frame_source import FrameSource
from .models import Candidate, Result, StageEvent, Window
from .progress import NullReporter, ProgressReporter
from .refiner import classify_appearance, refine_first_frame
from .scanner import coarse_scan, group_hits, pick_group


class PipelineError(Exception):
    """Fatal, user-facing. The CLI prints str(e) and exits 1."""


def confidence_for(source: str, ocr_score: float, window: Window | None) -> str:
    if source == "ocr":
        return "HIGH" if ocr_score >= 0.9 else "MEDIUM"
    if source == "audio":
        return "MEDIUM"
    return "LOW"


def _save_frames(src: FrameSource, index: int, cfg: Config) -> tuple[str, str]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.output_dir / f"frame_{index}.png"
    cv2.imwrite(str(p), src.frame_at(index))
    prev = ""
    if index > 0:
        pp = cfg.output_dir / f"frame_{index - 1}.png"
        cv2.imwrite(str(pp), src.frame_at(index - 1))
        prev = str(pp)
    return str(p), prev


def _default_extractor():
    from .ocr import EasyOCRExtractor
    return EasyOCRExtractor()


def run(source_spec: str, target: str, *, cfg: Config = DEFAULT, reporter: ProgressReporter | None = None,
        mode: str = "hybrid", occurrence: str = "first", local: bool = False,
        extractor=None, locator=None) -> Result:
    reporter = reporter or NullReporter()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # ---- download / open --------------------------------------------------
    try:
        if local:
            video = Path(source_spec)
            if not video.exists():
                raise DownloadError(f"File not found: {video}")
            reporter.emit(StageEvent("download", "ok", f"using local file {video.name}", 1.0))
        else:
            video = fetch_video(source_spec, cfg, reporter)
        info = probe(video)
    except DownloadError as e:
        reporter.emit(StageEvent("error", "error", str(e)))
        raise PipelineError(str(e)) from e
    timings["download"] = time.perf_counter() - t0

    # ---- locate by audio ----------------------------------------------------
    window: Window | None = None
    t1 = time.perf_counter()
    if mode in ("hybrid", "audio"):
        if locator is None and mode == "hybrid":
            try:
                from .audio_locator import WhisperLocator
                locator = WhisperLocator(cfg, reporter)
            except Exception as e:                      # module missing until Task 8, or import failure
                reporter.emit(StageEvent("locate", "skipped", f"audio locator unavailable: {e}"))
        if locator is not None:
            try:
                reporter.emit(StageEvent("transcribe", "running", "transcribing audio"))
                window = locator.locate(video, target)
                if window is None:
                    reporter.emit(StageEvent("locate", "fallback", "no audio match; will scan whole video"))
                else:
                    reporter.emit(StageEvent("locate", "ok", f"window {window.start_s:.1f}-{window.end_s:.1f}s "
                                             f"score {window.score:.2f}: '{window.matched_text}'", 1.0,
                                             {"window": window.__dict__}))
            except Exception as e:
                reporter.emit(StageEvent("locate", "fallback", f"audio stage failed: {e}"))
                window = None
    else:
        reporter.emit(StageEvent("locate", "skipped", f"mode={mode}"))
    timings["locate"] = time.perf_counter() - t1

    with FrameSource(video) as src:
        # ---- audio-only mode -------------------------------------------------
        if mode == "audio":
            if window is None:
                raise PipelineError("No audio match found for the target text (mode=audio).")
            idx = src.index_for_time(window.start_s)
            img, prev = _save_frames(src, idx, cfg)
            timings["total"] = time.perf_counter() - t0
            reporter.emit(StageEvent("done", "ok", "audio-only result"))
            return Result(src.time_for_index(idx), idx, window.matched_text, "MEDIUM", "audio",
                          "mode=audio; frame at first spoken word", src.fps, img, prev, "", window, [], [], timings)

        # ---- scan -------------------------------------------------------------
        ex = extractor or _default_extractor()
        t2 = time.perf_counter()
        if window is not None:
            a, b = max(0.0, window.start_s - cfg.window_pad_s), min(src.duration_s, window.end_s + cfg.window_pad_s)
            fps = cfg.window_fps
            reporter.emit(StageEvent("scan", "running", f"OCR {a:.1f}-{b:.1f}s at {fps} fps"))
        else:
            a, b, fps = 0.0, src.duration_s, cfg.fullscan_fps
            reporter.emit(StageEvent("scan", "running" if mode == "ocr" else "fallback",
                                     f"OCR whole video ({b:.0f}s) at {fps} fps"))
        cands = coarse_scan(src, ex, target, a, b, fps, cfg, reporter)
        groups = pick_group(group_hits(cands, cfg.ocr_match_threshold, cfg.hit_gap_s), occurrence)
        if not groups and window is not None and mode == "hybrid":
            # the window missed; one more try over the whole video before giving up on OCR
            reporter.emit(StageEvent("scan", "fallback", "no match in window; scanning whole video"))
            cands = coarse_scan(src, ex, target, 0.0, src.duration_s, cfg.fullscan_fps, cfg, reporter)
            groups = pick_group(group_hits(cands, cfg.ocr_match_threshold, cfg.hit_gap_s), occurrence)
        timings["scan"] = time.perf_counter() - t2
        step = max(1, int(round(src.fps / fps)))

        # ---- refine -----------------------------------------------------------
        t3 = time.perf_counter()
        if groups:
            refined: list[Candidate] = []
            for g in groups:
                hit = g[0]
                refined.append(refine_first_frame(src, ex, target, hit.frame_index, hit.frame_index - step, cfg))
            best = refined[0]
            reporter.emit(StageEvent("refine", "ok", f"first frame {best.frame_index} score {best.score:.2f}", 1.0,
                                     {"frame_index": best.frame_index}))
            appearance = classify_appearance(src, best.frame_index, cfg)
            img, prev = _save_frames(src, best.frame_index, cfg)
            timings["refine"] = time.perf_counter() - t3
            timings["total"] = time.perf_counter() - t0
            reporter.emit(StageEvent("done", "ok", "result ready"))
            return Result(best.timestamp_s, best.frame_index, best.text, confidence_for("ocr", best.score, window),
                          "ocr", "", src.fps, img, prev, appearance, window, cands, refined[1:], timings)

        # ---- fallbacks ----------------------------------------------------------
        if window is not None:
            idx = src.index_for_time(window.start_s)
            img, prev = _save_frames(src, idx, cfg)
            reporter.emit(StageEvent("refine", "fallback", "no on-screen match; using audio timestamp"))
            timings["total"] = time.perf_counter() - t0
            reporter.emit(StageEvent("done", "ok", "audio-fallback result"))
            return Result(src.time_for_index(idx), idx, window.matched_text, "MEDIUM", "audio",
                          "no on-screen text matched; frame at first spoken word", src.fps, img, prev, "",
                          window, cands, [], timings)
        weak = max(cands, key=lambda c: c.score) if cands else Candidate(0, 0.0, "", 0.0)
        img, prev = _save_frames(src, weak.frame_index, cfg)
        reporter.emit(StageEvent("refine", "fallback", f"no match anywhere; best effort frame {weak.frame_index}"))
        timings["total"] = time.perf_counter() - t0
        reporter.emit(StageEvent("done", "ok", "low-confidence result"))
        return Result(weak.timestamp_s, weak.frame_index, weak.text, "LOW", "ocr-weak",
                      f"best OCR similarity only {weak.score:.2f}", src.fps, img, prev, "", None, cands, [], timings)
```

- [ ] **Step 4: implement cli.py and __main__.py**

`backend/dialogue_finder/cli.py`:
```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT, Config
from .progress import PrintReporter


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dialogue_finder",
                                description="Find the first frame where a dialogue appears in a video.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="video URL (any yt-dlp supported site)")
    src.add_argument("--local", help="path to a local video file")
    p.add_argument("--text", required=True, help='target dialogue, e.g. "My mind rebels at stagnation"')
    p.add_argument("--mode", choices=["hybrid", "audio", "ocr"], default="hybrid")
    p.add_argument("--occurrence", choices=["first", "last", "all"], default="first")
    p.add_argument("--out", default="output", help="output directory (default: output)")
    p.add_argument("--json", action="store_true", help="also print result.json content to stdout")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:               # argparse exits 2 on usage error
        return int(e.code or 2)
    cfg = Config(output_dir=Path(args.out), cache_dir=DEFAULT.cache_dir)
    from .pipeline import PipelineError, run
    try:
        res = run(args.url or args.local, args.text, cfg=cfg, reporter=PrintReporter(args.verbose),
                  mode=args.mode, occurrence=args.occurrence, local=args.local is not None)
    except PipelineError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Error: interrupted", file=sys.stderr)
        return 130
    except Exception as e:                # last line of defence: never a traceback
        print(f"Error: unexpected failure ({type(e).__name__}: {str(e)[:200]})", file=sys.stderr)
        return 1
    print(res.format_block())
    for alt in res.alternatives:
        print(f"Also at   : {alt.frame_index} ({alt.timestamp_s:.3f}s) score {alt.score:.2f}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "result.json").write_text(json.dumps(res.to_dict(), indent=2, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, default=str))
    return 0
```

`backend/dialogue_finder/__main__.py`:
```python
import sys

from .cli import main

sys.exit(main())
```

- [ ] **Step 5: run tests**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest -q
```
Expected: all pass (≈ 28 tests; slow ones take 1-3 min). If `test_ground_truth_exact_frame` is off by one, the bug is in `FrameSource._seek` counting — print `res.candidates[-5:]` and `cache` from the refiner to see which side is wrong.

- [ ] **Step 6: run on the real video, OCR-only (measures the slow path)**

```bash
cd /c/Users/Asus/Quest1/backend && time ../.venv/Scripts/python -m dialogue_finder --url "https://ok.ru/video/248244667877" --text "My mind rebels at stagnation" --mode ocr --verbose
```
Expected: 5-line block + `Previous` line, `output/frame_<n>.png` and `frame_<n-1>.png`. Record in `docs/APPROACH.md` Phase 3: wall time, GPU or CPU, route taken, frame found. Open both PNGs with the Read tool and confirm n−1 has no text and n has it (or, if the video has no burned-in subtitles, record that the result is `ocr-weak` and why).

- [ ] **Step 7: commit**

```bash
cd /c/Users/Asus/Quest1 && git add -A && git commit -m "feat: pipeline orchestration, CLI, ground-truth tests, first real-video run"
```

---

### Task 8: audio locator (Whisper) + hybrid path

**Files:**
- Create: `backend/dialogue_finder/audio_locator.py`
- Test: `backend/tests/test_audio_locator.py`
- Modify: `docs/APPROACH.md` (Phase 3 measured fast-path timing)

**Interfaces:**
- Produces: `class Locator(Protocol): locate(video: Path, target: str) -> Window | None`; `class WhisperLocator(cfg, reporter)`; `extract_audio(video, wav)->Path`; `transcribe_words(wav, model, task, reporter)->list[Word]`; `words_cache_path(video, cfg)->Path`.
- Consumes: `best_word_window` (Task 2), `ensure_ffmpeg` (Task 3).

- [ ] **Step 1: write failing test (unit test on the window logic with a fake transcript + one slow smoke test)**

`backend/tests/test_audio_locator.py`:
```python
import json

import pytest

from dialogue_finder.config import DEFAULT
from dialogue_finder.audio_locator import WhisperLocator, words_cache_path
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
    from dialogue_finder.downloader import ensure_ffmpeg
    from dialogue_finder.audio_locator import transcribe_words
    ensure_ffmpeg()
    wav = tmp_path / "s.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "3", str(wav)],
                   check=True, capture_output=True)
    words = transcribe_words(wav, "base", "translate", NullReporter())
    assert isinstance(words, list)
```

- [ ] **Step 2: run to verify failure**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest tests/test_audio_locator.py -q
```
Expected: ImportError.

- [ ] **Step 3: implement audio_locator.py**

```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Protocol

from .config import Config
from .downloader import ensure_ffmpeg
from .matcher import best_word_window
from .models import StageEvent, Window, Word
from .progress import ProgressReporter


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


def _load_model(name: str):
    from faster_whisper import WhisperModel
    try:
        return WhisperModel(name, device="cuda", compute_type="float16"), "cuda"
    except Exception:
        return WhisperModel(name, device="cpu", compute_type="int8"), "cpu"


def transcribe_words(wav: Path, model_name: str, task: str, reporter: ProgressReporter) -> list[Word]:
    model, device = _load_model(model_name)
    reporter.emit(StageEvent("transcribe", "running", f"whisper {model_name} on {device}", 0.0))
    segments, info = model.transcribe(str(wav), task=task, word_timestamps=True, vad_filter=True)
    words: list[Word] = []
    total = getattr(info, "duration", 0) or 0
    for seg in segments:
        for w in (seg.words or []):
            words.append(Word(w.word.strip(), float(w.start), float(w.end)))
        reporter.emit(StageEvent("transcribe", "running", seg.text.strip()[:80],
                                 (seg.end / total) if total else None,
                                 {"start": seg.start, "end": seg.end, "text": seg.text.strip()}))
    reporter.emit(StageEvent("transcribe", "ok", f"{len(words)} words, language {getattr(info, 'language', '?')}", 1.0))
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
```

- [ ] **Step 4: run tests**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest tests/test_audio_locator.py -q
```
Expected: `2 passed` (first run downloads Whisper base ≈ 150 MB).

- [ ] **Step 5: run the hybrid path on the real video and compare with Task 7's frame**

```bash
cd /c/Users/Asus/Quest1/backend && time ../.venv/Scripts/python -m dialogue_finder --url "https://ok.ru/video/248244667877" --text "My mind rebels at stagnation" --verbose
```
Expected: `[locate:ok] window ...` then a short scan, same `Frame` as Task 7 (or, if the video has no subtitles, `source: audio` with the spoken-word frame). Record in APPROACH.md Phase 3: transcription time, device, window found, total time vs OCR-only time.

- [ ] **Step 6: commit**

```bash
cd /c/Users/Asus/Quest1 && git add -A && git commit -m "feat: whisper audio locator with transcript cache; hybrid path measured"
```

---

### Task 9: benchmark, real-video matrix, never-crash pass

**Files:**
- Create: `backend/bench/run_bench.py`, `docs/BENCHMARK.md` (generated)
- Test: `backend/tests/test_never_crash.py`
- Modify: `docs/APPROACH.md` (Phase 4)

**Interfaces:**
- Produces: `run_bench.py` writing a markdown table; `python -m bench.run_bench` from `backend/`.

- [ ] **Step 1: never-crash tests**

`backend/tests/test_never_crash.py`:
```python
from dialogue_finder.cli import main


def test_unreachable_url_is_clean(capsys):
    code = main(["--url", "https://example.invalid/video/1", "--text", "x", "--mode", "ocr"])
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error:") and "Traceback" not in err


def test_corrupt_file_is_clean(capsys, tmp_path):
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"not a video")
    code = main(["--local", str(bad), "--text", "x", "--mode", "ocr"])
    assert code == 1
    assert "Traceback" not in capsys.readouterr().err
```

Run: `cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest tests/test_never_crash.py -q` → expected `2 passed` (if a traceback leaks, the exception type is not caught in `cli.main` — it is, so a failure here means output went to stdout: check `format_block` isn't reached).

- [ ] **Step 2: run_bench.py**

```python
"""Run the pipeline (OCR mode) on synthetic variants; write docs/BENCHMARK.md with frame error per variant."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.make_clip import make_clip  # noqa: E402
from dialogue_finder.config import Config  # noqa: E402
from dialogue_finder.ocr import EasyOCRExtractor  # noqa: E402
from dialogue_finder.pipeline import run  # noqa: E402

TEXT = "My mind rebels at stagnation"
VARIANTS = {
    "baseline_640x360_24fps_bottom": dict(),
    "top_position": dict(position="top"),
    "center_position": dict(position="center"),
    "fade_12_frames": dict(fade_frames=12),
    "small_text_360p": dict(scale=0.6),
    "hd_1280x720": dict(size=(1280, 720)),
    "30fps": dict(fps=30),
    "60fps": dict(fps=60),
}


def main() -> None:
    out_dir = Path("bench_out"); out_dir.mkdir(exist_ok=True)
    ex = EasyOCRExtractor()
    rows = []
    for name, kw in VARIANTS.items():
        clip = out_dir / f"{name}.mp4"
        truth = make_clip(clip, text=TEXT, appear_s=5.0, **kw)
        cfg = Config(output_dir=out_dir / name, cache_dir=out_dir / "cache")
        t = time.perf_counter()
        try:
            res = run(str(clip), TEXT, cfg=cfg, mode="ocr", local=True, extractor=ex)
            err = res.frame_index - truth["frame"]
            rows.append((name, truth["frame"], res.frame_index, err, res.source, res.confidence, f"{time.perf_counter() - t:.1f}"))
        except Exception as e:
            rows.append((name, truth["frame"], "-", "-", f"error: {e}", "-", "-"))
    md = ["# Benchmark (synthetic ground truth, OCR mode)", "",
          "| variant | truth frame | found frame | error (frames) | source | confidence | seconds |",
          "|---|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    Path("../docs/BENCHMARK.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
```

Run: `cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m bench.run_bench` → expected a table with error 0 for pop-in variants and 0-12 for fade. Fix cheap failures (e.g. `center_position` needs the full-frame fallback in `read_dialogue`, already there); otherwise document the limit in APPROACH.md Phase 5.

- [ ] **Step 3: real-video matrix**

Pick a YouTube clip with burned-in English captions and one with speech but no captions (search "movie scene english subtitles" / any interview clip). Run each with `--verbose`, and fill a table in `docs/APPROACH.md` Phase 4: video, has subs?, route (`locate` ok/fallback, `scan` window/whole, `source`), frame, time, correct by eye (open the PNG).

- [ ] **Step 4: commit**

```bash
cd /c/Users/Asus/Quest1 && git add -A && git commit -m "test: never-crash cases; bench: synthetic variants table; real-video matrix in APPROACH"
```

---

### Task 10: Plan-1 docs pass

**Files:**
- Modify: `README.md`, `docs/APPROACH.md`, `docs/DECISIONS.md`, `prompts.txt`

- [ ] **Step 1: README (CLI section)**

```markdown
# Dialogue Frame Finder

Given a video URL and a line of dialogue, finds the **first frame** where that dialogue appears on screen.

## Run (CLI)
    py -3.14 -m venv .venv && .venv\Scripts\pip install -r requirements.txt
    cd backend
    ..\.venv\Scripts\python -m dialogue_finder --url "https://ok.ru/video/248244667877" --text "My mind rebels at stagnation"

Output:
    Timestamp : HH:MM:SS.sss
    Frame     : <n>            (0-based)
    Text      : "<extracted text>"
    Confidence: HIGH|MEDIUM|LOW  (source: ocr|audio|ocr-weak)
    Image     : output/frame_<n>.png
    Previous  : output/frame_<n-1>.png  (pop-in|fade-in)

Flags: `--local <file>`, `--mode hybrid|audio|ocr`, `--occurrence first|last|all`, `--verbose`, `--json`, `--out <dir>`.
First run downloads models (~250 MB) and the video (~400 MB); later runs use `cache/`.
Measured times: see docs/APPROACH.md Phase 3. GPU is optional (auto-detected).

## How it works (one paragraph)
Audio says *where* (Whisper transcript, fuzzy-matched → ~10 s window), OCR says *which second*
(EasyOCR on the subtitle band at 5 fps), binary search says *which frame* (OCR score between the last
non-matching and first matching sample). No audio match → whole-video OCR at 2 fps. No OCR match → frame
at the first spoken word, with the transcript words as the text. Always produces a result; never a traceback.

## Docs
- docs/APPROACH.md — phased design, measurements, limits
- docs/DECISIONS.md — where the human overrode the AI, and why
- docs/BENCHMARK.md — synthetic ground-truth results
- prompts.txt — every AI prompt used, verbatim
```

- [ ] **Step 2: APPROACH.md — fill Phases 1-5 from what was recorded in Tasks 3, 7, 8, 9** (facts and numbers only; each of the four evaluation bullets — where to look, which frame, how text is extracted, ambiguity — gets its own sub-heading under Phase 2). Phase 5 lists limits found (fade-in ±N frames, stylised fonts, dubbed audio) and extensions not built (multi-language, scene detection, hosting).

- [ ] **Step 3: DECISIONS.md — one entry per task where the plan changed while building** (e.g. SEEK_BACK value, EasyOCR vs RapidOCR, CUDA availability). `prompts.txt` — append every prompt used during execution.

- [ ] **Step 4: full test run + commit**

```bash
cd /c/Users/Asus/Quest1/backend && ../.venv/Scripts/python -m pytest -q && cd .. && git add -A && git commit -m "docs: README CLI usage, APPROACH phases 1-5, decisions, prompts"
```

---

## Self-review (done while writing)

- Spec coverage: download/cache/--local (T3, T7), audio locate + translate + cache (T8), coarse OCR band-first (T5, T6), exact first frame (T6 refiner), fallbacks both ways + `Text` rule (T7), 0-based frames + timestamp format (T2, T4), evidence pair n−1/n + pop-in/fade label (T6, T7), `--mode`/`--occurrence` seams (T7), never-crash (T7, T9), synthetic ground truth + benchmark (T4, T7, T9), real-video matrix (T9), DECISIONS/APPROACH/README/prompts (T1, T10). API + UI → Plan 2. Docker/hosting/multi-language: out of scope by spec.
- Placeholders: none; every code step has code.
- Type consistency: `Candidate(frame_index, timestamp_s, text, score)`, `Window(start_s, end_s, score, matched_text)`, `Result(...)` positional order used identically in T7's three `Result(...)` constructions; `read_dialogue(extractor, frame, cfg)` signature used by scanner/refiner/tests; `first_true(lo, hi, pred)` contract documented.

---

## Appendix A — prompts.txt seed (verbatim user prompts to Claude Code, brainstorming session, 2026-08-24)

```
# prompts.txt — every prompt given to an AI assistant while building this project, in order.
# Tool: Claude Code (model: Claude Fable 5), plugins: superpowers (brainstorming, writing-plans), advisor.
# Session 1 — brainstorming (2026-08-24)

[1]
/superpowers:brainstorming - C:\Users\Asus\Downloads\Problem statement - My mind rebels at stagnation.pdf - So this was the problem statement given to me by a company So I am planning to finish this task but right now let's keep this whole session only for brainstorming. C:\Users\Asus\Downloads\NOtes from the company.txt - So I was present in the pre-placement talk by the company. So this is what notes I took So I think there are very much useful stuff in this like what they like and what constraints are there.   Right now, what I've thought of is this workflow, but I know that like my brain is pretty disorganized, and I don't know if I'm going in the right path. You're obviously a better model, and you think about this and see if I'm right. But I think there is a much better way of doing this - Inputs:

1. Video URL
2. Target dialogue
   Example: "My mind rebels at stagnation"

We will interpret "on-screen dialogue" as the dialogue associated with the actors visible in the video — i.e., we want to identify when the actors speak the target dialogue and locate the corresponding video frame.

So that's probably gonna be like a million questions that arise like how are we gonna get the video from my URL and how are we gonna like extract the audio from it and transcribe it or like whatever approach that you want to take that we can take the the thing is in this brainstorming session I want us to come up with a clear idea of what we are gonna do and like like what approach we are gonna take and how we are gonna do things like how are we gonna get the video like how are we gonna like build I want us to think about like the whole architecture and the whole pipeline of this entire pipeline. So, this whole thing is just to help me think of everything and help me get my brain sorted and to come up with a good plan of what we are gonna do, and for me to have a clear idea and answer: like, like when I did thinking of myself, I had like 20 different questions like of what I'm gonna do. I want you to think of more than 20, 30, more than every questions, like how muchever you can, and then like discuss with me, and we can go back and forth, and we can see what we can do. And I'm kind of a beginner to this whole transcribing audio and this whole thing. So, you will be helping me with recommending stuff and helping me use my brain and choose the best option available

[2]
So yeah, that's the problem. We don't know what on-screen dialogue actually means like I don't know what they meant. So like it could be like you know embedded burnt in a subtitle on the screen like a text on the video itself or it may be like a person actually talking a dialogue I don't know like what do you think after analyzing the problem statement given in the PDF that I gave you what do you think they want and even if we are making a program or a product for that what approach do you think will satisfy all of the doubts that we have

[3]
Okay, let's say that the burnt and subtitle is for like translating a language. So let's say that in the Godfather he says something in Italian and they want us to know that what was said in Italian to English. So they burnt in a subtitle of whatever they're speaking in the video. So with your approach like when we transcribe the whole movie and we see the transcribed audio text and then we try to locate what was said in English that Italian translated thing it we can't find it because like originally the audio transcribing thing it'll be it will not be there right it will be in Italian right so what are we gonna do about that. Also like what if like it's just that like let's say the godfather is worried in the subtitle like the burnt subtitle but he never says anything right like like the dialogue where it's not spoken but it's just shown on the screen what are we gonna do about that

[4]
Okay, but don't you think this hybrid approach will take like a lot of time

[5]
Okay, can you use the advisor and ask it whatever is good like can you ask like general advice of whatever you're doing is okay or is it gonna push back ask it is there any pushbacks that are needed

[6] (answers to the assistant's questions)
VPN: ok.ru not Being able to reach was because of my internet problem until about that like my internet out that time
Interface: Wonderful UA with Next.js and pretty much a really pretty UI with use all the fend and skills that is needed for that and the interviewer should be able to give like a URL and the whole pipeline should you know just work like in front of them like all the visuals of what's happening and with like in the loading screen and all it should show and everything I want it like that not only this one

[7] (answers to the assistant's questions)
Hosting: Runs locally, one start script
Deadline: Within 3 days

[8]
give this plan to the advisor and ask what changes can be done.. don't worry about the time. consider u have like 10 days

[9]
Can you make, like, a small architecture diagram of whatever we are building? and give it to me... show it to me, like, like, a simple one, but with a lot of details. use whatever tool u want to draw

[10]
Okay, so I like the approach. The plan is fine. Okay, here's a problem, right? So, like, all my competitions, everyone is gonna use claude to finish this task. So, everyone's claude is gonna give the exact same approach, and like you know, what's very much similar approaches of what's gonna be done right? So, how are you the best claude that is compared to around 100 other cloths of other competitors like other students that are doing this task? What do you think you can do that no other plot could ever think of that would impress the interviewer that he just so he sees it and just takes me in for a job like figure it out what it is, think of it like think what can be done on this thing to help me be different and be much better and have like a unique selling point that no one else or no one claude thought of it

[11]
Okay, so without making the whole thing very much complex and hard for me to explain to the interviewer, like whatever thing you can fit into the plan, you put it into the plan

[12]
Okay, so why not we do it in the same session? I put it in the bypass permissions on. So, yeah, let's go on and start implementing it. But what I want you to do is follow the plan and like cover often, like have a local gate itself file. It's not put into GitHub right now. Let's have a local git. Okay, okay, what do you think? I just typed superpowers writing plans here itself and we just go on here. Shall I do that?
```
