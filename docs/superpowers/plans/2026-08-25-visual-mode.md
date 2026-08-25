# Visual Verification Mode ("hybrid" = audio + OCR + active speaker) — Plan 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth mode, `hybrid`, that verifies each spoken occurrence of the dialogue with on-screen text (OCR) **and** a visible speaking face (YuNet face tracks + LR-ASD), classifies every occurrence as valid / uncertain / invalid, picks the best, and refines the frame to the visual speech onset — without changing the three existing modes (`audio`, `ocr`, and the old hybrid, now named `audio+ocr`).

**Architecture:** `WhisperLocator.locate_all()` returns every transcript window ≥ 0.6. For each window the pipeline runs the existing OCR scan, builds face tracks (YuNet + IoU tracker) over the padded window, scores each track with LR-ASD against the window's MFCC, and `visual.verifier.classify()` applies the spec's rules. Selection is `valid > uncertain > invalid`, then ASR score, then earliest; the frame is the OCR first frame, the visual onset, or the first spoken word. LR-ASD never runs outside candidate windows. Without the ASD extras the mode degrades to `audio+ocr` behaviour with a `verify: skipped` event.

**Tech Stack:** existing stack + `cv2.FaceDetectorYN` (OpenCV core, YuNet ONNX ≈ 230 KB), LR-ASD (vendored PyTorch model, MIT, weights ≈ 3 MB; torch CPU wheel via `requirements-asd.txt`), `python_speech_features` (MFCC), Silero VAD already bundled in faster-whisper.

**Spec:** `docs/superpowers/specs/2026-08-25-visual-mode-design.md` (approved 2026-08-25; default mode = `hybrid`).

## Global Constraints

- Branch `plan-4-visual-mode` from `plan-2-web-ui` (HEAD 2bd2757). Local git only, never push. Commit after every task; code-simplifier pass after each implementer before review; SOLID/DRY/KISS; protocols are the only seams (`FaceDetector`, `SpeakerDetector`).
- Python `.venv/Scripts/python` (3.14, Windows). Tests from `backend/`: `../.venv/Scripts/python -m pytest -q` (74 passing at start; `-m "not slow"` fast subset).
- Modes: `audio`, `ocr`, `audio+ocr` (old hybrid, behaviour byte-identical), `hybrid` (new, default). Measured runs recorded before 2026-08-25 in APPROACH keep the label "hybrid" meaning audio+ocr — add one note, do not rewrite history.
- LR-ASD only inside candidate windows (window ± `window_pad_s`, plus 1 s before for onset search). Torch is CPU-only on Python 3.14/Windows — acceptable (0.84 M params).
- "No usable face" → `uncertain`, never `invalid`. Usable track: ≥ `min_track_s` (0.5 s) and median face height ≥ `min_face_px` (40).
- Thresholds only in `config.py`: `asd_threshold 0.5`, `asd_min_active 0.3`, `asd_onset_frames 3`, `min_track_s 0.5`, `min_face_px 40`, `max_occurrences 5`, `onset_lookback_s 1.0`, `models_dir = cache_dir / "models"`.
- Never a traceback: missing extras, missing weights, no faces, ASD failure → events + degrade, never a crash.
- Every prompt the user types is appended to `prompts.txt`.
- Weight/model downloads: retry with backoff; README documents the manual `curl` fallback into `cache/models/`.

---

## File structure

```
requirements-asd.txt                      # torch (CPU), python_speech_features — optional extras
backend/dialogue_finder/
  config.py                               # + visual knobs (above)
  models.py                               # + FaceTrack, Occurrence; Result + occurrence_class, speaker_box, speaker_image_path, occurrences
  audio/locator.py                        # + all_word_windows() (matcher) and WhisperLocator.locate_all()
  text/matcher.py                         # + all_word_windows(words, target, threshold, cap) -> list[Window]
  pipeline.py                             # mode rename; + _run_hybrid() branch
  cli.py                                  # mode choices; extra output lines
  visual/
    __init__.py                           # "Third kind of evidence: is a visible person speaking?"
    faces.py                              # FaceDetector protocol, YuNetDetector, IouTracker, build_tracks(), crop_face()
    audio_features.py                     # mfcc_100hz(), speech_mask()
    lrasd.py                              # SpeakerDetector protocol, LrAsdDetector (vendored model + weights), availability check
    lrasd_model.py                        # vendored LR-ASD network definition (MIT header) — from the Task 1 spike
    verifier.py                           # classify(), find_onset(), verify_window() — pure, fake-testable
backend/api/jobs.py                       # mode Literal
backend/tests/test_visual_faces.py, test_visual_verifier.py, test_matcher.py (+), test_pipeline.py (+), test_api.py (+)
frontend/index.html, app.js, styles.css   # mode option, Occurrences block, face box image, timeline class marks
docs/superpowers/spikes/2026-08-25-lrasd-spike.md
docs/APPROACH.md (+ Phase 7), docs/DECISIONS.md (+ Plan 4), README.md, prompts.txt
```

---

### Task 1: Spike — can LR-ASD run here, and what exactly does it need? (decision gate)

**Files:**
- Create: `docs/superpowers/spikes/2026-08-25-lrasd-spike.md`, `requirements-asd.txt`, `backend/dialogue_finder/visual/__init__.py`, `backend/dialogue_finder/visual/lrasd_model.py` (vendored, only if the spike succeeds)
- Scratch (git-ignored): `bench_out/lrasd_src/` (clone), `bench_out/spike_asd.py`

**Interfaces:**
- Produces: the facts every later task depends on, written in the spike note under fixed headings: `weights` (file name, URL, size, how loaded), `model_api` (constructor + forward signature of the vendored class), `preprocessing` (crop size, colour, normalisation; audio feature type/dims/rate; frames-per-video-frame ratio; how the reference demo aligns lengths), `torch` (version installed, `torch.load` flags needed), `cost` (seconds per 10 s window on CPU for detection+ASD), `decision` (GO with LR-ASD | FALLBACK to lip-motion heuristic).

- [ ] **Step 1: branch + extras** — `git checkout -b plan-4-visual-mode`; write `requirements-asd.txt`:
```
# Optional. Active-speaker detection for --mode hybrid (LR-ASD, CPU). Install after requirements.txt.
torch>=2.6
python_speech_features==0.6
```
Install: `.venv/Scripts/python -m pip install -r requirements-asd.txt` (torch CPU wheel ≈ 200 MB; run_in_background). Record the installed torch version.
- [ ] **Step 2: vendor** — `git clone --depth 1 https://github.com/Junhua-Liao/LR-ASD bench_out/lrasd_src`. Read `README.md`, the model file(s), the demo/inference script, and the data-loader to extract the preprocessing facts. Copy the network definition into `backend/dialogue_finder/visual/lrasd_model.py` with the MIT licence header and the upstream commit hash in a comment; remove training-only code. Download the pretrained weights the README points to into `cache/models/` (note the exact URL; if `requests` is reset by this network, use `curl -L`).
- [ ] **Step 3: spike script** — `bench_out/spike_asd.py`: open `cache/5f39d4605665a831.mp4`, take frames 7700-7950 (≈ 320.9-331.3 s, the Holmes line), detect faces per frame with `cv2.FaceDetectorYN.create(str(yunet_onnx), "", (w, h))` (download `face_detection_yunet_2023mar.onnx` from the OpenCV Zoo into `cache/models/`), keep the largest face per frame as one track, crop per the reference preprocessing, compute the audio feature from `cache/5f39d4605665a831.16k.wav` for the same span, run the vendored model, print per-frame scores + mean, and `time` each stage. Expect scores clearly above 0.5 while Holmes speaks.
- [ ] **Step 4: decision** — write the spike note with all six headings and the decision. GO if: weights load, scores on the Holmes window are high while he speaks and low elsewhere, and detection+ASD for a 10 s window costs ≤ 60 s on CPU. Otherwise FALLBACK: the note says which criterion failed; Tasks 5-6 then implement the lip-motion heuristic from the spec's §10 behind the same `SpeakerDetector` protocol (the controller re-plans those two tasks).
- [ ] **Step 5: commit** — `git add requirements-asd.txt backend/dialogue_finder/visual docs/superpowers/spikes && git commit -m "spike: LR-ASD feasibility on py3.14 CPU; vendored model + extras"` (weights and clone stay git-ignored under cache/ and bench_out/).

---

### Task 2: Mode rename + multi-window locator + config/models

**Files:**
- Modify: `backend/dialogue_finder/config.py`, `models.py`, `text/matcher.py`, `audio/locator.py`, `pipeline.py`, `cli.py`, `backend/api/jobs.py`, `frontend/index.html` (option text only), tests that use `mode="hybrid"` for the old behaviour, `docs/APPROACH.md` (one history note), `README.md` (mode table)
- Test: `backend/tests/test_matcher.py`, `test_audio_locator.py`, `test_pipeline.py`, `test_api.py`, `test_cli.py`

**Interfaces:**
- Produces: `Config` fields `asd_threshold=0.5, asd_min_active=0.3, asd_onset_frames=3, min_track_s=0.5, min_face_px=40, max_occurrences=5, onset_lookback_s=1.0` and property `models_dir -> cache_dir / "models"`; `models.FaceTrack(track_id: int, frames: list[int], boxes: list[tuple[int,int,int,int]], scores: list[float] = [])` with `start_index`, `end_index`, `median_height()`; `models.Occurrence(window: Window, klass: str, frame_index: int, ocr_score: float, faces: int, asd_mean: float, speaker_box: tuple|None, note: str)` with `to_dict()`; `Result` gains `occurrence_class: str = ""`, `speaker_box: list[int] | None = None`, `speaker_image_path: str = ""`, `occurrences: list[dict] = []` (all in `to_dict()`, and `format_block()` prints `Occurrence: <class>` and `Speaker  : x,y,w,h` lines when set); `matcher.all_word_windows(words, target, threshold, cap) -> list[Window]` (non-overlapping, sorted by score desc); `WhisperLocator.locate_all(video, target) -> list[Window]`; `locate()` unchanged (= first of locate_all or None); mode strings `audio | ocr | audio+ocr | hybrid` everywhere (`run()` default `"hybrid"`; the old code path now runs for `audio+ocr`; `hybrid` temporarily behaves like `audio+ocr` until Task 6 adds `_run_hybrid`).

- [ ] **Step 1: failing tests**
`test_matcher.py` append:
```python
def test_all_word_windows_returns_non_overlapping_sorted_capped():
    from dialogue_finder.text.matcher import all_word_windows
    words = [Word(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(
        ("my mind rebels at stagnation " * 3 + "the quick brown fox").split())]
    wins = all_word_windows(words, "My mind rebels at stagnation", threshold=0.6, cap=2)
    assert len(wins) == 2 and wins[0].score >= wins[1].score
    assert wins[0].end_s <= wins[1].start_s or wins[1].end_s <= wins[0].start_s


def test_all_word_windows_empty_below_threshold():
    from dialogue_finder.text.matcher import all_word_windows
    words = [Word(w, i, i + 0.5) for i, w in enumerate("the quick brown fox".split())]
    assert all_word_windows(words, "my mind rebels at stagnation", threshold=0.6, cap=5) == []
```
`test_audio_locator.py` append (reuse the cached-words fixture pattern from `test_locate_uses_cached_words_and_threshold`): `locate_all` on a transcript containing the line twice returns 2 windows; `locate` returns the best one.
`test_pipeline.py`: rename every `mode="hybrid"` that tests the old behaviour to `mode="audio+ocr"`; add `test_hybrid_without_extras_matches_audio_ocr` (fake locator + NoMatch extractor on the synthetic clip: `run(..., mode="hybrid")` returns `source == "audio"`, `frame_index == 120`, and a `verify` event with status `skipped` is emitted — this passes only after Task 6; mark it `@pytest.mark.xfail(strict=True, reason="Task 6")` now and remove the marker in Task 6).
`test_cli.py`: `--mode audio+ocr` accepted; `--mode bogus` exits 2. `test_api.py`: `{"mode": "audio+ocr"}` accepted (200), `"hybrid"` accepted.
- [ ] **Step 2: run → fail.**
- [ ] **Step 3: implement** — `config.py` fields + `models_dir` property; `models.py` dataclasses + Result fields + `format_block` lines; `matcher.all_word_windows`:
```python
def all_word_windows(words: list[Word], target: str, threshold: float, cap: int) -> list[Window]:
    """Every non-overlapping span scoring ≥ threshold, best first (greedy: take the best, drop overlaps, repeat)."""
    n = max(1, len(normalize(target).split()))
    spans: list[Window] = []
    for size in range(max(1, n - 1), n + 3):
        for i in range(0, max(1, len(words) - size + 1)):
            span = words[i:i + size]
            if not span:
                continue
            s = score_similar(target, " ".join(w.text for w in span))
            if s >= threshold:
                spans.append(Window(span[0].start, span[-1].end, s, " ".join(w.text for w in span)))
    spans.sort(key=lambda w: (-w.score, w.start_s))
    chosen: list[Window] = []
    for w in spans:
        if all(w.end_s <= c.start_s or w.start_s >= c.end_s for c in chosen):
            chosen.append(w)
        if len(chosen) >= cap:
            break
    return chosen
```
`locator.py`: `locate_all` = `all_word_windows(self._words(video), target, self.cfg.audio_match_threshold, self.cfg.max_occurrences)`; `locate` = `wins[0] if wins else None` (keep its `transcribe`/cache behaviour). Rename in `pipeline.py` (`mode in ("hybrid", "audio+ocr", "audio")` for locating; the widened-retry condition `mode in ("audio+ocr", "hybrid")`), `cli.py` choices `["hybrid", "audio+ocr", "audio", "ocr"]`, `api/jobs.py` Literal, `frontend/index.html` `<option>hybrid</option><option>audio+ocr</option><option>audio</option><option>ocr</option>`. APPROACH: one sentence at the top of Phase 3 ("Runs recorded before 2026-08-25 use 'hybrid' for what is now called audio+ocr."). README mode table (4 rows).
- [ ] **Step 4: run** — full suite green except the one strict xfail. **Step 5: commit** — `feat: mode rename (audio+ocr), multi-window locator, visual config/models`.

---

### Task 3: faces — YuNet detector, IoU tracker, crops

**Files:**
- Create: `backend/dialogue_finder/visual/faces.py`
- Test: `backend/tests/test_visual_faces.py`

**Interfaces:**
- Produces: `class FaceDetector(Protocol): def detect(self, frame: np.ndarray) -> list[tuple[int,int,int,int]]` (x, y, w, h); `class YuNetDetector(models_dir: Path, score_threshold=0.7)` (downloads `face_detection_yunet_2023mar.onnx` with retries into `models_dir`; `detect` resizes the detector input to the frame size lazily); `class IouTracker(iou_threshold=0.5, max_gap=3)` with `update(frame_index, boxes) -> None` and `tracks() -> list[FaceTrack]`; `build_tracks(source, detector, start_index, end_index, cfg, reporter=None, should_cancel=None) -> list[FaceTrack]` (every frame in range, usable-track filter applied: `len ≥ min_track_s·fps` and `median_height ≥ min_face_px`); `crop_face(frame, box, size) -> np.ndarray` (square crop around the box, grey, resized to `size`×`size` — the size and colour come from the spike note).

- [ ] **Step 1: failing tests** (fake detector; no models):
```python
from dialogue_finder.visual.faces import IouTracker, build_tracks, crop_face
from dialogue_finder.config import DEFAULT

class FakeSrc:
    fps = 24.0; frame_count = 100
    def iter_range(self, a, b, step):
        for i in range(a, b + 1, step): yield i, i
    def frame_at(self, i): return i

class FakeDet:
    def __init__(self, boxes_at): self.boxes_at = boxes_at
    def detect(self, frame): return self.boxes_at(frame)

def test_tracker_links_overlapping_boxes_and_tolerates_gaps():
    t = IouTracker(iou_threshold=0.5, max_gap=3)
    for i in range(10):
        t.update(i, [] if i == 4 else [(100 + i, 100, 60, 60)])
    tracks = t.tracks()
    assert len(tracks) == 1 and tracks[0].frames[0] == 0 and tracks[0].frames[-1] == 9

def test_tracker_splits_on_long_gap_and_far_boxes():
    t = IouTracker(iou_threshold=0.5, max_gap=3)
    for i in range(0, 5): t.update(i, [(100, 100, 60, 60)])
    for i in range(10, 15): t.update(i, [(400, 100, 60, 60)])
    assert len(t.tracks()) == 2

def test_build_tracks_filters_short_and_small():
    det = FakeDet(lambda i: [(10, 10, 80, 80)] + ([(300, 300, 20, 20)] if i < 30 else []))
    tracks = build_tracks(FakeSrc(), det, 0, 47, DEFAULT)   # 2 s at 24 fps
    assert len(tracks) == 1 and tracks[0].median_height() == 80

def test_crop_face_is_square_grey_sized():
    import numpy as np
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    c = crop_face(frame, (300, 100, 50, 70), 112)
    assert c.shape == (112, 112) and c.dtype == np.uint8
```
Plus one `@pytest.mark.slow` test: `YuNetDetector(DEFAULT.models_dir).detect(frame)` on the cached episode frame 7794 (`FrameSource(...)`) returns ≥ 1 box with height ≥ 40 (this downloads the ONNX once).
- [ ] **Step 2: run → fail. Step 3: implement** — IoU tracker: greedy match each new box to the open track with highest IoU ≥ threshold; unmatched boxes start tracks; tracks not updated for > `max_gap` frames close. `build_tracks` iterates `source.iter_range(start, end, 1)`, calls `detector.detect`, updates the tracker, then filters. YuNet: `cv2.FaceDetectorYN.create(model, "", (w, h), score_threshold)`; `setInputSize` when the frame size changes; convert its float rows to int boxes.
- [ ] **Step 4: run green. Step 5: commit** — `feat: face tracks (YuNet + IoU tracker) for visual verification`.

---

### Task 4: audio features — MFCC at 100 Hz and speech mask

**Files:**
- Create: `backend/dialogue_finder/visual/audio_features.py`
- Test: `backend/tests/test_visual_audio.py`

**Interfaces:**
- Produces: `mfcc_100hz(wav: Path, start_s: float, end_s: float) -> np.ndarray` (T×13 float32, 10 ms hop — parameters exactly as the spike note's `preprocessing` heading; `python_speech_features.mfcc(signal, 16000, numcep=13, winlen=0.025, winstep=0.010)` unless the spike says otherwise); `speech_mask(wav: Path, start_s: float, end_s: float, fps: float) -> list[bool]` (one entry per video frame in the span, from faster-whisper's bundled Silero VAD: `from faster_whisper.vad import get_speech_timestamps, VadOptions` on the 16 kHz slice, then rasterised to frames); `read_wav_slice(wav, start_s, end_s) -> np.ndarray` (int16 → float32, via the `wave` stdlib module).

- [ ] **Step 1: failing tests** — generate a 3 s 16 kHz wav in `tmp_path` with the stdlib `wave` module: 1 s silence, 1 s 440 Hz tone at amplitude 0.3, 1 s silence. `mfcc_100hz(wav, 0, 3).shape == (≈300, 13)`; `speech_mask(wav, 0, 3, 24)` has length 72 and is `False` for the first 20 frames (VAD may or may not flag a pure tone as speech — assert only the silence frames are False and the length); `read_wav_slice` returns 48000 samples for a 3 s span.
- [ ] **Step 2-5:** implement, run, commit `feat: MFCC + speech mask for visual verification`.

---

### Task 5: LR-ASD detector behind the SpeakerDetector protocol

**Files:**
- Create: `backend/dialogue_finder/visual/lrasd.py`
- Modify: `backend/dialogue_finder/visual/lrasd_model.py` (from Task 1, only if the API needs a thin wrapper)
- Test: `backend/tests/test_visual_lrasd.py`

**Interfaces:**
- Produces: `class SpeakerDetector(Protocol): def score(self, crops: np.ndarray, mfcc: np.ndarray) -> list[float]` (crops: N×S×S uint8 grey aligned to N video frames; mfcc: T×13 for the same span; returns N scores in [0, 1]); `asd_available() -> tuple[bool, str]` (torch importable, weights present or downloadable → `(True, "")`, else `(False, reason)`); `class LrAsdDetector(models_dir: Path)` — lazy model load on first `score`, weights fetched with 3 retries + backoff into `models_dir` (file name and URL from the spike note), aligns audio/video lengths exactly as the reference demo (trim to the shorter, `audio_frames_per_video_frame` from the spike), runs on CPU, converts logits to probabilities with the reference's rule.

- [ ] **Step 1: failing tests** — `asd_available()` returns `(False, reason)` when `sys.modules["torch"] = None` (monkeypatch) and the reason mentions "requirements-asd.txt"; `LrAsdDetector.score` with a monkeypatched `_forward` returning a fixed logits tensor of length N gives N probabilities in [0,1] (no real model); `@pytest.mark.slow`: real weights on the spike's Holmes window (frames 7700-7950 with YuNet largest-face crops via Task 3 + `mfcc_100hz` via Task 4) → mean score over frames 7794-7900 ≥ 0.5.
- [ ] **Step 2-5:** implement per the spike note; commit `feat: LR-ASD speaker detector (optional extras) behind SpeakerDetector`.

If Task 1 decided FALLBACK: this task instead implements `LipMotionDetector(models_dir)` — 106-point landmarks ONNX (`2d106det.onnx` from InsightFace, downloaded into `models_dir`) → mouth-aspect-ratio series per track → score per frame = normalised cross-correlation of MAR variance with the speech mask over a 0.5 s window — same protocol, same tests with adjusted thresholds recorded in the spike note.

---

### Task 6: verifier + pipeline `_run_hybrid` + CLI output

**Files:**
- Create: `backend/dialogue_finder/visual/verifier.py`
- Modify: `backend/dialogue_finder/pipeline.py`, `backend/dialogue_finder/cli.py` (prints the extra lines via `format_block`), `backend/tests/test_pipeline.py` (remove the xfail)
- Test: `backend/tests/test_visual_verifier.py`, `backend/tests/test_pipeline.py`

**Interfaces:**
- Produces: `classify(ocr_score: float, tracks: list[FaceTrack], speech: list[bool], cfg) -> tuple[str, FaceTrack | None, float]` → (`"valid-text" | "valid-speaker" | "invalid" | "uncertain"`, speaker track, its mean score over speech frames); `find_onset(track: FaceTrack, speech: list[bool], first_index: int, cfg) -> int | None` (first frame ≥ `first_index` where the track's score ≥ `asd_threshold` for `asd_onset_frames` consecutive frames and speech is True at that frame); `verify_window(src, window, target, extractor, detector, speaker, wav, cfg, reporter, should_cancel) -> Occurrence` (runs OCR via the existing `coarse_scan` + `refine_first_frame`, then tracks → crops → mfcc → scores → classify → frame); pipeline `_run_hybrid(src, video, windows, target, ex, cfg, reporter, timings, t0, occurrence, should_cancel) -> Result` (loops windows, emits `verify running/ok`, `occurrences ok`, selects by class → ASR score → earliest, honours `--occurrence`, writes `frame_<n>_speaker.png` with the box when a speaker exists, calls `_finish` with `source` in `ocr | audio+asd | audio`, `confidence` per the spec table, `occurrence_class`, `speaker_box`, `occurrences`). In `run()`: for `mode == "hybrid"`, if `asd_available()` is False emit `verify skipped <reason>` and continue down the `audio+ocr` path unchanged; otherwise use `locator.locate_all` and branch to `_run_hybrid` (no windows → the `audio+ocr` whole-video path).

- [ ] **Step 1: failing tests** — `classify` table: OCR 0.85 → valid-text; no OCR, one track with scores 0.8 on 50 % of speech frames → valid-speaker; no OCR, tracks present but scores ≤ 0.3 → invalid; no OCR, no tracks → uncertain; two qualifying tracks → the higher mean wins. `find_onset`: scores `[0.1,0.2,0.6,0.7,0.8,0.9]` with speech all True and `asd_onset_frames=3` → index of the 0.6 frame; speech False there → next qualifying; none → None. Pipeline (fake locator returning 2 windows at 2 s and 8 s on the synthetic clip; fake detector giving one 80-px box on every frame; fake speaker returning 0.9 for the second window and 0.1 for the first; NoMatch extractor): result `occurrence_class == "valid-speaker"`, frame within the second window, `source == "audio+asd"`, `occurrences` has 2 entries with classes `["invalid", "valid-speaker"]`; and with a detector returning no boxes → `uncertain`, frame = first spoken word of the higher-ASR window; and the un-xfailed `test_hybrid_without_extras_matches_audio_ocr`.
- [ ] **Step 2-5:** implement, run (full suite green), commit `feat: hybrid mode — per-window OCR + face tracks + LR-ASD verification, occurrence classification, visual onset`.

---

### Task 7: API + page — occurrences and the speaker frame

**Files:**
- Modify: `backend/api/main.py` (serve `frame_<n>_speaker.png` via the existing frames endpoint with `?box=1` drawing the box from the job result, or simply expose the saved PNG path through `/jobs/{id}/speaker.png`), `frontend/app.js`, `frontend/styles.css`, `frontend/index.html`
- Test: `backend/tests/test_api.py` (speaker image endpoint 404/200 with a fake result)

**Interfaces:**
- Consumes: events `verify` (`running` payload `{window_index, faces, asd_mean}`, then `ok|skipped|fallback`), `occurrences` (`ok`, payload `{"occurrences": [Occurrence.to_dict()...]}`), result fields from Task 2/6.
- Page: mode select order `hybrid, audio+ocr, audio, ocr`; a **Occurrences** block (hidden until the event) with one row per occurrence: `tc(start)–tc(end) · ASR 0.95 · OCR ✓/✗ · faces N · speaking ✓/✗/? · <badge class>` (badge colours: valid teal, uncertain muted, invalid danger); timeline: one 2-px mark per occurrence in its class colour (selected occurrence keeps the amber window); result card: when `speaker_box` exists show the speaker image (with the box) in place of the plain result frame and print `occurrence_class` next to the route. Copy for skipped verify: "Active-speaker check unavailable (install requirements-asd.txt) — using audio + OCR."
- [ ] Steps: test → implement → run the synthetic clip and the real episode through the page (Playwright if available; the episode now shows Occurrences + the boxed Holmes frame) → screenshot `docs/ui-hybrid.png` → commit `feat: page shows occurrences and the speaking face`.

---

### Task 8: validation + docs

- [ ] Real episode via CLI (`--mode hybrid`) and via the page: expect `valid-speaker`, onset frame recorded next to 7794 (state which is earlier and by how many frames), speaker box on Holmes, per-window cost measured (faces s, ASD s). Squid Game clip: `valid-text`, frame 297 unchanged. One voice-over clip (narration over footage — pick a short YouTube documentary/trailer clip; record URL + line): expect `invalid` (faces present, none speaking) or `uncertain` (no faces); record which and why. A run with the extras uninstalled (`pip uninstall -y torch python_speech_features` in a throwaway venv copy, or monkeypatch `asd_available`) → `verify: skipped`, answer identical to `audio+ocr`.
- [ ] Docs: `docs/APPROACH.md` Phase 7 (the third evidence; why LR-ASD (research table + sources); windows-only; the classification table; measured costs; validation table); `docs/DECISIONS.md` Plan 4 entries (first person: mode naming, LR-ASD vs heuristic, "no face ≠ not speaking", default mode); README (mode table, `requirements-asd.txt`, curl fallback for weights); BENCHMARK unchanged; `prompts.txt` append the brainstorm prompts for this feature ([27]…) verbatim.
- [ ] Full suite, commit `docs: hybrid mode validation, Phase 7, decisions`.

---

## Verification

- `pytest -q` green (fast + slow); the three old modes produce byte-identical CLI output on the synthetic clip and the episode compared with commit 2bd2757 (`audio+ocr` = old `hybrid`).
- `python -m dialogue_finder --local cache/5f39d4605665a831.mp4 --text "My mind rebels at stagnation"` (default hybrid) prints `Occurrence: valid-speaker`, a `Speaker` box line, and writes `frame_<n>_speaker.png`.
- Without extras: same command prints the old audio+ocr answer plus `[verify:skipped] ...` in verbose mode.
- Page: Occurrences block + boxed speaker frame on the episode; Squid Game still teal OCR.
- Docs: Phase 7 numbers traceable to the task reports; prompts.txt current.
