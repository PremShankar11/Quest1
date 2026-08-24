# Approach

## Phase 1 — Understand the problem

Spike on the real test video (https://ok.ru/video/248244667877, "The Adventures of
Sherlock Holmes: A Scandal in Bohemia", 3261s, 23.98 fps, 640x480 mp4 at max_height=480)
using `scripts/dump_frames.py`. Frames were dumped and inspected (Read tool) at
t=30, 45, 60, 90, 120, 150, 200, 250, 300, 400, 600, 700, 900, 1200, 1800, 2400, 3000
(the brief's original 9, the dialogue-hint set around "My mind rebels at stagnation"
in the first ~5 minutes, and the no-text fallback set).

What was seen: t=45 and t=60 show yellow opening-credit title cards ("Developed for
television by John Hawkesworth", "Dramatised by Alexander Baron") overlaid on the
picture — proving text-on-frame rendering and OCR are technically viable on this
source. t=150 is a black scene-transition frame. Every other sampled frame
(t=30, 90, 120, 200, 250, 300, 400, 600, 700, 900, 1200, 1800, 2400, 3000) shows plain
picture with no overlay text of any kind — no dialogue captions, no lower-third text.

Test video does not have burned-in subtitles (checked frames at t=30, 45, 60, 90, 120,
150, 200, 250, 300, 400, 600, 700, 900, 1200, 1800, 2400, 3000; only opening-credit
title cards render as on-screen text, never spoken dialogue); therefore the expected
route on this video is audio-fallback.

## Phase 2 — Design
## Phase 3 — Build

### Pipeline + CLI (Task 7)

Built `dialogue_finder/pipeline.py` (`run`, `confidence_for`, `PipelineError`),
`dialogue_finder/cli.py` (`main`, `build_parser`), and `dialogue_finder/__main__.py`,
wiring together the downloader, frame source, OCR scanner/refiner, and (once Task 8
lands) the audio locator. `cli.main`'s `except SystemExit` handler returns
`int(e.code) if e.code is not None else 2` so argparse's `SystemExit(0)` from
`--help` is preserved as exit 0, while usage errors (`SystemExit(2)`) still exit 2.

Full suite: 31 passed (25 existing + 6 new: `test_pipeline.py` x4,
`test_cli.py` x2) in ~52s, including the two `slow`-marked ground-truth tests
(`test_ground_truth_exact_frame`, `test_ground_truth_fade_in`) which build synthetic
clips and assert the exact frame/timestamp OCR lands on.

**Synthetic-clip end-to-end run** (proves the whole OCR path, first real invocation
of the CLI):

```
cd backend && ../.venv/Scripts/python -m dialogue_finder --local ../bench_out/chk.mp4 --text "My mind rebels at stagnation" --mode ocr --verbose
```

```
Timestamp : 00:00:05.000
Frame     : 120
Text      : "My mind rebels at stagnation"
Confidence: HIGH  (source: ocr)
Image     : output\frame_120.png
Previous  : output\frame_119.png  (pop-in)
```

Matches ground truth exactly: frame 120 / 5.000s, HIGH confidence, pop-in appearance.

**Real-video OCR timing (measured, not run end-to-end)**

This machine is CPU-only (onnxruntime, no CUDA). A full OCR scan of the real 3261.7s
/ 78 204-frame video (`cache/5f39d4605665a831.mp4`, fps=23.976) at the pipeline's
`fullscan_fps=2.0` would take 6 517 OCR calls (`78204 / (23.976/2.0)` sampled frames)
and was judged too slow to run as part of this task (>1 hour). Instead, timed
`read_dialogue()` directly over 100 frames sampled every 12th frame starting at frame
2400 (a throwaway script, not committed):

- mean time per OCR call: **0.598 s**
- extrapolated full-scan time: 0.598 s × 6 517 calls ≈ **3 897 s (≈ 65 min, ≈ 1.08 h)**

This confirms the >1 hour estimate and justifies skipping the full real-video scan
here. As established in Phase 1, this video has no burned-in subtitles — only
opening-credit title cards render as on-screen text, never spoken dialogue — so the
OCR-only route on this video is expected to end as `ocr-weak` / LOW confidence
regardless of scan time; the audio route (`WhisperLocator`, Task 8) is the one
that will actually answer "when does this line appear," by transcribing speech to
locate the *spoken* moment and returning that frame as `source="audio"`.

### Audio locator + hybrid path (Task 8)

Built `dialogue_finder/audio/locator.py` (`Locator` protocol, `words_cache_path`,
`extract_audio`, `_load_model`, `transcribe_words`, `WhisperLocator`) per the brief,
with one deviation forced by this machine's environment:

**Deviation 1 — CUDA fallback moved from model construction to first inference.**
The brief's `_load_model` wraps `WhisperModel(..., device="cuda")` construction in
try/except, assuming a CUDA failure surfaces at construction time. On this machine
`ctranslate2.get_cuda_device_count()` reports `1` (a GPU is enumerated) but the CUDA
runtime library (`cublas64_12.dll`) is not installed, so construction succeeds and
the failure (`RuntimeError: Library cublas64_12.dll is not found or cannot be
loaded`) only surfaces on the first `model.transcribe()` call, inside
`detect_language()`'s encode step — past the brief's try/except. Fixed by wrapping
the `model.transcribe(...)` call itself in `transcribe_words`: on `RuntimeError`
when `device == "cuda"`, reload as `WhisperModel(name, device="cpu",
compute_type="int8")` and retry once, emitting
`StageEvent("transcribe", "fallback", "cuda unavailable, retrying whisper base on cpu")`.
Confirmed firing in the real run's log (see below). No other change to the brief's
code.

Test suite after Task 8's own code: `pytest tests/test_audio_locator.py -q` → 2
passed (unit test on cached transcript + threshold; slow smoke test transcribing
3s of silence). Full suite: 33 passed.

**Deviation 2 — pipeline retry widened instead of whole-video (controller-directed
mid-task fix).** The first real-video run correctly transcribed and located the
audio window, but `pipeline.py`'s existing "window missed → retry whole video" rule
fired the ~65-minute, 6 517-call OCR scan predicted in Task 7's Phase 3 note above
(no burned-in subtitles on this video, so the window OCR scan always misses). The
controller killed that run and directed: add `Config.retry_pad_s: float = 15.0`
and replace the whole-video retry in `pipeline.run()` with a widened-window retry —
`[window.start_s - retry_pad_s, window.end_s + retry_pad_s]` at the same
`fullscan_fps`, so whole-video OCR scanning now only happens when `window is None`.
Added `test_hybrid_retries_widened_window_not_whole_video` in `test_pipeline.py`
(fake no-match extractor + fake locator on the synthetic clip) asserting
`source == "audio"`, `frame_index == 120`, and that no emitted scan event mentions
"whole video". Full suite: 34 passed.

**Deviation 3 — false-positive OCR match, found by the widened retry (controller-
directed fix).** The widened retry (rerun 1) surfaced a real bug: `frame 7459`
OCR-read a single stray character `"R"` (scene noise, not text) and
`text/matcher.py:score_contains` scored it `1.00` against the 29-character target,
because `rapidfuzz.fuzz.partial_ratio` returns 100 whenever the shorter string is a
substring-equivalent match of the longer one — a lone correct letter is enough. The
pipeline then reported `source: ocr`, frame 7458, text `"G"`, confidence HIGH — a
false positive (frames 7457/7458 show two men in a Victorian drawing room, no
on-screen text at all). Controller's fix: scale `score_contains`'s result by
`coverage = min(1.0, len(haystack) / len(target))`, penalizing short reads. Added
`test_score_contains_short_fragment_is_low` and
`test_score_contains_near_full_read_still_high` to `test_matcher.py`. Full suite:
36 passed.

**Real-video hybrid run** (`cache/5f39d4605665a831.mp4`, 54 min, fps=23.976, no
burned-in subtitles — see Phase 1):

- Transcription (first run, before the transcript cache existed): audio extraction
  + faster-whisper `base`/`translate` on **cpu (after a cuda→cpu fallback)**, wall
  time **≈ 2 m 44 s** (`.16k.wav` written 23:17:40 → `.words.json` written
  23:20:24, from file mtimes) for the full 54-minute video — well under the ~10-20
  min estimate, likely due to VAD-filtered silence skipping. Detected language:
  **en**. 4 098 words transcribed and cached to
  `cache/5f39d4605665a831.words.json`.
- Locate: `[locate:ok] window 325.1-327.8s score 0.95: 'My mind rebels its
  stagnation.'`
- Scan (post-fix, rerun 2, transcript cache hit): OCR 322.1-330.8s at 5.0 fps → no
  hit; widened retry 310-343s at 2.0 fps (`retry_pad_s=15.0`) → no hit (best score
  0.04, down from the pre-fix false-positive 1.00) → `[refine:fallback] no
  on-screen match; using audio timestamp`.
- Final printed block:
  ```
  Timestamp : 00:05:25.073
  Frame     : 7794
  Text      : "My mind rebels its stagnation."
  Confidence: MEDIUM  (source: audio; no on-screen text matched; frame at first spoken word)
  Image     : ..\output\frame_7794.png
  Previous  : ..\output\frame_7793.png  (frame before)
  ```
- Total CLI wall time (cached-transcript hybrid run, post-fix): **1 m 3.7 s**
  (`time` real).
- `output/frame_7794.png` and `output/frame_7793.png`: both show a close-up of the
  actor playing Sherlock Holmes (dark hair, three-quarter profile, dark suit and
  tie), indoor lighting, no subtitle or on-screen text visible in either frame —
  consistent with `source: audio` and this video having no burned-in captions.

Test progression across Task 8: 31 (Task 7 baseline) → 33 (locator) → 34
(widened-retry pipeline fix) → 36 (matcher coverage-scaling fix).

## Phase 4 — Test and measure
## Phase 5 — Reflect: limits and extensions
