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

## Phase 4 — Test and measure
## Phase 5 — Reflect: limits and extensions
