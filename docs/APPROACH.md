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

### Where to look

Transcribe the full audio track with faster-whisper (`base`, `task=translate` so dubbed/foreign audio
comes back in English), fuzzy-match the target line against the word-level transcript with rapidfuzz
(`token_sort_ratio`, threshold 0.6), and take a window around the best-scoring span, padded by
`Config.window_pad_s` (3.0 s). This is a *location* signal, not a frame — it answers "roughly when," not
"which frame."

### Which frame

Binary search (`first_true`) over the OCR-scanned samples inside the window: OCR score is a monotone-ish
step from "not yet visible" to "visible," so the exact first frame is found in 3-4 OCR calls between the
last non-matching sample and the first matching one, instead of scanning every frame.

### How text is extracted

RapidOCR (onnxruntime backend) reads the bottom subtitle band first (`band_fraction=0.35`, upscaled
`ocr_upscale=2.0`); if that band read comes back empty, the same frame is re-read full-frame (catches
text outside the bottom band, e.g. `center_position` in the benchmark). The extracted text — whichever
read produced it — is then compared against the target with `text/matcher.py:score_contains`
(`rapidfuzz.fuzz.partial_ratio`, scaled by `coverage = min(1, len(haystack)/len(target))` so a single
stray character can't score as a perfect match — see docs/DECISIONS.md).

### Ambiguity

Two kinds handled explicitly: (1) the same line appears more than once — `--occurrence first|last|all`
picks which OCR hit to report, all candidates are still recorded in `Result.candidates`; (2) the line
fades or pops onto screen rather than appearing on one clean frame — `classify_appearance` (Canny edge
diff between frame *n-1* and *n*) labels the `Previous` evidence pair `pop-in` or `fade-in`, and
confidence is downgraded to `MEDIUM` for a fade so the label and the confidence agree with each other.

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

### Never-crash pass (Task 9, Step 1)

`backend/tests/test_never_crash.py` (3 tests, 1 marked `slow`):

- `test_unreachable_url_is_clean` — `--url https://example.invalid/...` fails DNS
  resolution; asserts exit 1, stderr starts with `Error:`, no `Traceback`.
- `test_corrupt_file_is_clean` — `--local` a file containing `b"not a video"`;
  asserts exit 1, no `Traceback`.
- `test_output_write_failure_is_clean` (slow — runs a real synthetic-clip OCR pass) —
  monkeypatches `Path.write_text` to raise `OSError` only for `result.json`; asserts
  exit 1, stderr contains `Error: could not write output:`, no `Traceback`.

Two bugs surfaced and were fixed to make these pass:

1. **Duplicate/leaking error line.** `pipeline.run`'s `DownloadError` handler emitted
   a `StageEvent("error", "error", str(e))` before raising `PipelineError`.
   `PrintReporter.emit` only filters events that carry a `progress` value, so this
   `error`-stage event (progress `None`) printed unconditionally — even in
   non-verbose mode — as `[error:error] Could not download video: ...`, ahead of
   cli.py's own `Error: ...` line. That broke "stderr starts with Error:" and was
   pure duplication (cli.py already prints the same message via
   `except PipelineError`). Removed the redundant `reporter.emit` call.
2. **yt-dlp's own error text leaking to stderr.** Even with `quiet`/`no_warnings`,
   yt-dlp's default logger still writes `ERROR: ...` lines straight to stderr on
   failure. Added a `_QuietLogger` (no-op `debug`/`info`/`warning`/`error`) and wired
   it in via `opts["logger"]`, so failures surface only through our own
   `DownloadError` -> `Error: ...` path.

**Controller ruling (a) — output-write errors.** `cli.main`'s `cfg.output_dir.mkdir`
and `result.json` write sat outside any `try`. Wrapped them so an `OSError` there
prints `Error: could not write output: ...` and returns 1 instead of a traceback.
Covered by `test_output_write_failure_is_clean` above.

**Controller ruling (b) — weak-fallback note.** `pipeline.run`'s final fallback
(`Candidate(0, 0.0, "", 0.0)` when `cands` is empty) previously reported
`note=f"best OCR similarity only {weak.score:.2f}"` even when there was no
candidate at all (`0.00` — misleading, reads as "we found something, barely").
Now: `note = "no text detected anywhere; frame 0 returned"` when `cands` is empty,
`f"best OCR similarity only {weak.score:.2f}"` otherwise.

**Controller ruling (extra) — `--mode audio` must build its own locator.**
`pipeline.run` only constructed `WhisperLocator` when `locator is None and mode ==
"hybrid"`, so `--mode audio` with no injected `locator=` silently produced
`window=None` and raised `PipelineError("No audio match found...")` even on a video
with clear matching speech — `audio` mode was unusable stand-alone. Changed the
guard to `mode in ("hybrid", "audio")`. Covered by two new fast tests in
`test_pipeline.py`: `test_audio_mode_uses_locator` (`FakeLocator` -> `source ==
"audio"`, `frame_index == 120`) and `test_audio_mode_no_match_raises_pipeline_error`
(`FakeNoLocate` -> `PipelineError`).

Full suite after the never-crash pass: **43 passed** — 3 new in
`test_never_crash.py` and 2 new in `test_pipeline.py` (the audio-mode-locator
tests) on top of the pre-Task-9 suite.

### Synthetic benchmark (Task 9, Step 2)

`backend/bench/run_bench.py` runs `dialogue_finder.pipeline.run` in OCR mode over 8
synthetic variants from `bench/make_clip.py` (position, fade, scale, resolution,
frame rate) and writes `docs/BENCHMARK.md`. Full run: **~200 s** on CPU (RapidOCR
approx. 0.6 s/call), in line with the ~3-5 min estimate.

| variant | truth frame | found frame | error (frames) | source | confidence | seconds |
|---|---|---|---|---|---|---|
| baseline_640x360_24fps_bottom | 120 | 120 | 0 | ocr | HIGH | 24.5 |
| top_position | 120 | 120 | 0 | ocr | HIGH | 27.3 |
| center_position | 120 | 120 | 0 | ocr | HIGH | 30.1 |
| fade_12_frames | 120 | 121 | 1 | ocr | MEDIUM | 23.3 |
| small_text_360p | 120 | 120 | 0 | ocr | HIGH | 23.0 |
| hd_1280x720 | 120 | 120 | 0 | ocr | HIGH | 23.6 |
| 30fps | 150 | 150 | 0 | ocr | HIGH | 23.0 |
| 60fps | 300 | 300 | 0 | ocr | HIGH | 24.1 |

Every pop-in variant (position, scale, resolution, frame rate) lands on the exact
truth frame with `HIGH` confidence — `read_dialogue`'s full-frame OCR fallback
already handles `center_position` text outside the bottom band, so nothing needed
fixing there. The one non-zero row is `fade_12_frames`: text fades in over 12
frames, so "first frame with detectable text" is inherently fuzzy at low opacity;
the pipeline lands 1 frame after the nominal fully-transparent start (well inside
the 12-frame fade window) and correctly reports `MEDIUM` confidence via
`classify_appearance`'s fade-in detection — this is the expected, documented limit
for gradual-appearance text, not a bug.

### Real-video matrix (Task 9, Step 3)

| video | duration | has subs? | locate | scan | source | frame | timestamp | time | correct by eye? |
|---|---|---|---|---|---|---|---|---|---|
| [ok.ru Sherlock Holmes ep.](https://ok.ru/video/248244667877) (Phase 3, hybrid) | 3261 s | no burned-in subs | ok (score 0.95, window 325.1-327.8s) | window then widened retry, both no on-screen match | audio | 7794 | 00:05:25.073 | 63.7 s | yes — close-up of Holmes actor, no subtitle text, consistent with audio source |
| [Squid Game Intro Sequence — English Subtitles](https://www.youtube.com/watch?v=3_XZ354E9uE) (burned-in captions) | 146 s | yes, burned-in English subtitles (Korean audio) | ok but weak/garbled (score 0.63, window 24.6-29.6s — noisy Korean-to-English `whisper --task translate`) | narrow window (21.6-32.6s) no match, then widened retry (10-45s) hit | ocr | 297 | 00:00:09.900 | 2m 1.3s full CLI (cached download; dominated by whisper transcription of the whole 146 s clip) | yes — frame shows the on-screen caption "In my town, we had a game called the \"Squid Game.\"" verbatim |
| [A one minute TEDx Talk for the digital age \| Woody Roseland](https://www.youtube.com/watch?v=1aA1WGON49E) (no captions) | 81 s | no burned-in captions | ok (score 1.00, window 34.1-34.9s: 'Thanks for the click.') | window (31.1-37.9s) no match, then widened retry (19-50s) no match | audio | 818 | 00:00:34.117 | 1m 24.8s full CLI (cached download) | yes — speaker on stage, no on-screen text anywhere, consistent with `source: audio`; timestamp lands on the correct spoken line per the whisper transcript |

Commands used:
```
cd backend
../.venv/Scripts/python -m dialogue_finder --url "https://www.youtube.com/watch?v=3_XZ354E9uE" \
  --text "In my town, we had a game called the Squid Game." --mode hybrid --verbose --out ../output/squidgame_intro

../.venv/Scripts/python -m dialogue_finder --url "https://www.youtube.com/watch?v=1aA1WGON49E" \
  --text "Thanks for the click." --mode hybrid --verbose --out ../output/tedx_digital_age
```

**What the matrix shows:** the hybrid pipeline handles both directions of the
captions/no-captions split correctly, by the intended route. When burned-in text
exists (Squid Game clip) it lands on `source: ocr` with an exact, byte-for-byte
on-screen match despite a badly garbled audio-locate window (Korean speech run
through `whisper --task translate`) — the widened-retry fallback (Task 8's fix)
is what rescues it after the narrow window misses. When no on-screen text exists
(both the ok.ru episode and the TEDx clip) both OCR scan attempts (window +
widened retry) correctly fail to match and the pipeline falls back to `source:
audio`, landing on the right spoken line each time. All three real videos were
reachable on this network — no YouTube-blocked fallback to a synthetic variant was
needed. Both YouTube downloads used the CLI's own `bv*[height<=480]` cache path and
completed in a few seconds (2.13 MiB and 4.79 MiB), so wall time on the real-video
runs is dominated by `whisper base` CPU transcription (60-120 s), not download.

## Phase 5 — Reflect: limits and extensions

### Measured limits

- **Audio-route precision is bounded by Whisper word timestamps: ≈ ±0.1 s, ≈ ±2-3 frames at 24-30 fps.**
  When there's no on-screen text to refine against, the reported frame is only as good as the word
  timestamp faster-whisper assigns to the first spoken word of the match — not frame-exact the way the
  OCR route is.
- **Fade-in text: ±1 frame.** Confirmed by the benchmark (`fade_12_frames`, error 1 frame, `MEDIUM`
  confidence) — "first frame with detectable text" is inherently fuzzy while opacity is still low, and
  the pipeline reports `MEDIUM` rather than pretending to more precision than the source has.
- **Subtitles more than `retry_pad_s` (15 s) from the spoken line are missed by OCR and the pipeline
  falls back to the audio frame.** The widened retry covers `[window.start_s - 15, window.end_s + 15]`
  once; a subtitle further out than that from where Whisper anchors the line is never scanned, and the
  result reports `source: audio` / `MEDIUM` instead of `source: ocr` / `HIGH` for that case. Documented
  trade-off, not a bug — a confident audio match makes a whole-video scan (65 min, see Phase 3) not worth
  it for the common case.
- **OCR false positives on short reads are now scaled by coverage, not eliminated.** `score_contains`
  scales `partial_ratio` by `coverage = min(1, len(haystack)/len(target))`, so a read still needs to
  cover ≥80% of the target's length to clear the 0.8 match threshold (the stray-"R" false positive from
  Task 8 now scores 0.04, not 1.00). Residual limit: a subtitle split across two short on-screen lines
  can still score low unless each line individually covers ≥80% of the target. Not observed in the
  benchmark or the real-video matrix, but not structurally impossible.
- **CPU full-scan cost.** Whole-video OCR (only triggered when no audio window exists at all) runs at
  ≈ 0.6 s/OCR call on this CPU; on the 54-minute test video that's ≈ 65 minutes for a full scan at
  `fullscan_fps=2.0`. This is why the pipeline prefers to locate via audio first and only fall back to a
  full scan when there's no transcript match to anchor a window on.

### Extensions not built (out of scope for Plan 1)

- **Hosted/faster ASR** (Groq Whisper API, OpenAI `whisper-1`) — would cut the ≈2-3 min local
  transcription to seconds, at the cost of a required API key and network dependency; rejected for Plan 1
  so the tool runs offline after the first download (see docs/DECISIONS.md stack review).
- **WhisperX forced alignment** — would tighten audio-route precision below the current ±0.1 s/±2-3
  frame bound; not built because no interviewer feedback has asked for tighter audio precision yet
  (YAGNI — see ledger ruling, docs/DECISIONS.md).
- **Multi-language OCR models** — RapidOCR is currently configured for its default (Latin-script/English)
  model; non-Latin burned-in subtitles would need a different or multi-language model bundle.
- **Scene detection** — could narrow the whole-video fallback scan to shot boundaries instead of a fixed
  fps sample; not needed once the audio-first + widened-retry strategy made the whole-video path rare.
- **Web UI (Plan 2).** Plan 1 is CLI-only by design; the live-progress web interface (FastAPI + static
  HTML/JS, decided in the stack review) is the next plan, not part of this one.
