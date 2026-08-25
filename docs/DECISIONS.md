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

## Phase 1 — Build
- CUDA torch wheel unavailable for py3.14 on 2026-08-24; CPU path used; GPU noted as optional.
  (cu128 index has no torch==2.13.0 build for any Python version — only 2.9.0–2.11.0+cu128 — so no cp314 CUDA wheel exists at this pin regardless of interpreter.)

## Phase 1 — Stack review against alternatives (2026-08-24)

Trigger: I stopped implementation and asked whether we had actually compared alternatives. Rule adopted:
**lightest tool that does the job, every choice justified.** Research + a measured spike on this laptop (CPU).

- **OCR — AI proposed EasyOCR → I changed to RapidOCR (onnxruntime).** Measured on a synthetic subtitle band:
  RapidOCR loads in 0.5 s, reads "My mind rebels at stagnation" exactly, 0.87 s/frame; EasyOCR loads in 54 s,
  needs ~200 MB torch + 100 MB models, split the line into two pieces, 0.92 s/frame. Also considered: ready-made
  hardsub extractors (videocr, videocr-PaddleOCR, VideOCR, VideoSubFinder) — they sample frames at an interval, so
  they cannot guarantee the *first* frame, and the PaddleOCR-based ones have no Python 3.14 wheels; PaddleOCR 3.x
  (no py3.14 wheels); Tesseract (weak on stylised subtitle fonts); Windows built-in OCR via winocr (lightest of all,
  but Windows-only — the interviewer's machine may not be); cloud text-in-video detection (Google Video Intelligence
  TEXT_DETECTION ≈ $8 per 54-min video, AWS Rekognition ≈ $5.40 — both need bucket upload, neither accepts a URL);
  Gemini video understanding (samples at 1 fps → ~1 s precision, not frame-exact, cannot ingest ok.ru).
- **Audio locator — kept local faster-whisper (base, translate task).** Considered: Groq Whisper API (free tier,
  word timestamps, ~30 s for the episode — the best hosted option, needs an API key and audio < 25 MB), OpenAI
  whisper-1 ($0.36/h), Deepgram Nova-3 (no translation), AssemblyAI, Google STT v2, Gemini audio (MM:SS granularity,
  translates natively), whisper.cpp, WhisperX (best alignment, English-centric). Decisive finding: every provider
  documents that word timestamps on *translated* output are unreliable — so audio can only give a window; the exact
  frame must come from OCR. Local wins because the interviewer can run it with no key and no network after the first run.
- **End-to-end video platforms — rejected as the core.** Twelve Labs (600 free minutes, second-level timestamps),
  Azure Video Indexer (no public pricing), Google/AWS combined (≈ $7-11 per video, bucket upload). None is
  frame-exact; none accepts an ok.ru URL; all add cost and setup for something the local pipeline does exactly.
- **Video fetch — yt-dlp.** Only tool that turns an ok.ru/YouTube URL into a file; every alternative needs one anyway.
- **UI — AI proposed Next.js + Tailwind + shadcn → I changed to FastAPI serving one static index.html with vanilla
  JS.** The page has two inputs, one button, a live progress list and a result card; FastAPI ≥ 0.135 streams
  Server-Sent Events natively, so live progress needs no extra package and no Node. Considered: Gradio (least code,
  generic look, less control over the stage-by-stage view), Streamlit (no true push), htmx (fine, but plain
  EventSource is already ~20 lines), NiceGUI (heavier).
- **Exact-frame method — kept binary search on OCR score** (no existing tool does first-frame-exact detection).

Result: dependencies dropped — torch, torchvision, easyocr, Node, Next.js, Tailwind, shadcn. Final stack:
yt-dlp, static-ffmpeg, opencv-python, rapidocr + onnxruntime, faster-whisper, rapidfuzz, fastapi + uvicorn, pytest.

## Phase 3 — Build (Tasks 2-9)

### Task 2 — text matcher

- AI's `normalize()` proposed `replace("'", "'")` — a no-op, same character on both sides.
  → I changed it to map the curly apostrophe `’` to a straight `'`. → Left as written, dialogue
  typed with a curly apostrophe would never match a transcript using a straight one, silently.
- AI's `score_similar` proposed `rapidfuzz.fuzz.token_set_ratio`. → I changed it to
  `token_sort_ratio`. → `token_set_ratio` returns 1.0 for any subset span (a short phrase entirely
  contained in a longer one scores as a perfect match), which would pick the wrong audio window;
  `token_sort_ratio` still tolerates word-order noise from translation without that false-perfect case.

### Task 3 — demo narrative

- AI's plan assumed the given test video would prove the OCR route end-to-end. → I changed the demo
  narrative once Task 3's frame spike found the video has no burned-in subtitles anywhere: audio is the
  hero on this video, and OCR is proven separately, on synthetic clips plus a real YouTube clip with
  burned-in subtitles (Task 9's matrix). → Forcing OCR-only proof onto a video that structurally can't
  supply it would mean either a faked result or an unproven path; audio-frame precision (≈ ±0.1 s ≈ ±2-3
  frames, bounded by Whisper word timestamps) is documented as a limit instead — WhisperX forced
  alignment stays a documented extension, not built now (YAGNI until asked for).

### Task 5 — OCR band preprocessing

- AI's `read_dialogue` prep step converted the subtitle band to grayscale before OCR. → I changed it to
  keep colour, upscale only. → Measured: grayscale made RapidOCR drop a space ("mindrebels" instead of
  "mind rebels"); the same band in colour reads the line exactly, confidence 0.98.

### Task 7 — CLI exit codes

- AI's plan text for `cli.main`'s `SystemExit` handling was `int(e.code or 2)`. → I changed it to
  `int(e.code) if e.code is not None else 2`. → `e.code or 2` turns argparse's `SystemExit(0)` from
  `--help` into exit code 2 (`0 or 2 == 2`) — wrong; `--help` must exit 0.

### Task 8 — audio locator + hybrid retry

- AI's `_load_model` wrapped `WhisperModel(..., device="cuda")` **construction** in try/except, assuming
  a CUDA failure surfaces there. → I changed the fallback to wrap the `model.transcribe(...)` call
  itself, reloading on `device="cpu"` and retrying once: it catches any exception whose message mentions
  cuda/cublas/cudnn/gpu, retries once on CPU; other errors propagate. → On this machine
  `ctranslate2` enumerates a GPU so construction succeeds, but the CUDA runtime DLL (`cublas64_12.dll`)
  is missing — the failure only surfaces on first inference, past the plan's try/except. Confirmed firing
  in the real run's log.
- AI's pipeline retry rule was: OCR misses in the audio window → retry over the **whole video**.
  → I changed it to retry once over the window widened by `±retry_pad_s` (15 s) at the same
  `fullscan_fps`, falling back to the audio timestamp if that also misses too; whole-video scan now only
  fires when no audio window exists at all. → The first real run showed a confident audio match still
  triggering the ≈65-minute whole-video scan, because this video has no burned-in subtitles at all —
  pointless. Cost if wrong: a subtitle appearing more than 15 s from the spoken line is missed and falls
  back to the audio frame (documented limit, docs/APPROACH.md Phase 5).
- AI's `score_contains` scored any OCR read that fuzzy-matched inside the target as up to 1.0
  (`rapidfuzz.fuzz.partial_ratio`). → I changed it to scale that score by
  `coverage = min(1, len(haystack)/len(target))`. → The widened retry surfaced a real false positive:
  frame 7459 OCR-read a single stray "R" and scored 1.00 against a 29-character target, reporting
  `source: ocr`, `HIGH` confidence on a frame with no text at all.

### Task 9 — `--mode audio`

- AI's `pipeline.run` only built a `WhisperLocator` when `locator is None and mode == "hybrid"`.
  → I changed the guard to `mode in ("hybrid", "audio")`. → `--mode audio` with no injected locator
  silently produced `window=None` and always raised `PipelineError`, even on a video with clear matching
  speech — audio-only mode was unusable stand-alone before this fix.

## Plan 1 — Final review fixes

- AI's `refine_first_frame` treated the coarse-scan sample immediately before a hit as a reliable
  "text not yet visible" anchor for the binary search. → I changed it to prove that first: hop
  back up to `MAX_BACK_HOPS` (8) samples while the text is still on screen, looking for a real
  no-match anchor, before binary-searching between it and the hit. → A coarse sample can itself
  already contain the text (the scan step can be several frames wide), so binary-searching against
  it would silently report a frame that isn't actually the first one. When the hop budget runs out
  still on-screen, the result is reported as the last confirmed-matching hop with `exact=False`;
  `pipeline.run` turns that into confidence `MEDIUM` and a note ("text already visible at scan
  start; first frame may be earlier") instead of asserting a frame-exact result it can't back up.

## Plan 2 — Service hardening (2026-08-25)

- AI's first pass left `Config.cache_dir` / `output_dir` as bare relative `Path("cache")` /
  `Path("output")`. → I changed the defaults to `REPO_ROOT / "cache"` / `REPO_ROOT / "output"`
  (`REPO_ROOT = Path(__file__).resolve().parents[2]`, computed once from `config.py`'s own
  location). → A relative path resolves against the process's current working directory, which
  differs between the CLI (run from `backend/`) and the future FastAPI server (run from wherever
  `uvicorn` is started) — a real run surfaced exactly this split: the downloaded mp4 landed in
  `cache/` at the repo root while a transcript landed in `backend/cache/`, two different
  directories for the same conceptual cache.
- AI's `pipeline.run` let `DownloadError`, generic Python exceptions, and (via `coarse_scan`) any
  future cancellation signal all propagate with their native types. → I changed `run()` to raise
  only `PipelineError`: known failures convert as before, `CancelledError` (a `PipelineError`
  subclass defined in `models.py`, not `pipeline.py`, so `text/scanner.py` can raise it without a
  circular import) passes through unchanged, and anything else is caught, logged as an `error`
  `StageEvent`, and re-raised as `PipelineError(f"unexpected failure (...)")`. → The web API (Plan
  2, later tasks) needs exactly one exception type to translate into an HTTP error response;
  without this, every new failure mode downstream (a codec `cv2` doesn't support, a corrupt
  download, an OCR engine crash) would need its own handler in the API layer instead of one.
- AI's pipeline had no way to stop a run once started. → I added `should_cancel: Callable[[], bool]
  | None` threaded through `run()` and `coarse_scan()`, checked after download, after audio
  locate, and once per OCR sample inside the scan loop. → A browser tab closing mid-run (Plan 2's
  UI) needs the backend job to actually stop, not keep OCR-scanning a 15-minute video after nobody
  is listening; checking once per scan sample bounds the worst-case stop latency to a single OCR
  call (under a second) rather than waiting for the whole scan or transcode to finish.

## Phase 1 — Build notes

- **static-ffmpeg download can fail on some networks.** `static_ffmpeg.add_paths()` fetches ffmpeg/ffprobe from
  GitHub with Python `requests`; on my ISP that connection was reset twice (`ConnectionResetError 10054`) while
  `curl` to the same URL worked. Workaround used here: download the zip with curl and unpack it into
  `.venv/Lib/site-packages/static_ffmpeg/bin/win32/`. On a normal network the automatic download works; if it
  doesn't, the README troubleshooting line points here. The code path is unchanged.
