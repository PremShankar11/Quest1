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
