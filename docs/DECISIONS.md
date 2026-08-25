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

## Plan 2 — Web UI (2026-08-25)

- AI's first design for live progress used a WebSocket. → I changed it to Server-Sent Events. → Progress
  only ever flows server→client (stage events); a WebSocket's bidirectionality buys nothing here but a
  reconnect protocol I'd have to write by hand. SSE needs no extra dependency (FastAPI streams
  `text/event-stream` natively) and the browser's built-in `EventSource` already reconnects and resends
  `Last-Event-ID` on its own.
- AI's first API wrote each result/candidate frame to a file under `output/` and served it as a static
  path. → I changed it to render frames on demand from the cached video, `GET /jobs/{id}/frames/{n}.png`.
  → Nothing was cleaning those files up, so concurrent jobs would leak PNGs into `output/` forever; on
  demand, a ten-thumbnail filmstrip costs ten PNG-encode calls against the already-cached video, not ten
  files nobody deletes.
- AI's first `JobReporter` forwarded every `StageEvent` straight through, including a per-OCR-sample
  progress tick from `scan`. → I moved a 200 ms debounce into `JobReporter.emit`, gated to payload-less
  `running` ticks only (in practice `download`'s progress ticks), not into the pipeline. OCR `scan`
  events don't need this debounce at all — the scanner itself already throttles them (`sample_num % 10
  == 0 or is_hit`), so a 45-sample scan emits a handful of events, not 45. → The pipeline shouldn't know
  it has a UI-specific consumer at all — the CLI's `PrintReporter` wants every tick (it just doesn't
  print non-verbose ones), the web job store wants at most 5/s of them. Both reporters can decide that
  independently only if `pipeline.py` stays unaware either exists.
- AI's job store spawned a new thread per `POST /jobs` with no coordination between them. → I added a
  `threading.Lock` in `JobStore` so jobs run one at a time. → RapidOCR (onnxruntime) and faster-whisper
  are both CPU-bound and not documented as re-entrant-safe; two jobs racing on the same process would
  either fight over the CPU with no user-visible benefit or corrupt shared model state — better to queue
  than to guess.
- AI's frontend draft included a light/dark theme toggle. → I cut it: one dark theme, no toggle. → The
  design direction (this plan's brief) commits to a single deliberate look — graphite ground, amber for
  the audio route, teal for OCR — and the only user is an interviewer watching one run once; a toggle
  would be effort spent on a preference nobody here has expressed (same YAGNI discipline as Plan 1's
  WhisperX call).
- AI kept CPU as the contract for Whisper transcription (`_load_model` fell back to CPU on any
  CUDA-flavoured error and stopped there). → I added GPU as the primary path with CPU fallback, not a
  replacement for it. → this machine has an RTX 3050 and `nvidia-smi` works, but faster-whisper's CUDA
  backend (ctranslate2) needs cuBLAS 12 + cuDNN 9 DLLs on the DLL search path that a plain
  `pip install faster-whisper` never provides — without them `WhisperModel(..., device="cuda")`
  constructs fine but `.transcribe()` throws a cublas load error, which is exactly the failure the
  existing fallback already catches. Installing `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` and pointing
  the process at their DLLs (via a new `_ensure_cuda_path()` in `locator.py`, called only when
  `device == "cuda"`) turns that caught failure into a working GPU run instead (measured: earlier cold
  run 2 m 44 s; same-session forced-CPU rerun 141.2 s; GPU 48.9 s — all on the cached test episode's
  `.16k.wav`, `base`/`translate`). Packaging: the CUDA
  packages are ~1 GB and Windows/NVIDIA-specific, so they live in a separate `requirements-gpu.txt`
  rather than `requirements.txt` — a CPU-only machine never downloads them, and `_ensure_cuda_path` is
  guarded (`os.name == "nt"`, and it's a no-op when no `nvidia/*/bin` directory is on `sys.path`) so
  nothing changes for a machine that skips the extras file.

## Plan 4 — Visual verification (2026-08-25)

- AI's brainstorm proposal was to rename nothing and just add a fourth mode. → I decided instead:
  the *new* full mode (audio + OCR + active-speaker) takes over the name **`hybrid`** and becomes
  the default, and the mode that mode name used to mean is renamed **`audio+ocr`**, behaviour
  untouched. → I said it directly: "there are already three modes and none of which needs to be
  changed... i want this to be the fourth mode called as hybrid and the prev hybrid mode can now
  be called audio+ocr and we need not change any of its features" — `hybrid` should mean the *real*
  hybrid (all three evidence types), not stay pinned to its old, narrower meaning; every existing
  test and doc that used `mode="hybrid"` for the old audio+OCR behaviour was swept to
  `mode="audio+ocr"` (Task 2), and history in APPROACH.md's earlier phases keeps the old label with
  a note explaining it, since those runs predate the rename and rewriting history would desync them
  from their own commit record.
- AI's default-mode question was left open in the design gate. → I chose **`hybrid` stays the
  default**, not `audio+ocr`. → A plain run should get the full evidence stack by default; without
  the ASD extras installed it degrades gracefully to the old `audio+ocr` answer plus a
  `[verify:skipped]` line (proven in this task's validation run 5), so nobody without `torch` loses
  functionality by taking the new default.
- AI's initial model idea (in my own brainstorm prompt) was "some model like L or ASD, I don't know
  whether this is good — you do your own research." → I let the AI's own spike (Task 1) settle it
  on **LR-ASD** over the alternatives it considered (Light-ASD, TalkNet, LoCoNet/SPELL, SyncNet, a
  lip-motion heuristic, cloud APIs). → LR-ASD is the only option that is simultaneously CPU-viable
  (0.84 M params, measured ≈0.85-0.9 s forward pass per 10 s window), ships its own weights inside
  the repo (no separate download gate to fail), and is a genuine model rather than a hand-tuned
  heuristic — the lip-motion heuristic stayed on as the documented fallback (spec §10) behind the
  same `SpeakerDetector` protocol, in case the spike had failed its GO/NO-GO gate; it didn't, so the
  heuristic was never built.
- AI's spec draft could have treated "no face detected in the window" the same as "a face was seen
  but wasn't talking." → I ruled explicitly: **"no usable face" is never treated as "not
  speaking."** → I said in the brainstorm: "if faces there and speaking, that is a high match, if
  not, then that is a low match" — but a person can be on screen facing away from the camera, or
  the face can simply be too small/short-lived to track; conflating "we saw nobody" with "we saw
  someone and they weren't talking" would silently downgrade genuinely valid off-camera-face lines
  to the same `invalid` bucket as a real "narrator, not the person on screen" case. The
  classification table keeps them as two separate classes: `uncertain` (no usable track) vs
  `invalid` (a usable track exists, none qualifies).
- AI's `verify_window` runs OCR + face/ASD scoring only inside the *same padded window* Whisper's
  `locate_all()` returned (± `window_pad_s`, 3 s), never widened. → I accepted this as a documented
  limit rather than asking for a fix in this task. → This task's own validation (run 3, the Squid
  Game clip) is what surfaced it: the audio locate window landed on a garbled Korean-to-English
  translation span that doesn't contain the on-screen caption at all, and only `audio+ocr`'s
  separate ±15 s widened-retry fallback (a Plan 1 mechanism `verify_window` never inherited) rescues
  it. The result is user-visible — the new default mode gives a worse answer than the old mode on
  that one clip — and is written up as a Limit in APPROACH.md Phase 7 rather than silently patched,
  since this task's scope is docs and validation, not code.
- AI's LR-ASD probability rule, read naively off the upstream demo script, would have thresholded
  the *raw logit* (`head(outsAV)[:, 1] >= 0`) the way `Columbia_test.py`'s visualization does. → I
  had the spike correct this to the calibrated **softmax probability**
  (`softmax(head(outsAV), dim=-1)[:, 1] >= asd_threshold`). → The raw-logit rule and the softmax
  rule aren't equivalent (`softmax(x)[1] >= 0.5 ⟺ x[1] >= x[0]`, not `x[1] >= 0`); using the
  uncalibrated logit against a probability-shaped config constant (`asd_threshold: 0.5`) would have
  made the threshold meaningless. A later review note corrected the spike's own framing further:
  upstream's published Columbia ASD F1 scores were themselves computed with the raw-logit rule, so
  it's upstream's actual benchmark methodology, not merely a "demo-only visualization shortcut" —
  this codebase still deliberately uses the softmax probability, because it's what makes
  `asd_threshold` a real, comparable number.
- AI's default `Config` had no knobs for the visual stage. → I had `asd_threshold` (0.5),
  `asd_min_active` (0.3), `asd_onset_frames` (3), `min_track_s` (0.5 s), `min_face_px` (40 px),
  `max_occurrences` (5), and `onset_lookback_s` (1.0 s) added as named, documented `Config` fields
  rather than inline constants. → Every other pipeline stage's tunables already live in `Config`
  (`audio_match_threshold`, `ocr_match_threshold`, `window_pad_s`, ...); the visual stage
  shouldn't be the one stage whose thresholds are buried in a module and invisible to anyone
  reading `config.py`.
- A background security review flagged that both model downloads (`faces.py`'s YuNet ONNX and
  `lrasd.py`'s LR-ASD weights) fetched over HTTP with no integrity check. → I ruled: **pin
  SHA-256 hashes for both**, verify after every download and again on every load, delete-and-raise
  on mismatch (not silently retried — a hash mismatch is a security failure, not a transient
  network blip). → An upstream mirror compromise, MITM, or plain corrupted download would otherwise
  load straight into `cv2.FaceDetectorYN` or `torch.load` with no signal to the user; the shared
  `visual/model_files.py:fetch_verified()` helper (added by Task 5's simplifier pass) now carries
  this pattern for both files so a third model added later gets it for free.
- AI's `4:1 audio:video frame ratio` implementation for MFCC extraction, before the spike's
  drift finding, would have used a fixed 100 Hz hop rate (`winstep=0.010`) regardless of the
  video's actual fps. → I had Task 4 change it to **fps-aware**: `winstep = 1 / (4 × fps)`, so
  audio frames are always exactly 4× video frames by construction for the video's *real* frame
  rate (23.976 fps), not the 25 fps LR-ASD was trained on. → The spike measured this mismatch as
  ~4.1% timestamp drift (~0.2-0.25 s per 10 s window) if fed through unmodified; resampling the
  video with a per-window `ffmpeg -r 25` pass would fix the drift but cost an extra encode per
  window for no measured benefit to scoring — the ruling accepted the residual ±0.25 s onset
  accuracy instead of paying that cost, and documented it rather than hiding it.
- AI's first pass on the OCR-hit result in a `hybrid` occurrence reused `Result.text` from the
  ASR-matched transcript line, even for `valid-text` occurrences. → A task review caught this and
  I had it changed: `Occurrence` gained its own `text` field, set from the OCR-read string (not the
  spoken-word transcript) when the occurrence classifies as `valid-text`. → A `valid-text` result
  should show the reader what was actually **on screen**, not what Whisper heard — those two
  strings can differ (translation artefacts, OCR reading the literal on-screen wording), and
  showing the ASR text on an on-screen-text result silently misrepresents which piece of evidence
  won.
- AI's `--occurrence` flag semantics for `hybrid` were unspecified beyond "ranks by class, then ASR
  score." → I ruled `--occurrence` **keeps its pre-Plan-4 meaning within the selected class**:
  class order (`valid > uncertain > invalid`) picks which bucket wins, then `first`/`last`/`all`
  work exactly as they did before — `first`/`all` by highest ASR score then earliest window,
  `last` by temporally latest window in that class. → Reusing the existing flag's meaning inside
  the new class dimension means the flag's contract doesn't silently change for users of the older
  modes, and a `--occurrence last` on a `hybrid` run stays predictable rather than needing a new
  flag or a new meaning to learn.

## Phase 1 — Build notes

- **static-ffmpeg download can fail on some networks.** `static_ffmpeg.add_paths()` fetches ffmpeg/ffprobe from
  GitHub with Python `requests`; on my ISP that connection was reset twice (`ConnectionResetError 10054`) while
  `curl` to the same URL worked. Workaround used here: download the zip with curl and unpack it into
  `.venv/Lib/site-packages/static_ffmpeg/bin/win32/`. On a normal network the automatic download works; if it
  doesn't, the README troubleshooting line points here. The code path is unchanged.
