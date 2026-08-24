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
