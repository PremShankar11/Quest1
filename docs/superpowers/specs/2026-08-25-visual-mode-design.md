# Visual verification mode ("hybrid" = audio + OCR + active speaker) — design

Date: 2026-08-25. Status: approved in brainstorm; implementation = Plan 4.

## 1. Purpose

Today the pipeline decides "where the dialogue appears" from two kinds of evidence: the spoken line (Whisper) and on-screen text (OCR). A spoken line can come from someone off screen (voice-over, phone, a character out of frame). The new mode adds a third kind of evidence — **is a visible person speaking the line?** — and uses all three to rank the occurrences of the dialogue and to pick the exact first frame.

Nothing that exists changes behaviour. Only names move:

| CLI/API `mode` | Meaning | Status |
|---|---|---|
| `audio` | Whisper only | unchanged |
| `ocr` | OCR full scan only | unchanged |
| `audio+ocr` | Whisper locates → OCR confirms (was called `hybrid`) | renamed, behaviour identical |
| `hybrid` | **new**: Whisper locates → OCR + face tracks + active-speaker detection → classify → pick → refine | new default |

**Default mode:** stays `hybrid` — which is now the full mode. A plain run therefore gains the verify stage; without the ASD extras it emits `verify: skipped` and produces exactly the old `audio+ocr` answer. (Open question for the user at the review gate: keep `hybrid` as default, or make `audio+ocr` the default?)

Rename scope: CLI choices (`cli.py`), API `JobRequest` Literal (`backend/api/jobs.py`), the page's mode select, README/APPROACH mode tables, and every test that used `mode="hybrid"` for the old behaviour (→ `audio+ocr`). **Measured runs recorded before 2026-08-25 in APPROACH Phases 3, 4 and 6 keep the label `hybrid`, meaning audio+ocr** — history is not rewritten; one note says so.

## 2. Flow of the new `hybrid` mode

1. **Locate (existing locator, extended):** `WhisperLocator.locate_all(video, target) -> list[Window]` returns every non-overlapping transcript window with `score_similar ≥ cfg.audio_match_threshold` (0.6), sorted by score, capped at `cfg.max_occurrences` (5). `locate()` stays as `locate_all(...)[0] or None` so `audio+ocr` is untouched.
2. **Per window** (window ± `window_pad_s`, existing 3 s):
   - **OCR** — existing `coarse_scan` at `window_fps` and, if a hit, the existing binary-search refiner. Produces `ocr_hit: Candidate | None`.
   - **Faces** — YuNet (`cv2.FaceDetectorYN`, OpenCV core, model file ≈ 230 KB fetched once into `cache/models/`) on every frame of the padded window; IoU tracker (threshold 0.5, tolerate 3 missed frames) → `FaceTrack(id, frames: list[(index, bbox)], crops: 112×112 grey)`. Tracks shorter than `cfg.min_track_s` (0.5 s) or with median face height < `cfg.min_face_px` (40 px) are discarded as unusable.
   - **Active speaker** — LR-ASD (Liao et al., IJCV 2025, MIT; 0.84 M params; PyTorch CPU) scores each usable track per frame from its crops + the window's 13-d MFCC at 100 Hz (from the existing 16 kHz wav; `python_speech_features`). Output per track: `scores: list[float]` aligned to frames. LR-ASD runs **only inside candidate windows** — never over the whole video.
   - **Speech mask** — Silero VAD (bundled with faster-whisper, onnxruntime) over the padded window → per-frame `speech: bool`.
3. **Classify the occurrence** (rules in §3) → `Occurrence(window, ocr_hit, tracks, speaker_track, asd_mean, klass, frame_index, note)`.
4. **Select:** class order `valid > uncertain > invalid`; within a class, higher ASR score, then earliest window. `--occurrence last|all` keeps its meaning over the selected class.
5. **Refine the frame** (§4) and build the `Result` (§5).

If the ASD extras are not installed (torch or the weights missing), the `verify` stage emits `skipped` with the reason and the mode behaves exactly like `audio+ocr` (classification uses OCR + audio only; every occurrence without OCR is `uncertain`).

## 3. Classification rules (per occurrence)

| Evidence in the window | Class | Frame | Confidence |
|---|---|---|---|
| OCR hit ≥ `ocr_match_threshold` (0.8) | **valid** (on-screen text) | OCR first frame (existing refiner) | HIGH (≥ 0.9) / MEDIUM |
| No OCR; a usable track has LR-ASD score ≥ `asd_threshold` (0.5) on ≥ `asd_min_active` (30 %) of the window's speech frames | **valid** (on-screen speaker) | visual onset (§4) | HIGH if that track's mean score over speech frames ≥ 0.7, else MEDIUM |
| No OCR; ≥ 1 usable track, none meets the rule | **invalid / off-screen** | first spoken word frame | LOW |
| No OCR; no usable track (none detected, all too short/small, or ASD unavailable) | **uncertain** | first spoken word frame | MEDIUM |

"No usable face" is never treated as "not speaking": a person can be on screen facing away.

Speaker track choice when several qualify: highest mean score over speech frames.

## 4. Exact frame for an on-screen speaker

Search range: `[window.start_s − 1.0 s, window.end_s]` at native fps, on the chosen track. Onset = the first frame `i` such that the track's score ≥ `asd_threshold` for `asd_onset_frames` (3) consecutive frames **and** the speech mask is true at `i`. If no onset is found in range (track begins mid-line, e.g. a cut), fall back to the first spoken word frame and note `"speaker visible; onset not observed (cut)"`. Frames are 0-based; timestamp = `index / fps` as everywhere else.

## 5. Result and events

`Result` gains: `occurrence_class: str` (`valid-text | valid-speaker | uncertain | invalid`), `speaker_box: list[int] | None` (x, y, w, h on the result frame), `occurrences: list[dict]` (per window: `start_s, end_s, asr_score, ocr_score, faces, asd_mean, klass, frame_index`). `source` values: existing `ocr | audio | ocr-weak` plus `audio+asd` for valid-speaker. The saved PNG pair stays; a third image `frame_<n>_speaker.png` with the face box drawn is written when a speaker was found.

New StageEvents: `verify` (`running` per window with `{window_index, faces, asd_mean}`, then `ok | skipped | fallback`) and `occurrences` (`ok`, payload = the list). Existing stages/events are unchanged so the `audio+ocr` UI path does not move.

## 6. Components

```
backend/dialogue_finder/visual/          # third kind of evidence: who is visibly speaking
  __init__.py
  faces.py            # YuNetDetector (cv2.FaceDetectorYN), IouTracker, FaceTrack, crop_112
  audio_features.py   # mfcc_100hz(wav, start_s, end_s) -> np.ndarray (T×13); vad_mask(wav, ...) via faster_whisper.vad
  lrasd.py            # vendored LR-ASD model (MIT), weight download, SpeakerDetector protocol, LrAsdDetector.score(track, mfcc)
  verifier.py         # verify_window(...) -> Occurrence; classify(); pure rules, unit-tested with fakes
backend/dialogue_finder/audio/locator.py # + locate_all()
backend/dialogue_finder/pipeline.py      # + _run_hybrid() branch; mode rename
backend/dialogue_finder/config.py        # + asd_threshold 0.5, asd_min_active 0.3, asd_onset_frames 3, min_track_s 0.5, min_face_px 40, max_occurrences 5, models_dir
requirements-asd.txt                     # torch (CPU wheel), python_speech_features
frontend/                                # mode option, Occurrences block, face box overlay, timeline occurrence marks
```

Also touched: `backend/api/jobs.py` (mode Literal), `backend/dialogue_finder/cli.py` (choices), tests.

Protocols (the only seams): `SpeakerDetector.score(crops: np.ndarray, mfcc: np.ndarray) -> list[float]`; `FaceDetector.detect(frame) -> list[bbox]`. Fakes implement them in tests.

**Spike-owned constants.** The LR-ASD preprocessing above (112×112 grey crops, 13-d MFCC at 100 Hz, `python_speech_features`, 25 fps ↔ 4 audio frames per video frame, weight file name/size, `torch.load` compatibility on Python 3.14 CPU torch, per-window CPU cost) is reconstructed from the TalkNet lineage and is **confirmed or corrected by Plan 4's Task 1 spike** from the vendored repository; the 23.976 fps video is aligned by trimming to the shorter of the two streams as the reference demo does. If the spike fails (weights do not load, or a 10 s window costs > 60 s on CPU), the fallback engine is the lip-motion heuristic of §10 behind the same `SpeakerDetector` protocol — a decision gate, not a rewrite.

**Weight download.** GitHub downloads via Python `requests` were connection-reset on this network once before (static-ffmpeg, Plan 1); `lrasd.py` fetches with retries and the README documents the manual `curl` fallback into `cache/models/`.

## 7. UI

Mode select: `hybrid` (default), `audio+ocr`, `audio`, `ocr`. New **Occurrences** block after Stages: one row per window — timecode range, ASR score, OCR ✓/✗, faces count, speaking face ✓/✗/?, class badge (valid teal, uncertain grey, invalid red). Timeline: each window drawn as a thin mark in the class colour; the amber window stays for the selected one. Result card: the speaker image (face box) replaces the plain result frame when present; `occurrence_class` shown beside the route.

## 8. Testing and evidence

- Unit: `verifier.classify` table (all four rows, ties, several qualifying tracks), onset search (found / not found / cut), `IouTracker` (continuity, gap tolerance, discard rules), `locate_all` (non-overlap, cap, threshold), pipeline selection order with fake detector/ASD/locator on the synthetic clip.
- Integration (slow, real models): the episode window at 05:25 — Holmes visible and speaking → `valid-speaker`, onset frame recorded next to the audio frame 7794; one voice-over clip (narration over B-roll, to be picked in Plan 4) → `invalid` or `uncertain`; the Squid Game clip still → `valid-text`.
- Docs: APPROACH Phase 7 (why a third evidence, why LR-ASD, why windows only, measured per-window cost), DECISIONS entries, README mode table, BENCHMARK unchanged.

## 9. Non-goals

Whole-video ASD; lip-reading; speaker identity; multi-language; running LR-ASD on GPU (torch CPU only on Python 3.14/Windows — documented).

## 10. Alternatives considered

TalkNet-ASD (heavier, licence unclear), Light-ASD (superseded by LR-ASD, same author), LoCoNet/SPELL (34 M params, GPU), SyncNet (lip-sync offset as a proxy), a no-model lip-motion heuristic (YuNet + 106-pt landmarks + MAR/VAD correlation — explainable, no torch, but not a model and weak on profiles), cloud APIs (none fuse audio with faces). Research notes with sources: DECISIONS.md (Plan 4).
