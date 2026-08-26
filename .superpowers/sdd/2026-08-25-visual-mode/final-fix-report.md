# Final fix wave report (plan 4)

Branch `plan-4-visual-mode`, starting HEAD `88a5f0a`, working tree was clean. TDD per item (test
first, watched it fail, implemented, watched it pass); fast subset while iterating, full suite
once before commit, real episode run before commit.

## I-1 — Lazy `faster_whisper` import

`backend/dialogue_finder/visual/audio_features.py:31` — `from faster_whisper.vad import
get_speech_timestamps, VadOptions` moved from module top level into `speech_mask()`
(`audio_features.py:147`). Confirmed empirically that `faster_whisper.vad` alone (no other
import) pulls in `torch` transitively.

Covering test (fast): `backend/tests/test_pipeline.py::test_importing_pipeline_does_not_pull_in_torch`
— runs `sys.modules.pop("torch", None); importlib.import_module("dialogue_finder.pipeline");
assert "torch" not in sys.modules` in a subprocess (`cwd=backend/`), asserts returncode 0.

Import-time (fresh subprocess, `import dialogue_finder.pipeline`), 3 runs each:
- **Before**: 2.10 s (single measured run; brief estimated 2.04 s)
- **After**: 0.246 s, 0.232 s, 0.229 s (~0.23 s)

## I-2 — `--occurrence last|all` in hybrid

`backend/dialogue_finder/pipeline.py::_select_occurrence` — added 4 unit tests in
`test_pipeline.py` (`_occ`/`_three_occurrences` helpers: valid-speaker@5s score 0.9,
invalid@2s score 0.99, valid-speaker@9s score 0.7):
- `test_select_occurrence_first_picks_highest_score_then_earliest` — `first` → the 5 s one
  (highest score, not accident of position; the invalid@2s scores higher but its class loses)
- `test_select_occurrence_last_picks_temporally_last_of_selected_class` — `last` → the 9 s one
- `test_select_occurrence_all_reports_first_pick_and_alternatives` — `all` → same pick as
  `first`, `alternatives` = the other valid-speaker
- `test_select_occurrence_never_picks_invalid_while_valid_exists` — all three modes

**No bug found.** All 4 passed on first run — the existing `_select_occurrence` implementation
already matched the brief's rules exactly.

## I-3 — Speech denominator = the located window, not the padded region

`backend/dialogue_finder/visual/verifier.py::classify` — added `window_start_index: int | None
= None` / `window_end_index: int | None = None` params (default `None` → falls back to
`first_index` / `first_index + len(speech) - 1`, i.e. the whole `speech` span, so all 5
pre-existing `classify` tests are unaffected). The denominator (`n_speech`), the numerator
(`active`), and the mean (`speech_scores`) are now all restricted to frames where
`window_start_index <= first_index + i <= window_end_index` — a track active only in the ±3 s
padding no longer counts either way. `verify_window` (`verifier.py:270-272`) now passes
`window_start_index=src.index_for_time(window.start_s)`,
`window_end_index=src.index_for_time(window.end_s)`.

Covering tests in `test_visual_verifier.py`:
- `test_classify_line_active_only_qualifies_when_denominator_is_windowed` — 26-frame line
  inside 90 frames of continuous padded speech, track active only during the line: old
  denominator 26/90 = 0.289 < 0.3 (would misclassify `invalid`); new denominator 26/26 = 1.0 →
  `valid-speaker`
- `test_classify_track_active_only_in_padding_does_not_qualify` — track active only outside
  the window → `invalid`, not `valid-speaker`

Docs updated: `docs/superpowers/specs/2026-08-25-visual-mode-design.md` §3 and
`docs/APPROACH.md` Phase 7 classification table, both changed "of the window's speech frames"
→ "of the speech frames inside the located window (not the ±3 s padding)".

## I-4 — Missing weights + offline → degrade to audio+ocr, once

- `backend/dialogue_finder/visual/model_files.py` — new `VisualStageUnavailable(RuntimeError)`
  base class.
- `backend/dialogue_finder/visual/lrasd.py` — new `SpeakerDetectorUnavailable
  (VisualStageUnavailable)`; `LrAsdDetector.__init__` gained `self._load_error`; `_load()`
  checks it first and re-raises immediately, otherwise wraps only `_ensure_weights`'s `OSError`
  (download failure / `WeightsVerificationError`) into `SpeakerDetectorUnavailable` and caches
  it — a torch/state-dict error (unrelated to weights availability) is NOT wrapped, so it still
  falls through to `verify_window`'s generic `except Exception` → per-window `uncertain`.
- `backend/dialogue_finder/visual/faces.py` — new `FaceDetectorUnavailable
  (VisualStageUnavailable)`; `YuNetDetector.__init__` gained `self._load_error`; `_ensure()`
  wraps `_model_path()`'s `OSError`/`RuntimeError` (download failure / hash mismatch) the same
  way, cached.
- `backend/dialogue_finder/visual/verifier.py::verify_window` — `except (CancelledError,
  PipelineError, VisualStageUnavailable): raise` added before the generic `except Exception`,
  so `VisualStageUnavailable` propagates instead of becoming `uncertain`.
- `backend/dialogue_finder/pipeline.py::run` — the `mode == "hybrid" and hybrid_ready and
  windows` branch now wraps `_run_hybrid(...)` in `try/except VisualStageUnavailable as e:
  reporter.emit(StageEvent("verify", "skipped", str(e)))` and falls through (no `return`) to
  the existing audio+ocr scan code below, which already has `window`/`windows` set from the
  locate step — identical to the `hybrid_ready=False` path.

Covering tests:
- `test_pipeline.py::test_hybrid_degrades_to_audio_ocr_when_speaker_detector_unavailable` — a
  fake `SpeakerDetector.score()` raising `SpeakerDetectorUnavailable("weights missing
  (offline)")` → result equals the audio+ocr answer (`source="audio"`, `frame_index=120`), one
  `verify skipped` event, no `occurrences` event
- `test_visual_lrasd.py::test_load_wraps_weights_download_failure` and
  `test_load_failure_is_cached_and_not_retried` — monkeypatch `lrasd.fetch_verified` to raise
  `IOError`; second `_load()` call doesn't call `fetch_verified` again (call count stays 1)
- `test_visual_faces.py::test_ensure_caches_load_failure_and_does_not_retry` — same pattern for
  `YuNetDetector._ensure()` / `faces._download_yunet`

## Minor items — all fixed

- **YuNet box clipping** — `faces.py::YuNetDetector.detect` now clips via a new
  `_clip_box(f, w, h)` static method: `x2, y2 = min(w, x+bw), min(h, y+bh)` computed *before*
  clamping `x, y = max(0, x), max(0, y)`, so `w, h = x2-x, y2-y` never poke past the frame.
  Tests: `test_detect_clips_negative_origin_to_zero` (raw `218,-15,304,410` on a 1000×1000
  frame → `(218, 0, 304, 395)`, the exact real-episode repro) and
  `test_detect_clips_box_extending_past_frame_edges` (box overflowing the right/bottom edge).
  Confirmed on the real episode run below: `Speaker : 218,0,304,395` (was `218,-15,304,410`).
- **Confidence parity** — `Occurrence` gained `exact: bool = True`
  (`backend/dialogue_finder/models.py`); `verifier._ocr_occurrence` now passes `exact=exact`
  through; `confidence_for_occurrence` returns `MEDIUM` whenever `occ.exact is False`,
  regardless of `ocr_score`, matching `pipeline.confidence_for`'s existing override. Tests:
  `test_confidence_for_occurrence_inexact_refine_is_medium_even_at_high_score` (score 0.97,
  `exact=False` → MEDIUM) and `..._exact_high_score_refine_is_high` (score 0.97, `exact=True`
  → HIGH).
- **`refine ok` + appearance in hybrid** — `pipeline._run_hybrid` now emits
  `StageEvent("refine", "ok", ..., {"frame_index": selected.frame_index})` and sets
  `appearance = classify_appearance(src, selected.frame_index, cfg)` for `valid-text`/
  `valid-speaker` selections only (uncertain/invalid keep `appearance=""`, no refine event —
  matching the old audio-fallback path). Tests:
  `test_hybrid_valid_text_emits_refine_ok_and_sets_appearance` and
  `test_hybrid_uncertain_result_has_no_appearance_or_refine_event`.
- **Speaker endpoint hardening** — `JobStore` (`backend/api/jobs.py`) gained `cfg: Config =
  DEFAULT`, stored as `self.cfg` and threaded through `_serialised_run`'s `run_job(job,
  self.cfg)` calls (previously always `DEFAULT` regardless of what was passed to `JobStore()`).
  `api/main.py::job_speaker_image` now 404s unless
  `Path(path).resolve().is_relative_to(store.cfg.output_dir.resolve())` in addition to existing.
  Tests: `test_speaker_png_200_from_result_path` rewritten to monkeypatch `store.cfg` to a tmp
  `Config` and write the PNG under `tmp_cfg.output_dir / job.id /` (matching `run_job`'s
  `job_cfg.output_dir`); new `test_speaker_png_404_outside_output_dir` (a real file outside
  `output_dir` → 404).
- **Stale docstring** — `test_pipeline.py::test_hybrid_without_extras_matches_audio_ocr`'s
  docstring "no verify stage at all" → "verify: skipped then the audio+ocr answer".
- **`scan running` before `fallback` in hybrid** — `verifier._ocr_occurrence` now emits
  `StageEvent("scan", "running", f"OCR {a:.1f}-{b:.1f}s at {fps} fps")` (no payload, mirroring
  `run()`'s own message format) immediately before `coarse_scan`. Test:
  `test_hybrid_ocr_occurrence_emits_scan_running_before_scan_fallback` (flip-clip widened-retry
  scenario) asserts a payload-less `scan running` event precedes the `scan fallback` event.
  Confirmed on the real episode run below (`[scan:running] OCR 322.1-330.8s at 5.0 fps` before
  `[scan:fallback] no match in window; retrying...`).

## Test counts

- Baseline (HEAD `88a5f0a`): 131 total (124 fast + 7 slow), all passing.
- Fast subset (`-m "not slow"`) after every item: 135 → 137 → 139 → 141 → 142 → 143 (I-1 through
  the last minor), all green throughout.
- Full suite (`../.venv/Scripts/python -m pytest -q`): **150 passed**, 0 failed, in 83 s.

## Real episode run (before commit)

```
../.venv/Scripts/python -m dialogue_finder --local ../cache/5f39d4605665a831.mp4 \
  --text "My mind rebels at stagnation"
```

```
[transcribe:running] transcribing audio
[verify:running] window 0: 325.1-327.8s
[scan:running] OCR 322.1-330.8s at 5.0 fps
[scan:fallback] no match in window; retrying 310-343s at 2.0 fps
[verify:ok] window 0: valid-speaker
[occurrences:ok] 1 occurrences classified
[done:ok] hybrid result
Timestamp : 00:05:25.365
Frame     : 7801
Text      : "My mind rebels its stagnation."
Confidence: HIGH  (source: audio+asd; on-screen speaker verified (LR-ASD mean 0.89))
Image     : C:\Users\Asus\Quest1\output\frame_7801.png
Previous  : C:\Users\Asus\Quest1\output\frame_7800.png  (fade-in)
Occurrence: valid-speaker
Speaker   : 218,0,304,395
```

Matches the documented pre-existing answer (`valid-speaker`, frame 7801) exactly, with two
visible effects of this wave: the `[scan:running]` event now precedes `[scan:fallback]` (Minor:
scan running before fallback), and `Speaker : 218,0,304,395` — `y` clipped from `-15` to `0`
and `h` from `410` to `395` (Minor: YuNet box clipping), confirming the fix on real data.
Confidence stayed HIGH (asd_mean 0.89 ≥ 0.7) — the I-3 windowing change did not flip it on this
clip, which is within the documented acceptable range (a HIGH↔MEDIUM flip would also have been
acceptable per the review).

## Commit

One commit, only the 16 files touched (no output/cache artifacts): `backend/api/jobs.py`,
`backend/api/main.py`, `backend/dialogue_finder/models.py`, `backend/dialogue_finder/pipeline.py`,
`backend/dialogue_finder/visual/{audio_features,faces,lrasd,model_files,verifier}.py`,
`backend/tests/test_{api,pipeline,visual_faces,visual_lrasd,visual_verifier}.py`,
`docs/APPROACH.md`, `docs/superpowers/specs/2026-08-25-visual-mode-design.md`, and this report.

## Not done / deviations

None — every brief item (I-1 through I-4, all 6 minors) was completed exactly as specified. The
"Do NOT" list was respected: `lrasd_model.py`, the frontend, and requirements files were
untouched; `best_word_window`/helper relocation was not refactored; the verify stage order was
not changed.
