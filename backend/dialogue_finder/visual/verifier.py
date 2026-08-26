"""Per-window verification: OCR + face tracks + LR-ASD classification (Plan 4, Task 6).

`verify_window()` is the per-window entry point: it runs OCR (the existing `coarse_scan` +
`refine_first_frame`), then -- only if OCR missed -- builds face tracks over the padded window
(`visual.faces`), scores each usable track with LR-ASD against the window's MFCC
(`visual.lrasd` / `visual.audio_features`), and hands the result to the pure `classify()` rule
(spec section 3) to produce an `Occurrence`. `find_onset()` implements the visual-onset search
(spec section 4).

Never raises for expected per-window trouble (no faces, no OCR, an ASD score that never
qualifies): those all produce `invalid`/`uncertain` Occurrences per the classification table.
An *unexpected* exception during the visual scan (a bad frame decode, a model hiccup) is also
converted to an `uncertain` Occurrence with the error captured in `note` and a `verify fallback`
event -- one bad window must never abort the whole run (plan's Global Constraints: "never a
traceback"). `CancelledError`/`PipelineError` are not caught here; they propagate so the run
aborts on user cancellation. `VisualStageUnavailable` (missing weights/model file, offline) also
propagates uncaught: it isn't "one bad window" trouble -- without the detector no window can ever
be scored -- so `pipeline._run_hybrid` catches it once for the whole run and degrades to
audio+ocr (I-4).

Frame-alignment convention (plan's Global Constraints): `FaceTrack.frames[j]` are absolute
video indices, aligned to `scores[j]`. A `speech` list's index `i` corresponds to absolute frame
`first_index + i` -- `first_index` travels alongside every `speech` list passed around in this
module (defaults to 0, the common case in unit tests, where `speech` is already 0-based).
"""
from __future__ import annotations

import statistics
from typing import Callable

import numpy as np

from ..config import Config
from ..models import CancelledError, FaceTrack, Occurrence, PipelineError, StageEvent, Window
from ..progress import ProgressReporter
from ..text.refiner import refine_first_frame
from ..text.scanner import coarse_scan, group_hits, pick_group
from .audio_features import mfcc_for_video, speech_mask
from .faces import FaceDetector, build_tracks, crop_face
from .lrasd import SpeakerDetector
from .model_files import VisualStageUnavailable

CROP_SIZE = 112
SMOOTH_KERNEL = 13   # track-box median filter kernel (controller ruling: lives here, not faces.py)


def _median_smooth_boxes(boxes: list[tuple[int, int, int, int]],
                         kernel: int = SMOOTH_KERNEL) -> list[tuple[int, int, int, int]]:
    """Median-filter each box's centre and size along the track (kernel `kernel`, clamped to an
    odd number <= len(boxes)), before cropping -- smooths YuNet's per-frame jitter without
    touching detection or tracking. Pure function of the box sequence; independent of frame
    numbers, so it applies whether or not the track has gaps."""
    n = len(boxes)
    if n == 0:
        return []
    k = min(kernel, n if n % 2 == 1 else n - 1)
    k = max(1, k)
    half = k // 2
    cx = [b[0] + b[2] / 2 for b in boxes]
    cy = [b[1] + b[3] / 2 for b in boxes]
    w = [float(b[2]) for b in boxes]
    h = [float(b[3]) for b in boxes]

    def smoothed(series: list[float]) -> list[float]:
        return [statistics.median(series[max(0, i - half):min(n, i + half + 1)]) for i in range(n)]

    scx, scy, sw, sh = smoothed(cx), smoothed(cy), smoothed(w), smoothed(h)
    return [(int(round(scx[i] - sw[i] / 2)), int(round(scy[i] - sh[i] / 2)),
             int(round(sw[i])), int(round(sh[i]))) for i in range(n)]


def _fill_gaps(frames: list[int],
               boxes: list[tuple[int, int, int, int]]) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    """Expand a (possibly gappy) track's frames/boxes to every frame in `[frames[0], frames[-1]]`,
    filling a missing frame with its nearest known neighbour's box -- LR-ASD needs a contiguous
    crop sequence 1:1 with the window's contiguous MFCC (`IouTracker` tolerates up to 3 missed
    frames per track, so gaps are short)."""
    if not frames:
        return [], []
    box_by_frame = dict(zip(frames, boxes))
    known = sorted(box_by_frame)
    full_frames = list(range(frames[0], frames[-1] + 1))
    full_boxes = [box_by_frame[f] if f in box_by_frame else box_by_frame[min(known, key=lambda k: abs(k - f))]
                 for f in full_frames]
    return full_frames, full_boxes


def classify(ocr_score: float, tracks: list[FaceTrack], speech: list[bool], cfg: Config,
            first_index: int = 0, window_start_index: int | None = None,
            window_end_index: int | None = None) -> tuple[str, FaceTrack | None, float]:
    """Spec section 3's classification table, evaluated for one window.

    `speech[i]` is the speech mask at absolute frame `first_index + i`. Returns
    `(klass, speaker_track, mean_score)`: `speaker_track`/`mean_score` are populated only for
    "valid-speaker" (`None`/`0.0` otherwise). A track "qualifies" when its score is
    >= `cfg.asd_threshold` on >= `cfg.asd_min_active` of the speech frames *inside the located
    window* `[window_start_index, window_end_index]` (both inclusive, absolute frame indices) --
    NOT the padded region `speech` may extend into; a face speaking only during the ±3 s padding
    must not count either way (denominator, numerator, or the mean). `window_start_index`/
    `window_end_index` default to `speech`'s whole span (via `first_index`), so callers that pass
    an already-windowed `speech` list (every existing unit test) are unaffected. The qualifying
    track with the highest mean score over in-window speech frames wins ties.
    """
    if ocr_score >= cfg.ocr_match_threshold:
        return "valid-text", None, 0.0

    if window_start_index is None:
        window_start_index = first_index
    if window_end_index is None:
        window_end_index = first_index + len(speech) - 1

    def in_window(i: int) -> bool:
        abs_frame = first_index + i
        return window_start_index <= abs_frame <= window_end_index

    n_speech = sum(1 for i, s in enumerate(speech) if s and in_window(i))
    best_track: FaceTrack | None = None
    best_mean = 0.0
    if n_speech > 0:
        for t in tracks:
            score_by_frame = dict(zip(t.frames, t.scores))
            active = 0
            speech_scores: list[float] = []
            for i, sp in enumerate(speech):
                if not sp or not in_window(i):
                    continue
                sc = score_by_frame.get(first_index + i)
                if sc is None:
                    continue
                speech_scores.append(sc)
                if sc >= cfg.asd_threshold:
                    active += 1
            if active / n_speech >= cfg.asd_min_active:
                mean = sum(speech_scores) / len(speech_scores) if speech_scores else 0.0
                if best_track is None or mean > best_mean:
                    best_track, best_mean = t, mean

    if best_track is not None:
        return "valid-speaker", best_track, best_mean
    if tracks:
        return "invalid", None, 0.0
    return "uncertain", None, 0.0


def find_onset(track: FaceTrack, speech: list[bool], first_index: int, cfg: Config) -> int | None:
    """First absolute frame `i` (>= `first_index`) such that `track`'s score is
    >= `cfg.asd_threshold` for `cfg.asd_onset_frames` consecutive frames starting at `i`, and
    `speech` is True at `i` (spec section 4). `speech[k]` is speech at absolute frame
    `first_index + k`. Returns `None` if no such run exists in `speech`'s range."""
    score_by_frame = dict(zip(track.frames, track.scores))
    k = cfg.asd_onset_frames
    for i, sp in enumerate(speech):
        if not sp:
            continue
        if all((sc := score_by_frame.get(first_index + i + j)) is not None and sc >= cfg.asd_threshold
              for j in range(k)):
            return first_index + i
    return None


def confidence_for_occurrence(occ: Occurrence) -> str:
    """Spec section 3's confidence column, evaluated from a classified `Occurrence`. An inexact
    OCR refine (`occ.exact is False` -- text already visible at scan start, first frame may be
    earlier) forces MEDIUM regardless of score, matching audio+ocr's `confidence_for` override."""
    if occ.klass == "valid-text":
        if not occ.exact:
            return "MEDIUM"
        return "HIGH" if occ.ocr_score >= 0.9 else "MEDIUM"
    if occ.klass == "valid-speaker":
        return "HIGH" if occ.asd_mean >= 0.7 else "MEDIUM"
    if occ.klass == "invalid":
        return "LOW"
    return "MEDIUM"   # uncertain


def _ocr_occurrence(src, window: Window, target: str, extractor, a: float, b: float, fps: float,
                    cfg: Config, reporter: ProgressReporter,
                    should_cancel: Callable[[], bool] | None) -> Occurrence | None:
    """`verify_window`'s OCR stage: `coarse_scan` the padded window and, on a miss, retry ONCE
    over the window widened by `cfg.retry_pad_s` at `cfg.fullscan_fps` -- the identical widened
    retry `run()`'s audio+ocr branch uses (`pipeline.retry_ocr_scan_if_missed`; a subtitle just
    outside the ASD window, e.g. `window_pad_s` away, is otherwise missed and the occurrence
    misclassified `invalid`/`uncertain` -- regression fix). Then `refine_first_frame`, same as
    `audio+ocr`. Returns the `valid-text` Occurrence for the first hit (post-retry), or None if
    OCR missed even after the retry.

    `pipeline.retry_ocr_scan_if_missed` is imported here (not at module level): `pipeline.py`
    already imports this module at its own top level, so a top-level import here would be
    circular; by the time this function runs, `pipeline` is always already fully loaded.
    """
    from ..pipeline import retry_ocr_scan_if_missed

    reporter.emit(StageEvent("scan", "running", f"OCR {a:.1f}-{b:.1f}s at {fps} fps"))
    cands = coarse_scan(src, extractor, target, a, b, fps, cfg, reporter, should_cancel=should_cancel)
    groups = pick_group(group_hits(cands, cfg.ocr_match_threshold, cfg.hit_gap_s), "first")
    cands, groups, fps = retry_ocr_scan_if_missed(src, extractor, target, cands, groups, fps, window, cfg,
                                                  reporter, "first", should_cancel)
    if not groups:
        return None
    first_hit = groups[0][0]
    step = max(1, int(round(src.fps / fps)))   # fps may now be cfg.fullscan_fps if a retry hit
    prev_index = max(0, first_hit.frame_index - step)
    ocr_cand, exact = refine_first_frame(src, extractor, target, first_hit.frame_index, prev_index, cfg, step=step)
    note = "" if exact else "text already visible at scan start; first frame may be earlier"
    return Occurrence(window=window, klass="valid-text", frame_index=ocr_cand.frame_index,
                      ocr_score=ocr_cand.score, faces=0, asd_mean=0.0, speaker_box=None, note=note,
                      text=ocr_cand.text, exact=exact)


def _score_tracks(src, detector: FaceDetector, speaker: SpeakerDetector, wav, a: float, b: float,
                  start_index: int, cfg: Config, reporter: ProgressReporter,
                  should_cancel: Callable[[], bool] | None) -> tuple[list[FaceTrack], list[bool], int]:
    """Builds face tracks over the padded window and scores each against the window's MFCC
    (smoothing boxes, filling gaps, and cropping first -- see `_median_smooth_boxes`/`_fill_gaps`).
    Returns `(scored_tracks, speech, faces)`; `faces` is the raw track count (before scoring),
    which stays the `Occurrence.faces` value even for a class that has no speaker track."""
    end_index = src.index_for_time(b)
    if hasattr(src, "prefetch_range"):
        src.prefetch_range(start_index, end_index)
    tracks = build_tracks(src, detector, start_index, end_index, cfg, reporter, should_cancel)
    speech = speech_mask(wav, a, b, src.fps)
    scored_tracks: list[FaceTrack] = []
    if tracks:
        mfcc = mfcc_for_video(wav, a, b, src.fps)
        for t in tracks:
            smoothed = _median_smooth_boxes(t.boxes)
            full_frames, full_boxes = _fill_gaps(t.frames, smoothed)
            crops = np.stack([crop_face(src.frame_at(f), box, CROP_SIZE)
                              for f, box in zip(full_frames, full_boxes)])
            rel = full_frames[0] - start_index
            track_mfcc = mfcc[4 * rel: 4 * rel + 4 * len(full_frames)]
            scores = speaker.score(crops, track_mfcc)
            n = len(scores)
            scored_tracks.append(FaceTrack(track_id=t.track_id, frames=full_frames[:n],
                                           boxes=full_boxes[:n], scores=scores))
    return scored_tracks, speech, len(tracks)


def _valid_speaker_occurrence(src, window: Window, speaker_track: FaceTrack, asd_mean: float, speech: list[bool],
                              a: float, start_index: int, fallback_frame: int, faces: int,
                              cfg: Config) -> Occurrence:
    """Builds the `valid-speaker` Occurrence via `find_onset()`'s visual-onset search over the
    lookback window, falling back to `fallback_frame` (first spoken word) if no onset is found."""
    onset_start_s = max(a, window.start_s - cfg.onset_lookback_s)
    onset_start_index = src.index_for_time(onset_start_s)
    end_search_index = src.index_for_time(window.end_s)
    offset = onset_start_index - start_index
    onset_speech = speech[offset: offset + max(0, end_search_index - onset_start_index + 1)]
    onset = find_onset(speaker_track, onset_speech, onset_start_index, cfg)
    if onset is not None:
        frame_index = onset
        note = f"on-screen speaker verified (LR-ASD mean {asd_mean:.2f})"
    else:
        frame_index = fallback_frame
        note = "speaker visible; onset not observed (cut)"
    box_by_frame = dict(zip(speaker_track.frames, speaker_track.boxes))
    nearest = min(box_by_frame, key=lambda f: abs(f - frame_index)) if box_by_frame else None
    speaker_box = box_by_frame.get(nearest) if nearest is not None else None
    return Occurrence(window=window, klass="valid-speaker", frame_index=frame_index, ocr_score=0.0,
                      faces=faces, asd_mean=asd_mean, speaker_box=speaker_box, note=note)


def verify_window(src, window: Window, target: str, extractor, detector: FaceDetector,
                  speaker: SpeakerDetector, wav, cfg: Config, reporter: ProgressReporter,
                  should_cancel: Callable[[], bool] | None = None) -> Occurrence:
    """OCR + face tracks + LR-ASD for one candidate window -> a classified `Occurrence`.

    OCR runs first (`_ocr_occurrence`); an OCR hit short-circuits straight to `valid-text`
    without touching the visual stage at all (the classification table's first row). Otherwise
    scores face tracks (`_score_tracks`), classifies (`classify`), and -- for "valid-speaker" --
    locates the visual onset (`_valid_speaker_occurrence`).
    """
    a, b = window.padded(src.duration_s, cfg.window_pad_s)
    fps = cfg.window_fps
    fallback_frame = src.index_for_time(window.start_s)

    ocr_occ = _ocr_occurrence(src, window, target, extractor, a, b, fps, cfg, reporter, should_cancel)
    if ocr_occ is not None:
        return ocr_occ

    start_index = src.index_for_time(a)
    try:
        scored_tracks, speech, faces = _score_tracks(src, detector, speaker, wav, a, b, start_index, cfg,
                                                      reporter, should_cancel)
        klass, speaker_track, asd_mean = classify(0.0, scored_tracks, speech, cfg, first_index=start_index,
                                                  window_start_index=src.index_for_time(window.start_s),
                                                  window_end_index=src.index_for_time(window.end_s))
    except (CancelledError, PipelineError, VisualStageUnavailable):
        raise
    except Exception as e:
        reporter.emit(StageEvent("verify", "fallback",
                                 f"window {a:.1f}-{b:.1f}s: {type(e).__name__}: {e}"))
        return Occurrence(window=window, klass="uncertain", frame_index=fallback_frame, ocr_score=0.0,
                          faces=0, asd_mean=0.0, speaker_box=None,
                          note=f"visual stage error: {type(e).__name__}: {e}")

    if klass == "valid-speaker":
        return _valid_speaker_occurrence(src, window, speaker_track, asd_mean, speech, a, start_index,
                                         fallback_frame, faces, cfg)

    if klass == "invalid":
        return Occurrence(window=window, klass=klass, frame_index=fallback_frame, ocr_score=0.0,
                          faces=faces, asd_mean=0.0, speaker_box=None,
                          note="faces visible but none speaking; frame at first spoken word")

    return Occurrence(window=window, klass="uncertain", frame_index=fallback_frame, ocr_score=0.0,
                      faces=faces, asd_mean=0.0, speaker_box=None,
                      note="no usable face in the window; frame at first spoken word")
