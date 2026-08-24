from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from .config import DEFAULT, Config
from .video.downloader import DownloadError, fetch_video, probe
from .video.frame_source import FrameSource
from .models import Candidate, Result, StageEvent, Window
from .progress import NullReporter, ProgressReporter
from .text.refiner import classify_appearance, refine_first_frame
from .text.scanner import coarse_scan, group_hits, pick_group

if TYPE_CHECKING:
    from .text.ocr import TextExtractor


class PipelineError(Exception):
    """Fatal, user-facing. The CLI prints str(e) and exits 1."""


def confidence_for(source: str, ocr_score: float, window: Window | None) -> str:
    if source == "ocr":
        return "HIGH" if ocr_score >= 0.9 else "MEDIUM"
    if source == "audio":
        return "MEDIUM"
    return "LOW"


def _save_frame_images(src: FrameSource, index: int, cfg: Config) -> tuple[str, str]:
    """Write the frame at `index` (and its predecessor, if any) as PNGs; return (path, prev_path)."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.output_dir / f"frame_{index}.png"
    cv2.imwrite(str(p), src.frame_at(index))
    prev = ""
    if index > 0:
        pp = cfg.output_dir / f"frame_{index - 1}.png"
        cv2.imwrite(str(pp), src.frame_at(index - 1))
        prev = str(pp)
    return str(p), prev


def _default_extractor() -> TextExtractor:
    from .text.ocr import RapidOCRExtractor
    return RapidOCRExtractor()


def _scan_for_groups(src: FrameSource, ex: TextExtractor, target: str, a: float, b: float, fps: float,
                     cfg: Config, reporter: ProgressReporter,
                     occurrence: str) -> tuple[list[Candidate], list[list[Candidate]]]:
    """OCR-scan [a, b] and group the hits into candidate occurrences."""
    cands = coarse_scan(src, ex, target, a, b, fps, cfg, reporter)
    groups = pick_group(group_hits(cands, cfg.ocr_match_threshold, cfg.hit_gap_s), occurrence)
    return cands, groups


def _finish(src: FrameSource, cfg: Config, reporter: ProgressReporter, timings: dict[str, float], t0: float,
           frame_index: int, done_msg: str, **result_kwargs) -> Result:
    """Save the result frame + its predecessor, stamp total elapsed time, emit `done`, and build the Result."""
    img, prev = _save_frame_images(src, frame_index, cfg)
    timings["total"] = time.perf_counter() - t0
    reporter.emit(StageEvent("done", "ok", done_msg))
    return Result(frame_index=frame_index, fps=src.fps, image_path=img, prev_image_path=prev,
                 timings_s=timings, **result_kwargs)


def run(source_spec: str, target: str, *, cfg: Config = DEFAULT, reporter: ProgressReporter | None = None,
        mode: str = "hybrid", occurrence: str = "first", local: bool = False,
        extractor=None, locator=None) -> Result:
    reporter = reporter or NullReporter()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # ---- download / open --------------------------------------------------
    try:
        if local:
            video = Path(source_spec)
            if not video.exists():
                raise DownloadError(f"File not found: {video}")
            reporter.emit(StageEvent("download", "ok", f"using local file {video.name}", 1.0))
        else:
            video = fetch_video(source_spec, cfg, reporter)
        probe(video)  # validates the file is a readable video; raises DownloadError otherwise
    except DownloadError as e:
        # no reporter.emit here: cli.main already prints "Error: {e}" for PipelineError,
        # and StageEvent(..., progress=None) bypasses PrintReporter's non-verbose filter,
        # which would duplicate the message on stderr and break "starts with Error:" checks.
        raise PipelineError(str(e)) from e
    timings["download"] = time.perf_counter() - t0

    # ---- locate by audio ----------------------------------------------------
    window: Window | None = None
    t1 = time.perf_counter()
    if mode in ("hybrid", "audio"):
        if locator is None and mode in ("hybrid", "audio"):
            try:
                from .audio.locator import WhisperLocator
                locator = WhisperLocator(cfg, reporter)
            except Exception as e:                      # module missing until Task 8, or import failure
                reporter.emit(StageEvent("locate", "skipped", f"audio locator unavailable: {e}"))
        if locator is not None:
            try:
                reporter.emit(StageEvent("transcribe", "running", "transcribing audio"))
                window = locator.locate(video, target)
                if window is None:
                    reporter.emit(StageEvent("locate", "fallback", "no audio match; will scan whole video"))
                else:
                    reporter.emit(StageEvent("locate", "ok", f"window {window.start_s:.1f}-{window.end_s:.1f}s "
                                             f"score {window.score:.2f}: '{window.matched_text}'", 1.0,
                                             {"window": window.__dict__}))
            except Exception as e:
                reporter.emit(StageEvent("locate", "fallback", f"audio stage failed: {e}"))
                window = None
    else:
        reporter.emit(StageEvent("locate", "skipped", f"mode={mode}"))
    timings["locate"] = time.perf_counter() - t1

    with FrameSource(video) as src:
        # ---- audio-only mode -------------------------------------------------
        if mode == "audio":
            if window is None:
                raise PipelineError("No audio match found for the target text (mode=audio).")
            idx = src.index_for_time(window.start_s)
            return _finish(src, cfg, reporter, timings, t0, idx, "audio-only result",
                           timestamp_s=src.time_for_index(idx), text=window.matched_text, confidence="MEDIUM",
                           source="audio", note="mode=audio; frame at first spoken word", window=window)

        # ---- scan -------------------------------------------------------------
        ex = extractor or _default_extractor()
        t2 = time.perf_counter()
        if window is not None:
            a, b = max(0.0, window.start_s - cfg.window_pad_s), min(src.duration_s, window.end_s + cfg.window_pad_s)
            fps = cfg.window_fps
            reporter.emit(StageEvent("scan", "running", f"OCR {a:.1f}-{b:.1f}s at {fps} fps"))
        else:
            a, b, fps = 0.0, src.duration_s, cfg.fullscan_fps
            reporter.emit(StageEvent("scan", "running" if mode == "ocr" else "fallback",
                                     f"OCR whole video ({b:.0f}s) at {fps} fps"))
        cands, groups = _scan_for_groups(src, ex, target, a, b, fps, cfg, reporter, occurrence)
        if not groups and window is not None and mode == "hybrid":
            # the window missed; retry a widened window around it (not the whole video) before giving up on OCR
            r_a = max(0.0, window.start_s - cfg.retry_pad_s)
            r_b = min(src.duration_s, window.end_s + cfg.retry_pad_s)
            reporter.emit(StageEvent("scan", "fallback",
                                     f"no match in window; retrying {r_a:.0f}-{r_b:.0f}s at {cfg.fullscan_fps} fps"))
            cands, groups = _scan_for_groups(src, ex, target, r_a, r_b, cfg.fullscan_fps, cfg,
                                             reporter, occurrence)
            if groups:
                fps = cfg.fullscan_fps
        timings["scan"] = time.perf_counter() - t2
        step = max(1, int(round(src.fps / fps)))

        # ---- refine -----------------------------------------------------------
        t3 = time.perf_counter()
        if groups:
            refined: list[Candidate] = []
            for g in groups:
                hit = g[0]
                refined.append(refine_first_frame(src, ex, target, hit.frame_index, hit.frame_index - step, cfg))
            best = refined[0]
            reporter.emit(StageEvent("refine", "ok", f"first frame {best.frame_index} score {best.score:.2f}", 1.0,
                                     {"frame_index": best.frame_index}))
            appearance = classify_appearance(src, best.frame_index, cfg)
            timings["refine"] = time.perf_counter() - t3
            return _finish(src, cfg, reporter, timings, t0, best.frame_index, "result ready",
                           timestamp_s=best.timestamp_s, text=best.text,
                           confidence=confidence_for("ocr", best.score, window), source="ocr",
                           appearance=appearance, window=window, candidates=cands, alternatives=refined[1:])

        # ---- fallbacks ----------------------------------------------------------
        if window is not None:
            idx = src.index_for_time(window.start_s)
            reporter.emit(StageEvent("refine", "fallback", "no on-screen match; using audio timestamp"))
            return _finish(src, cfg, reporter, timings, t0, idx, "audio-fallback result",
                           timestamp_s=src.time_for_index(idx), text=window.matched_text, confidence="MEDIUM",
                           source="audio", note="no on-screen text matched; frame at first spoken word",
                           window=window, candidates=cands)
        weak = max(cands, key=lambda c: c.score) if cands else Candidate(0, 0.0, "", 0.0)
        note = (f"best OCR similarity only {weak.score:.2f}" if cands
                else "no text detected anywhere; frame 0 returned")
        reporter.emit(StageEvent("refine", "fallback", f"no match anywhere; best effort frame {weak.frame_index}"))
        return _finish(src, cfg, reporter, timings, t0, weak.frame_index, "low-confidence result",
                       timestamp_s=weak.timestamp_s, text=weak.text, confidence="LOW", source="ocr-weak",
                       note=note, candidates=cands)
