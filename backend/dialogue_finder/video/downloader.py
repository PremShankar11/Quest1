from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from ..config import Config
from ..models import StageEvent, VideoInfo
from ..progress import ProgressReporter


class DownloadError(Exception):
    """Raised when the video cannot be fetched or read. Message is user-facing."""


class _QuietLogger:
    """Swallow yt-dlp's own console output; failures surface as DownloadError -> 'Error: ...' instead."""

    def debug(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


def ensure_ffmpeg() -> None:
    import static_ffmpeg
    static_ffmpeg.add_paths()          # downloads ffmpeg+ffprobe once, then adds them to PATH


def cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def fetch_video(url: str, cfg: Config, reporter: ProgressReporter) -> Path:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    target = cfg.cache_dir / f"{cache_key(url)}.mp4"
    if target.exists() and target.stat().st_size > 0:
        reporter.emit(StageEvent("download", "running", f"cache hit {target.name}", 1.0, {"path": str(target)}))
        return target
    ensure_ffmpeg()
    import yt_dlp

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            frac = (done / total) if total else None
            reporter.emit(StageEvent("download", "running", "downloading", frac))

    opts = {
        "format": f"bv*[height<={cfg.max_height}]+ba/b[height<={cfg.max_height}]/b",
        "outtmpl": str(target.with_suffix("")) + ".%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "logger": _QuietLogger(),
    }
    reporter.emit(StageEvent("download", "running", f"fetching {url}", 0.0))
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:  # yt-dlp raises many types; all are fatal here
        raise DownloadError(f"Could not download video: {str(e).splitlines()[0][:200]}") from e
    if not target.exists():
        found = [f for f in cfg.cache_dir.glob(f"{cache_key(url)}.*") if not f.name.endswith((".part", ".ytdl"))]
        if not found:
            raise DownloadError("Download finished but no file was produced.")
        found[0].rename(target)
    reporter.emit(StageEvent("download", "running", f"saved {target.name}", 1.0, {"path": str(target)}))
    return target


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DownloadError(f"Cannot open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or count <= 0:
        raise DownloadError(f"Video has no readable frames: {path}")
    return VideoInfo(fps=fps, frame_count=count, width=w, height=h, duration_s=count / fps)
