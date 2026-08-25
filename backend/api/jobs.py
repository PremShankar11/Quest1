"""Job store, reporter and runner for the web API. In-memory: fine for a single-user demo."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from dialogue_finder.config import DEFAULT, Config
from dialogue_finder.models import StageEvent
from dialogue_finder.pipeline import PipelineError, run
from dialogue_finder.progress import ProgressReporter

DEBOUNCE_S = 0.2


class JobRequest(BaseModel):
    url: str = Field(min_length=1)
    text: str = Field(min_length=1)
    mode: Literal["hybrid", "audio", "ocr"] = "hybrid"
    occurrence: Literal["first", "last", "all"] = "first"


class Job:
    def __init__(self, req: JobRequest) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.req = req
        self.status = "queued"
        self.events: list[StageEvent] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.video_path: str | None = None
        self.cancel = threading.Event()
        self.cond = threading.Condition()
        self.started = time.monotonic()

    def add(self, event: StageEvent) -> None:
        with self.cond:
            self.events.append(event)
            self.cond.notify_all()

    def finish(self, status: str) -> None:
        with self.cond:
            self.status = status
            self.cond.notify_all()

    def to_dict(self) -> dict:
        return {"id": self.id, "status": self.status, "error": self.error, "result": self.result}


class JobReporter:
    """ProgressReporter that stores events on the job, stamping seq/t and debouncing payload-less progress ticks."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self._seq = 0
        self._last_tick: dict[str, float] = {}

    def emit(self, event: StageEvent) -> None:
        now = time.monotonic()
        is_tick = event.status == "running" and event.progress is not None and not event.payload
        if is_tick and now - self._last_tick.get(event.stage, -1.0) < DEBOUNCE_S:
            return
        if is_tick:
            self._last_tick[event.stage] = now
        self._seq += 1
        event.seq = self._seq
        event.t = round(now - self.job.started, 3)
        if event.stage == "download" and event.status == "ok" and event.payload.get("path"):
            self.job.video_path = event.payload["path"]
        self.job.add(event)


def _is_local(source: str) -> bool:
    return "://" not in source


def run_job(job: Job, cfg: Config = DEFAULT) -> None:
    job.status = "running"
    job_cfg = Config(cache_dir=cfg.cache_dir, output_dir=cfg.output_dir / job.id)
    reporter = JobReporter(job)
    try:
        result = run(job.req.url, job.req.text, cfg=job_cfg, reporter=reporter, mode=job.req.mode,
                     occurrence=job.req.occurrence, local=_is_local(job.req.url), should_cancel=job.cancel.is_set)
        job.result = result.to_dict()
        status = "done"
    except PipelineError as e:
        job.error = str(e)
        status = "cancelled" if str(e) == "cancelled" else "error"
    reporter.emit(StageEvent("end", status, job.error or ""))
    job.finish(status)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._run_lock = threading.Lock()      # one pipeline at a time: OCR/Whisper are CPU-bound

    def create(self, req: JobRequest) -> Job:
        job = Job(req)
        self._jobs[job.id] = job
        threading.Thread(target=self._serialised_run, args=(job,), daemon=True).start()
        return job

    def _serialised_run(self, job: Job) -> None:
        with self._run_lock:
            run_job(job)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def clear(self) -> None:
        self._jobs.clear()
