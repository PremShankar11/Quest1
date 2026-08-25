"""FastAPI app: serves the page and a small job API with Server-Sent Events."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.jobs import JobRequest, JobStore
from dialogue_finder.config import REPO_ROOT
from dialogue_finder.video.frame_source import FrameSource

FRONTEND = REPO_ROOT / "frontend"
KEEPALIVE_S = 15

app = FastAPI(title="Dialogue Frame Finder")
store = JobStore()
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.post("/jobs")
def create_job(req: JobRequest) -> dict:
    return {"id": store.create(req).id}


def _job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _job(job_id).to_dict()


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = _job(job_id)
    job.cancel.set()
    return {"id": job.id, "cancelling": True}


def _sse(event) -> str:
    return f"id: {event.seq}\nevent: {event.stage}\ndata: {json.dumps(asdict(event))}\n\n"


@app.get("/jobs/{job_id}/events")
def job_events(job_id: str, request: Request) -> StreamingResponse:
    job = _job(job_id)
    try:
        start = int(request.headers.get("last-event-id", "0") or 0)
    except ValueError:
        start = 0

    def stream():
        i = start
        while True:
            with job.cond:
                if i >= len(job.events):
                    job.cond.wait(timeout=KEEPALIVE_S)
                batch = job.events[i:]
            if not batch:
                yield ": keepalive\n\n"
                continue
            for event in batch:
                yield _sse(event)
                i += 1
                if event.stage == "end":
                    return

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/jobs/{job_id}/frames/{index}.png")
def job_frame(job_id: str, index: int, w: int | None = None) -> Response:
    job = _job(job_id)
    if not job.video_path or not Path(job.video_path).exists():
        raise HTTPException(404, "video not available for this job")
    with FrameSource(job.video_path) as src:
        frame = src.frame_at(index)
    if w:
        h = int(frame.shape[0] * w / frame.shape[1])
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise HTTPException(500, "could not encode frame")
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
