import dataclasses
import json
import time

import pytest
from fastapi.testclient import TestClient

from api import jobs as jobs_mod
from api.main import app, store
from dialogue_finder.models import Result, StageEvent


def fake_run_factory(events, result=None, error=None, delay=0.0):
    def fake_run(source, text, *, cfg, reporter, mode, occurrence, local, should_cancel=None, **_):
        for e in (dataclasses.replace(ev) for ev in events):     # fresh copies: the reporter mutates seq/t
            if should_cancel and should_cancel():
                from dialogue_finder.pipeline import PipelineError
                raise PipelineError("cancelled")
            reporter.emit(e)
            time.sleep(delay)
        if error:
            from dialogue_finder.pipeline import PipelineError
            raise PipelineError(error)
        return result
    return fake_run


EVENTS = [
    StageEvent("download", "ok", "video ready", 1.0, {"path": "x.mp4", "fps": 24.0, "frame_count": 360, "duration_s": 15.0}),
    StageEvent("locate", "ok", "window", 1.0, {"window": {"start_s": 5.0, "end_s": 6.0, "score": 0.9, "matched_text": "hi"}}),
    StageEvent("scan", "running", "frame 120", 0.5, {"frame_index": 120, "score": 0.95, "text": "hi", "best": 0.95}),
    StageEvent("done", "ok", "result ready"),
]
RESULT = Result(5.0, 120, "hi", "HIGH", "ocr", fps=24.0, image_path="a.png", prev_image_path="b.png")


@pytest.fixture(autouse=True)
def fresh_store():
    store.clear()
    yield
    store.clear()


def _wait_done(client, job_id, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/jobs/{job_id}").json()["status"]
        if s in ("done", "error", "cancelled"):
            return s
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_index_served():
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200 and "Find frame" in r.text


def test_job_lifecycle_and_sse(monkeypatch):
    monkeypatch.setattr(jobs_mod, "run", fake_run_factory(EVENTS, RESULT))
    with TestClient(app) as c:
        r = c.post("/jobs", json={"url": "x.mp4", "text": "hi"})
        assert r.status_code == 200
        job_id = r.json()["id"]
        assert _wait_done(c, job_id) == "done"
        body = c.get(f"/jobs/{job_id}").json()
        assert body["result"]["frame_index"] == 120 and body["result"]["timestamp"] == "00:00:05.000"
        with c.stream("GET", f"/jobs/{job_id}/events") as s:
            text = "".join(chunk for chunk in s.iter_text())
    frames = [b for b in text.strip().split("\n\n") if b.strip()]
    assert frames[0].startswith("id: 1\nevent: download\n")
    assert any("event: end" in f for f in frames)
    data = json.loads(frames[0].split("data: ", 1)[1])
    assert data["payload"]["duration_s"] == 15.0 and data["seq"] == 1


def test_sse_replays_from_last_event_id(monkeypatch):
    monkeypatch.setattr(jobs_mod, "run", fake_run_factory(EVENTS, RESULT))
    with TestClient(app) as c:
        job_id = c.post("/jobs", json={"url": "x.mp4", "text": "hi"}).json()["id"]
        _wait_done(c, job_id)
        with c.stream("GET", f"/jobs/{job_id}/events", headers={"Last-Event-ID": "2"}) as s:
            text = "".join(s.iter_text())
    assert "event: download" not in text and "event: scan" in text


def test_pipeline_error_becomes_job_error(monkeypatch):
    monkeypatch.setattr(jobs_mod, "run", fake_run_factory(EVENTS[:1], error="Could not download video: nope"))
    with TestClient(app) as c:
        job_id = c.post("/jobs", json={"url": "bad", "text": "hi"}).json()["id"]
        assert _wait_done(c, job_id) == "error"
        assert "Could not download" in c.get(f"/jobs/{job_id}").json()["error"]


def test_cancel(monkeypatch):
    monkeypatch.setattr(jobs_mod, "run", fake_run_factory(EVENTS * 50, RESULT, delay=0.02))
    with TestClient(app) as c:
        job_id = c.post("/jobs", json={"url": "x.mp4", "text": "hi"}).json()["id"]
        time.sleep(0.05)
        assert c.post(f"/jobs/{job_id}/cancel").status_code == 200
        assert _wait_done(c, job_id) == "cancelled"


def test_unexpected_exception_ends_job_with_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(jobs_mod, "run", boom)
    with TestClient(app) as c:
        job_id = c.post("/jobs", json={"url": "x.mp4", "text": "hi"}).json()["id"]
        assert _wait_done(c, job_id) == "error"
        body = c.get(f"/jobs/{job_id}").json()
        assert "unexpected failure" in body["error"] and "RuntimeError" in body["error"]
        with c.stream("GET", f"/jobs/{job_id}/events") as s:
            text = "".join(s.iter_text())
        assert "event: end" in text


def test_cancel_queued_job_does_not_wait_for_first(monkeypatch):
    # First job holds the serialised-run lock for a while (many slow events); the second job is
    # cancelled while still queued behind it and must reach "cancelled" quickly, not after the first.
    monkeypatch.setattr(jobs_mod, "run", fake_run_factory(EVENTS * 200, RESULT, delay=0.05))
    with TestClient(app) as c:
        first_id = c.post("/jobs", json={"url": "x.mp4", "text": "hi"}).json()["id"]
        second_id = c.post("/jobs", json={"url": "y.mp4", "text": "hi"}).json()["id"]
        assert c.post(f"/jobs/{second_id}/cancel").status_code == 200
        t0 = time.time()
        assert _wait_done(c, second_id, timeout=1.0) == "cancelled"
        assert time.time() - t0 < 1.0
        c.post(f"/jobs/{first_id}/cancel")   # don't leak a ~10 s lock hold into later tests
        _wait_done(c, first_id, timeout=5.0)


def test_download_failure_gets_friendly_message_and_detail(monkeypatch):
    original = "Could not download video: ERROR: [generic] x"
    monkeypatch.setattr(jobs_mod, "run", fake_run_factory(EVENTS[:0], error=original))
    with TestClient(app) as c:
        job_id = c.post("/jobs", json={"url": "bad", "text": "hi"}).json()["id"]
        assert _wait_done(c, job_id) == "error"
        body = c.get(f"/jobs/{job_id}").json()
        assert body["error"] == "Could not download that URL. Check it opens in a browser, or paste a local file path."
        assert body["detail"] == original


def test_sse_replays_terminal_end_past_last_event_id():
    with TestClient(app) as c:
        req = jobs_mod.JobRequest(url="x", text="y")
        job = jobs_mod.Job(req)
        store._jobs[job.id] = job
        job.add(StageEvent("download", "ok", "video ready", seq=1))
        job.add(StageEvent("end", "done", "", seq=2))
        job.finish("done")
        with c.stream("GET", f"/jobs/{job.id}/events", headers={"Last-Event-ID": "2"}) as s:
            text = "".join(s.iter_text())
        assert "event: end" in text


def test_frames_404_without_video():
    with TestClient(app) as c:
        assert c.get("/jobs/nope/frames/1.png").status_code == 404


def test_frame_png_from_synthetic_clip(monkeypatch, synthetic_clip):
    path, truth = synthetic_clip
    ev = [StageEvent("download", "ok", "video ready", 1.0, {"path": str(path), "fps": 24.0, "frame_count": 360, "duration_s": 15.0})]
    monkeypatch.setattr(jobs_mod, "run", fake_run_factory(ev, RESULT))
    with TestClient(app) as c:
        job_id = c.post("/jobs", json={"url": str(path), "text": "hi"}).json()["id"]
        _wait_done(c, job_id)
        r = c.get(f"/jobs/{job_id}/frames/{truth['frame']}.png?w=160")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png" and len(r.content) > 500


def test_validation_errors_are_json():
    with TestClient(app) as c:
        assert c.post("/jobs", json={"url": "", "text": ""}).status_code == 422
        assert c.post("/jobs", json={"url": "x", "text": "y", "mode": "weird"}).status_code == 422


def test_mode_audio_ocr_and_hybrid_accepted(monkeypatch):
    monkeypatch.setattr(jobs_mod, "run", fake_run_factory(EVENTS, RESULT))
    with TestClient(app) as c:
        assert c.post("/jobs", json={"url": "x.mp4", "text": "hi", "mode": "audio+ocr"}).status_code == 200
        assert c.post("/jobs", json={"url": "x.mp4", "text": "hi", "mode": "hybrid"}).status_code == 200


def test_reporter_debounces_only_payloadless_running_events():
    job = jobs_mod.Job(jobs_mod.JobRequest(url="x", text="y"))
    rep = jobs_mod.JobReporter(job)
    for i in range(20):
        rep.emit(StageEvent("download", "running", "downloading", i / 20))
    rep.emit(StageEvent("download", "ok", "video ready", 1.0, {"path": "p"}))
    kinds = [(e.stage, e.status) for e in job.events]
    assert kinds.count(("download", "running")) <= 2 and kinds[-1] == ("download", "ok")
    assert [e.seq for e in job.events] == list(range(1, len(job.events) + 1))
