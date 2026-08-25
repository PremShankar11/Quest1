const $ = (id) => document.getElementById(id);
const form = $("form"), status = $("status"), hint = $("hint"), error = $("error");
let es = null, jobId = null, duration = 0, fps = 0;
const seenCandidates = new Set();   // the widened retry re-scans frames; never add a thumbnail twice

const tc = (s) => {
  const ms = Math.round(s * 1000), h = Math.floor(ms / 3.6e6), m = Math.floor(ms % 3.6e6 / 6e4), sec = Math.floor(ms % 6e4 / 1000), r = ms % 1000;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}.${String(r).padStart(3, "0")}`;
};
const pct = (s) => duration ? `${Math.min(100, Math.max(0, s / duration * 100))}%` : "0%";
const setState = (s, label) => { status.dataset.state = s; status.textContent = label || s; };
const showError = (msg) => { error.textContent = msg; error.hidden = false; setState("error", "error"); };

function reset() {
  error.hidden = true; $("result").hidden = true; $("cands-block").hidden = true; $("transcript-block").hidden = true;
  $("filmstrip").innerHTML = ""; $("ticks").innerHTML = ""; $("window").hidden = true; $("marker").hidden = true;
  delete $("timeline").dataset.route; delete $("result").dataset.route;
  $("tc-end").textContent = "--:--:--"; duration = 0; fps = 0; seenCandidates.clear();
  document.querySelectorAll("#stages li").forEach((li) => { li.dataset.status = ""; li.querySelector(".msg").textContent = ""; });
  hint.textContent = "Starting…";
}

function stage(name, statusName, msg) {
  const li = document.querySelector(`#stages li[data-stage="${name}"]`);
  if (!li) return;
  li.dataset.status = statusName;
  if (msg) li.querySelector(".msg").textContent = msg;
}

function onEvent(e) {
  const ev = JSON.parse(e.data), p = ev.payload || {};
  stage(ev.stage, ev.status, ev.message);
  if (ev.stage === "download" && ev.status === "ok" && p.duration_s) {
    duration = p.duration_s; fps = p.fps; $("tc-end").textContent = tc(duration);
    hint.textContent = `${Math.round(duration)} s of video at ${fps.toFixed(3)} fps — listening for the line…`;
  }
  if (ev.stage === "transcribe" && p.text) { $("transcript-block").hidden = false; $("transcript").textContent = p.text; }
  if (ev.stage === "locate" && ev.status === "ok" && p.window) {
    const w = p.window, el = $("window");
    el.style.left = pct(w.start_s); el.style.width = `calc(${pct(w.end_s)} - ${pct(w.start_s)})`; el.hidden = false;
    $("transcript-block").hidden = false; $("transcript").innerHTML = `<mark>${escapeHtml(w.matched_text)}</mark> <span class="mono muted">${tc(w.start_s)}–${tc(w.end_s)} · score ${w.score.toFixed(2)}</span>`;
    hint.textContent = "Audio found the line — reading the frames around it…";
  }
  if (ev.stage === "locate" && ev.status !== "ok") hint.textContent = "No usable audio match — reading frames across the video…";
  if (ev.stage === "scan" && p.frame_index !== undefined && fps) {
    const t = p.frame_index / fps, i = document.createElement("i");
    i.style.left = pct(t); if (p.score >= 0.8) i.className = "hit"; $("ticks").appendChild(i);
    if (p.score >= 0.8) addCandidate(p.frame_index, p.score, p.text);
    hint.textContent = `Scanning… frame ${p.frame_index}, best match ${p.best.toFixed(2)}`;
  }
  if (ev.stage === "refine" && p.frame_index !== undefined && fps) placeMarker(p.frame_index / fps);   // ok AND fallback routes
  if (ev.stage === "error") showError(ev.message);
  if (ev.stage === "end") finish(ev.status, ev.message);
}

function addCandidate(frame, score, text) {
  if (seenCandidates.has(frame)) return;
  seenCandidates.add(frame);
  $("cands-block").hidden = false;
  const d = document.createElement("figure"); d.className = "cand hit";
  d.innerHTML = `<img src="/jobs/${jobId}/frames/${frame}.png?w=320" alt="frame ${frame}" loading="lazy"><figcaption class="mono">${frame} · ${score.toFixed(2)}</figcaption>`;
  d.title = text || ""; $("filmstrip").appendChild(d);
}

function placeMarker(t) { const m = $("marker"); m.style.left = pct(t); $("marker-tc").textContent = tc(t); m.hidden = false; }

async function finish(st, msg) {
  if (es) { es.close(); es = null; }
  $("go").disabled = false; $("go").textContent = "Find frame"; $("cancel").hidden = true;
  if (st === "cancelled") { setState("idle", "cancelled"); hint.textContent = "Stopped."; return; }
  if (st === "error") { showError(msg || "The run failed. Check the server log."); return; }
  let job;
  try {
    job = await (await fetch(`/jobs/${jobId}`)).json();
  } catch {
    showError("Finished, but the result could not be loaded. Refresh the page and run again.");
    return;
  }
  const r = job.result;
  setState("done", "done"); hint.textContent = "Done.";
  placeMarker(r.timestamp_s);
  $("r-tc").textContent = r.timestamp; $("r-frame").textContent = `frame ${r.frame_index}`;
  $("r-route").textContent = `${r.source} · ${r.confidence}`;
  const route = r.source.startsWith("ocr") ? "ocr" : "audio";
  $("result").dataset.route = route; $("timeline").dataset.route = route;
  $("r-img").src = `/jobs/${jobId}/frames/${r.frame_index}.png`; $("r-img-cap").textContent = `frame ${r.frame_index} — ${r.timestamp}`;
  if (r.frame_index > 0) { $("r-prev").src = `/jobs/${jobId}/frames/${r.frame_index - 1}.png`; $("r-prev-cap").textContent = `frame ${r.frame_index - 1}${r.appearance ? " — " + r.appearance : ""}`; }
  $("r-text").textContent = r.text; $("r-note").textContent = r.note || "";
  $("r-alts").textContent = r.alternatives?.length ? "Also at: " + r.alternatives.map((a) => `frame ${a.frame_index} (${tc(a.timestamp_s)})`).join(", ") : "";
  $("result").hidden = false;
}

function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

form.addEventListener("submit", async (e) => {
  e.preventDefault(); reset();
  $("go").disabled = true; $("go").textContent = "Finding…"; setState("running", "running");
  const body = { url: $("url").value.trim(), text: $("text").value.trim(), mode: $("mode").value, occurrence: $("occurrence").value };
  const res = await fetch("/jobs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) { showError("The server rejected the request — check both fields are filled in."); $("go").disabled = false; $("go").textContent = "Find frame"; return; }
  jobId = (await res.json()).id; $("cancel").hidden = false;
  es = new EventSource(`/jobs/${jobId}/events`);
  ["download", "transcribe", "locate", "scan", "refine", "done", "error", "end"].forEach((n) => es.addEventListener(n, onEvent));
  es.onerror = () => { if (es && es.readyState === EventSource.CLOSED) showError("Lost the connection to the server. Restart it and try again."); };
});
$("cancel").addEventListener("click", () => jobId && fetch(`/jobs/${jobId}/cancel`, { method: "POST" }));
