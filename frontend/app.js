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
const showError = (msg, detail) => {
  error.innerHTML = escapeHtml(msg) + (detail ? `<br><small class="muted">${escapeHtml(detail)}</small>` : "");
  error.hidden = false; setState("error", "error");
};

function reset() {
  error.hidden = true; $("result").hidden = true; $("cands-block").hidden = true; $("transcript-block").hidden = true;
  $("occurrences-block").hidden = true; $("occurrences").innerHTML = ""; $("occ-marks").innerHTML = "";
  $("player-block").hidden = true;
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

const VERIFY_SKIPPED_MSG = "Active-speaker check unavailable (install requirements-asd.txt) — using audio + OCR.";

function onEvent(e) {
  const ev = JSON.parse(e.data), p = ev.payload || {};
  const msg = (ev.stage === "verify" && ev.status === "skipped") ? VERIFY_SKIPPED_MSG : ev.message;
  stage(ev.stage, ev.status, msg);
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
  if (ev.stage === "occurrences" && p.occurrences) renderOccurrences(p.occurrences);
  if (ev.stage === "error") showError(ev.message);
  if (ev.stage === "end") finish(ev.status, ev.message, p);
}

function speakMarkFor(klass) {
  if (klass === "valid-speaker") return "✓";
  if (klass === "invalid") return "✗";
  return "?";
}

let ytPlayer = null, ytReady = false, pendingVideoSync = null, currentVideoUrl = "";

function parseYouTubeId(raw) {
  if (!raw) return null;
  const s = raw.trim();
  try {
    const url = new URL(s.startsWith("http") ? s : `https://${s}`);
    if (url.hostname.includes("youtube.com")) {
      if (url.searchParams.has("v")) {
        const id = url.searchParams.get("v");
        if (id && id.length >= 11) return id.slice(0, 11);
      }
      const parts = url.pathname.split("/").filter(Boolean);
      if (parts[0] === "embed" || parts[0] === "v" || parts[0] === "shorts") {
        if (parts[1] && parts[1].length >= 11) return parts[1].slice(0, 11);
      }
    }
    if (url.hostname.includes("youtu.be")) {
      const parts = url.pathname.split("/").filter(Boolean);
      if (parts[0] && parts[0].length >= 11) return parts[0].slice(0, 11);
    }
  } catch {}
  const m = s.match(/(?:youtu\.be\/|youtube\.com\/(?:watch\?.*v=|embed\/|v\/|shorts\/))([a-zA-Z0-9_-]{11})/);
  return m ? m[1] : null;
}

function isDirectVideo(url) {
  if (!url) return false;
  return /\.(mp4|webm|ogg|m4v|mov)(\?.*)?$/i.test(url) || url.startsWith("blob:") || url.startsWith("data:") || url.startsWith("file://") || /^[A-Za-z]:[\\\/]/.test(url);
}

function initYouTubeApi() {
  if (window.YT && window.YT.Player) {
    ytReady = true; return;
  }
  const tag = document.createElement("script");
  tag.src = "https://www.youtube.com/iframe_api";
  const firstScriptTag = document.getElementsByTagName("script")[0];
  if (firstScriptTag && firstScriptTag.parentNode) {
    firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
  } else {
    document.head.appendChild(tag);
  }
  window.onYouTubeIframeAPIReady = () => {
    ytReady = true;
    if (pendingVideoSync) {
      syncVideoPlayer(pendingVideoSync.url, pendingVideoSync.timestamp_s);
      pendingVideoSync = null;
    }
  };
}
initYouTubeApi();

function syncVideoPlayer(url, timestamp_s = 0, shouldScroll = false, autoplay = false) {
  if (!url) return;
  const playerBlock = $("player-block");
  if (!playerBlock) return;
  
  const formattedTc = tc(timestamp_s);
  const startSec = Math.max(0, Math.floor(timestamp_s));
  $("player-time-badge").textContent = `Synced: ${formattedTc}`;

  const ytWrap = $("yt-player-wrap");
  const ytIframe = $("yt-iframe");
  const html5Player = $("html5-player");
  const fallbackPlayer = $("fallback-player");
  const directLink = $("player-direct-link");
  const extLink = $("external-video-link");

  const ytId = parseYouTubeId(url);
  if (ytId && ytId.length === 11) {
    playerBlock.hidden = false;
    ytWrap.hidden = false;
    html5Player.hidden = true;
    fallbackPlayer.hidden = true;

    const autoplayParam = autoplay ? 1 : 0;
    const ytEmbedUrl = `https://www.youtube.com/embed/${ytId}?start=${startSec}&autoplay=${autoplayParam}&rel=0&enablejsapi=1`;
    if (ytIframe) {
      if (ytIframe.dataset.videoId !== ytId || ytIframe.dataset.startSec != startSec || autoplay) {
        ytIframe.dataset.videoId = ytId;
        ytIframe.dataset.startSec = startSec;
        ytIframe.src = ytEmbedUrl;
      }
      if (ytIframe.contentWindow) {
        try {
          ytIframe.contentWindow.postMessage(JSON.stringify({ event: "command", func: "seekTo", args: [timestamp_s, true] }), "*");
          if (autoplay) {
            ytIframe.contentWindow.postMessage(JSON.stringify({ event: "command", func: "playVideo", args: [] }), "*");
          } else {
            ytIframe.contentWindow.postMessage(JSON.stringify({ event: "command", func: "pauseVideo", args: [] }), "*");
          }
        } catch {}
      }
    }

    const watchUrl = `https://www.youtube.com/watch?v=${ytId}&t=${startSec}s`;
    if (directLink) {
      directLink.href = watchUrl;
      directLink.textContent = `Open on YouTube (${formattedTc}) ↗`;
      directLink.hidden = false;
    }
  } else if (isDirectVideo(url)) {
    playerBlock.hidden = false;
    ytWrap.hidden = true;
    html5Player.hidden = false;
    fallbackPlayer.hidden = true;

    if (html5Player.src !== url) {
      html5Player.src = url;
    }
    html5Player.currentTime = timestamp_s;
    if (autoplay) {
      html5Player.play().catch(() => {});
    } else {
      html5Player.pause();
    }

    if (directLink) {
      directLink.href = url;
      directLink.textContent = `Open direct video file ↗`;
      directLink.hidden = false;
    }
  } else if (url.startsWith("http://") || url.startsWith("https://")) {
    playerBlock.hidden = false;
    ytWrap.hidden = true;
    html5Player.hidden = true;
    fallbackPlayer.hidden = false;
    if (directLink) directLink.hidden = true;

    let extUrl = url;
    if (url.includes("?") && !url.includes("t=")) {
      extUrl = `${url}&t=${startSec}`;
    } else if (!url.includes("t=")) {
      extUrl = `${url}#t=${startSec}`;
    }
    if (extLink) {
      extLink.href = extUrl;
      extLink.textContent = `Open Video at ${formattedTc} ↗`;
    }
  }

  if (shouldScroll && !playerBlock.hidden) {
    playerBlock.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderOccurrences(occurrences) {
  const list = $("occurrences"), marks = $("occ-marks");
  list.innerHTML = ""; marks.innerHTML = "";
  occurrences.forEach((occ) => {
    const w = occ.window;
    const ocrMark = occ.klass === "valid-text" ? "✓" : "✗";
    const speakMark = speakMarkFor(occ.klass);
    const li = document.createElement("li");
    li.className = "occ-row"; li.dataset.klass = occ.klass;
    li.title = "Click to jump and play video at this scene";
    li.addEventListener("click", () => {
      syncVideoPlayer($("url").value.trim(), w.start_s, true, true);
    });

    const main = document.createElement("span");
    main.className = "mono";
    main.textContent = `${tc(w.start_s)}–${tc(w.end_s)} · ASR ${w.score.toFixed(2)} · OCR ${ocrMark} · faces ${occ.faces} · speaking ${speakMark}`;
    const badge = document.createElement("span");
    badge.className = "badge"; badge.textContent = occ.klass;
    li.append(main, badge);
    if (occ.text) {
      const q = document.createElement("q");
      q.className = "mono muted"; q.textContent = occ.text;
      li.append(" ", q);
    }
    list.appendChild(li);

    const mark = document.createElement("i");
    mark.dataset.klass = occ.klass; mark.style.left = pct(w.start_s);
    marks.appendChild(mark);

    if (occ.frame_index !== undefined) {
      addCandidate(occ.frame_index, occ.window.score, occ.text || w.matched_text, w.start_s);
    }
  });
  $("occurrences-block").hidden = false;
}

function addCandidate(frame, score, text, timestamp_s) {
  if (seenCandidates.has(frame)) return;
  seenCandidates.add(frame);
  $("cands-block").hidden = false;
  const d = document.createElement("figure"); d.className = "cand hit";
  const timeLabel = timestamp_s !== undefined ? tc(timestamp_s) : (fps ? tc(frame / fps) : "");
  const timePrefix = timeLabel ? `${timeLabel} · ` : "";
  d.innerHTML = `<img src="/jobs/${jobId}/frames/${frame}.png?w=320" alt="frame ${frame}" loading="lazy"><figcaption class="mono">${timePrefix}${frame} · ${score.toFixed(2)}</figcaption>`;
  d.title = text ? `${text} (Click to jump and play video)` : "Click to jump and play video";
  d.addEventListener("click", () => {
    const t = timestamp_s !== undefined ? timestamp_s : (fps ? frame / fps : 0);
    syncVideoPlayer($("url").value.trim(), t, true, true);
  });
  $("filmstrip").appendChild(d);
}

function placeMarker(t) { const m = $("marker"); m.style.left = pct(t); $("marker-tc").textContent = tc(t); m.hidden = false; }

async function finish(st, msg, payload) {
  if (es) { es.close(); es = null; }
  $("go").disabled = false; $("go").textContent = "Find frame"; $("cancel").hidden = true;
  document.querySelectorAll('#stages li[data-status="running"]').forEach((li) => {
    li.dataset.status = st === "done" ? "ok" : "skipped";
  });
  if (st === "cancelled") { setState("idle", "cancelled"); hint.textContent = "Stopped."; return; }
  if (st === "error") { showError(msg || "The run failed. Check the server log.", payload && payload.detail); return; }
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
  $("r-route").textContent = [r.source, r.confidence, r.occurrence_class].filter(Boolean).join(" · ");
  const route = r.source.startsWith("ocr") ? "ocr" : "audio";
  $("result").dataset.route = route; $("timeline").dataset.route = route;
  if (r.window) {   // hybrid can select a different window than the one `locate` first drew
    const w = r.window, el = $("window");
    el.style.left = pct(w.start_s); el.style.width = `calc(${pct(w.end_s)} - ${pct(w.start_s)})`; el.hidden = false;
  }
  if (r.speaker_box) {
    $("r-img").src = `/jobs/${jobId}/speaker.png`; $("r-img-cap").textContent = `frame ${r.frame_index} — ${r.timestamp} (speaker boxed)`;
  } else {
    $("r-img").src = `/jobs/${jobId}/frames/${r.frame_index}.png`; $("r-img-cap").textContent = `frame ${r.frame_index} — ${r.timestamp}`;
  }
  if (r.frame_index > 0) { $("r-prev").src = `/jobs/${jobId}/frames/${r.frame_index - 1}.png`; $("r-prev-cap").textContent = `frame ${r.frame_index - 1}${r.appearance ? " — " + r.appearance : ""}`; }
  $("r-text").textContent = r.text; $("r-note").textContent = r.note || "";
  $("r-alts").textContent = r.alternatives?.length ? "Also at: " + r.alternatives.map((a) => `frame ${a.frame_index} (${tc(a.timestamp_s)})`).join(", ") : "";
  $("result").hidden = false;

  // Make result and previous frame clickable to jump video
  $("r-tc").style.cursor = "pointer";
  $("r-tc").title = "Click to jump and play video at this result";
  $("r-tc").onclick = () => syncVideoPlayer($("url").value.trim(), r.timestamp_s, true, true);

  $("r-img").style.cursor = "pointer";
  $("r-img").title = "Click to jump and play video at this result";
  $("r-img").onclick = () => syncVideoPlayer($("url").value.trim(), r.timestamp_s, true, true);

  if (r.frame_index > 0) {
    const prevTime = fps ? Math.max(0, (r.frame_index - 1) / fps) : Math.max(0, r.timestamp_s - 0.033);
    $("r-prev").style.cursor = "pointer";
    $("r-prev").title = "Click to jump and play video at this frame before";
    $("r-prev").onclick = () => syncVideoPlayer($("url").value.trim(), prevTime, true, true);
  }

  syncVideoPlayer($("url").value.trim(), r.timestamp_s, false);
}

function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

// Preload player preview as user types or pastes video URL
$("url").addEventListener("input", (e) => {
  const u = e.target.value.trim();
  if (u) syncVideoPlayer(u, 0, false);
});
$("url").addEventListener("change", (e) => {
  const u = e.target.value.trim();
  if (u) syncVideoPlayer(u, 0, false);
});

form.addEventListener("submit", async (e) => {
  e.preventDefault(); reset();
  const inputUrl = $("url").value.trim();
  syncVideoPlayer(inputUrl, 0, false);
  $("go").disabled = true; $("go").textContent = "Finding…"; setState("running", "running");
  const body = { url: inputUrl, text: $("text").value.trim(), mode: $("mode").value, occurrence: $("occurrence").value };
  const res = await fetch("/jobs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) { showError("The server rejected the request — check both fields are filled in."); $("go").disabled = false; $("go").textContent = "Find frame"; return; }
  jobId = (await res.json()).id; $("cancel").hidden = false;
  es = new EventSource(`/jobs/${jobId}/events`);
  ["download", "transcribe", "locate", "scan", "verify", "occurrences", "refine", "done", "error", "end"].forEach((n) => es.addEventListener(n, onEvent));
  es.onerror = () => { if (es && es.readyState === EventSource.CLOSED) showError("Lost the connection to the server. Restart it and try again."); };
});
$("cancel").addEventListener("click", () => jobId && fetch(`/jobs/${jobId}/cancel`, { method: "POST" }));

// Load initial video preview if URL field has initial value
const initialUrl = $("url") ? $("url").value.trim() : "";
if (initialUrl) {
  syncVideoPlayer(initialUrl, 0, false);
}
