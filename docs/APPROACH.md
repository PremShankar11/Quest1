# Approach

## Phase 1 — Understand the problem

Spike on the real test video (https://ok.ru/video/248244667877, "The Adventures of
Sherlock Holmes: A Scandal in Bohemia", 3261s, 23.98 fps, 640x480 mp4 at max_height=480)
using `scripts/dump_frames.py`. Frames were dumped and inspected (Read tool) at
t=30, 45, 60, 90, 120, 150, 200, 250, 300, 400, 600, 700, 900, 1200, 1800, 2400, 3000
(the brief's original 9, the dialogue-hint set around "My mind rebels at stagnation"
in the first ~5 minutes, and the no-text fallback set).

What was seen: t=45 and t=60 show yellow opening-credit title cards ("Developed for
television by John Hawkesworth", "Dramatised by Alexander Baron") overlaid on the
picture — proving text-on-frame rendering and OCR are technically viable on this
source. t=150 is a black scene-transition frame. Every other sampled frame
(t=30, 90, 120, 200, 250, 300, 400, 600, 700, 900, 1200, 1800, 2400, 3000) shows plain
picture with no overlay text of any kind — no dialogue captions, no lower-third text.

Test video does not have burned-in subtitles (checked frames at t=30, 45, 60, 90, 120,
150, 200, 250, 300, 400, 600, 700, 900, 1200, 1800, 2400, 3000; only opening-credit
title cards render as on-screen text, never spoken dialogue); therefore the expected
route on this video is audio-fallback.

## Phase 2 — Design
## Phase 3 — Build
## Phase 4 — Test and measure
## Phase 5 — Reflect: limits and extensions
