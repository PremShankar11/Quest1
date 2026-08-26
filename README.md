<div align="center">

# 🎬 Dialogue Frame Finder

**Give it a video and a line of dialogue. It gives you back the exact frame.**

[![tests](https://img.shields.io/badge/tests-151%20passing-4FD1C5?style=for-the-badge)](#-verify-it-yourself)
[![python](https://img.shields.io/badge/python-3.14-2b5b84?style=for-the-badge&logo=python&logoColor=white)](#-install)
[![modes](https://img.shields.io/badge/modes-4-F2B33D?style=for-the-badge)](#-the-four-modes)
[![runs](https://img.shields.io/badge/runs-100%25%20local-8F95A0?style=for-the-badge)](#-install)

<img src="docs/media/ui-hybrid-result.png" alt="Result card: timecode, frame number, the frame before and after, with the verified on-screen speaker boxed" width="850">

*The answer, with its evidence: the frame before, the frame itself, and the face the model confirmed was speaking.*

</div>

---

## ⚡ Quick start

```bash
py -3.14 -m venv .venv && .venv\Scripts\pip install -r requirements.txt
.\start.ps1
```

That's it — the browser opens at `http://127.0.0.1:8000`. Paste a URL, type a line, press **Find frame**.

> [!TIP]
> **Windows blocks the script?** Run `powershell -ExecutionPolicy Bypass -File start.ps1`.
> **macOS / Linux?** `./start.sh` does the same thing.

> [!NOTE]
> The first run downloads the speech model (~150 MB), the OCR models (~10 MB) and the video itself into `cache/`.
> Every run after that reuses them — a repeat query on the same video takes seconds.

---

## 🎥 See it work

Four recordings, one per mode, on the same Spider-Man trailer.

<table>
<tr>
<td width="50%">

### 🟡 `hybrid` — the full pipeline
Audio finds every occurrence, OCR checks for on-screen text, and an active-speaker model checks **whether a visible person is actually saying it**.

<video src="docs/media/demo-hybrid.mp4" controls width="100%"></video>

▶️ [demo-hybrid.mp4](docs/media/demo-hybrid.mp4)

</td>
<td width="50%">

### 🟢 `audio+ocr` — locate, then confirm
Whisper narrows the search to a few seconds, OCR pins the exact frame where the subtitle appears.

<video src="docs/media/demo-audio-ocr.mp4" controls width="100%"></video>

▶️ [demo-audio-ocr.mp4](docs/media/demo-audio-ocr.mp4)

</td>
</tr>
<tr>
<td width="50%">

### 🔵 `ocr` — pure on-screen text
No audio at all. Scans frames for the text — title cards, captions, signs.

<video src="docs/media/demo-ocr.mp4" controls width="100%"></video>

▶️ [demo-ocr.mp4](docs/media/demo-ocr.mp4)

</td>
<td width="50%">

### 🟣 `audio` — spoken word only
Straight to the transcript. Fastest answer when you only care about who said what, when.

<video src="docs/media/demo-audio.mp4" controls width="100%"></video>

▶️ [demo-audio.mp4](docs/media/demo-audio.mp4)

</td>
</tr>
</table>

> [!IMPORTANT]
> If a video doesn't play inline in your browser, click the link underneath it — the files live in `docs/media/`.

---

## 🖼️ The interface

<table>
<tr>
<td width="50%" align="center">
<img src="docs/media/ui-idle.png" alt="Idle state" width="100%"><br>
<sub><b>Paste and go</b> — two fields, one button</sub>
</td>
<td width="50%" align="center">
<img src="docs/media/ui-player-sync.png" alt="Video player synced to the result timestamp" width="100%"><br>
<sub><b>Player sync</b> — the video jumps to the moment it found</sub>
</td>
</tr>
<tr>
<td width="50%" align="center">
<img src="docs/media/ui-hybrid-stages.png" alt="Live stage list and timeline" width="100%"><br>
<sub><b>Watch it think</b> — every stage, live, with its numbers</sub>
</td>
<td width="50%" align="center">
<img src="docs/media/ui-hybrid-occurrences.png" alt="Occurrences list with classification badges" width="100%"><br>
<sub><b>Every candidate, judged</b> — and why each was accepted or rejected</sub>
</td>
</tr>
<tr>
<td width="50%" align="center">
<img src="docs/media/ui-ocr-candidates.png" alt="Candidate filmstrip" width="100%"><br>
<sub><b>Filmstrip</b> — the frames it considered, with scores</sub>
</td>
<td width="50%" align="center">
<img src="docs/media/ui-ocr-result.png" alt="OCR result showing the extracted text" width="100%"><br>
<sub><b>Text extracted</b> — read straight off the frame</sub>
</td>
</tr>
</table>

<details>
<summary><b>More screenshots</b> — mode picker, occurrence picker, title-card detection</summary>
<br>
<table>
<tr>
<td align="center"><img src="docs/media/ui-modes.png" width="100%"><br><sub>Four modes, one dropdown</sub></td>
<td align="center"><img src="docs/media/ui-occurrence.png" width="100%"><br><sub>first / last / all occurrences</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/media/ui-ocr-titlecard.png" width="100%"><br><sub>Title cards and burned-in text</sub></td>
<td align="center"><img src="docs/media/ui-ocr-player.png" width="100%"><br><sub>Result synced back to the source video</sub></td>
</tr>
</table>
</details>

---

## 🧠 How it finds the frame

```mermaid
flowchart LR
    A["🔗 Video URL<br/>or local file"] --> B["📥 Download<br/>+ cache"]
    B --> C["🎙️ Transcribe<br/>word timestamps"]
    C --> D{"Candidate<br/>windows"}
    D --> E["🔤 OCR<br/>subtitle band"]
    D --> F["🙂 Face tracks<br/>YuNet"]
    F --> G["🗣️ Active speaker<br/>LR-ASD"]
    E --> H{"Classify each<br/>occurrence"}
    G --> H
    H --> I["✅ valid-text"]
    H --> J["✅ valid-speaker"]
    H --> K["❔ uncertain"]
    H --> L["❌ invalid"]
    I --> M["🎯 Binary search<br/>→ first frame"]
    J --> N["🎯 Visual onset<br/>→ first frame"]
    M --> O["🖼️ Result<br/>timecode · frame · text"]
    N --> O

    style A fill:#23272E,stroke:#4FD1C5,color:#ECEAE4
    style O fill:#23272E,stroke:#F2B33D,color:#ECEAE4
    style G fill:#23272E,stroke:#F2B33D,color:#ECEAE4
    style H fill:#23272E,stroke:#4FD1C5,color:#ECEAE4
```

| The hard question | How this answers it |
|---|---|
| **Where do I even look?** | Whisper transcribes once, fuzzy-matching narrows a 54-minute video to a few seconds. No blind frame-by-frame scan. |
| **Which frame exactly?** | Binary search on the OCR score between the last miss and the first hit — 3-4 reads, not thousands. For a speaker, the frame where the lips start moving. |
| **How is the text extracted?** | RapidOCR on the subtitle band first, full frame if that misses. |
| **What if it's ambiguous?** | Every occurrence is scored and classed, the best one wins, and the answer always says *how* it was found and how confident it is. |

---

## 🎛️ The four modes

| Mode | What it uses | Best for | Extra install |
|:---|:---|:---|:---|
| 🟡 **`hybrid`** *(default)* | audio **+** OCR **+** active speaker | "Is a real person on screen saying this?" | `requirements-asd.txt` |
| 🟢 **`audio+ocr`** | audio **+** OCR | Burned-in subtitles, dubbed films | — |
| 🔵 **`ocr`** | OCR only | Title cards, signs, silent text | — |
| 🟣 **`audio`** | transcript only | Fastest; spoken-word lookup | — |

Add `--occurrence first | last | all` when the line is said more than once.

> [!NOTE]
> Without the optional extras, `hybrid` doesn't break — it reports `verify: skipped` and returns the `audio+ocr` answer.

---

## 💻 Prefer the terminal?

```bash
cd backend
..\.venv\Scripts\python -m dialogue_finder --url "https://ok.ru/video/248244667877" --text "My mind rebels at stagnation"
```

```text
Timestamp : 00:05:25.365
Frame     : 7801
Text      : "My mind rebels its stagnation."
Confidence: HIGH  (source: audio+asd; on-screen speaker verified (LR-ASD mean 0.89))
Image     : ..\output\frame_7801.png
Previous  : ..\output\frame_7800.png  (frame before)
Occurrence: valid-speaker
Speaker   : 218,0,304,395
```

`--local <file>` · `--mode hybrid|audio+ocr|audio|ocr` · `--occurrence first|last|all` · `--verbose` · `--json` · `--out <dir>`

Frames are 0-based, timestamps are `HH:MM:SS.sss`, and a wrong answer is always labelled as low-confidence rather than dressed up as a right one.

<details>
<summary><b>HTTP API</b> — the same pipeline, five endpoints</summary>

| Method | Endpoint | Does |
|---|---|---|
| `POST` | `/jobs` | Start a job — `{url, text, mode, occurrence}` → `{id}` |
| `GET` | `/jobs/{id}` | Status and result |
| `GET` | `/jobs/{id}/events` | Server-Sent Events, one per pipeline stage (resumable) |
| `GET` | `/jobs/{id}/frames/{n}.png` | Any frame, rendered on demand |
| `POST` | `/jobs/{id}/cancel` | Stop a running job |

</details>

---

## 📦 Install

<table>
<tr><th align="left" width="30%">Step</th><th align="left">Command</th></tr>
<tr><td><b>1. Environment</b></td><td><code>py -3.14 -m venv .venv</code></td></tr>
<tr><td><b>2. Core</b></td><td><code>.venv\Scripts\pip install -r requirements.txt</code></td></tr>
<tr><td><b>3. Speaker detection</b> <sub>(optional)</sub></td><td><code>.venv\Scripts\pip install -r requirements-asd.txt</code></td></tr>
<tr><td><b>4. GPU</b> <sub>(optional)</sub></td><td><code>.venv\Scripts\pip install -r requirements-gpu.txt</code></td></tr>
<tr><td><b>5. Run</b></td><td><code>.\start.ps1</code></td></tr>
</table>

Everything runs on your machine — no API keys, no cloud, no data leaving the laptop. GPU is detected automatically and falls back to CPU on its own (measured: transcription 48.9 s on GPU vs 141 s on CPU).

<details>
<summary><b>Troubleshooting</b></summary>
<br>

| Symptom | Fix |
|---|---|
| `start.ps1` refuses to run | `powershell -ExecutionPolicy Bypass -File start.ps1` |
| ffmpeg download fails (connection reset) | See `docs/DECISIONS.md` → "static-ffmpeg download can fail on some networks" |
| Speaker-detection models won't download | `curl -L -o cache/models/finetuning_TalkSet.model https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/1b6dcd2d8fc2895683de6508ec6294ec47d388ca/weight/finetuning_TalkSet.model`<br>`curl -L -o cache/models/face_detection_yunet_2023mar.onnx https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` |
| A run takes minutes | First run on a long video: download + transcription. Repeat runs hit the cache. |

Model files are SHA-256 verified on every download **and** every load — a mismatch deletes the file and stops rather than loading something tampered with.

</details>

---

## ✅ Verify it yourself

```bash
cd backend && ..\.venv\Scripts\python -m pytest -q
```

| Check | Result |
|---|---|
| Test suite | **151 passing** |
| 54-min episode, `hybrid` | `valid-speaker` · frame 7801 · speaker verified (LR-ASD 0.89) |
| Trailer with burned-in title card, `ocr` | `valid-text` · frame 466 · text read as *"MARVEL STUDIOS"* |
| Voice-over clip (narrator off screen) | `uncertain` — correctly refuses to claim a visible speaker |
| Bad URL / missing file | One-line error, exit code 1 — never a traceback |

---

## 📚 Deeper reading

| Document | What's inside |
|---|---|
| [`docs/approach_final.md`](docs/approach_final.md) · [PDF](docs/approach_final.pdf) | **Start here** — the technical approach: problem, architecture, evidence model, exact-frame selection, validation, limitations |
| [`docs/APPROACH.md`](docs/APPROACH.md) · [PDF](docs/APPROACH.pdf) | The full engineering log: every phase, measurement and investigation |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Every decision and why, including the ones I overruled |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | Ground-truth accuracy across fonts, positions, fades, resolutions, frame rates |
| [`prompts.txt`](prompts.txt) | Every prompt used to build this, verbatim |

---

## ⚠️ Known limits

- Onset accuracy for a speaking face is about ±0.25 s — honest precision, not a frame-exact claim.
- Profile faces and hard cuts weaken speaker detection; it reports `uncertain` instead of guessing.
- Demo scope: single user, jobs held in memory, one job at a time.

## 🙏 Built on

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [RapidOCR](https://github.com/RapidAI/RapidOCR) · [LR-ASD](https://github.com/Junhua-Liao/LR-ASD) (MIT) · [OpenCV YuNet](https://github.com/opencv/opencv_zoo) · [yt-dlp](https://github.com/yt-dlp/yt-dlp) · [FastAPI](https://fastapi.tiangolo.com/)
