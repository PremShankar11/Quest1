# Spike: LR-ASD feasibility on Python 3.14, Windows, torch CPU-only

Date: 2026-08-25. Task 1 of Plan 4 (`docs/superpowers/plans/2026-08-25-visual-mode.md`). Decision gate for Tasks 2-8.

Ran `bench_out/spike_asd.py` against `cache/5f39d4605665a831.mp4` (Holmes episode, 640x480, 23.976 fps), frames 7700-7950 (~321.15-328.99 s of the actually-detected track), audio from `cache/5f39d4605665a831.16k.wav`. Full console output: `bench_out/spike_output.log` (not committed, scratch). Face crop used for scoring: `bench_out/spike_face.png` (not committed, scratch, referenced here for the record).

## weights

Not a separate download — LR-ASD ships its pretrained weights **inside the repository itself**, in `weight/`. `git clone --depth 1 https://github.com/Junhua-Liao/LR-ASD` (commit `1b6dcd2d8fc2895683de6508ec6294ec47d388ca`, 2025-03-23) pulled real binary files, not Git LFS pointers.

| File | Size | Copied to |
|---|---|---|
| `weight/pretrain_AVA.model` | 3,426,337 bytes | `cache/models/pretrain_AVA.model` |
| `weight/finetuning_TalkSet.model` | 3,426,337 bytes | `cache/models/finetuning_TalkSet.model` |

Both are `torch.save`d `OrderedDict[str, Tensor]` (168 keys), no custom pickle classes — safe to load with `weights_only=True`. Load: `torch.load(path, map_location="cpu", weights_only=True)`. Key prefixes in the checkpoint:
- `model.*` (164 keys) → the backbone (`ASD_Model` in the vendored file); strip the `model.` prefix.
- `lossAV.FC.weight` (2, 128), `lossAV.FC.bias` (2,) → the classification head (`AVScoreHead` in the vendored file); strip `lossAV.` (keep `FC.`).
- `lossV.FC.*` (2 keys) → visual-only auxiliary head, training-only, not vendored, not loaded.

`README.md` names `pretrain_AVA.model` as the default (AVA val mAP 94.45%) and `finetuning_TalkSet.model` as better out-of-domain (Columbia ASD F1 96.4% vs 86.1%). Both were scored in the spike (see `cost` and `decision`); **`finetuning_TalkSet.model` is the recommended default for Task 2+** since our footage (TV episode, not the AVA/Columbia benchmark distribution) is exactly the "out-of-domain" case that weight set was fine-tuned for, and it produced the larger absolute margin above `asd_threshold` on both spans.

YuNet face detector (used for face tracks, not part of LR-ASD): `face_detection_yunet_2023mar.onnx`, 232,589 bytes, from `opencv/opencv_zoo`. The plain `raw.githubusercontent.com` URL serves the file via Git LFS and a naive `curl -L` there can silently return a ~134-byte pointer stub — fetch instead from `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` (confirmed binary ONNX, opset marker `pytorch1.7`), saved to `cache/models/face_detection_yunet_2023mar.onnx`. Loaded with `cv2.FaceDetectorYN.create(str(path), "", (w, h))` — the OpenCV 5.0.0 installed here (`cv2.__version__`) still exposes this name; no API move.

## model_api

Vendored file: `backend/dialogue_finder/visual/lrasd_model.py` (MIT header, upstream commit `1b6dcd2d8fc2895683de6508ec6294ec47d388ca` in a comment). Classes copied verbatim from `model/Model.py`, `model/Encoder.py`, `model/Classifier.py`: `Audio_Block`, `Visual_Block`, `visual_encoder`, `audio_encoder`, `Fusion`, `Detector`, `ASD_Model`. Training wrapper `ASD.py` (`.cuda()`, optimizer, scheduler, `train_network`, BCELoss) is **not** vendored — CPU-only, no training code, per the brief.

```python
class ASD_Model(nn.Module):
    def __init__(self): ...
    def forward(self, audioFeature: Tensor, visualFeature: Tensor) -> tuple[Tensor, Tensor]:
        # audioFeature:  (B, T_a, 13)        T_a = 100 Hz MFCC frame count
        # visualFeature: (B, T_v, 112, 112)  grey crops, 0..255 range (not pre-normalised)
        # constraint: T_a == 4 * T_v exactly
        # returns (outsAV, outsV), each (B*T_v, 128); outsAV is the fused embedding used for scoring
        ...
    def forward_audio_frontend(self, x): ...   # used standalone in the spike for staged timing
    def forward_visual_frontend(self, x): ...
    def forward_audio_visual_backend(self, x1, x2): ...

class AVScoreHead(nn.Module):
    """Not in upstream Model.py; reconstructed from the `lossAV` training module
    (loss.py) minus its BCELoss, since ASD_Model.forward stops at the 128-d
    embedding and the checkpoint's classification weights live in `lossAV.FC`."""
    def __init__(self): self.FC = nn.Linear(128, 2)
    def forward(self, x: Tensor) -> Tensor: return self.FC(x)   # raw 2-class logits, (N, 2)
```

Construction:
```python
model = ASD_Model(); model.load_state_dict(backbone_state)   # backbone_state = {k[6:]: v for k,v in ckpt.items() if k.startswith("model.")}
head  = AVScoreHead(); head.load_state_dict(head_state)      # head_state = {k[7:]: v for k,v in ckpt.items() if k.startswith("lossAV.FC.")}
model.eval(); head.eval()   # MANDATORY: Detector has nn.Dropout(0.5); a train-mode forward silently corrupts scores
```

**Logits → probability rule (spike finding, corrects a naive reading of the upstream demo):** the correct per-frame active-speaker probability is
```python
probs = torch.softmax(head(outsAV), dim=-1)[:, 1]   # in [0, 1]; asd_threshold (0.5) applies directly to this
```
The upstream demo script (`Columbia_test.py`, `evaluate_network` + `visualization`) instead colours faces using `raw_logit = head(outsAV)[:, 1]` thresholded at `>= 0`, which is **not equivalent** — `softmax(x)[1] >= 0.5 ⟺ x[1] >= x[0]`, not `x[1] >= 0`. That raw-logit/`>=0` rule is a demo-only visualization shortcut from the code path taken when `labels=None`; the code path taken during evaluation *with* labels (`ASD.evaluate_network` in `ASD.py`) computes the same `softmax(x)[:, 1]` this spike uses. The spike printed both columns (`prob`, `raw`) per frame; they agree on sign of separation but only `prob` is a calibrated 0..1 score comparable to `config.py`'s `asd_threshold: 0.5`.

## preprocessing

- **Face crop → model input:** replicate `Columbia_test.py`'s `crop_video` geometry exactly, per frame's largest YuNet bbox `(x, y, w, h)`:
  - `s = max(w, h) / 2`, `cx, cy = x + w/2, y + h/2`, `cs = 0.40` (crop scale, upstream constant).
  - Pad the frame by `bsi = int(s * (1 + 2*cs))` on all sides with constant value 110 (grey, all 3 channels).
  - Crop `[my - s : my + s*(1+2cs), mx - s*(1+cs) : mx + s*(1+cs)]` in padded coordinates (`my, mx` = padded center) — a square crop of side `2*s*(1+cs)`, vertically offset to include more chin/torso than forehead (typical talking-head framing).
  - Resize to 224×224 (colour), then convert to grayscale, then take the center 112×112 (`[56:168, 56:168]`). **This final 112×112 grayscale image is the model's visual input**, not the 224×224 intermediate.
  - Normalisation happens inside `ASD_Model.forward_visual_frontend`, not in preprocessing: `(x / 255 - 0.4161) / 0.1688`, applied to raw 0-255 grayscale pixel values.
  - Upstream's `crop_video` also median-filters (`kernel=13`) the bbox center/scale across the *whole* track before cropping, to smooth jitter; the spike skips this (single-frame bboxes, no smoothing) — a track builder for Task 2 should add it back for production quality since it measurably steadies the crop.
- **Audio feature:** `python_speech_features.mfcc(audio, sr=16000, numcep=13, winlen=0.025, winstep=0.010)` → `(T, 13)` at 100 Hz (10 ms hop), matching spec §6/§10. `sr` confirmed from the wav header (16000).
- **Frames-per-video-frame ratio:** exactly **4 audio (100 Hz) frames per 1 video frame**, and it is architectural, not a convention: `audio_encoder` applies two `MaxPool3d(kernel=(1,1,3), stride=(1,1,2))` stages along the time axis (confirmed by tracing `Audio_Block`/`audio_encoder.forward`), each halving the time dimension, so a 100 Hz stream is downsampled to 25 Hz before `Fusion.forward` does `torch.cat((audioEmbed, visualEmbed), dim=2)` — this **requires the two embeddings to have identical frame counts**, i.e. `T_audio_raw == 4 * T_video` exactly, or the concat raises.
- **How the reference demo aligns lengths** (`evaluate_network` in `Columbia_test.py`): `length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])`, then trims both streams to that length — i.e. **trim to the shorter of the two streams**, in whole "quad-frames" of audio. The spike replicates this in frame-count terms rather than seconds (see next point), matching the plan's Global Constraint that the 23.976 fps video is "aligned by trimming to the shorter of the two streams as the reference demo does": `n_video_used = min(n_video, n_audio_raw // 4)`, `n_audio_used = n_video_used * 4`. On the spike's 251-frame track this trimmed 0 video frames and 38 (of 1042) trailing audio frames.
- **23.976 vs 25 fps drift (spike-owned correction to the spec's assumption):** LR-ASD was trained assuming 25 fps video (its own demo re-encodes input video to 25 fps with `ffmpeg -r 25` before ever detecting faces). This pipeline instead feeds native 23.976 fps frames directly into the 4:1 audio:video ratio described above — the ratio is a frame-*count* constraint the architecture enforces exactly, not a frame-*rate* constraint, so the forward pass runs without error at 23.976 fps. The cost is a small timestamp drift against true audio time: over a 10 s window, 10 s × 23.976 fps ≈ 240 native frames map to audio meant for 25 fps × 10 s = 250 frames — about **4.1% short, ~5-6 frames (~0.2-0.25 s) of drift per 10 s window**. This did not visibly hurt scoring in the spike (scores separate cleanly around the true speech onset at 325.4 s), but Task 4/5's onset search should treat onset timestamps as accurate to roughly ±0.25 s per window rather than frame-exact, and should not "fix" this by resampling to 25 fps (out of scope, adds an ffmpeg pass per window).

## torch

- Installed via `requirements-asd.txt` (`torch>=2.6`): resolved to **`torch==2.13.0+cpu`** (`pip install torch` on Python 3.14/Windows gives a CPU wheel directly, ~no CUDA extras pulled). `python_speech_features==0.6` installed clean (pure Python, no build issues on 3.14).
- `torch.load(path, map_location="cpu", weights_only=True)` loads both `.model` checkpoints without error or warning — they are plain tensor dicts, no custom classes, so the `weights_only=True` default (torch ≥ 2.6) is compatible; no `map_location` failure since we always pass it explicitly (a bare `torch.load(path)` on a CPU-only machine would otherwise raise if the checkpoint carries CUDA storage tags — not tested since we always pass `map_location="cpu"`).
- Backbone parameter count: **0.837 M params** (`ASD_Model` + `AVScoreHead`), matching the spec's "0.84 M params" claim.
- No GPU involved anywhere in this spike; `ASD_Model`/`AVScoreHead` as vendored have no `.cuda()` calls (unlike the upstream `ASD` training wrapper, which is why it isn't vendored).

## cost

Window: frames 7700-7950 requested (250 native frames, 23.976 fps ⇒ 10.43 s of video); 251 frames actually iterated (`iter_range` is inclusive of both ends) and every single one had ≥1 YuNet detection (251/251) at `setScoreThreshold(0.6)` on 640×480 frames.

Per-stage wall time (this machine, CPU, single run, `time.perf_counter()`):

| Stage | Time | Notes |
|---|---|---|
| YuNet detect + crop, 251 frames | 5.20 s | ~20.7 ms/frame; dominates the budget |
| MFCC extraction (10.43 s of audio) | 0.054 s | negligible |
| Model load (`torch.load` + `load_state_dict` + `.eval()`) | 0.044-0.098 s | one-time per process, not per-window |
| Forward pass, 251 frames, `pretrain_AVA` | 0.900 s | |
| Forward pass, 251 frames, `finetuning_TalkSet` | 0.847 s | same size, timing noise |

**Detection+ASD for a ~10 s window: ≈ 6.0-6.1 s total** (5.20 s detect+crop + 0.05 s MFCC + ~0.85-0.90 s forward), well under the 60 s/10 s-window gate. Scaled linearly to exactly 10 s: ≈ 5.04 s detect+crop+mfcc + ≈ 0.85 s forward ≈ **5.9 s per 10 s window**. Model load is excluded from per-window cost since a long-lived process loads weights once.

## decision

**GO with LR-ASD.**

| Gate criterion (from the brief) | Result |
|---|---|
| Weights load | Yes — both `.model` files load via `torch.load(map_location="cpu", weights_only=True)` with no key-mismatch after the `model.`/`lossAV.FC.` prefix split. |
| Scores high while speaking, low elsewhere | Yes, clearly, both weight sets: |

| Weights | mean prob, speaking (7794-7860, n=67) | mean prob, quiet (7700-7780, n=81) | separation | quiet frames ≥ 0.5 (false positives) | speaking frames ≥ 0.5 (of 67) |
|---|---|---|---|---|---|
| `pretrain_AVA` | **0.819** | 0.006 | 0.812 | 0 / 81 | 55 / 67 |
| `finetuning_TalkSet` | **0.950** | 0.140 | 0.810 | 0 / 81 | 67 / 67 |

Per-frame trace (both weight sets) shows probability sitting near 0 through the quiet span, then rising sharply to ~1.0 within a few frames of true speech onset (~325.3-325.4 s, matching the brief's ≈325.1-327.8 s Holmes line), staying high through the line, and dropping back to near 0 within a few frames after it ends (~327.9-328.2 s) — this is the onset-search signal Task 4/5 needs (§4 of the spec), not just a coarse mean. The classifier (spec §3) counts frames crossing `asd_threshold` (0.5) against `asd_min_active` (30% of speech frames), not means, so the crossing counts matter more than the means: both weight sets have **zero false-positive crossings on the quiet span**, and `finetuning_TalkSet` covers 100% of speaking frames (vs 82% for `pretrain_AVA`) — reinforcing it as the safer default (see Recommendation below), not just the one with the bigger mean.

- **Cost ≤ 60 s / 10 s window:** Yes — measured ≈ 5.9-6.1 s / 10 s window, roughly **10x margin**.

**Recommendation for Task 2+:** use `finetuning_TalkSet.model` as the default weight file (larger absolute scores in both directions on out-of-domain footage; same separation magnitude as `pretrain_AVA`), keep `pretrain_AVA.model` available as an alternative, and implement `SpeakerDetector`/`LrAsdDetector` per the spec's protocol using the exact preprocessing and probability rule recorded above.

**Concerns carried forward (not blockers):**
1. Detect+crop (YuNet) is the dominant per-window cost (~87% of the 5.9-6.1 s); still ~10x under budget, but if `max_occurrences` (5) windows each pad to several seconds, total per-video ASD cost is roughly `5 × window_duration_s × 0.5 s/s` — worth a coarse check against real episode-length runs in Task 8, not re-measuring here.
2. The spike's face tracker is a placeholder ("largest face per frame" as one track, no IoU/gap tolerance, no bbox smoothing) — Task 2's real `IouTracker` + median-filtered crop centers (per spec §6) should be at least as clean as this, likely cleaner.
3. 23.976-vs-25 fps drift (~0.2-0.25 s per 10 s window, see `preprocessing`) means onset frames from Task 4's search should be treated as approximate to roughly a quarter-second, not frame-exact, when native-fps video is fed directly into a 25 fps-trained model.
