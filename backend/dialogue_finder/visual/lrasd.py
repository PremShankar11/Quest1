"""LR-ASD active-speaker detector behind the `SpeakerDetector` protocol (Plan 4, Task 5).

Loads the vendored network (`visual/lrasd_model.py`) and its pretrained checkpoint lazily
on first `score()` call, per the plan's Global Constraints: no top-level `import torch` (or
`python_speech_features`) here, so `pipeline.py` can import `asd_available` at module level
on a machine without the optional `requirements-asd.txt` extras installed.

Preprocessing, tensor shapes, the frame-alignment rule (trim to
`min(n_video, n_audio // 4)`, as the reference `Columbia_test.py` demo does), and the
logits -> probability rule (softmax over the 2 classes, taking the "speaking" column, NOT
the upstream demo's uncalibrated `raw_logit >= 0` shortcut) are all per
`docs/superpowers/spikes/2026-08-25-lrasd-spike.md`.

Weights: LR-ASD ships its pretrained weights inside its own repo (`weight/`), not as a
separate release asset (spike note `weights` heading) -- fetched here from
raw.githubusercontent.com at the pinned commit, ordinary git blobs (confirmed not
Git-LFS-tracked in the spike, unlike YuNet's opencv_zoo copy). SHA-256 hashes below are
pinned from the spike's own downloaded copies (`cache/models/`, hashed 2026-08-25) and are
verified after every download and again on every `_load()`, deleting and re-raising on
mismatch so a corrupted or tampered checkpoint is never fed to `torch.load`.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import DEFAULT as _DEFAULT_CONFIG
from .model_files import VisualStageUnavailable, fetch_verified

_LRASD_COMMIT = "1b6dcd2d8fc2895683de6508ec6294ec47d388ca"
# Ordinary git blobs at this commit (confirmed not Git-LFS-tracked in the spike), so a plain
# raw.githubusercontent.com URL serves the real binary -- unlike YuNet's opencv_zoo copy.
# Manual fallback if the download below fails after retries:
#   curl -L -o cache/models/finetuning_TalkSet.model \
#     https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/1b6dcd2d8fc2895683de6508ec6294ec47d388ca/weight/finetuning_TalkSet.model
_WEIGHTS_BASE_URL = f"https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/{_LRASD_COMMIT}/weight"

# sha256 of each weights file at _WEIGHTS_BASE_URL, hashed 2026-08-25 from the spike's copies
# (docs/superpowers/spikes/2026-08-25-lrasd-spike.md, `weights` heading). Pinned so a
# compromised/mirrored/corrupted download is rejected instead of silently `torch.load`ed.
_WEIGHTS_SHA256 = {
    "pretrain_AVA.model": "85e6c77fc981595234790d1e128ebb60352d37726b2445e0ef8891e2512fe9e3",
    "finetuning_TalkSet.model": "6b4ef53694e874e96cf630198dc479c78aebb3993bbf166aee3d926dfe7d9342",
}
# Recommended default (spike note `weights` heading): better out-of-domain (our TV-episode
# footage) separation than pretrain_AVA -- 100% of speaking frames crossed asd_threshold vs
# 82%, same zero false positives on the quiet span.
DEFAULT_WEIGHTS = "finetuning_TalkSet.model"


class SpeakerDetector(Protocol):
    def score(self, crops: np.ndarray, mfcc: np.ndarray) -> list[float]:
        """crops: N x S x S uint8 grey, aligned to N video frames. mfcc: (4N) x 13 MFCC for
        the same span. Returns N active-speaker probabilities in [0, 1], aligned to crops."""
        ...


class WeightsVerificationError(IOError):
    """A weights file's SHA-256 didn't match the pinned value; the file has been deleted."""


class SpeakerDetectorUnavailable(VisualStageUnavailable):
    """Raised by `LrAsdDetector._load()` when the LR-ASD weights cannot be obtained -- a
    download failure (offline, unreachable) or a hash mismatch after retries -- wrapping the
    underlying error's message. Cached on the instance (`_load_error`) so a second `score()`
    call re-raises immediately instead of retrying the download/timeout."""


def _weights_url(name: str) -> str:
    return f"{_WEIGHTS_BASE_URL}/{name}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_weights(path: Path, name: str) -> None:
    """Raise `WeightsVerificationError` (and delete `path`) if its sha256 doesn't match the
    pinned value for `name`. A `name` with no pinned hash (a custom weights file) passes
    through unchecked."""
    expected = _WEIGHTS_SHA256.get(name)
    if expected is None:
        return
    actual = _sha256(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise WeightsVerificationError(
            f"{path.name} failed SHA-256 verification (expected {expected}, got {actual}) -- "
            f"deleted.\nManual fallback: curl -L -o {path} {_weights_url(name)}"
        )


def _download_weights(dest: Path, name: str) -> None:
    fetch_verified(
        _weights_url(name),
        dest,
        lambda tmp: _verify_weights(tmp, name),
        f"LR-ASD weights ({name})",
        reraise=(WeightsVerificationError,),   # a hash mismatch is not retried
    )


def _ensure_weights(models_dir: Path, name: str) -> Path:
    path = Path(models_dir) / name
    if path.exists():
        _verify_weights(path, name)   # re-checked on every load, not just freshly-downloaded files
        return path
    _download_weights(path, name)
    return path


def asd_available(models_dir: Path | None = None) -> tuple[bool, str]:
    """(False, reason) if torch isn't importable (reason mentions `requirements-asd.txt`).
    Otherwise (True, ""), or (True, "weights missing -- will download on first use") if the
    weights file isn't present yet -- this only checks presence, it never downloads."""
    if importlib.util.find_spec("torch") is None:
        return False, (
            "torch is not installed -- active-speaker detection needs the optional extras: "
            "pip install -r requirements-asd.txt"
        )
    models_dir = Path(models_dir) if models_dir is not None else _DEFAULT_CONFIG.models_dir
    weights_path = models_dir / DEFAULT_WEIGHTS
    if weights_path.exists():
        return True, ""
    return True, "weights missing -- will download on first use"


def _softmax_speaking_column(logits: np.ndarray) -> np.ndarray:
    """softmax(logits, axis=-1)[:, 1] -- the calibrated active-speaker probability (spike note
    `model_api` heading), not the upstream demo's uncalibrated `logits[:, 1] >= 0` shortcut."""
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits - m)
    return (exp / exp.sum(axis=-1, keepdims=True))[:, 1]


class LrAsdDetector:
    """`SpeakerDetector` backed by LR-ASD. Model load and weights download are lazy (first
    `score()` call), so constructing this on a machine without the extras is always safe."""

    def __init__(self, models_dir: Path, weights: str = DEFAULT_WEIGHTS) -> None:
        self.models_dir = Path(models_dir)
        self.weights = weights
        self._model = None
        self._head = None
        self._load_error: SpeakerDetectorUnavailable | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise self._load_error

        import torch

        from .lrasd_model import ASD_Model, AVScoreHead

        try:
            weights_path = _ensure_weights(self.models_dir, self.weights)
        except OSError as e:      # download failure (offline) or a WeightsVerificationError
            self._load_error = SpeakerDetectorUnavailable(f"LR-ASD weights unavailable: {e}")
            raise self._load_error from e
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        backbone_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        head_state = {k[len("lossAV."):]: v for k, v in state.items() if k.startswith("lossAV.FC.")}

        model = ASD_Model()
        model.load_state_dict(backbone_state)
        head = AVScoreHead()
        head.load_state_dict(head_state)
        model.eval()
        head.eval()   # MANDATORY: Detector has nn.Dropout(0.5); a train-mode forward corrupts scores
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = model.to(device)
            head = head.to(device)
            self._device = device
        except Exception:
            self._device = "cpu"
            model = model.to("cpu")
            head = head.to("cpu")
        self._model = model
        self._head = head

    def _forward(self, audio_tensor, video_tensor) -> np.ndarray:
        """Runs the network on GPU (with CPU fallback) under `torch.no_grad()` and returns
        raw (N, 2) logits as a numpy array. `audio_tensor`: (1, 4N, 13) float tensor.
        `video_tensor`: (1, N, 112, 112) float tensor, 0..255 range (normalisation happens
        inside `ASD_Model.forward_visual_frontend`, not here)."""
        import torch

        device = getattr(self, "_device", "cpu")
        try:
            audio_tensor = audio_tensor.to(device)
            video_tensor = video_tensor.to(device)
        except Exception:
            pass
        with torch.no_grad():
            audio_embed = self._model.forward_audio_frontend(audio_tensor)
            visual_embed = self._model.forward_visual_frontend(video_tensor)
            outs_av = self._model.forward_audio_visual_backend(audio_embed, visual_embed)
            logits = self._head(outs_av)
        if hasattr(logits, "cpu"):
            logits = logits.cpu()
        return logits.numpy()

    def score(self, crops: np.ndarray, mfcc: np.ndarray) -> list[float]:
        self._load()

        n = min(len(crops), mfcc.shape[0] // 4)   # reference demo's alignment rule (spike note)
        if n <= 0:
            return []

        import torch

        video_tensor = torch.FloatTensor(np.asarray(crops[:n], dtype=np.float32)).unsqueeze(0)
        audio_tensor = torch.FloatTensor(np.asarray(mfcc[: n * 4], dtype=np.float32)).unsqueeze(0)

        logits = self._forward(audio_tensor, video_tensor)
        probs = _softmax_speaking_column(logits)
        return [float(p) for p in probs]
