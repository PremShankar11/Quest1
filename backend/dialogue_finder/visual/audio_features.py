"""Audio features for visual verification: MFCC and speech activity detection.

Extracts MFCC (Mel-Frequency Cepstral Coefficients) and speech activity masks
from 16 kHz WAV files, aligned to video frames for LR-ASD inference.

IMPORTANT: Frame-rate-aware MFCC extraction (Controller ruling, 2026-08-25)
---
LR-ASD was trained on 25 fps video with 4 audio frames per video frame
(100 Hz MFCC → 25 Hz after model's time reductions). When processing
23.976 fps video directly (not resampled to 25 fps), the frame-count
constraint T_audio == 4 * T_video is architectural and must be enforced
exactly, but timestamps drift at ~0.25 s per 10 s window vs. true audio time.

Implementation uses fps-aware hop size: winstep = 1.0 / (4.0 * fps)
- mfcc_for_video(fps=25.0) → standard 100 Hz (winstep=0.010)
- mfcc_for_video(fps=23.976) → 95.904 Hz (winstep≈0.010421) for native 23.976 fps
- Both pad/trim to exactly 4 * n_video_frames rows to maintain alignment
---

Interfaces:
- read_wav_slice(wav, start_s, end_s) → float32 [-1, 1] samples at 16 kHz
- mfcc_100hz(wav, start_s, end_s) → MFCC at standard 25 fps (thin wrapper)
- mfcc_for_video(wav, start_s, end_s, fps) → MFCC at fps-aware hop size
- speech_mask(wav, start_s, end_s, fps) → bool per video frame (VAD-based)
"""

from pathlib import Path
import numpy as np
import wave

from faster_whisper.vad import get_speech_timestamps, VadOptions


def read_wav_slice(wav: Path, start_s: float, end_s: float) -> np.ndarray:
    """Read a time slice from a 16 kHz mono WAV file, returning float32 samples.

    Args:
        wav: Path to 16 kHz mono WAV file.
        start_s: Start time in seconds (clamped to [0, duration]).
        end_s: End time in seconds (clamped to [0, duration]).

    Returns:
        float32 numpy array in range [-1, 1], shape (n_samples,).
    """
    with wave.open(str(wav), "rb") as f:
        sr = f.getframerate()
        n_frames = f.getnframes()

        # Clamp to file bounds
        start_s = max(0.0, start_s)
        end_s = min(end_s, n_frames / sr)

        # Convert to frame indices
        start_frame = int(start_s * sr)
        end_frame = int(end_s * sr)

        # Read the slice
        f.setpos(start_frame)
        data_int16 = f.readframes(end_frame - start_frame)

        # Decode int16 → float32 in [-1, 1]
        samples = np.frombuffer(data_int16, dtype=np.int16).astype(np.float32) / 32768.0
        return samples


def mfcc_for_video(wav: Path, start_s: float, end_s: float, fps: float) -> np.ndarray:
    """Extract MFCC aligned to video frames at a given frame rate.

    Computes MFCC with fps-aware hop size to maintain exactly 4 audio frames
    per video frame, as required by LR-ASD's architecture.

    Args:
        wav: Path to 16 kHz mono WAV file.
        start_s: Start time in seconds.
        end_s: End time in seconds.
        fps: Video frame rate (e.g., 25.0, 23.976).

    Returns:
        MFCC array, shape (4 * n_video_frames, 13), dtype float32, where
        n_video_frames = round((end_s - start_s) * fps).
    """
    import python_speech_features

    signal = read_wav_slice(wav, start_s, end_s)
    sr = 16000

    # Parameters from spike note (preprocessing heading)
    winlen = 0.025      # 25 ms
    numcep = 13         # 13 MFCCs

    # Fps-aware hop size: ensures 4 audio frames per video frame
    winstep = 1.0 / (4.0 * fps)

    # Compute raw MFCC at dynamic hop size
    mfcc = python_speech_features.mfcc(
        signal, sr, numcep=numcep, winlen=winlen, winstep=winstep
    )

    # Pad/trim to exactly 4 * n_video_frames rows
    duration_s = end_s - start_s
    n_video_frames = round(duration_s * fps)
    target_rows = 4 * n_video_frames

    if mfcc.shape[0] < target_rows:
        # Pad with zeros to the target length
        pad_rows = target_rows - mfcc.shape[0]
        mfcc = np.vstack([mfcc, np.zeros((pad_rows, numcep), dtype=np.float32)])
    elif mfcc.shape[0] > target_rows:
        # Trim to target length
        mfcc = mfcc[:target_rows]

    return mfcc.astype(np.float32)


def mfcc_100hz(wav: Path, start_s: float, end_s: float) -> np.ndarray:
    """Extract MFCC at standard 100 Hz (10 ms hop, 25 fps model).

    Thin wrapper around mfcc_for_video with fps=25.0 for compatibility
    with the LR-ASD reference training configuration.

    Args:
        wav: Path to 16 kHz mono WAV file.
        start_s: Start time in seconds.
        end_s: End time in seconds.

    Returns:
        MFCC array, shape (T, 13) at 100 Hz, dtype float32.
    """
    return mfcc_for_video(wav, start_s, end_s, fps=25.0)


def speech_mask(wav: Path, start_s: float, end_s: float, fps: float) -> list[bool]:
    """Detect which video frames contain speech using Silero VAD.

    Rasterizes VAD speech segments to a boolean mask with one entry per video frame.
    A frame is True if its center time falls inside any detected speech segment.

    Args:
        wav: Path to 16 kHz mono WAV file.
        start_s: Start time in seconds.
        end_s: End time in seconds.
        fps: Video frame rate.

    Returns:
        List of bool, length = round((end_s - start_s) * fps), where mask[i] is True
        if video frame i (at absolute time start_s + i/fps) falls within a speech segment.
    """
    signal = read_wav_slice(wav, start_s, end_s)
    sr = 16000

    # Run VAD with default options
    vad_options = VadOptions()
    speech_chunks = get_speech_timestamps(signal, vad_options, sampling_rate=sr)

    # Build a set of sample indices that are marked as speech
    speech_samples = set()
    for chunk in speech_chunks:
        start_sample = chunk["start"]
        end_sample = chunk["end"]
        speech_samples.update(range(start_sample, end_sample))

    # Rasterize to video frames
    duration_s = end_s - start_s
    n_video_frames = round(duration_s * fps)
    mask = []

    for frame_idx in range(n_video_frames):
        # Center time of this frame (in seconds relative to start_s)
        frame_center_s = (frame_idx + 0.5) / fps
        # Convert to sample index within the signal
        frame_center_sample = int(frame_center_s * sr)

        # Check if this sample falls in a speech chunk
        is_speech = frame_center_sample in speech_samples
        mask.append(is_speech)

    return mask
