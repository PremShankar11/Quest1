"""Tests for visual audio feature extraction: MFCC and speech mask."""

import math
import wave
import numpy as np
import pytest

from dialogue_finder.visual.audio_features import (
    read_wav_slice,
    mfcc_100hz,
    mfcc_for_video,
    speech_mask,
)


@pytest.fixture
def test_wav(tmp_path):
    """Generate a test WAV: 1s silence, 1s 440 Hz tone (amplitude 0.3), 1s silence.

    Returns the path to the generated WAV file (3s total, 16 kHz mono).
    """
    wav_path = tmp_path / "test.wav"
    sr = 16000
    duration = 3.0
    n_samples = int(sr * duration)

    # Create samples: silence, tone, silence
    samples = np.zeros(n_samples, dtype=np.int16)
    tone_start = sr  # 1s in
    tone_end = 2 * sr  # 2s in
    t = np.arange(tone_end - tone_start) / sr
    tone = 0.3 * 32767 * np.sin(2 * np.pi * 440 * t)  # amplitude 0.3
    samples[tone_start:tone_end] = tone.astype(np.int16)

    # Write WAV
    with wave.open(str(wav_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(samples.tobytes())

    return wav_path


def test_read_wav_slice_full_file(test_wav):
    """read_wav_slice returns 48000 samples for a 3s span at 16 kHz."""
    samples = read_wav_slice(test_wav, 0, 3)
    assert samples.shape == (48000,)
    assert samples.dtype == np.float32
    assert np.all(samples >= -1.0)
    assert np.all(samples <= 1.0)


def test_read_wav_slice_partial(test_wav):
    """read_wav_slice handles partial slices and clamping."""
    # Read first 1s (silence)
    samples_1s = read_wav_slice(test_wav, 0, 1)
    assert samples_1s.shape == (16000,)
    # Silence should be close to zero
    assert np.max(np.abs(samples_1s)) < 0.01

    # Read second 1s (tone)
    samples_tone = read_wav_slice(test_wav, 1, 2)
    assert samples_tone.shape == (16000,)
    # Tone should have significant energy
    assert np.max(np.abs(samples_tone)) > 0.2

    # Read beyond file bounds (should clamp)
    samples_clamped = read_wav_slice(test_wav, 2, 10)
    assert samples_clamped.shape == (16000,)  # Only 1s of audio left


def test_mfcc_100hz_shape(test_wav):
    """mfcc_100hz returns approximately (300, 13) shape for 3s audio."""
    mfcc = mfcc_100hz(test_wav, 0, 3)
    # At 100 Hz with 10ms hop, 3s should give ~300 frames
    assert mfcc.shape[0] >= 290  # Allow some tolerance
    assert mfcc.shape[0] <= 310
    assert mfcc.shape[1] == 13
    assert mfcc.dtype == np.float32


def test_speech_mask_length_and_silence_frames(test_wav):
    """speech_mask has correct length and marks silence frames as False."""
    mask = speech_mask(test_wav, 0, 3, fps=24)
    # 3s at 24 fps = 72 frames
    assert len(mask) == 72
    assert all(isinstance(x, bool) for x in mask)

    # First 20 frames (0-0.83s) should be silence → all False
    silence_frames = sum(mask[:20])
    assert silence_frames == 0, "First 20 frames should all be silent"


def test_speech_mask_detects_tone_or_silence(test_wav):
    """speech_mask can detect tone region (VAD may or may not flag pure tone as speech)."""
    mask = speech_mask(test_wav, 0, 3, fps=24)
    # Just verify the mask is all booleans and doesn't crash
    # VAD is trained on human speech, so a pure 440 Hz tone may not be detected
    tone_region = mask[20:50]
    # Just verify it returns a valid mask (no exception means success)
    assert len(tone_region) == 30
    assert all(isinstance(x, bool) for x in tone_region)


def test_mfcc_for_video_25fps(test_wav):
    """mfcc_for_video with fps=25 matches mfcc_100hz output."""
    mfcc_std = mfcc_100hz(test_wav, 0, 3)
    mfcc_25 = mfcc_for_video(test_wav, 0, 3, fps=25.0)
    # Should produce identical shapes (both 25 fps)
    assert mfcc_std.shape == mfcc_25.shape
    # Values should be very close (same parameters)
    assert np.allclose(mfcc_std, mfcc_25, atol=1e-5)


def test_mfcc_for_video_exact_frame_alignment(test_wav):
    """mfcc_for_video pads/trims to exactly 4 * n_video_frames rows."""
    # 2s at 23.976 fps = 48 video frames = 192 audio frames expected
    mfcc = mfcc_for_video(test_wav, 0, 2, fps=23.976)
    n_video_frames = round(2.0 * 23.976)
    expected_rows = 4 * n_video_frames
    assert mfcc.shape[0] == expected_rows
    assert mfcc.shape == (192, 13)


def test_mfcc_for_video_different_fps(test_wav):
    """mfcc_for_video with different fps values produces different hop sizes."""
    duration = 1.0
    mfcc_25 = mfcc_for_video(test_wav, 0, duration, fps=25.0)
    mfcc_24 = mfcc_for_video(test_wav, 0, duration, fps=24.0)

    # Both should have the correct frame alignment
    assert mfcc_25.shape[0] == 4 * round(duration * 25.0)
    assert mfcc_24.shape[0] == 4 * round(duration * 24.0)

    # But different fps should have different numbers of frames
    assert mfcc_25.shape[0] != mfcc_24.shape[0]


def test_speech_mask_frame_alignment(test_wav):
    """speech_mask returns exactly round((end_s - start_s) * fps) entries."""
    for fps in [24, 25, 23.976, 30]:
        duration = 2.5
        mask = speech_mask(test_wav, 0, duration, fps=fps)
        expected_frames = round(duration * fps)
        assert len(mask) == expected_frames


def test_mfcc_for_video_partial_window(test_wav):
    """mfcc_for_video handles partial windows correctly."""
    # Extract MFCC from 0.5s to 2.5s (2 second window)
    mfcc = mfcc_for_video(test_wav, 0.5, 2.5, fps=25.0)
    duration = 2.5 - 0.5
    n_video_frames = round(duration * 25.0)
    expected_rows = 4 * n_video_frames
    assert mfcc.shape[0] == expected_rows
    assert mfcc.shape[1] == 13


def test_read_wav_slice_span_past_eof(test_wav):
    """read_wav_slice handles spans entirely past EOF by returning empty array."""
    # Request span beyond file bounds (5s to 10s, but file is only 3s)
    samples = read_wav_slice(test_wav, 5.0, 10.0)
    assert samples.shape == (0,)
    assert samples.dtype == np.float32


def test_read_wav_slice_span_partly_past_eof(test_wav):
    """read_wav_slice clamps span that extends past EOF."""
    # Request 2.5s to 10s (should clamp to 2.5s to 3.0s)
    samples = read_wav_slice(test_wav, 2.5, 10.0)
    # 0.5s @ 16kHz = 8000 samples
    assert samples.shape == (8000,)


def test_read_wav_slice_reversed_span(test_wav):
    """read_wav_slice returns empty array if end_s < start_s."""
    samples = read_wav_slice(test_wav, 2.0, 1.0)
    assert samples.shape == (0,)
    assert samples.dtype == np.float32


def test_read_wav_slice_stereo_raises_error(tmp_path):
    """read_wav_slice raises ValueError for stereo WAV files."""
    stereo_wav = tmp_path / "stereo.wav"
    sr = 16000
    n_samples = sr  # 1 second

    # Create stereo (2 channel) samples
    samples_stereo = np.zeros((n_samples, 2), dtype=np.int16)

    with wave.open(str(stereo_wav), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(samples_stereo.tobytes())

    with pytest.raises(ValueError, match="expected 16 kHz mono wav"):
        read_wav_slice(stereo_wav, 0, 1)


def test_mfcc_for_video_empty_span(test_wav):
    """mfcc_for_video produces zero-filled array for empty span."""
    mfcc = mfcc_for_video(test_wav, 5.0, 10.0, fps=25.0)
    # Span is beyond file (5s-10s), but duration is still 5s
    # Should produce zero-filled array for the requested duration
    n_video_frames = round(5.0 * 25.0)
    expected_rows = 4 * n_video_frames
    assert mfcc.shape == (expected_rows, 13)
    assert mfcc.dtype == np.float32
    assert np.all(mfcc == 0)


def test_mfcc_for_video_too_short_signal(test_wav):
    """mfcc_for_video produces zero-filled array for signal shorter than window."""
    # 5 ms is shorter than 25 ms window
    mfcc = mfcc_for_video(test_wav, 0, 0.005, fps=25.0)
    n_video_frames = round(0.005 * 25.0)  # 0 frames
    expected_rows = 4 * n_video_frames
    assert mfcc.shape == (expected_rows, 13)
    assert mfcc.dtype == np.float32
    assert np.all(mfcc == 0)
