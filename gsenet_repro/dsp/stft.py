"""STFT/iSTFT utilities for GSENet reproduction."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

MODEL_STFT: Dict[str, int] = {
    "n_fft": 320,
    "win_length": 320,
    "hop_length": 160,
}

LOSS_STFT: Dict[str, int] = {
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 256,
}


def _to_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def stft(
    x: np.ndarray,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str = "hann",
    center: bool = False,
) -> np.ndarray:
    """Compute STFT.

    Args:
        x: 1D waveform (float32 preferred).
        n_fft: FFT size.
        win_length: Window length.
        hop_length: Hop length.
        window: Window type, only \"hann\" is supported.
        center: If True, zero-pad by win_length//2 before STFT. Default False.
            For near-perfect roundtrip with Hann windows, use center=True.

    Returns:
        Complex STFT of shape (n_fft // 2 + 1, frames) in np.complex64.
    """
    x = _to_float32(x)
    if x.ndim != 1:
        raise ValueError("stft expects a 1D array")
    if window != "hann":
        raise ValueError("Only 'hann' window is supported")

    if center:
        pad = win_length // 2
        x = np.pad(x, (pad, pad), mode="constant")

    if x.size == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.complex64)

    n_frames = 1 + max(0, int(np.ceil((x.shape[0] - win_length) / hop_length)))
    total_len = (n_frames - 1) * hop_length + win_length
    if total_len > x.shape[0]:
        x = np.pad(x, (0, total_len - x.shape[0]), mode="constant")

    window_vals = np.hanning(win_length).astype(np.float32)
    stft_matrix = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)

    for idx in range(n_frames):
        start = idx * hop_length
        frame = x[start : start + win_length]
        frame = frame * window_vals
        if win_length < n_fft:
            frame = np.pad(frame, (0, n_fft - win_length), mode="constant")
        spectrum = np.fft.rfft(frame, n=n_fft)
        stft_matrix[:, idx] = spectrum.astype(np.complex64)

    return stft_matrix


def istft(
    X: np.ndarray,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str = "hann",
    length: Optional[int] = None,
    center: bool = False,
) -> np.ndarray:
    """Inverse STFT.

    Args:
        X: Complex STFT of shape (n_fft // 2 + 1, frames).
        n_fft: FFT size.
        win_length: Window length.
        hop_length: Hop length.
        window: Window type, only \"hann\" is supported.
        length: Optional output length after trimming.
        center: If True, removes win_length//2 samples from both ends.
            Should match the center flag used in stft.

    Returns:
        Time-domain reconstruction in np.float32.
    """
    if X.ndim != 2:
        raise ValueError("istft expects a 2D complex STFT array")
    if window != "hann":
        raise ValueError("Only 'hann' window is supported")

    n_frames = X.shape[1]
    if n_frames == 0:
        return np.zeros((0,), dtype=np.float32)

    expected_len = (n_frames - 1) * hop_length + win_length
    y = np.zeros(expected_len, dtype=np.float32)
    win_sum = np.zeros(expected_len, dtype=np.float32)
    window_vals = np.hanning(win_length).astype(np.float32)

    for idx in range(n_frames):
        start = idx * hop_length
        frame = np.fft.irfft(X[:, idx], n=n_fft).astype(np.float32)
        frame = frame[:win_length] * window_vals
        y[start : start + win_length] += frame
        win_sum[start : start + win_length] += window_vals**2

    nonzero = win_sum > 1e-8
    y[nonzero] /= win_sum[nonzero]

    if center:
        pad = win_length // 2
        if y.size >= 2 * pad:
            y = y[pad:-pad]
        else:
            y = np.array([], dtype=y.dtype)

    if length is not None:
        y = y[:length]
        if y.size < length:
            y = np.pad(y, (0, length - y.size), mode="constant")

    return np.asarray(y, dtype=np.float32)
