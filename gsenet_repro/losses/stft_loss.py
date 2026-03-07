"""STFT-based reconstruction losses."""

from __future__ import annotations

from typing import Literal

import numpy as np

from gsenet_repro.dsp.stft import LOSS_STFT, stft


def _stft_mag(x: np.ndarray, n_fft: int, win_length: int, hop_length: int) -> np.ndarray:
    return np.abs(stft(x, n_fft=n_fft, win_length=win_length, hop_length=hop_length))


def stft_reconstruction_loss(
    y_hat: np.ndarray,
    y_ref: np.ndarray,
    n_fft: int = LOSS_STFT["n_fft"],
    win_length: int = LOSS_STFT["win_length"],
    hop_length: int = LOSS_STFT["hop_length"],
    kind: Literal["l1", "l2"] = "l1",
) -> float:
    """Compute single-scale STFT magnitude reconstruction loss.

    Args:
        y_hat: Estimated waveform, shape (T,) or (B, T).
        y_ref: Reference waveform, same shape as y_hat.
        n_fft: FFT size.
        win_length: Window length.
        hop_length: Hop length.
        kind: "l1" (mean absolute error) or "l2" (mean squared error).

    Returns:
        Loss as a Python float.
    """
    y_hat = np.asarray(y_hat, dtype=np.float32)
    y_ref = np.asarray(y_ref, dtype=np.float32)
    if y_hat.shape != y_ref.shape:
        raise ValueError("y_hat and y_ref must have the same shape")

    if y_hat.ndim == 1:
        mag_hat = _stft_mag(y_hat, n_fft, win_length, hop_length)
        mag_ref = _stft_mag(y_ref, n_fft, win_length, hop_length)
        diff = mag_hat - mag_ref
        if kind == "l1":
            return float(np.mean(np.abs(diff)))
        if kind == "l2":
            return float(np.mean(diff**2))
        raise ValueError("kind must be 'l1' or 'l2'")

    if y_hat.ndim == 2:
        losses = [
            stft_reconstruction_loss(
                y_hat[idx],
                y_ref[idx],
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                kind=kind,
            )
            for idx in range(y_hat.shape[0])
        ]
        return float(np.mean(losses))

    raise ValueError("y_hat and y_ref must be 1D or 2D arrays")
