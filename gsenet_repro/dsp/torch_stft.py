"""Torch STFT/iSTFT utilities for GSENet reproduction."""
from __future__ import annotations

from typing import Optional

import torch


def stft(
    x: torch.Tensor,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str = "hann",
    center: bool = False,
) -> torch.Tensor:
    """Compute torch STFT.

    Args:
        x: 1D waveform tensor.
        n_fft: FFT size.
        win_length: Window length.
        hop_length: Hop length.
        window: Window type, only "hann" is supported.
        center: If True, pad by win_length // 2 before STFT.

    Returns:
        Complex STFT of shape (n_fft // 2 + 1, frames).
    """
    if x.ndim != 1:
        raise ValueError("stft expects a 1D tensor")
    if window != "hann":
        raise ValueError("Only 'hann' window is supported")
    if x.numel() == 0:
        complex_dtype = torch.complex64 if x.dtype == torch.float32 else torch.complex128
        return torch.zeros((n_fft // 2 + 1, 0), dtype=complex_dtype, device=x.device)

    window_tensor = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_tensor,
        center=center,
        return_complex=True,
    )


def istft(
    X: torch.Tensor,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str = "hann",
    length: Optional[int] = None,
    center: bool = False,
) -> torch.Tensor:
    """Inverse torch STFT.

    Args:
        X: Complex STFT of shape (n_fft // 2 + 1, frames).
        n_fft: FFT size.
        win_length: Window length.
        hop_length: Hop length.
        window: Window type, only "hann" is supported.
        length: Optional output length after trimming.
        center: If True, remove win_length // 2 samples from both ends.

    Returns:
        Time-domain reconstruction.
    """
    if X.ndim != 2:
        raise ValueError("istft expects a 2D complex STFT tensor")
    if window != "hann":
        raise ValueError("Only 'hann' window is supported")
    if X.numel() == 0:
        return torch.zeros((0,), dtype=X.real.dtype, device=X.device)

    window_tensor = torch.hann_window(win_length, device=X.device, dtype=X.real.dtype)
    return torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_tensor,
        center=center,
        length=length,
    )
