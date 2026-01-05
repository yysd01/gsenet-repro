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
        x: Waveform tensor of shape (T,) or (B, T).
        n_fft: FFT size.
        win_length: Window length.
        hop_length: Hop length.
        window: Window type, only "hann" is supported.
        center: If True, pad by win_length // 2 before STFT.

    Returns:
        Complex STFT of shape (n_fft // 2 + 1, frames) for 1D input or
        (B, n_fft // 2 + 1, frames) for batched input.
    """
    if x.ndim not in (1, 2):
        raise ValueError("stft expects a 1D or 2D tensor")
    if window != "hann":
        raise ValueError("Only 'hann' window is supported")
    if x.numel() == 0:
        complex_dtype = torch.complex64 if x.dtype == torch.float32 else torch.complex128
        if x.ndim == 1:
            return torch.zeros((n_fft // 2 + 1, 0), dtype=complex_dtype, device=x.device)
        return torch.zeros(
            (x.shape[0], n_fft // 2 + 1, 0), dtype=complex_dtype, device=x.device
        )

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
        X: Complex STFT of shape (n_fft // 2 + 1, frames) or
            (B, n_fft // 2 + 1, frames).
        n_fft: FFT size.
        win_length: Window length.
        hop_length: Hop length.
        window: Window type, only "hann" is supported.
        length: Optional output length after trimming.
        center: If True, remove win_length // 2 samples from both ends.

    Returns:
        Time-domain reconstruction of shape (T,) or (B, T).
    """
    if X.ndim not in (2, 3):
        raise ValueError("istft expects a 2D or 3D complex STFT tensor")
    if window != "hann":
        raise ValueError("Only 'hann' window is supported")
    if X.numel() == 0:
        if X.ndim == 2:
            out_shape = (0,) if length is None else (length,)
            return torch.zeros(out_shape, dtype=X.real.dtype, device=X.device)
        batch = X.shape[0]
        if length is None:
            return torch.zeros((batch, 0), dtype=X.real.dtype, device=X.device)
        return torch.zeros((batch, length), dtype=X.real.dtype, device=X.device)

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


torch_stft = stft
torch_istft = istft
