"""Torch STFT/iSTFT utilities with causal settings."""
from __future__ import annotations

from typing import Optional

import torch

from gsenet_repro.dsp.stft import MODEL_STFT


def _hann_window(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    window = torch.hann_window(length, periodic=True, device=device, dtype=dtype)
    return window


def torch_stft(
    x: torch.Tensor,
    n_fft: int = MODEL_STFT["n_fft"],
    win_length: int = MODEL_STFT["win_length"],
    hop_length: int = MODEL_STFT["hop_length"],
    center: bool = False,
) -> torch.Tensor:
    """Compute torch STFT with causal alignment (center=False).

    Args:
        x: Waveform tensor, shape (T,) or (B, T).
        n_fft: FFT size.
        win_length: Window length.
        hop_length: Hop length.
        center: Must be False for causal usage.
    """
    if center:
        raise ValueError("torch_stft must be called with center=False for causality.")
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True
    elif x.dim() == 2:
        squeeze = False
    else:
        raise ValueError("torch_stft expects a 1D or 2D tensor")

    window = _hann_window(win_length, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )
    spec = spec.to(torch.complex64)
    if squeeze:
        return spec.squeeze(0)
    return spec


def torch_istft(
    X: torch.Tensor,
    n_fft: int = MODEL_STFT["n_fft"],
    win_length: int = MODEL_STFT["win_length"],
    hop_length: int = MODEL_STFT["hop_length"],
    length: Optional[int] = None,
    center: bool = False,
) -> torch.Tensor:
    """Inverse torch STFT with causal alignment (center=False)."""
    if center:
        raise ValueError("torch_istft must be called with center=False for causality.")
    if X.dim() == 2:
        X = X.unsqueeze(0)
        squeeze = True
    elif X.dim() == 3:
        squeeze = False
    else:
        raise ValueError("torch_istft expects a 2D or 3D complex tensor")

    window = _hann_window(win_length, device=X.device, dtype=X.real.dtype)
    y = torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        length=length,
    )
    if squeeze:
        return y.squeeze(0)
    return y
