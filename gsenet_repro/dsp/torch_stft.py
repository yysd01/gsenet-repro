"""Torch STFT/iSTFT utilities for GSENet reproduction."""

from __future__ import annotations

from typing import Optional

import torch


def _make_window(
    win_length: int,
    device: torch.device,
    dtype: torch.dtype,
    center: bool,
    eps: float = 1e-4,
) -> torch.Tensor:
    window = torch.hann_window(win_length, periodic=True, device=device, dtype=dtype)
    # With center=False, there is no analysis padding and the first sample is
    # multiplied by window[0]. Standard periodic Hann has window[0] == 0, which
    # makes boundary samples non-invertible. Lift only the endpoints in the
    # center=False path to keep center=True behavior unchanged.
    if not center:
        window = window.clone()
        window[0] = eps
        window[-1] = torch.maximum(
            window[-1], torch.tensor(eps, device=device, dtype=dtype)
        )
    return window


def stft(
    x: torch.Tensor,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str = "hann",
    center: bool = False,
) -> torch.Tensor:
    if x.ndim not in (1, 2):
        raise ValueError("stft expects a 1D or 2D tensor")
    if window != "hann":
        raise ValueError("Only 'hann' window is supported")
    if x.numel() == 0:
        complex_dtype = torch.complex64 if x.dtype == torch.float32 else torch.complex128
        if x.ndim == 1:
            return torch.zeros((n_fft // 2 + 1, 0), dtype=complex_dtype, device=x.device)
        return torch.zeros((x.shape[0], n_fft // 2 + 1, 0), dtype=complex_dtype, device=x.device)

    window_tensor = _make_window(win_length, device=x.device, dtype=x.dtype, center=center)
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_tensor,
        center=center,
        return_complex=True,
    )


def _istft_center_false_manual(
    X: torch.Tensor,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window_tensor: torch.Tensor,
    length: Optional[int],
) -> torch.Tensor:
    squeeze = X.ndim == 2
    if squeeze:
        X = X.unsqueeze(0)
    B, F, T = X.shape
    if F != n_fft // 2 + 1:
        raise ValueError("STFT frequency dimension mismatch")

    frames = torch.fft.irfft(X, n=n_fft, dim=1)[:, :win_length, :]  # (B, win, T)
    frames = frames * window_tensor.view(1, win_length, 1)
    out_len = win_length + hop_length * max(T - 1, 0)
    y = torch.zeros((B, out_len), dtype=X.real.dtype, device=X.device)
    norm = torch.zeros((out_len,), dtype=X.real.dtype, device=X.device)

    w2 = window_tensor.square()
    for t in range(T):
        start = t * hop_length
        end = start + win_length
        y[:, start:end] += frames[:, :, t]
        norm[start:end] += w2

    y = y / norm.clamp_min(1e-8).unsqueeze(0)
    if length is not None:
        if y.shape[-1] < length:
            y = torch.nn.functional.pad(y, (0, length - y.shape[-1]))
        y = y[..., :length]
    return y.squeeze(0) if squeeze else y


def istft(
    X: torch.Tensor,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str = "hann",
    length: Optional[int] = None,
    center: bool = False,
) -> torch.Tensor:
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

    window_tensor = _make_window(win_length, device=X.device, dtype=X.real.dtype, center=center)
    if not center:
        try:
            return torch.istft(
                X,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window_tensor,
                center=False,
                length=length,
            )
        except RuntimeError:
            return _istft_center_false_manual(X, n_fft, win_length, hop_length, window_tensor, length)
    return torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_tensor,
        center=True,
        length=length,
    )


torch_stft = stft
torch_istft = istft
