"""Torch utilities for MCWF-style preprocessing."""
from __future__ import annotations

import torch
from torch import Tensor


def windowed_power(input_stft: Tensor, window_len: int = 4) -> Tensor:
    """Compute causal windowed power for complex STFT input.

    Args:
        input_stft: Complex tensor of shape (B, F, T, C).
        window_len: Number of causal frames for averaging.

    Returns:
        Windowed power tensor with shape (B, F, T, C).
    """
    if input_stft.ndim != 4:
        raise ValueError("input_stft must have shape (B, F, T, C)")
    if window_len <= 0:
        raise ValueError("window_len must be positive")

    power = input_stft.abs().pow(2)
    if window_len == 1:
        return power

    pad_shape = (power.shape[0], power.shape[1], window_len - 1, power.shape[3])
    pad = torch.zeros(pad_shape, dtype=power.dtype, device=power.device)
    power_pad = torch.cat([pad, power], dim=2)
    cumulative = torch.cumsum(power_pad, dim=2)
    zero = torch.zeros_like(cumulative[:, :, :1, :])
    cumulative = torch.cat([zero, cumulative], dim=2)
    window_sum = cumulative[:, :, window_len:, :] - cumulative[:, :, :-window_len, :]
    return window_sum / float(window_len)
