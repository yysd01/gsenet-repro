"""Simplified multi-channel Wiener filter (MCWF) in STFT domain."""
from __future__ import annotations

from typing import Union

import numpy as np


def _as_float_array(x: Union[float, np.ndarray]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _validate_input(input_stft: np.ndarray, stft_win_length: int, stft_hop_size: int) -> None:
    if not isinstance(input_stft, np.ndarray):
        raise TypeError("input_stft must be a numpy array")
    if input_stft.ndim != 4:
        raise ValueError("input_stft must have shape (B, F, T, 3)")
    if input_stft.shape[-1] != 3:
        raise ValueError("input_stft must have 3 microphone channels")
    if stft_win_length <= 0 or stft_hop_size <= 0:
        raise ValueError("stft_win_length and stft_hop_size must be positive")


def mcwf(
    input_stft: np.ndarray,
    stft_win_length: int,
    stft_hop_size: int,
    noise_pow: Union[float, np.ndarray],
    signal_pow: Union[float, np.ndarray],
) -> np.ndarray:
    """Apply a simplified multi-channel Wiener filter in the STFT domain.

    Args:
        input_stft: Complex STFT input with shape (B, F, T, 3).
        stft_win_length: STFT window length (kept for interface parity).
        stft_hop_size: STFT hop size (kept for interface parity).
        noise_pow: Noise power estimate (scalar or broadcastable array).
        signal_pow: Signal power estimate (scalar or broadcastable array).

    Returns:
        Filtered power spectrum with the same shape as input_stft.
    """
    _validate_input(input_stft, stft_win_length, stft_hop_size)

    power = np.abs(input_stft).astype(np.float32) ** 2
    window_len = 4

    power_pad = np.pad(power, ((0, 0), (0, 0), (window_len - 1, 0), (0, 0)))
    cumulative = np.cumsum(power_pad, axis=2, dtype=np.float32)
    cumulative = np.pad(cumulative, ((0, 0), (0, 0), (1, 0), (0, 0)), mode="constant")
    window_sum = cumulative[:, :, window_len:, :] - cumulative[:, :, :-window_len, :]
    windowed_power = window_sum / float(window_len)

    signal_pow_arr = _as_float_array(signal_pow)
    noise_pow_arr = _as_float_array(noise_pow)
    eps = np.float32(1e-8)
    gain = signal_pow_arr / (signal_pow_arr + noise_pow_arr + eps)

    output = windowed_power * gain
    return np.asarray(output, dtype=np.float32)
