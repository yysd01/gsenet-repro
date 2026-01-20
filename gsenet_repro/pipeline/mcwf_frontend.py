from __future__ import annotations

import importlib.util
from typing import Dict

import numpy as np

from gsenet_repro.dsp import MODEL_STFT
from gsenet_repro.dsp.mcwf import apply_mcwf_mask
from gsenet_repro.dsp.stft import istft, stft

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    import torch
else:  # pragma: no cover
    torch = None


def _windowed_stats(
    power: np.ndarray,
    window_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    if window_len <= 0:
        raise ValueError("window_len must be positive")
    power_pad = np.pad(power, ((0, 0), (0, 0), (window_len - 1, 0), (0, 0)))
    cumulative = np.cumsum(power_pad, axis=2, dtype=np.float32)
    cumulative = np.pad(cumulative, ((0, 0), (0, 0), (1, 0), (0, 0)), mode="constant")
    window_sum = cumulative[:, :, window_len:, :] - cumulative[:, :, :-window_len, :]
    mean = window_sum / float(window_len)

    power_sq = power.astype(np.float32) ** 2
    power_sq_pad = np.pad(power_sq, ((0, 0), (0, 0), (window_len - 1, 0), (0, 0)))
    cumulative_sq = np.cumsum(power_sq_pad, axis=2, dtype=np.float32)
    cumulative_sq = np.pad(
        cumulative_sq, ((0, 0), (0, 0), (1, 0), (0, 0)), mode="constant"
    )
    window_sum_sq = cumulative_sq[:, :, window_len:, :] - cumulative_sq[:, :, :-window_len, :]
    mean_sq = window_sum_sq / float(window_len)
    var = np.maximum(mean_sq - mean**2, 0.0)
    return mean, var


def mcwf_make_y0(
    x_mics: np.ndarray | "torch.Tensor",
    stft_params: Dict[str, int] | None,
    causal_frames: int = 4,
) -> np.ndarray | "torch.Tensor":
    """Generate a single-channel MCWF output from multi-mic waveforms.

    This is a simplified, reproducible MCWF implementation that applies a
    causal windowed power estimate (default 4 frames), computes a Wiener-style
    gain, and averages the filtered channels into a single output.
    """
    input_is_torch = torch is not None and torch.is_tensor(x_mics)
    x_np = x_mics.detach().cpu().numpy() if input_is_torch else np.asarray(x_mics)

    if x_np.ndim == 2:
        x_np = x_np[None, ...]
    if x_np.ndim != 3 or x_np.shape[1] < 2:
        raise ValueError("x_mics must have shape (C, T) or (B, C, T) with C>=2")

    params = dict(stft_params) if stft_params is not None else dict(MODEL_STFT)
    n_fft = int(params["n_fft"])
    win_length = int(params["win_length"])
    hop_length = int(params["hop_length"])

    batch, channels, length = x_np.shape
    stft_list = []
    for b in range(batch):
        mic_stfts = []
        for mic_idx in range(channels):
            mic_stfts.append(
                stft(
                    x_np[b, mic_idx],
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    center=False,
                )
            )
        mic_stack = np.stack(mic_stfts, axis=-1)
        stft_list.append(mic_stack)
    input_stft = np.stack(stft_list, axis=0)

    power = np.abs(input_stft).astype(np.float32) ** 2
    signal_pow, noise_pow = _windowed_stats(power, window_len=causal_frames)
    filtered = apply_mcwf_mask(input_stft, noise_pow, signal_pow)
    y0_stft = np.mean(filtered, axis=-1)

    y0_list = []
    for b in range(batch):
        y0_list.append(
            istft(
                y0_stft[b],
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=False,
                length=length,
            )
        )
    y0 = np.stack(y0_list, axis=0).astype(np.float32)
    if input_is_torch:
        return torch.tensor(y0, device=x_mics.device, dtype=x_mics.dtype)
    if x_mics.ndim == 2:
        return y0[0]
    return y0
