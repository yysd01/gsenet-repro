from __future__ import annotations

import importlib.util
from typing import Dict

import numpy as np

from gsenet_repro.dsp import MODEL_STFT
from gsenet_repro.dsp.mcwf import apply_mcwf_mask, mcwf
from gsenet_repro.dsp.stft import istft, stft

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    import torch
else:  # pragma: no cover
    torch = None


def _windowed_power(
    input_stft: np.ndarray,
    window_len: int,
) -> np.ndarray:
    power = np.abs(input_stft).astype(np.float32) ** 2
    power_pad = np.pad(power, ((0, 0), (0, 0), (window_len - 1, 0), (0, 0)))
    cumulative = np.cumsum(power_pad, axis=2, dtype=np.float32)
    cumulative = np.pad(cumulative, ((0, 0), (0, 0), (1, 0), (0, 0)), mode="constant")
    window_sum = cumulative[:, :, window_len:, :] - cumulative[:, :, :-window_len, :]
    return window_sum / float(window_len)


def mcwf_make_y0(
    x_mics: np.ndarray | "torch.Tensor",
    stft_params: Dict[str, int] | None,
    causal_frames: int = 4,
) -> np.ndarray | "torch.Tensor":
    """Generate a single-channel MCWF output from 3-mic waveforms.

    This is a simplified placeholder that reuses gsenet_repro.dsp.mcwf and
    collapses the three channels with a noise-power-weighted average.
    """
    input_is_torch = torch is not None and torch.is_tensor(x_mics)
    x_np = x_mics.detach().cpu().numpy() if input_is_torch else np.asarray(x_mics)

    if x_np.ndim == 2:
        x_np = x_np[None, ...]
    if x_np.ndim != 3 or x_np.shape[1] != 3:
        raise ValueError("x_mics must have shape (3, T) or (B, 3, T)")

    params = dict(stft_params) if stft_params is not None else dict(MODEL_STFT)
    n_fft = int(params["n_fft"])
    win_length = int(params["win_length"])
    hop_length = int(params["hop_length"])

    batch, _, length = x_np.shape
    stft_list = []
    for b in range(batch):
        mic_stfts = []
        for mic_idx in range(3):
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

    windowed_power = mcwf(
        input_stft,
        stft_win_length=win_length,
        stft_hop_size=hop_length,
        noise_pow=0.0,
        signal_pow=1.0,
    )
    if causal_frames != 4:
        windowed_power = _windowed_power(input_stft, window_len=causal_frames)

    signal_pow = windowed_power.mean(axis=2, keepdims=True)
    noise_pow = windowed_power.var(axis=2, keepdims=True)
    filtered = apply_mcwf_mask(input_stft, noise_pow, signal_pow)

    noise_pow_mic = np.mean(noise_pow, axis=(1, 2))
    weights = 1.0 / (noise_pow_mic + 1e-8)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    y0_stft = np.sum(filtered * weights[:, None, None, :], axis=-1)

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
