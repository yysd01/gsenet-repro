"""Torch STFT reconstruction loss."""

from __future__ import annotations

from typing import Dict

import torch

from gsenet_repro.dsp import LOSS_STFT
from gsenet_repro.dsp.torch_stft import torch_stft


def stft_magnitude_loss(
    y_hat: torch.Tensor,
    y_ref: torch.Tensor,
    stft_params: Dict[str, int] | None = None,
) -> torch.Tensor:
    """Single-scale STFT magnitude reconstruction loss.

    Args:
        y_hat: Estimated waveform tensor of shape (B, T).
        y_ref: Reference waveform tensor of shape (B, T).
        stft_params: Optional STFT parameter overrides.

    Returns:
        Scalar loss tensor.
    """
    if y_hat.ndim != 2 or y_ref.ndim != 2:
        raise ValueError("y_hat and y_ref must have shape (B, T)")
    if y_hat.shape != y_ref.shape:
        raise ValueError("y_hat and y_ref must have the same shape")

    params = dict(stft_params) if stft_params is not None else dict(LOSS_STFT)
    params.pop("center", None)

    Y_hat = torch_stft(y_hat, **params, center=False)
    Y_ref = torch_stft(y_ref, **params, center=False)
    mag_hat = torch.abs(Y_hat)
    mag_ref = torch.abs(Y_ref)
    return torch.mean(torch.abs(mag_hat - mag_ref))
