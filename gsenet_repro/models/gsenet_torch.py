"""Minimal offline GSENet-like torch model."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from gsenet_repro.dsp import MODEL_STFT
from gsenet_repro.dsp.torch_stft import torch_istft, torch_stft


class MinimalGSENet(nn.Module):
    """A compact causal Conv2D mask estimator operating on STFT features."""

    def __init__(self, stft_params: Dict[str, int] | None = None) -> None:
        super().__init__()
        self.stft_params = dict(stft_params) if stft_params is not None else dict(MODEL_STFT)

        self.conv1 = nn.Conv2d(4, 8, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3))
        self.conv_out = nn.Conv2d(8, 2, kernel_size=(1, 1))

    @staticmethod
    def _causal_conv(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        kernel_f, kernel_t = conv.kernel_size
        pad_f = kernel_f // 2
        pad_t = kernel_t - 1
        x = F.pad(x, (pad_t, 0, pad_f, pad_f))
        return conv(x)

    def forward(self, y0: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
        if y0.ndim != 2 or y1.ndim != 2:
            raise ValueError("y0 and y1 must have shape (B, T)")
        if y0.shape != y1.shape:
            raise ValueError("y0 and y1 must have the same shape")

        X0 = torch_stft(y0, **self.stft_params, center=False)
        X1 = torch_stft(y1, **self.stft_params, center=False)

        features = torch.stack([X0.real, X0.imag, X1.real, X1.imag], dim=1)

        x = F.relu(self._causal_conv(features, self.conv1))
        x = F.relu(self._causal_conv(x, self.conv2))
        x = self.conv_out(x)

        X_hat = torch.complex(x[:, 0], x[:, 1])
        return torch_istft(
            X_hat,
            **self.stft_params,
            center=False,
            length=y0.shape[1],
        )
