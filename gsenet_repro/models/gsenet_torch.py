"""Minimal offline GSENet-like torch model."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from gsenet_repro.dsp import MODEL_STFT, mcwf_windowed_power
from gsenet_repro.dsp.torch_stft import torch_istft, torch_stft


class MinimalGSENet(nn.Module):
    """A compact causal Conv2D mask estimator operating on STFT features."""

    def __init__(
        self,
        stft_params: Dict[str, int] | None = None,
        mcwf_window_len: int = 4,
        mcwf_gain_min: float = 0.05,
        mcwf_gain_max: float = 1.0,
        mcwf_noise_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.stft_params = dict(stft_params) if stft_params is not None else dict(MODEL_STFT)
        self.mcwf_window_len = mcwf_window_len
        self.mcwf_gain_min = mcwf_gain_min
        self.mcwf_gain_max = mcwf_gain_max
        self.mcwf_noise_scale = mcwf_noise_scale

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

    def _apply_mcwf(
        self,
        X0: torch.Tensor,
        X1: torch.Tensor,
        X2: torch.Tensor,
        noise_level: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mcwf_windowed_power is None:
            raise RuntimeError("mcwf_windowed_power requires torch to be installed.")
        input_stft = torch.stack([X0, X1, X2], dim=-1)
        power = mcwf_windowed_power(input_stft, window_len=self.mcwf_window_len)

        avg_power = power.mean(dim=-1)
        signal_pow = avg_power.mean(dim=2, keepdim=True)
        noise_pow = avg_power.var(dim=2, keepdim=True)
        if noise_level is not None:
            if noise_level.ndim == 1:
                noise_level = noise_level[:, None, None]
            noise_pow = noise_pow * noise_level
        noise_pow = noise_pow * self.mcwf_noise_scale
        gain = signal_pow / (signal_pow + noise_pow + 1e-8)
        gain = gain.clamp(self.mcwf_gain_min, self.mcwf_gain_max)

        X0_enh = X0 * gain
        X1_enh = X1 * gain
        return X0_enh, X1_enh

    def forward(
        self,
        y0: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
        noise_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y0.ndim != 2 or y1.ndim != 2 or y2.ndim != 2:
            raise ValueError("y0, y1, y2 must have shape (B, T)")
        if y0.shape != y1.shape or y0.shape != y2.shape:
            raise ValueError("y0, y1, y2 must have the same shape")

        X0 = torch_stft(y0, **self.stft_params, center=False)
        X1 = torch_stft(y1, **self.stft_params, center=False)
        X2 = torch_stft(y2, **self.stft_params, center=False)

        X0, X1 = self._apply_mcwf(X0, X1, X2, noise_level=noise_level)

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
