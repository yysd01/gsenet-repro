"""Minimal offline GSENet implementation in PyTorch."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from gsenet_repro.dsp.stft import MODEL_STFT
from gsenet_repro.dsp.torch_stft import torch_istft, torch_stft


@dataclass(frozen=True)
class GSENetConfig:
    n_fft: int = MODEL_STFT["n_fft"]
    win_length: int = MODEL_STFT["win_length"]
    hop_length: int = MODEL_STFT["hop_length"]


class CausalConv2d(nn.Module):
    """Causal 2D convolution with left padding on time axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kf, kt = self.kernel_size
        pad_freq = kf // 2
        pad_time = kt - 1
        x = F.pad(x, (pad_time, 0, pad_freq, pad_freq))
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.conv = CausalConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class GSENetTorch(nn.Module):
    """Minimal offline GSENet-like network with causal time convolutions."""

    def __init__(self, config: GSENetConfig | None = None) -> None:
        super().__init__()
        self.config = config or GSENetConfig()

        self.enc1 = ConvBlock(4, 16)
        self.enc2 = ConvBlock(16, 32, stride=(1, 2))
        self.bottleneck = ConvBlock(32, 32)
        self.dec1 = ConvBlock(32 + 16, 16)
        self.out = CausalConv2d(16, 2, kernel_size=(1, 1))

    def forward(self, y0: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
        if y0.shape != y1.shape:
            raise ValueError("y0 and y1 must have the same shape")
        if y0.dim() != 2:
            raise ValueError("y0 and y1 must have shape (B, T)")

        x0 = torch_stft(
            y0,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            center=False,
        )
        x1 = torch_stft(
            y1,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            center=False,
        )

        feats = torch.stack(
            [x0.real, x0.imag, x1.real, x1.imag],
            dim=1,
        )
        feats = feats.float()

        enc1 = self.enc1(feats)
        enc2 = self.enc2(enc1)
        bottleneck = self.bottleneck(enc2)
        up = F.interpolate(bottleneck, size=(bottleneck.shape[-2], enc1.shape[-1]), mode="nearest")
        merged = torch.cat([up, enc1], dim=1)
        dec1 = self.dec1(merged)
        out = self.out(dec1)

        x_hat = torch.complex(out[:, 0], out[:, 1])
        y_hat = torch_istft(
            x_hat,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            length=y0.shape[-1],
            center=False,
        )
        return y_hat
