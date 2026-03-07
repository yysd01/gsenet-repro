"""Paper-scale GSENet implementation aligned with arXiv:2303.07486."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from gsenet_repro.dsp import MODEL_STFT
from gsenet_repro.dsp.torch_stft import torch_istft, torch_stft

PAPER_ENCODER_BLOCKS: tuple[dict[str, object], ...] = (
    {"cin": 16, "cout": 32, "stime": 1, "sfreq": 2, "dtime": False},
    {"cin": 32, "cout": 48, "stime": 2, "sfreq": 2, "dtime": False},
    {"cin": 48, "cout": 48, "stime": 1, "sfreq": 2, "dtime": True},
    {"cin": 48, "cout": 96, "stime": 1, "sfreq": 2, "dtime": True},
    {"cin": 96, "cout": 96, "stime": 1, "sfreq": 2, "dtime": True},
)

PAPER_DECODER_BLOCKS: tuple[dict[str, object], ...] = (
    {"cin": 96, "cout": 96, "stime": 1, "sfreq": 2, "dtime": True},
    {"cin": 96, "cout": 48, "stime": 1, "sfreq": 2, "dtime": True},
    {"cin": 48, "cout": 48, "stime": 1, "sfreq": 2, "dtime": True},
    {"cin": 48, "cout": 32, "stime": 2, "sfreq": 2, "dtime": False},
    {"cin": 32, "cout": 16, "stime": 1, "sfreq": 2, "dtime": False},
)


@dataclass(frozen=True)
class BlockConfig:
    cin: int
    cout: int
    stime: int
    sfreq: int
    dtime: bool


def _to_block_configs(
    blocks: Optional[Iterable[Dict[str, object]]], fallback: Iterable[dict[str, object]]
) -> List[BlockConfig]:
    selected = blocks if blocks is not None else fallback
    configs: List[BlockConfig] = []
    for block in selected:
        configs.append(
            BlockConfig(
                cin=int(block["cin"]),
                cout=int(block["cout"]),
                stime=int(block["stime"]),
                sfreq=int(block["sfreq"]),
                dtime=bool(block["dtime"]),
            )
        )
    return configs


def _make_stft_params(stft_params: Dict[str, object] | None) -> Dict[str, object]:
    params = dict(stft_params) if stft_params is not None else dict(MODEL_STFT)
    params.setdefault("window", "hann")
    params.setdefault("center", False)
    return params


class CausalConv2d(nn.Module):
    """Conv2D with left-only padding on the time axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_f, kernel_t = self.kernel_size
        dilation_f, dilation_t = self.dilation
        pad_f = (kernel_f - 1) * dilation_f // 2
        pad_t = (kernel_t - 1) * dilation_t
        if pad_f or pad_t:
            x = F.pad(x, (pad_t, 0, pad_f, pad_f))
        return self.conv(x)


class TimeDilationBlock(nn.Module):
    """Residual time-dilated block to expand temporal receptive field."""

    def __init__(self, channels: int, slope: float) -> None:
        super().__init__()
        self.conv1 = CausalConv2d(
            channels,
            channels,
            kernel_size=(3, 3),
            dilation=(1, 3),
        )
        self.conv2 = CausalConv2d(
            channels,
            channels,
            kernel_size=(3, 3),
            dilation=(1, 3),
        )
        self.activation = nn.LeakyReLU(slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class EncoderBlock(nn.Module):
    def __init__(self, config: BlockConfig, slope: float) -> None:
        super().__init__()
        self.conv1 = CausalConv2d(config.cin, config.cin, kernel_size=(3, 3))
        self.conv2 = CausalConv2d(
            config.cin,
            config.cout,
            kernel_size=(3, 3),
            dilation=(1, 9),
        )
        self.time_dilation = TimeDilationBlock(config.cout, slope) if config.dtime else None
        self.downsample = CausalConv2d(
            config.cout,
            config.cout,
            kernel_size=(3, 3),
            stride=(config.sfreq, config.stime),
        )
        self.activation = nn.LeakyReLU(slope)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        if self.time_dilation is not None:
            x = self.time_dilation(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, config: BlockConfig, slope: float) -> None:
        super().__init__()
        self.scale = (config.sfreq, config.stime)
        self.conv1 = CausalConv2d(in_channels, config.cin, kernel_size=(3, 3))
        self.conv2 = CausalConv2d(
            config.cin,
            config.cout,
            kernel_size=(3, 3),
            dilation=(1, 9),
        )
        self.time_dilation = TimeDilationBlock(config.cout, slope) if config.dtime else None
        self.activation = nn.LeakyReLU(slope)

    @staticmethod
    def _match_shape(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_f, target_t = target.shape[-2], target.shape[-1]
        current_f, current_t = x.shape[-2], x.shape[-1]
        if current_f > target_f:
            x = x[:, :, :target_f, :]
        elif current_f < target_f:
            pad_total = target_f - current_f
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            x = F.pad(x, (0, 0, pad_top, pad_bottom))
        if current_t > target_t:
            x = x[:, :, :, :target_t]
        elif current_t < target_t:
            x = F.pad(x, (0, target_t - current_t, 0, 0))
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        x = self._match_shape(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        if self.time_dilation is not None:
            x = self.time_dilation(x)
        return x


class GSENetPaperScale(nn.Module):
    """U-Net style GSENet aligned with paper-scale hyperparameters."""

    def __init__(
        self,
        stft_params: Dict[str, object] | None = None,
        leaky_relu_slope: float = 0.3,
        encoder_blocks: Optional[Iterable[Dict[str, object]]] = None,
        decoder_blocks: Optional[Iterable[Dict[str, object]]] = None,
        stem_channels: int = 16,
        head_channels: int = 2,
        remove_dc: bool = False,
    ) -> None:
        super().__init__()
        self.stft_params = _make_stft_params(stft_params)
        self.remove_dc = remove_dc
        self.leaky_relu_slope = leaky_relu_slope

        self.stem = CausalConv2d(4, stem_channels, kernel_size=(7, 7))
        self.stem_activation = nn.LeakyReLU(leaky_relu_slope)

        self.encoder_configs = _to_block_configs(encoder_blocks, PAPER_ENCODER_BLOCKS)
        self.decoder_configs = _to_block_configs(decoder_blocks, PAPER_DECODER_BLOCKS)

        self.encoders = nn.ModuleList(
            [EncoderBlock(config, leaky_relu_slope) for config in self.encoder_configs]
        )

        decoder_in_channels = []
        prev_channels = self.encoder_configs[-1].cout
        for idx, config in enumerate(self.decoder_configs):
            skip_channels = self.encoder_configs[-(idx + 1)].cout
            decoder_in_channels.append(prev_channels + skip_channels)
            prev_channels = config.cout
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(in_ch, config, leaky_relu_slope)
                for in_ch, config in zip(decoder_in_channels, self.decoder_configs)
            ]
        )

        self.head = CausalConv2d(
            self.decoder_configs[-1].cout,
            head_channels,
            kernel_size=(7, 7),
        )

    def _remove_dc(self, y: torch.Tensor) -> torch.Tensor:
        return y - y.mean(dim=-1, keepdim=True)

    def forward(
        self,
        y0: torch.Tensor,
        y1: torch.Tensor,
        noise_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = noise_level
        if y0.ndim != 2 or y1.ndim != 2:
            raise ValueError("y0 and y1 must have shape (B, T)")
        if y0.shape != y1.shape:
            raise ValueError("y0 and y1 must have the same shape")

        if self.remove_dc:
            y0 = self._remove_dc(y0)
            y1 = self._remove_dc(y1)

        X0 = torch_stft(y0, **self.stft_params)
        X1 = torch_stft(y1, **self.stft_params)

        features = torch.stack([X1.real, X1.imag, X0.real, X0.imag], dim=1)

        x = self.stem_activation(self.stem(features))
        skips = []
        for block in self.encoders:
            x, skip = block(x)
            skips.append(skip)

        for block, skip in zip(self.decoders, reversed(skips)):
            x = block(x, skip)

        x = self.head(x)
        X_hat = torch.complex(x[:, 0], x[:, 1])
        return torch_istft(
            X_hat,
            **self.stft_params,
            length=y1.shape[1],
        )
