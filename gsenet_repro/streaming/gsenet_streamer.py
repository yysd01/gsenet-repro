"""Torch-based streaming wrapper for the minimal GSENet model."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from gsenet_repro.dsp import MODEL_STFT

try:  # pragma: no cover - import guarded for torch availability
    import torch
except ImportError as exc:  # pragma: no cover - import guarded for torch availability
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


class GSENetStreamer:
    """Streaming adapter for GSENet-like torch models.

    The streamer maintains a lookback buffer and an output FIFO to provide
    fixed-latency, chunked outputs aligned by ``algorithmic_delay``.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        sample_rate: int = 16000,
        chunk_size: int = 1600,
        lookback: Optional[int] = None,
        algorithmic_delay: Optional[int] = None,
    ) -> None:
        if torch is None:  # pragma: no cover - torch not installed
            raise ImportError(
                "GSENetStreamer requires torch to be installed."
            ) from _TORCH_IMPORT_ERROR

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.model = model
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.win_length = MODEL_STFT["win_length"]
        self.lookback = lookback if lookback is not None else 4096
        if self.lookback < self.win_length:
            raise ValueError("lookback must be >= MODEL_STFT win_length")
        self.algorithmic_delay = (
            algorithmic_delay if algorithmic_delay is not None else self.win_length
        )
        if self.algorithmic_delay < 0:
            raise ValueError("algorithmic_delay must be non-negative")

        self.reset()

    def reset(self) -> None:
        """Reset internal buffers."""
        self._buffer_y0: Optional[torch.Tensor] = None
        self._buffer_y1: Optional[torch.Tensor] = None
        self._buffer_y2: Optional[torch.Tensor] = None
        self._output_fifo: Deque[torch.Tensor] = deque()

    def _ensure_batch(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim == 1:
            return chunk.unsqueeze(0)
        if chunk.ndim == 2:
            return chunk
        raise ValueError("chunk must have shape (T,) or (B, T)")

    def _prepare_buffer(self, buffer: Optional[torch.Tensor], chunk: torch.Tensor) -> torch.Tensor:
        if buffer is None:
            history = torch.zeros((chunk.shape[0], 0), dtype=chunk.dtype, device=chunk.device)
        else:
            history = buffer
        return torch.cat([history, chunk], dim=-1)

    def process(
        self, y0_chunk: "torch.Tensor", y1_chunk: "torch.Tensor", y2_chunk: "torch.Tensor"
    ) -> "torch.Tensor":
        """Process a chunk and return the aligned output chunk."""
        if torch is None:  # pragma: no cover - torch not installed
            raise ImportError(
                "GSENetStreamer requires torch to be installed."
            ) from _TORCH_IMPORT_ERROR

        y0_chunk = self._ensure_batch(y0_chunk)
        y1_chunk = self._ensure_batch(y1_chunk)
        y2_chunk = self._ensure_batch(y2_chunk)
        if y0_chunk.shape != y1_chunk.shape or y0_chunk.shape != y2_chunk.shape:
            raise ValueError("y0_chunk, y1_chunk, y2_chunk must have the same shape")
        if y0_chunk.shape[1] != self.chunk_size:
            raise ValueError("chunk length must match chunk_size")

        y0_full = self._prepare_buffer(self._buffer_y0, y0_chunk)
        y1_full = self._prepare_buffer(self._buffer_y1, y1_chunk)
        y2_full = self._prepare_buffer(self._buffer_y2, y2_chunk)

        self.model.eval()
        with torch.no_grad():
            y_hat = self.model(y0_full, y1_full, y2_full)

        total_len = y_hat.shape[-1]
        end = max(0, total_len - self.algorithmic_delay)
        start = max(0, end - self.chunk_size)
        y_hat_chunk = y_hat[:, start:end]
        if y_hat_chunk.shape[1] < self.chunk_size:
            pad = torch.zeros(
                (y_hat_chunk.shape[0], self.chunk_size - y_hat_chunk.shape[1]),
                dtype=y_hat_chunk.dtype,
                device=y_hat_chunk.device,
            )
            y_hat_chunk = torch.cat([pad, y_hat_chunk], dim=-1)

        self._output_fifo.append(y_hat_chunk)

        available = torch.cat(tuple(self._output_fifo), dim=-1)
        if available.shape[1] >= self.chunk_size:
            out = available[:, : self.chunk_size]
            remaining = available[:, self.chunk_size :]
            self._output_fifo.clear()
            if remaining.numel() > 0:
                self._output_fifo.append(remaining)
        else:
            pad = torch.zeros(
                (available.shape[0], self.chunk_size - available.shape[1]),
                dtype=available.dtype,
                device=available.device,
            )
            out = torch.cat([pad, available], dim=-1)
            self._output_fifo.clear()

        self._buffer_y0 = y0_full[:, -self.lookback :]
        self._buffer_y1 = y1_full[:, -self.lookback :]
        self._buffer_y2 = y2_full[:, -self.lookback :]

        return out
