"""Frame-wise streaming MCWF implementation."""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import torch


class MCWFStreamer:
    """Stateful frame-wise MCWF streamer for multi-mic input."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 320,
        win_length: int = 320,
        hop_length: int = 160,
        causal_frames: int = 4,
        window: str = "hann",
        center: bool = False,
        eps: float = 1e-8,
        num_mics: int = 4,
    ) -> None:
        if window != "hann":
            raise ValueError("Only 'hann' window is supported")
        if center:
            raise ValueError("MCWFStreamer requires center=False for causal streaming")
        if win_length <= 0 or hop_length <= 0:
            raise ValueError("win_length and hop_length must be positive")
        if hop_length > win_length:
            raise ValueError("hop_length must be <= win_length")
        if causal_frames <= 0:
            raise ValueError("causal_frames must be positive")
        if num_mics < 2:
            raise ValueError("num_mics must be >= 2")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.causal_frames = causal_frames
        self.window = window
        self.center = center
        self.eps = eps
        self.num_mics = num_mics
        self.algorithmic_delay_samples = win_length - hop_length

        self._window_cpu = torch.hann_window(win_length, periodic=False)
        self.reset()

    def reset(self) -> None:
        """Reset internal buffers."""
        self._input_buffer: Optional[torch.Tensor] = None
        self._ola_buffer: Optional[torch.Tensor] = None
        self._ola_window_sum: Optional[torch.Tensor] = None
        self._output_fifo: Optional[torch.Tensor] = None
        self._power_history: Deque[torch.Tensor] = deque(maxlen=self.causal_frames)
        self._batch_size: Optional[int] = None

    def _ensure_batch(self, x_chunk: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x_chunk.ndim == 2:
            return x_chunk.unsqueeze(0), True
        if x_chunk.ndim == 3:
            return x_chunk, False
        raise ValueError("x_chunk must have shape (C, T) or (B, C, T)")

    def _get_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._window_cpu.to(device=device, dtype=dtype)

    def _init_output_fifo(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._output_fifo is not None:
            return
        if self.algorithmic_delay_samples > 0:
            self._output_fifo = torch.zeros(
                (batch_size, self.algorithmic_delay_samples), device=device, dtype=dtype
            )
        else:
            self._output_fifo = torch.zeros((batch_size, 0), device=device, dtype=dtype)

    def process(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """Process a chunk of audio and return the aligned output chunk."""
        x_chunk, squeeze = self._ensure_batch(x_chunk)
        if x_chunk.shape[1] != self.num_mics:
            raise ValueError(f"x_chunk must have {self.num_mics} microphone channels")

        batch_size, _, chunk_len = x_chunk.shape
        if self._batch_size is None:
            self._batch_size = batch_size
        elif batch_size != self._batch_size:
            raise ValueError("batch size must remain constant across calls")

        if self._input_buffer is None:
            self._input_buffer = torch.zeros(
                (batch_size, self.num_mics, 0), device=x_chunk.device, dtype=x_chunk.dtype
            )
        self._input_buffer = torch.cat([self._input_buffer, x_chunk], dim=-1)

        window = self._get_window(x_chunk.device, x_chunk.dtype)
        outputs = []

        if self._ola_buffer is None:
            self._ola_buffer = torch.zeros(
                (batch_size, self.win_length), device=x_chunk.device, dtype=x_chunk.dtype
            )
            self._ola_window_sum = torch.zeros_like(self._ola_buffer)

        while self._input_buffer.shape[-1] >= self.win_length:
            frame = self._input_buffer[:, :, : self.win_length]
            windowed = frame * window
            if self.win_length < self.n_fft:
                pad = torch.zeros(
                    (batch_size, self.num_mics, self.n_fft - self.win_length),
                    device=x_chunk.device,
                    dtype=x_chunk.dtype,
                )
                windowed = torch.cat([windowed, pad], dim=-1)
            spectrum = torch.fft.rfft(windowed, n=self.n_fft, dim=-1)
            spectrum = spectrum.transpose(1, 2)  # (B, F, C)

            power = spectrum.abs().pow(2)
            self._power_history.append(power)
            history = list(self._power_history)
            if len(history) < self.causal_frames:
                pad = torch.zeros_like(power)
                history = [pad] * (self.causal_frames - len(history)) + history
            power_stack = torch.stack(history, dim=0)
            signal_pow = power_stack.mean(dim=0)
            noise_pow = power_stack.var(dim=0, unbiased=False)
            gain = signal_pow / (signal_pow + noise_pow + self.eps)

            filtered = spectrum * gain
            y_frame = filtered.mean(dim=-1)
            time_frame = torch.fft.irfft(y_frame, n=self.n_fft, dim=-1)[..., : self.win_length]
            time_frame = time_frame * window

            self._ola_buffer[:, : self.win_length] += time_frame
            self._ola_window_sum[:, : self.win_length] += window.pow(2)

            hop_audio = self._ola_buffer[:, : self.hop_length]
            hop_sum = self._ola_window_sum[:, : self.hop_length]
            hop_out = torch.where(
                hop_sum > self.eps,
                hop_audio / hop_sum,
                torch.zeros_like(hop_audio),
            )
            outputs.append(hop_out)

            remaining = self.win_length - self.hop_length
            self._ola_buffer = torch.cat(
                [
                    self._ola_buffer[:, self.hop_length :],
                    torch.zeros(
                        (batch_size, self.hop_length),
                        device=x_chunk.device,
                        dtype=x_chunk.dtype,
                    ),
                ],
                dim=-1,
            )
            self._ola_window_sum = torch.cat(
                [
                    self._ola_window_sum[:, self.hop_length :],
                    torch.zeros(
                        (batch_size, self.hop_length),
                        device=x_chunk.device,
                        dtype=x_chunk.dtype,
                    ),
                ],
                dim=-1,
            )
            if self._ola_buffer.shape[1] != self.win_length:
                raise RuntimeError("OLA buffer size mismatch")
            if remaining != self.win_length - self.hop_length:
                raise RuntimeError("Unexpected hop/window configuration")

            self._input_buffer = self._input_buffer[:, :, self.hop_length :]

        if outputs:
            new_audio = torch.cat(outputs, dim=-1)
        else:
            new_audio = torch.zeros((batch_size, 0), device=x_chunk.device, dtype=x_chunk.dtype)

        self._init_output_fifo(batch_size, x_chunk.device, x_chunk.dtype)
        if self._output_fifo is None:
            raise RuntimeError("Output FIFO was not initialized")
        self._output_fifo = torch.cat([self._output_fifo, new_audio], dim=-1)

        if self._output_fifo.shape[1] >= chunk_len:
            out = self._output_fifo[:, :chunk_len]
            self._output_fifo = self._output_fifo[:, chunk_len:]
        else:
            pad = torch.zeros(
                (batch_size, chunk_len - self._output_fifo.shape[1]),
                device=x_chunk.device,
                dtype=x_chunk.dtype,
            )
            out = torch.cat([self._output_fifo, pad], dim=-1)
            self._output_fifo = torch.zeros((batch_size, 0), device=x_chunk.device, dtype=x_chunk.dtype)

        return out[0] if squeeze else out
