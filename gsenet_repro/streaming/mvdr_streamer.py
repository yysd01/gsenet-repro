"""Online stateful single-stream MVDR/LCMV streamer with target-likeness gated Rnn EMA."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gsenet_repro.dsp.beamform import diag_load, hermitian, lcmv_weights, mvdr_weights
from gsenet_repro.dsp.rtf_lib import get_d_from_lib, load_rtf_lib


class MVDRStreamer:
    """Stateful real-time beamformer for one stream.

    This class maintains recurrent covariance/OLA state and is intentionally single-stream.
    Call :meth:`process` with ``(C, T)`` or ``(1, C, T)`` only.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 320,
        win_length: int = 320,
        hop_length: int = 160,
        window: str = "hann",
        center: bool = False,
        num_mics: int = 4,
        ref_ch: int = 1,
        diag_load: float = 1e-2,
        rnn_alpha: float = 0.98,
        update_interval_frames: int = 4,
        coh_fmin_hz: float = 200.0,
        coh_fmax_hz: float = 5000.0,
        coh_t0: float = 0.15,
        coh_t1: float = 0.35,
        mode: str = "mvdr",
        lcmv_span_deg: int = 30,
        lcmv_k: int = 3,
    ) -> None:
        if window != "hann":
            raise ValueError("Only 'hann' window is supported")
        if center:
            raise ValueError("MVDRStreamer requires center=False for real-time processing")
        if mode not in {"mvdr", "lcmv"}:
            raise ValueError("mode must be 'mvdr' or 'lcmv'")
        if lcmv_k < 1:
            raise ValueError("lcmv_k must be >= 1")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = center
        self.num_mics = num_mics
        self.ref_ch = ref_ch
        self.diag_load_value = diag_load
        self.rnn_alpha = rnn_alpha
        self.update_interval_frames = max(1, int(update_interval_frames))
        self.coh_t0 = float(coh_t0)
        self.coh_t1 = float(coh_t1)
        self.mode = mode
        self.lcmv_span_deg = int(lcmv_span_deg)
        self.lcmv_k = int(lcmv_k)
        self.algorithmic_delay_samples = win_length - hop_length

        self._window_cpu = torch.hann_window(win_length, periodic=False)
        self._freqs_hz = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
        self._band_mask_cpu = (self._freqs_hz >= float(coh_fmin_hz)) & (
            self._freqs_hz <= float(coh_fmax_hz)
        )

        self.rtf_lib: Optional[dict[str, np.ndarray]] = None
        self.target_doa: Optional[int] = None
        self.last_score: Optional[torch.Tensor] = None
        self.last_target_like: Optional[torch.Tensor] = None
        self.last_coh: Optional[torch.Tensor] = None
        self.reset()

    @classmethod
    def from_config(cls, config: dict) -> "MVDRStreamer":
        data = config.get("data", {})
        stft = config.get("stft_model", {})
        stft_streaming = config.get("stft_streaming")
        if isinstance(stft_streaming, dict):
            keys = ("n_fft", "win_length", "hop_length")
            mismatched = [k for k in keys if k in stft_streaming and int(stft_streaming[k]) != int(stft.get(k, stft_streaming[k]))]
            if mismatched:
                mismatch_str = ", ".join(mismatched)
                raise ValueError(f"stft_streaming mismatch with stft_model for keys: {mismatch_str}")
        frontend = config.get("frontend", {})
        return cls(
            sample_rate=int(data.get("sample_rate", 16000)),
            n_fft=int(stft.get("n_fft", 320)),
            win_length=int(stft.get("win_length", 320)),
            hop_length=int(stft.get("hop_length", 160)),
            num_mics=int(data.get("num_mics", 4)),
            ref_ch=int(frontend.get("ref_ch", data.get("ref_mic_index", 1))),
            diag_load=float(frontend.get("diag_load_v", 1e-2)),
        )

    def reset(self) -> None:
        self._input_buffer: Optional[torch.Tensor] = None
        self._ola_buffer: Optional[torch.Tensor] = None
        self._ola_window_sum: Optional[torch.Tensor] = None
        self._output_fifo: Optional[torch.Tensor] = None
        self._batch_size: Optional[int] = None
        self._Rnn: Optional[torch.Tensor] = None
        self._d: Optional[torch.Tensor] = None
        self._w: Optional[torch.Tensor] = None
        self._frame_counter: int = 0

    def _validate_rtf_metadata(self, rtf_lib: dict[str, np.ndarray]) -> None:
        missing = tuple(rtf_lib.get("missing_metadata", ()))
        if missing:
            raise ValueError(
                f"rtf_lib is missing metadata {missing}; please rebuild with matching STFT settings. "
                "请用相同 STFT 参数重新 build rtf_lib"
            )
        if int(rtf_lib["num_mics"]) != self.num_mics:
            raise ValueError(
                f"rtf_lib num_mics={rtf_lib['num_mics']} != streamer num_mics={self.num_mics}; "
                "请用相同 STFT 参数重新 build rtf_lib"
            )
        lib_nfft = int(rtf_lib["n_fft"])
        if lib_nfft != self.n_fft and (lib_nfft // 2 + 1) != (self.n_fft // 2 + 1):
            raise ValueError(
                f"rtf_lib n_fft={lib_nfft} mismatch streamer n_fft={self.n_fft}; "
                "请用相同 STFT 参数重新 build rtf_lib"
            )
        if int(rtf_lib["sample_rate"]) != self.sample_rate:
            raise ValueError(
                f"rtf_lib sample_rate={rtf_lib['sample_rate']} mismatch streamer sample_rate={self.sample_rate}; "
                "请用相同 STFT 参数重新 build rtf_lib"
            )
        if str(rtf_lib["window"]) != self.window or bool(rtf_lib["center"]) != self.center:
            raise ValueError(
                f"rtf_lib window/center=({rtf_lib['window']},{rtf_lib['center']}) mismatch "
                f"streamer ({self.window},{self.center}); 请用相同 STFT 参数重新 build rtf_lib"
            )

    def load_rtf_lib(self, path: str | Path) -> None:
        lib = load_rtf_lib(path)
        self._validate_rtf_metadata(lib)
        self.rtf_lib = lib

    def set_target_doa(self, doa_deg: int) -> None:
        if self.rtf_lib is None:
            raise ValueError("RTF library is not loaded")
        d_np = get_d_from_lib(self.rtf_lib, doa_deg)
        self.target_doa = int(doa_deg)
        self._d = torch.from_numpy(d_np.astype(np.complex64))

    def _ensure_batch(self, x_chunk: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x_chunk.ndim == 2:
            return x_chunk.unsqueeze(0), True
        if x_chunk.ndim == 3:
            return x_chunk, False
        raise ValueError("x_chunk must have shape (C, T) or (B, C, T)")

    def _init_output_fifo(self, batch: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._output_fifo is None:
            self._output_fifo = torch.zeros(
                (batch, max(0, self.algorithmic_delay_samples)), device=device, dtype=dtype
            )

    def _ensure_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._Rnn is None:
            eye = torch.eye(self.num_mics, dtype=torch.complex64, device=device)
            self._Rnn = eye.unsqueeze(0).repeat(self.n_fft // 2 + 1, 1, 1)
        if self._d is None:
            d = torch.zeros(
                (self.n_fft // 2 + 1, self.num_mics), dtype=torch.complex64, device=device
            )
            d[:, self.ref_ch] = 1.0 + 0.0j
            self._d = d
        else:
            self._d = self._d.to(device=device)

        if self._ola_buffer is None:
            self._ola_buffer = torch.zeros((batch, self.win_length), device=device, dtype=dtype)
            self._ola_window_sum = torch.zeros_like(self._ola_buffer)

    def _constraint_angles(self, doa: int) -> list[int]:
        if self.lcmv_k <= 1:
            return [int(doa)]
        offsets = np.linspace(-self.lcmv_span_deg, self.lcmv_span_deg, self.lcmv_k)
        return [int(round(doa + float(off))) % 360 for off in offsets]

    def _compute_w(self) -> torch.Tensor:
        if self._Rnn is None or self._d is None:
            raise RuntimeError("state not initialized")
        if self.mode == "mvdr" or self.target_doa is None or self.rtf_lib is None:
            return mvdr_weights(self._Rnn, self._d, ref_ch=self.ref_ch)

        try:
            d_list = [
                torch.from_numpy(get_d_from_lib(self.rtf_lib, a).astype(np.complex64)).to(
                    self._Rnn.device
                )
                for a in self._constraint_angles(self.target_doa)
            ]
            D = torch.stack(d_list, dim=-1)
            g = torch.ones((D.shape[-1],), dtype=torch.float32, device=self._Rnn.device)
            return lcmv_weights(self._Rnn, D, g)
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"WARNING: LCMV constraints failed ({exc}); fallback to MVDR")
            return mvdr_weights(self._Rnn, self._d, ref_ch=self.ref_ch)

    def process(self, x_chunk: torch.Tensor, target_doa: int | None = None) -> torch.Tensor:
        if target_doa is not None:
            self.set_target_doa(target_doa)

        x_chunk, squeeze = self._ensure_batch(x_chunk)
        if x_chunk.shape[0] != 1:
            raise ValueError("MVDRStreamer is single-stream; pass (C,T) or (1,C,T)")
        if x_chunk.shape[1] != self.num_mics:
            raise ValueError(f"x_chunk must have {self.num_mics} channels")

        batch, _, chunk_len = x_chunk.shape
        if self._batch_size is None:
            self._batch_size = batch
        elif self._batch_size != batch:
            raise ValueError("batch size must remain constant")

        if self._input_buffer is None:
            self._input_buffer = torch.zeros(
                (batch, self.num_mics, 0), device=x_chunk.device, dtype=x_chunk.dtype
            )
        self._input_buffer = torch.cat([self._input_buffer, x_chunk], dim=-1)

        self._ensure_state(batch, x_chunk.device, x_chunk.dtype)
        self._init_output_fifo(batch, x_chunk.device, x_chunk.dtype)

        window = self._window_cpu.to(device=x_chunk.device, dtype=x_chunk.dtype)
        band_mask = self._band_mask_cpu.to(device=x_chunk.device)
        outputs = []

        while self._input_buffer.shape[-1] >= self.win_length:
            frame = self._input_buffer[:, :, : self.win_length]
            windowed = frame * window
            spectrum = torch.fft.rfft(windowed, n=self.n_fft, dim=-1).transpose(1, 2)  # (1,F,C)
            x_fc = spectrum.squeeze(0)

            d = self._d.to(device=x_chunk.device)
            proj = torch.sum(d.conj() * x_fc, dim=-1)
            d_norm = torch.sum(d.conj() * d, dim=-1).real
            x_norm = torch.sum(x_fc.conj() * x_fc, dim=-1).real
            coh = (proj.abs() ** 2) / (d_norm * x_norm + 1e-8)
            score = coh[band_mask].mean() if band_mask.any() else coh.mean()
            target_like = torch.clamp(
                (score - self.coh_t0) / max(self.coh_t1 - self.coh_t0, 1e-8), 0.0, 1.0
            )
            noise_gate = 1.0 - target_like

            xxh = x_fc.unsqueeze(-1) * x_fc.conj().unsqueeze(-2)
            self._Rnn = (
                self.rnn_alpha * self._Rnn
                + (1.0 - self.rnn_alpha) * noise_gate.to(torch.float32) * xxh
            )
            self._Rnn = torch.as_tensor(
                diag_load(hermitian(self._Rnn), self.diag_load_value),
                dtype=self._Rnn.dtype,
                device=self._Rnn.device,
            )

            if self._w is None or (self._frame_counter % self.update_interval_frames == 0):
                self._w = self._compute_w()
            self._frame_counter += 1

            y_frame_f = torch.einsum("fc,bfc->bf", self._w.conj(), spectrum)
            y_time = (
                torch.fft.irfft(y_frame_f, n=self.n_fft, dim=-1)[..., : self.win_length] * window
            )

            self._ola_buffer[:, : self.win_length] += y_time
            self._ola_window_sum[:, : self.win_length] += window.pow(2)

            hop_audio = self._ola_buffer[:, : self.hop_length]
            hop_sum = self._ola_window_sum[:, : self.hop_length]
            hop_out = torch.where(hop_sum > 1e-8, hop_audio / hop_sum, torch.zeros_like(hop_audio))
            outputs.append(hop_out)

            self._ola_buffer = torch.cat(
                [
                    self._ola_buffer[:, self.hop_length :],
                    torch.zeros(
                        (batch, self.hop_length), device=x_chunk.device, dtype=x_chunk.dtype
                    ),
                ],
                dim=-1,
            )
            self._ola_window_sum = torch.cat(
                [
                    self._ola_window_sum[:, self.hop_length :],
                    torch.zeros(
                        (batch, self.hop_length), device=x_chunk.device, dtype=x_chunk.dtype
                    ),
                ],
                dim=-1,
            )
            self._input_buffer = self._input_buffer[:, :, self.hop_length :]

            self.last_coh = coh.detach().cpu()
            self.last_score = score.detach().cpu()
            self.last_target_like = target_like.detach().cpu()

        new_audio = (
            torch.cat(outputs, dim=-1)
            if outputs
            else torch.zeros((batch, 0), device=x_chunk.device, dtype=x_chunk.dtype)
        )
        self._output_fifo = torch.cat([self._output_fifo, new_audio], dim=-1)

        if self._output_fifo.shape[1] >= chunk_len:
            out = self._output_fifo[:, :chunk_len]
            self._output_fifo = self._output_fifo[:, chunk_len:]
        else:
            pad = torch.zeros(
                (batch, chunk_len - self._output_fifo.shape[1]),
                device=x_chunk.device,
                dtype=x_chunk.dtype,
            )
            out = torch.cat([self._output_fifo, pad], dim=-1)
            self._output_fifo = torch.zeros((batch, 0), device=x_chunk.device, dtype=x_chunk.dtype)

        return out[0] if squeeze else out
