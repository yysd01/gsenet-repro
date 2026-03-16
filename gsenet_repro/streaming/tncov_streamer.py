"""Online trace-normalized covariance beamformer streamer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gsenet_repro.dsp.rtf_lib import get_d_from_lib, load_rtf_lib
from gsenet_repro.pipeline.frontend import DEFAULT_MIC_POSITIONS
from gsenet_repro.pipeline.gates import estimate_sector_gates


class TraceNormCovStreamer:
    """Stateful real-time trace-normalized covariance beamformer for one stream."""

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
        alpha_y: float = 0.92,
        alpha_v: float = 0.98,
        diag_load_v: float = 1e-2,
        diag_load_x: float = 1e-3,
        eps_trace: float = 1e-6,
        psd_project: bool = False,
        gate_mode: str = "vad",
        sector_half_angle_deg: float = 60.0,
        mic_pairs: list[list[int]] | None = None,
        mic_positions: np.ndarray | None = None,
        coh_fmin_hz: float = 200.0,
        coh_fmax_hz: float = 5000.0,
        coh_t0: float = 0.15,
        coh_t1: float = 0.35,
        vad_db_thresh: float = -35.0,
        vad_smooth: float = 6.0,
        log_interval_frames: int = 100,
    ) -> None:
        if window != "hann":
            raise ValueError("Only 'hann' window is supported")
        if center:
            raise ValueError("TraceNormCovStreamer requires center=False for real-time processing")
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.window = window
        self.center = center
        self.num_mics = int(num_mics)
        self.ref_ch = int(ref_ch)
        self.alpha_y = float(alpha_y)
        self.alpha_v = float(alpha_v)
        self.diag_load_v = float(diag_load_v)
        self.diag_load_x = float(diag_load_x)
        self.eps_trace = float(eps_trace)
        self.psd_project = bool(psd_project)
        self.gate_mode = str(gate_mode)
        self.sector_half_angle_deg = float(sector_half_angle_deg)
        self.mic_pairs = mic_pairs
        self.mic_positions = DEFAULT_MIC_POSITIONS if mic_positions is None else np.asarray(mic_positions, dtype=np.float32)
        self.coh_t0 = float(coh_t0)
        self.coh_t1 = float(coh_t1)
        self.vad_db_thresh = float(vad_db_thresh)
        self.vad_smooth = float(max(vad_smooth, 1e-6))
        self.log_interval_frames = int(max(1, log_interval_frames))
        self.algorithmic_delay_samples = self.win_length - self.hop_length

        self._window_cpu = torch.hann_window(self.win_length, periodic=False)
        self._freqs_hz = torch.linspace(0.0, self.sample_rate / 2.0, self.n_fft // 2 + 1)
        self._band_mask_cpu = (self._freqs_hz >= float(coh_fmin_hz)) & (
            self._freqs_hz <= float(coh_fmax_hz)
        )

        self.rtf_lib: Optional[dict[str, np.ndarray]] = None
        self.target_doa: Optional[int] = None
        self.last_score: Optional[torch.Tensor] = None
        self.last_target_like: Optional[torch.Tensor] = None
        self.last_noise_gate: Optional[torch.Tensor] = None
        self.last_trace_den: Optional[torch.Tensor] = None
        self.last_fallback_ratio: float = 0.0
        self.reset()

    @classmethod
    def from_config(cls, config: dict) -> "TraceNormCovStreamer":
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
        ref_ch = int(frontend.get("ref_ch", data.get("ref_mic_index", 1)))
        return cls(
            sample_rate=int(data.get("sample_rate", 16000)),
            n_fft=int(stft.get("n_fft", 320)),
            win_length=int(stft.get("win_length", 320)),
            hop_length=int(stft.get("hop_length", 160)),
            num_mics=int(data.get("num_mics", 4)),
            ref_ch=ref_ch,
            alpha_y=float(frontend.get("alpha_y", 0.92)),
            alpha_v=float(frontend.get("alpha_v", 0.98)),
            diag_load_v=float(frontend.get("diag_load_v", 1e-2)),
            diag_load_x=float(frontend.get("diag_load_x", 1e-3)),
            eps_trace=float(frontend.get("eps_trace", 1e-6)),
            psd_project=bool(frontend.get("psd_project", False)),
            gate_mode=str(frontend.get("gate_mode", "vad")),
            sector_half_angle_deg=float(frontend.get("sector_half_angle_deg", 60.0)),
            mic_pairs=frontend.get("mic_pairs"),
            mic_positions=np.asarray(frontend.get("mic_positions", DEFAULT_MIC_POSITIONS), dtype=np.float32),
        )

    def reset(self) -> None:
        self._input_buffer: Optional[torch.Tensor] = None
        self._ola_buffer: Optional[torch.Tensor] = None
        self._ola_window_sum: Optional[torch.Tensor] = None
        self._output_fifo: Optional[torch.Tensor] = None
        self._batch_size: Optional[int] = None
        self._phi_y: Optional[torch.Tensor] = None
        self._phi_v: Optional[torch.Tensor] = None
        self._d: Optional[torch.Tensor] = None
        self._frame_counter: int = 0

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
        f_bins = self.n_fft // 2 + 1
        if self._phi_y is None:
            eye = torch.eye(self.num_mics, dtype=torch.complex64, device=device)
            self._phi_y = eye.unsqueeze(0).repeat(f_bins, 1, 1)
            self._phi_v = eye.unsqueeze(0).repeat(f_bins, 1, 1)
        if self._d is None:
            d = torch.zeros((f_bins, self.num_mics), dtype=torch.complex64, device=device)
            d[:, self.ref_ch] = 1.0 + 0.0j
            self._d = d
        else:
            self._d = self._d.to(device=device)

        if self._ola_buffer is None:
            self._ola_buffer = torch.zeros((batch, self.win_length), device=device, dtype=dtype)
            self._ola_window_sum = torch.zeros_like(self._ola_buffer)

    def _validate_rtf_metadata(self, rtf_lib: dict[str, np.ndarray]) -> None:
        missing = tuple(rtf_lib.get("missing_metadata", ()))
        if missing:
            raise ValueError(f"rtf_lib is missing metadata {missing}")
        if int(rtf_lib["num_mics"]) != self.num_mics:
            raise ValueError("rtf_lib num_mics mismatch")
        if int(rtf_lib["sample_rate"]) != self.sample_rate:
            raise ValueError("rtf_lib sample_rate mismatch")

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

    @staticmethod
    def _hermitian(m: torch.Tensor) -> torch.Tensor:
        return 0.5 * (m + m.conj().transpose(-1, -2))

    def _diag_load(self, m: torch.Tensor, load: float) -> torch.Tensor:
        c = m.shape[-1]
        tr = m.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real / float(c)
        eye = torch.eye(c, dtype=m.dtype, device=m.device).unsqueeze(0)
        return m + (float(load) * tr).unsqueeze(-1).unsqueeze(-1) * eye

    def _psd_project(self, m: torch.Tensor) -> torch.Tensor:
        evals, evecs = torch.linalg.eigh(self._hermitian(m))
        evals = torch.clamp(evals.real, min=0.0)
        return torch.matmul(evecs * evals.unsqueeze(-2), evecs.conj().transpose(-1, -2))

    def _noise_gate(self, x_fc: torch.Tensor, band_mask: torch.Tensor, frame_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gate_mode == "coherence":
            if self.rtf_lib is None or self.target_doa is None or self._d is None:
                raise ValueError("gate_mode='coherence' requires loaded rtf_lib and target_doa")
            d = self._d.to(device=x_fc.device)
            proj = torch.sum(d.conj() * x_fc, dim=-1)
            d_norm = torch.sum(d.conj() * d, dim=-1).real
            x_norm = torch.sum(x_fc.conj() * x_fc, dim=-1).real
            coh = (proj.abs() ** 2) / (d_norm * x_norm + 1e-8)
            score = coh[band_mask].mean() if band_mask.any() else coh.mean()
            target_like = torch.clamp((score - self.coh_t0) / max(self.coh_t1 - self.coh_t0, 1e-8), 0.0, 1.0)
            return 1.0 - target_like, target_like
        if self.gate_mode == "sector":
            x_np = frame_time.squeeze(0).detach().cpu().numpy()
            target, noise = estimate_sector_gates(
                x_np,
                sample_rate=self.sample_rate,
                hop_length=self.win_length,
                win_length=self.win_length,
                mic_positions=self.mic_positions,
                ref_ch=self.ref_ch,
                mic_pairs=self.mic_pairs,
                sector_half_angle_deg=self.sector_half_angle_deg,
            )
            target_like = torch.tensor(float(target[0]), device=x_fc.device, dtype=torch.float32)
            noise_gate = torch.tensor(float(noise[0]), device=x_fc.device, dtype=torch.float32)
            return noise_gate, target_like

        frame_power = torch.mean(torch.abs(x_fc) ** 2)
        frame_db = 10.0 * torch.log10(frame_power + 1e-12)
        z = (self.vad_db_thresh - frame_db) / self.vad_smooth
        noise_gate = torch.sigmoid(z)
        return noise_gate, 1.0 - noise_gate

    def _compute_weights(self, phi_v: torch.Tensor, phi_x: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(self.num_mics, dtype=phi_v.dtype, device=phi_v.device).unsqueeze(0)
        try:
            a = torch.linalg.solve(phi_v, phi_x)
        except RuntimeError:
            a = torch.linalg.solve(phi_v + 1e-3 * eye, phi_x)
        tau = a.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        den_real = tau.real
        den = torch.clamp(den_real, min=self.eps_trace)
        w = a[..., self.ref_ch] / den.unsqueeze(-1)

        invalid = (~torch.isfinite(w).all(dim=-1)) | (~torch.isfinite(den_real)) | (
            den_real <= self.eps_trace
        )
        if torch.any(invalid):
            w_fb = torch.zeros_like(w)
            w_fb[:, self.ref_ch] = 1.0 + 0.0j
            w = torch.where(invalid.unsqueeze(-1), w_fb, w)

        self.last_trace_den = den_real.detach().cpu()
        self.last_fallback_ratio = float(invalid.to(torch.float32).mean().item())
        return w

    def process(self, x_chunk: torch.Tensor, target_doa: int | None = None) -> torch.Tensor:
        if target_doa is not None:
            self.set_target_doa(target_doa)

        x_chunk, squeeze = self._ensure_batch(x_chunk)
        if x_chunk.shape[0] != 1:
            raise ValueError("TraceNormCovStreamer is single-stream; pass (C,T) or (1,C,T)")
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

            noise_gate, target_like = self._noise_gate(x_fc, band_mask, frame)
            xxh = x_fc.unsqueeze(-1) * x_fc.conj().unsqueeze(-2)

            self._phi_y = self.alpha_y * self._phi_y + (1.0 - self.alpha_y) * xxh
            self._phi_v = self.alpha_v * self._phi_v + (1.0 - self.alpha_v) * noise_gate * xxh

            phi_y = self._diag_load(self._hermitian(self._phi_y), self.diag_load_x)
            phi_v = self._diag_load(self._hermitian(self._phi_v), self.diag_load_v)
            phi_x = self._hermitian(phi_y - phi_v)
            if self.psd_project:
                phi_x = self._psd_project(phi_x)
            phi_x = self._diag_load(phi_x, self.diag_load_x)

            w = self._compute_weights(phi_v, phi_x)

            y_frame_f = torch.einsum("fc,bfc->bf", w.conj(), spectrum)
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
                    torch.zeros((batch, self.hop_length), device=x_chunk.device, dtype=x_chunk.dtype),
                ],
                dim=-1,
            )
            self._ola_window_sum = torch.cat(
                [
                    self._ola_window_sum[:, self.hop_length :],
                    torch.zeros((batch, self.hop_length), device=x_chunk.device, dtype=x_chunk.dtype),
                ],
                dim=-1,
            )
            self._input_buffer = self._input_buffer[:, :, self.hop_length :]

            self.last_target_like = target_like.detach().cpu()
            self.last_noise_gate = noise_gate.detach().cpu()

            self._frame_counter += 1
            if self._frame_counter % self.log_interval_frames == 0:
                trace_stats = self.last_trace_den
                trace_mean = float(trace_stats.mean().item()) if trace_stats is not None else float("nan")
                print(
                    "[TraceNormCovStreamer] "
                    f"frame={self._frame_counter} "
                    f"target_like={float(target_like):.3f} "
                    f"noise_gate={float(noise_gate):.3f} "
                    f"trace_mean={trace_mean:.3e} "
                    f"fallback_ratio={self.last_fallback_ratio:.3f}"
                )

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
