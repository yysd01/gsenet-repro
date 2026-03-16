from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from gsenet_repro.dsp import MODEL_STFT
from gsenet_repro.dsp.mvdr import MVDRConfig, make_mvdr_y0_stft
from gsenet_repro.dsp.stft import istft, stft
from gsenet_repro.dsp.trace_norm import make_trace_norm_y0_stft
from gsenet_repro.pipeline.gates import estimate_sector_gates, estimate_vad_gates

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    import torch
else:  # pragma: no cover
    torch = None

DEFAULT_MIC_POSITIONS = np.array(
    [[0.00, 0.00, 0.00], [0.04, 0.00, 0.00], [0.01, 0.035, 0.00], [-0.03, 0.01, 0.00]],
    dtype=np.float32,
)


def _match_frames(g: np.ndarray, frames: int, *, fill_value: float) -> np.ndarray:
    g = np.asarray(g, dtype=np.float32)
    if g.shape[0] == frames:
        return g
    if g.shape[0] > frames:
        return g[:frames]
    if g.shape[0] == 0:
        return np.full((frames,), fill_value, dtype=np.float32)
    return np.pad(g, (0, frames - g.shape[0]), mode="edge").astype(np.float32)


def _coherence_gate(X: np.ndarray, frontend_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    d = frontend_cfg.get("d")
    if d is None:
        raise ValueError("gate_mode='coherence' requires frontend_cfg['d'] (or rtf_lib/target_doa wiring)")
    d = np.asarray(d)
    if d.shape != (X.shape[0], X.shape[2]):
        raise ValueError("frontend_cfg['d'] must have shape (F,C)")
    proj = np.sum(np.conjugate(d)[:, None, :] * X, axis=-1)
    d_norm = np.sum(np.conjugate(d) * d, axis=-1).real[:, None]
    x_norm = np.sum(np.conjugate(X) * X, axis=-1).real
    coh = (np.abs(proj) ** 2) / (d_norm * x_norm + 1e-8)
    score = coh.mean(axis=0)
    t0 = float(frontend_cfg.get("coh_t0", 0.15))
    t1 = float(frontend_cfg.get("coh_t1", 0.35))
    target = np.clip((score - t0) / max(t1 - t0, 1e-8), 0.0, 1.0).astype(np.float32)
    noise = (1.0 - target).astype(np.float32)
    return target, noise


def _estimate_gates(
    x: np.ndarray,
    X: np.ndarray,
    frontend_cfg: dict[str, Any],
    stft_cfg: dict[str, int],
    data_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(frontend_cfg.get("gate_mode", "sector"))
    ref_ch = int(frontend_cfg.get("ref_ch", data_cfg.get("ref_mic_index", 1)))
    if mode == "sector":
        mic_positions = np.asarray(frontend_cfg.get("mic_positions", data_cfg.get("mic_positions", DEFAULT_MIC_POSITIONS)), dtype=np.float32)
        target, noise = estimate_sector_gates(
            x,
            sample_rate=int(data_cfg.get("sample_rate", 16000)),
            hop_length=int(stft_cfg["hop_length"]),
            win_length=int(stft_cfg["win_length"]),
            mic_positions=mic_positions,
            ref_ch=ref_ch,
            mic_pairs=frontend_cfg.get("mic_pairs"),
            sector_half_angle_deg=float(frontend_cfg.get("sector_half_angle_deg", 60.0)),
            theta_s=float(frontend_cfg.get("theta_s", 0.6)),
            theta_t=float(frontend_cfg.get("theta_t", 0.6)),
            theta_i=float(frontend_cfg.get("theta_i", 0.4)),
            theta_n=float(frontend_cfg.get("theta_n", 0.3)),
            beta_speech_interf=float(frontend_cfg.get("beta_speech_interf", 0.5)),
        )
        return _match_frames(target, X.shape[1], fill_value=1.0), _match_frames(noise, X.shape[1], fill_value=1.0)
    if mode == "vad":
        target, noise = estimate_vad_gates(
            x,
            hop_length=int(stft_cfg["hop_length"]),
            win_length=int(stft_cfg["win_length"]),
            ref_ch=ref_ch,
            vad_db_thresh=float(frontend_cfg.get("vad_db_thresh", -35.0)),
            vad_smooth=float(frontend_cfg.get("vad_smooth", 6.0)),
        )
        return _match_frames(target, X.shape[1], fill_value=1.0), _match_frames(noise, X.shape[1], fill_value=1.0)
    if mode == "coherence":
        return _coherence_gate(X, frontend_cfg)
    raise ValueError(f"Unknown gate_mode: {mode}")


def make_y0_from_frontend(
    x_mics: np.ndarray | "torch.Tensor",
    frontend_cfg: dict[str, Any] | None,
    stft_cfg: dict[str, int] | None,
    data_cfg: dict[str, Any] | None,
) -> np.ndarray | "torch.Tensor":
    input_is_torch = torch is not None and torch.is_tensor(x_mics)
    x_np = x_mics.detach().cpu().numpy() if input_is_torch else np.asarray(x_mics)
    squeeze = x_np.ndim == 2
    if squeeze:
        x_np = x_np[None, ...]
    if x_np.ndim != 3:
        raise ValueError("x_mics must be (C,T) or (B,C,T)")

    stft_params = dict(MODEL_STFT if stft_cfg is None else stft_cfg)
    cfg = dict(frontend_cfg or {})
    data = dict(data_cfg or {})
    frontend_type = str(cfg.get("type", "none"))
    ref_ch = int(cfg.get("ref_ch", data.get("ref_mic_index", 1)))

    y0_list: list[np.ndarray] = []
    for b in range(x_np.shape[0]):
        x = x_np[b]
        if frontend_type == "none":
            y0_list.append(x[ref_ch].astype(np.float32))
            continue
        X = np.stack(
            [
                stft(x[ch], n_fft=int(stft_params["n_fft"]), win_length=int(stft_params["win_length"]), hop_length=int(stft_params["hop_length"]), center=False)
                for ch in range(x.shape[0])
            ],
            axis=-1,
        )
        target_gate, noise_gate = _estimate_gates(x, X, cfg, stft_params, data)

        if frontend_type == "mvdr":
            y_stft = make_mvdr_y0_stft(
                X,
                target_gate,
                noise_gate,
                cfg=MVDRConfig(ref_ch=ref_ch, diag_load=float(cfg.get("diag_load_v", 1e-2))),
            )
        elif frontend_type == "trace_norm":
            y_stft = make_trace_norm_y0_stft(
                X,
                noise_gate,
                ref_ch=ref_ch,
                alpha_y=float(cfg.get("alpha_y", 0.92)),
                alpha_v=float(cfg.get("alpha_v", 0.98)),
                diag_load_v=float(cfg.get("diag_load_v", 1e-2)),
                diag_load_x=float(cfg.get("diag_load_x", 1e-3)),
                eps_trace=float(cfg.get("eps_trace", 1e-6)),
                psd_project=bool(cfg.get("psd_project", True)),
            )
        else:
            raise ValueError(f"Unknown frontend.type: {frontend_type}")
        y0 = istft(
            y_stft,
            n_fft=int(stft_params["n_fft"]),
            win_length=int(stft_params["win_length"]),
            hop_length=int(stft_params["hop_length"]),
            center=False,
            length=x.shape[-1],
        ).astype(np.float32)
        y0_list.append(y0)

    y0 = np.stack(y0_list, axis=0)
    if input_is_torch:
        out = torch.tensor(y0, device=x_mics.device, dtype=x_mics.dtype)
        return out[0] if squeeze else out
    return y0[0] if squeeze else y0
