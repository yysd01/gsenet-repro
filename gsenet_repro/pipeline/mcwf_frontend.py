from __future__ import annotations

import importlib.util
from typing import Dict

import numpy as np

from gsenet_repro.dsp import MODEL_STFT
from gsenet_repro.dsp.mvdr import MVDRConfig, make_mvdr_y0_stft
from gsenet_repro.dsp.stft import istft, stft

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    import torch
else:  # pragma: no cover
    torch = None


DEFAULT_MIC_POSITIONS = np.array(
    [[0.00, 0.00, 0.00], [0.04, 0.00, 0.00], [0.01, 0.035, 0.00], [-0.03, 0.01, 0.00]],
    dtype=np.float32,
)


def _frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if x.shape[-1] < frame_length:
        pad = frame_length - x.shape[-1]
        x = np.pad(x, (0, pad))
    n_frames = 1 + (x.shape[-1] - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return x[idx]


def _gcc_phat_tau(frame_a: np.ndarray, frame_b: np.ndarray, fs: int) -> float:
    n = int(2 ** np.ceil(np.log2(len(frame_a) * 2 - 1)))
    A = np.fft.rfft(frame_a, n=n)
    B = np.fft.rfft(frame_b, n=n)
    G = A * np.conj(B)
    G /= np.maximum(np.abs(G), 1e-8)
    cc = np.fft.irfft(G, n=n)
    max_shift = n // 2
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return float(shift / fs)


def gates_from_probs(
    p_speech: np.ndarray,
    p_tar: np.ndarray,
    theta_s: float,
    theta_t: float,
    theta_i: float,
    theta_n: float,
    beta_speech_interf: float,
) -> tuple[np.ndarray, np.ndarray]:
    target_gate = ((p_speech > theta_s) & (p_tar > theta_t)).astype(np.float32)
    is_noise = ((p_tar < theta_i) | (p_speech < theta_n)).astype(np.float32)
    noise_beta = np.where(
        target_gate > 0.5,
        0.0,
        np.where(is_noise > 0.5, 1.0, beta_speech_interf),
    ).astype(np.float32)
    return target_gate, noise_beta


def _estimate_gates(
    x_mics: np.ndarray,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    mic_positions: np.ndarray,
    ref_ch: int,
    *,
    theta_s: float = 0.6,
    theta_t: float = 0.6,
    theta_i: float = 0.4,
    theta_n: float = 0.3,
    beta_speech_interf: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    channels = x_mics.shape[0]
    ref_ch = int(np.clip(ref_ch, 0, channels - 1))
    ref = x_mics[ref_ch]
    frames = _frame_signal(ref, win_length, hop_length)
    energy = np.mean(frames**2, axis=1)
    energy = (energy - np.min(energy)) / (np.ptp(energy) + 1e-8)

    spec = np.abs(np.fft.rfft(frames, axis=1)) + 1e-8
    sfm = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
    p_speech = np.clip(0.7 * energy + 0.3 * (1.0 - sfm), 0.0, 1.0)

    pairs = [(0, 1), (0, 2), (1, 3)] if channels >= 4 else [(0, 1)]
    votes = np.zeros(frames.shape[0], dtype=np.float32)
    sin60 = np.sin(np.deg2rad(60.0))
    c = 343.0
    for i, j in pairs:
        fi = _frame_signal(x_mics[i], win_length, hop_length)
        fj = _frame_signal(x_mics[j], win_length, hop_length)
        dist = float(np.linalg.norm(mic_positions[i] - mic_positions[j]))
        tau_lim = (dist / c) * sin60
        for t in range(frames.shape[0]):
            tau = abs(_gcc_phat_tau(fi[t], fj[t], sample_rate))
            votes[t] += 1.0 if tau <= tau_lim else 0.0
    p_tar = votes / float(len(pairs))
    return gates_from_probs(
        p_speech,
        p_tar,
        theta_s=theta_s,
        theta_t=theta_t,
        theta_i=theta_i,
        theta_n=theta_n,
        beta_speech_interf=beta_speech_interf,
    )


def _match_frames(g: np.ndarray, frames: int) -> np.ndarray:
    if g.shape[0] == frames:
        return g.astype(np.float32)
    if g.shape[0] > frames:
        return g[:frames].astype(np.float32)
    if g.shape[0] == 0:
        return np.ones(frames, dtype=np.float32)
    return np.pad(g, (0, frames - g.shape[0]), mode="edge").astype(np.float32)


def mcwf_make_y0(
    x_mics: np.ndarray | "torch.Tensor",
    stft_params: Dict[str, int] | None,
    causal_frames: int = 4,
    ref_ch: int = 1,
    diag_load: float = 1e-2,
    sample_rate: int | None = None,
    mic_positions: np.ndarray | None = None,
) -> np.ndarray | "torch.Tensor":
    """Generate y0 with 4-mic frequency-domain MVDR beamforming (legacy API name)."""
    del causal_frames
    input_is_torch = torch is not None and torch.is_tensor(x_mics)
    x_np = x_mics.detach().cpu().numpy() if input_is_torch else np.asarray(x_mics)

    if x_np.ndim == 2:
        x_np = x_np[None, ...]
    if x_np.ndim != 3 or x_np.shape[1] < 2:
        raise ValueError("x_mics must have shape (C, T) or (B, C, T) with C>=2")

    params = dict(stft_params) if stft_params is not None else dict(MODEL_STFT)
    n_fft = int(params["n_fft"])
    win_length = int(params["win_length"])
    hop_length = int(params["hop_length"])
    sample_rate = 16000 if sample_rate is None else int(sample_rate)

    positions = DEFAULT_MIC_POSITIONS if mic_positions is None else np.asarray(mic_positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("mic_positions must have shape (C, 3) or (>=C, 3)")

    batch, channels, length = x_np.shape
    if positions.shape[0] < channels:
        raise ValueError("mic_positions must provide at least one row per channel")

    y0_list = []
    for b in range(batch):
        mic_stfts = [
            stft(
                x_np[b, mic_idx],
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=False,
            )
            for mic_idx in range(channels)
        ]
        X_np = np.stack(mic_stfts, axis=-1)  # (F,T,C)
        target_gate, noise_gate = _estimate_gates(
            x_np[b],
            sample_rate=sample_rate,
            hop_length=hop_length,
            win_length=win_length,
            mic_positions=positions[:channels],
            ref_ch=ref_ch,
        )
        target_gate = _match_frames(target_gate, X_np.shape[1])
        noise_gate = _match_frames(noise_gate, X_np.shape[1])
        y0_stft = make_mvdr_y0_stft(
            X_np,
            target_gate,
            noise_gate,
            cfg=MVDRConfig(ref_ch=ref_ch, diag_load=diag_load),
        )
        y0 = istft(
            y0_stft,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=False,
            length=length,
        )
        y0_list.append(y0)

    y0 = np.stack(y0_list, axis=0).astype(np.float32)
    if input_is_torch:
        return torch.tensor(y0, device=x_mics.device, dtype=x_mics.dtype)
    if x_mics.ndim == 2:
        return y0[0]
    return y0
