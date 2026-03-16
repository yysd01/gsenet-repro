from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_MIC_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 3))


def _frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if x.shape[-1] < frame_length:
        x = np.pad(x, (0, frame_length - x.shape[-1]))
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
    noise_beta = np.where(target_gate > 0.5, 0.0, np.where(is_noise > 0.5, 1.0, beta_speech_interf))
    return target_gate.astype(np.float32), noise_beta.astype(np.float32)


def estimate_sector_gates(
    x_mics: np.ndarray,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    mic_positions: np.ndarray,
    ref_ch: int,
    mic_pairs: Sequence[Sequence[int]] | None = None,
    sector_half_angle_deg: float = 60.0,
    theta_s: float = 0.6,
    theta_t: float = 0.6,
    theta_i: float = 0.4,
    theta_n: float = 0.3,
    beta_speech_interf: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    channels = x_mics.shape[0]
    ref_ch = int(np.clip(ref_ch, 0, channels - 1))
    frames = _frame_signal(x_mics[ref_ch], win_length, hop_length)
    energy = np.mean(frames**2, axis=1)
    energy = (energy - np.min(energy)) / (np.ptp(energy) + 1e-8)

    spec = np.abs(np.fft.rfft(frames, axis=1)) + 1e-8
    sfm = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
    p_speech = np.clip(0.7 * energy + 0.3 * (1.0 - sfm), 0.0, 1.0)

    pairs = mic_pairs if mic_pairs is not None else DEFAULT_MIC_PAIRS
    valid_pairs = [(int(i), int(j)) for (i, j) in pairs if int(i) < channels and int(j) < channels and int(i) != int(j)]
    if not valid_pairs:
        valid_pairs = [(0, min(1, channels - 1))]

    votes = np.zeros(frames.shape[0], dtype=np.float32)
    sin_sector = np.sin(np.deg2rad(float(sector_half_angle_deg)))
    c = 343.0
    for i, j in valid_pairs:
        fi = _frame_signal(x_mics[i], win_length, hop_length)
        fj = _frame_signal(x_mics[j], win_length, hop_length)
        dist = float(np.linalg.norm(mic_positions[i] - mic_positions[j]))
        tau_lim = (dist / c) * sin_sector
        for t in range(frames.shape[0]):
            tau = abs(_gcc_phat_tau(fi[t], fj[t], sample_rate))
            votes[t] += 1.0 if tau <= tau_lim else 0.0

    p_tar = votes / float(len(valid_pairs))
    return gates_from_probs(
        p_speech,
        p_tar,
        theta_s=theta_s,
        theta_t=theta_t,
        theta_i=theta_i,
        theta_n=theta_n,
        beta_speech_interf=beta_speech_interf,
    )


def estimate_vad_gates(
    x_mics: np.ndarray,
    hop_length: int,
    win_length: int,
    ref_ch: int,
    vad_db_thresh: float = -35.0,
    vad_smooth: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    frames = _frame_signal(x_mics[int(ref_ch)], win_length, hop_length)
    power = np.mean(frames**2, axis=1)
    frame_db = 10.0 * np.log10(power + 1e-12)
    z = (float(vad_db_thresh) - frame_db) / max(float(vad_smooth), 1e-6)
    noise_gate = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    return (1.0 - noise_gate).astype(np.float32), noise_gate
