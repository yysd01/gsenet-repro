"""Lightweight audio quality metrics (numpy/scipy only).

Note: These are proxy implementations intended for regression testing and
relative comparisons when PESQ/STOI dependencies are unavailable.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def snr_db(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-12) -> float:
    """Compute global SNR in dB."""
    reference = np.asarray(reference, dtype=np.float32)
    estimate = np.asarray(estimate, dtype=np.float32)
    noise = reference - estimate
    ref_pow = np.mean(reference**2) + eps
    noise_pow = np.mean(noise**2) + eps
    return float(10.0 * np.log10(ref_pow / noise_pow))


def _frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if x.size < frame_len:
        x = np.pad(x, (0, frame_len - x.size), mode="constant")
    n_frames = 1 + max(0, int(np.floor((x.size - frame_len) / hop)))
    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    for idx in range(n_frames):
        start = idx * hop
        frames[idx] = x[start : start + frame_len]
    return frames


def pesq_proxy(reference: np.ndarray, estimate: np.ndarray, fs: int = 16000) -> float:
    """Approximate PESQ score using segmental SNR mapping."""
    frame_len = int(0.02 * fs)
    hop = int(0.01 * fs)
    ref_frames = _frame_signal(np.asarray(reference, dtype=np.float32), frame_len, hop)
    est_frames = _frame_signal(np.asarray(estimate, dtype=np.float32), frame_len, hop)
    snrs = []
    for ref, est in zip(ref_frames, est_frames):
        snrs.append(snr_db(ref, est))
    seg_snr = float(np.mean(np.clip(snrs, -10.0, 35.0)))
    pesq = 1.0 + (seg_snr + 10.0) / 45.0 * 3.5
    return float(np.clip(pesq, 1.0, 4.5))


def stoi_proxy(reference: np.ndarray, estimate: np.ndarray, fs: int = 16000) -> float:
    """Approximate STOI score using short-time envelope correlation."""
    ref = np.asarray(reference, dtype=np.float32)
    est = np.asarray(estimate, dtype=np.float32)
    ref_env = np.abs(hilbert(ref))
    est_env = np.abs(hilbert(est))

    frame_len = int(0.03 * fs)
    hop = int(0.015 * fs)
    ref_frames = _frame_signal(ref_env, frame_len, hop)
    est_frames = _frame_signal(est_env, frame_len, hop)

    corrs = []
    for ref_frame, est_frame in zip(ref_frames, est_frames):
        ref_centered = ref_frame - np.mean(ref_frame)
        est_centered = est_frame - np.mean(est_frame)
        denom = np.linalg.norm(ref_centered) * np.linalg.norm(est_centered) + 1e-8
        corrs.append(float(np.dot(ref_centered, est_centered) / denom))

    stoi = np.mean(corrs)
    return float(np.clip(stoi, 0.0, 1.0))
