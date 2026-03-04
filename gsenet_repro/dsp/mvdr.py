from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MVDRConfig:
    smoothing_rnn: float = 0.96
    smoothing_rtf: float = 0.04
    beta_speech_interf: float = 0.5
    diag_load: float = 1e-2
    ref_ch: int = 1


def _hermitian(R: np.ndarray) -> np.ndarray:
    return 0.5 * (R + np.swapaxes(np.conjugate(R), -1, -2))


def estimate_rnn(
    X: np.ndarray,
    mask_or_gate: np.ndarray,
    smoothing: float = 0.96,
) -> np.ndarray:
    """Estimate noise covariance with EMA.

    `smoothing` follows a decay/retention meaning: larger values keep more
    history (e.g. 0.96 keeps ~96% of the previous covariance per gated frame).
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (F,T,C)")
    F, T, C = X.shape
    gate = np.asarray(mask_or_gate, dtype=np.float32)
    if gate.shape != (T,):
        raise ValueError("mask_or_gate must have shape (T,)")

    R = np.tile(np.eye(C, dtype=np.complex64)[None, :, :], (F, 1, 1))
    for t in range(T):
        x = X[:, t, :]
        xxh = x[:, :, None] * np.conjugate(x[:, None, :])
        alpha = float((1.0 - smoothing) * gate[t])
        R = (1.0 - alpha) * R + alpha * xxh
        R = _hermitian(R)
    return R


def estimate_rtf(
    X: np.ndarray,
    gate: np.ndarray,
    ref_ch: int,
    smoothing: float = 0.04,
    min_frames: int = 4,
) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError("X must have shape (F,T,C)")
    F, T, C = X.shape
    if not 0 <= ref_ch < C:
        raise ValueError("ref_ch out of range")
    g = np.asarray(gate, dtype=np.float32)
    if g.shape != (T,):
        raise ValueError("gate must have shape (T,)")

    Rs = np.tile(np.eye(C, dtype=np.complex64)[None, :, :], (F, 1, 1))
    count = 0
    for t in range(T):
        if g[t] <= 0.5:
            continue
        x = X[:, t, :]
        xxh = x[:, :, None] * np.conjugate(x[:, None, :])
        eta = float(smoothing * g[t])
        Rs = (1.0 - eta) * Rs + eta * xxh
        Rs = _hermitian(Rs)
        count += 1

    d = np.zeros((F, C), dtype=np.complex64)
    if count < min_frames:
        d[:, ref_ch] = 1.0 + 0.0j
    else:
        for f in range(F):
            eigvals, eigvecs = np.linalg.eigh(Rs[f])
            v = eigvecs[:, np.argmax(eigvals)]
            ref = v[ref_ch] if np.abs(v[ref_ch]) > 1e-8 else 1.0 + 0.0j
            d[f] = v / ref

    if F >= 3:
        d_pad = np.pad(d, ((1, 1), (0, 0)), mode="edge")
        d = (d_pad[:-2] + d_pad[1:-1] + d_pad[2:]) / 3.0
    return d


def mvdr_weights(
    R_nn: np.ndarray,
    d: np.ndarray,
    diag_load: float = 1e-2,
) -> np.ndarray:
    F, C, _ = R_nn.shape
    w = np.zeros((F, C), dtype=np.complex64)
    eye = np.eye(C, dtype=np.complex64)
    for f in range(F):
        tr = float(np.trace(R_nn[f]).real)
        Rl = _hermitian(R_nn[f] + (diag_load * tr / C) * eye)
        u = np.linalg.solve(Rl, d[f])
        denom = np.vdot(d[f], u)
        if np.abs(denom) < 1e-8:
            w[f] = 0
            w[f, 0] = 1.0
        else:
            w[f] = u / denom
    return w


def apply_beamformer(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    if w.ndim != 2 or X.ndim != 3:
        raise ValueError("w must be (F,C), X must be (F,T,C)")
    return np.einsum("fc,ftc->ft", np.conjugate(w), X)


def make_mvdr_y0_stft(
    X: np.ndarray,
    target_gate: np.ndarray,
    noise_gate: np.ndarray,
    *,
    cfg: MVDRConfig,
) -> np.ndarray:
    rnn = estimate_rnn(X, noise_gate, smoothing=cfg.smoothing_rnn)
    d = estimate_rtf(X, target_gate, ref_ch=cfg.ref_ch, smoothing=cfg.smoothing_rtf)
    w = mvdr_weights(rnn, d, diag_load=cfg.diag_load)
    return apply_beamformer(w, X)
