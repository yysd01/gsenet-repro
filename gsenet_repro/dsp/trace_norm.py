from __future__ import annotations

import numpy as np

from gsenet_repro.dsp.stft import istft, stft


def hermitian_np(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + np.swapaxes(np.conjugate(m), -1, -2))


def diag_load_np(m: np.ndarray, load: float) -> np.ndarray:
    c = m.shape[-1]
    tr = m.diagonal(axis1=-2, axis2=-1).sum(axis=-1).real / float(c)
    eye = np.eye(c, dtype=m.dtype)[None, :, :]
    return m + (float(load) * tr)[:, None, None] * eye


def psd_project_np(m: np.ndarray) -> np.ndarray:
    h = hermitian_np(m)
    evals, evecs = np.linalg.eigh(h)
    evals = np.maximum(evals.real, 0.0)
    return np.matmul(evecs * evals[:, None, :], np.swapaxes(np.conjugate(evecs), -1, -2))


def _normalize_gate(gate: np.ndarray, T: int, default: float) -> np.ndarray:
    g = np.asarray(gate, dtype=np.float32).reshape(-1)
    if g.shape[0] == T:
        return g
    if g.shape[0] > T:
        return g[:T]
    if g.shape[0] == 0:
        return np.full((T,), default, dtype=np.float32)
    return np.pad(g, (0, T - g.shape[0]), mode="edge").astype(np.float32)


def estimate_phi_y(X: np.ndarray, alpha_y: float = 0.92) -> np.ndarray:
    F, T, C = X.shape
    R = np.tile(np.eye(C, dtype=np.complex64)[None, :, :], (F, 1, 1))
    for t in range(T):
        xxh = X[:, t, :, None] * np.conjugate(X[:, t, None, :])
        R = float(alpha_y) * R + (1.0 - float(alpha_y)) * xxh
    return hermitian_np(R)


def estimate_phi_v(X: np.ndarray, noise_gate: np.ndarray, alpha_v: float = 0.98) -> np.ndarray:
    F, T, C = X.shape
    g = _normalize_gate(noise_gate, T, default=1.0)
    R = np.tile(np.eye(C, dtype=np.complex64)[None, :, :], (F, 1, 1))
    for t in range(T):
        xxh = X[:, t, :, None] * np.conjugate(X[:, t, None, :])
        R = float(alpha_v) * R + (1.0 - float(alpha_v)) * float(g[t]) * xxh
    return hermitian_np(R)


def estimate_phi_x(phi_y: np.ndarray, phi_v: np.ndarray, psd_project: bool = False) -> np.ndarray:
    phi_x = hermitian_np(phi_y - phi_v)
    if psd_project:
        phi_x = psd_project_np(phi_x)
    return hermitian_np(phi_x)


def trace_norm_weights(
    phi_v: np.ndarray,
    phi_x: np.ndarray,
    ref_ch: int,
    diag_load_v: float = 1e-2,
    diag_load_x: float = 1e-3,
    eps_trace: float = 1e-6,
    psd_project: bool = True,
) -> np.ndarray:
    c = phi_v.shape[-1]
    ref_ch = int(np.clip(ref_ch, 0, c - 1))
    phi_v_loaded = diag_load_np(hermitian_np(phi_v), diag_load_v)
    phi_xh = hermitian_np(phi_x)
    if psd_project:
        phi_xh = psd_project_np(phi_xh)
    phi_x_loaded = diag_load_np(phi_xh, diag_load_x)

    eye = np.eye(c, dtype=phi_v.dtype)[None, :, :]
    try:
        A = np.linalg.solve(phi_v_loaded, phi_x_loaded)
    except np.linalg.LinAlgError:
        A = np.linalg.solve(phi_v_loaded + 1e-3 * eye, phi_x_loaded)

    tau = A.diagonal(axis1=-2, axis2=-1).sum(axis=-1)
    den_real = tau.real
    den = np.clip(den_real, a_min=float(eps_trace), a_max=None)
    w = A[..., ref_ch] / den[:, None]

    invalid = (~np.isfinite(w).all(axis=-1)) | (~np.isfinite(den_real)) | (den_real <= float(eps_trace))
    if np.any(invalid):
        w_fb = np.zeros_like(w)
        w_fb[:, ref_ch] = 1.0 + 0.0j
        w[invalid] = w_fb[invalid]
    return w.astype(np.complex64)


def apply_beamformer_np(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.einsum("fc,ftc->ft", np.conjugate(w), X)


def make_trace_norm_y0_stft(
    X: np.ndarray,
    noise_gate: np.ndarray,
    *,
    ref_ch: int,
    alpha_y: float = 0.92,
    alpha_v: float = 0.98,
    diag_load_v: float = 1e-2,
    diag_load_x: float = 1e-3,
    eps_trace: float = 1e-6,
    psd_project: bool = True,
) -> np.ndarray:
    phi_y = estimate_phi_y(X, alpha_y=alpha_y)
    phi_v = estimate_phi_v(X, noise_gate, alpha_v=alpha_v)
    phi_x = estimate_phi_x(phi_y, phi_v, psd_project=psd_project)
    w = trace_norm_weights(
        phi_v,
        phi_x,
        ref_ch=ref_ch,
        diag_load_v=diag_load_v,
        diag_load_x=diag_load_x,
        eps_trace=eps_trace,
        psd_project=psd_project,
    )
    return apply_beamformer_np(w, X)


def make_trace_norm_y0(
    x_mics: np.ndarray,
    noise_gate: np.ndarray,
    stft_params: dict[str, int],
    *,
    ref_ch: int,
    alpha_y: float,
    alpha_v: float,
    diag_load_v: float,
    diag_load_x: float,
    eps_trace: float,
    psd_project: bool,
) -> np.ndarray:
    n_fft = int(stft_params["n_fft"])
    win_length = int(stft_params["win_length"])
    hop_length = int(stft_params["hop_length"])
    X = np.stack([
        stft(x_mics[ch], n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
        for ch in range(x_mics.shape[0])
    ], axis=-1)
    y_stft = make_trace_norm_y0_stft(
        X,
        noise_gate,
        ref_ch=ref_ch,
        alpha_y=alpha_y,
        alpha_v=alpha_v,
        diag_load_v=diag_load_v,
        diag_load_x=diag_load_x,
        eps_trace=eps_trace,
        psd_project=psd_project,
    )
    return istft(y_stft, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False, length=x_mics.shape[-1]).astype(np.float32)
