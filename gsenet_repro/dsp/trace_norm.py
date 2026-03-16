from __future__ import annotations

import importlib.util

import numpy as np

from gsenet_repro.dsp.stft import istft, stft

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    import torch
else:  # pragma: no cover
    torch = None


DEFAULT_SOLVE_FALLBACK_JITTER = 1e-3


def _as_batched_covariance_torch(m: "torch.Tensor") -> tuple["torch.Tensor", bool]:
    if m.ndim == 2:
        return m.unsqueeze(0), True
    if m.ndim == 3:
        return m, False
    raise ValueError(f"Expected covariance shape (C, C) or (F, C, C), got {tuple(m.shape)}")


def _restore_covariance_shape_torch(m: "torch.Tensor", squeeze: bool) -> "torch.Tensor":
    return m.squeeze(0) if squeeze else m


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


def hermitian_torch(m: "torch.Tensor") -> "torch.Tensor":
    """Return Hermitian symmetrization for covariance tensors.

    Args:
        m: Covariance tensor with shape ``(C, C)`` or ``(F, C, C)``.

    Returns:
        Tensor with the same shape as input.
    """
    if torch is None:
        raise ImportError("hermitian_torch requires torch")
    mb, squeeze = _as_batched_covariance_torch(m)
    hb = 0.5 * (mb + mb.conj().transpose(-1, -2))
    return _restore_covariance_shape_torch(hb, squeeze)


def diag_load_torch(m: "torch.Tensor", load: float) -> "torch.Tensor":
    """Apply trace-proportional diagonal loading.

    Args:
        m: Covariance tensor with shape ``(C, C)`` or ``(F, C, C)``.
        load: Scalar loading factor.

    Returns:
        Tensor with same shape as input after diagonal loading.
    """
    if torch is None:
        raise ImportError("diag_load_torch requires torch")
    mb, squeeze = _as_batched_covariance_torch(m)
    c = mb.shape[-1]
    tr = mb.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real / float(c)
    eye = torch.eye(c, dtype=mb.dtype, device=mb.device).unsqueeze(0)
    out = mb + (float(load) * tr).unsqueeze(-1).unsqueeze(-1) * eye
    return _restore_covariance_shape_torch(out, squeeze)


def psd_project_torch(m: "torch.Tensor") -> "torch.Tensor":
    """Project covariance tensor to the positive semidefinite cone.

    Args:
        m: Covariance tensor with shape ``(C, C)`` or ``(F, C, C)``.

    Returns:
        Tensor with same shape as input after eigenvalue clipping.
    """
    if torch is None:
        raise ImportError("psd_project_torch requires torch")
    mb, squeeze = _as_batched_covariance_torch(m)
    evals, evecs = torch.linalg.eigh(hermitian_torch(mb))
    evals = torch.clamp(evals.real, min=0.0)
    out = torch.matmul(evecs * evals.unsqueeze(-2), evecs.conj().transpose(-1, -2))
    return _restore_covariance_shape_torch(out, squeeze)


def trace_norm_weights_torch(
    phi_v: "torch.Tensor",
    phi_x: "torch.Tensor",
    *,
    ref_ch: int,
    diag_load_v: float = 1e-2,
    diag_load_x: float = 1e-3,
    eps_trace: float = 1e-6,
    psd_project: bool = True,
    solve_fallback_jitter: float = DEFAULT_SOLVE_FALLBACK_JITTER,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Compute trace-normalized beamforming weights.

    Args:
        phi_v: Noise covariance with shape ``(C, C)`` or ``(F, C, C)``.
        phi_x: Speech covariance with shape ``(C, C)`` or ``(F, C, C)``.
        ref_ch: Reference channel index used to select ``A[..., ref_ch]``.
        diag_load_v: Diagonal load for ``phi_v``.
        diag_load_x: Diagonal load for ``phi_x``.
        eps_trace: Minimum real trace in denominator clamp.
        psd_project: Whether to PSD-project ``phi_x`` before solve.
        solve_fallback_jitter: Extra identity load used only if initial solve fails.

    Returns:
        ``(w, den_real, invalid)`` where each has shape ``(C,)``/``()``/``()`` for
        single covariance input and ``(F, C)``/``(F,)``/``(F,)`` for batched input.
        Invalid bins fall back to a one-hot reference-channel weight.
    """
    if torch is None:
        raise ImportError("trace_norm_weights_torch requires torch")
    phi_vb, squeeze = _as_batched_covariance_torch(phi_v)
    phi_xb, squeeze_x = _as_batched_covariance_torch(phi_x)
    if squeeze != squeeze_x:
        raise ValueError("phi_v and phi_x must both be 2D or both be 3D covariance tensors")

    c = phi_vb.shape[-1]
    ref_idx = int(np.clip(ref_ch, 0, c - 1))

    phi_v_loaded = diag_load_torch(hermitian_torch(phi_vb), diag_load_v)
    phi_xh = hermitian_torch(phi_xb)
    if psd_project:
        phi_xh = psd_project_torch(phi_xh)
    phi_x_loaded = diag_load_torch(phi_xh, diag_load_x)

    eye = torch.eye(c, dtype=phi_vb.dtype, device=phi_vb.device).unsqueeze(0)
    try:
        A = torch.linalg.solve(phi_v_loaded, phi_x_loaded)
    except RuntimeError:
        A = torch.linalg.solve(phi_v_loaded + float(solve_fallback_jitter) * eye, phi_x_loaded)

    tau = A.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    den_real = tau.real
    den = torch.clamp(den_real, min=float(eps_trace))
    w = A[..., ref_idx] / den.unsqueeze(-1)

    invalid = (~torch.isfinite(w).all(dim=-1)) | (~torch.isfinite(den_real)) | (den_real <= float(eps_trace))
    if torch.any(invalid):
        w_fb = torch.zeros_like(w)
        w_fb[:, ref_idx] = 1.0 + 0.0j
        w = torch.where(invalid.unsqueeze(-1), w_fb, w)

    if squeeze:
        return w.squeeze(0), den_real.squeeze(0), invalid.squeeze(0)
    return w, den_real, invalid


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
