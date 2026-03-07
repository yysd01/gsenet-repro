from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - optional torch path
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _is_torch(x: Any) -> bool:
    return torch is not None and torch.is_tensor(x)


def _to_numpy(x: np.ndarray | "torch.Tensor") -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def hermitian(R: np.ndarray | "torch.Tensor") -> np.ndarray | "torch.Tensor":
    if _is_torch(R):
        return 0.5 * (R + R.conj().transpose(-1, -2))
    arr = _to_numpy(R)
    return 0.5 * (arr + np.swapaxes(np.conjugate(arr), -1, -2))


def diag_load(R: np.ndarray | "torch.Tensor", delta: float) -> np.ndarray | "torch.Tensor":
    if _is_torch(R):
        arr = hermitian(R)
        C = arr.shape[-1]
        tr = torch.diagonal(arr, dim1=-2, dim2=-1).real.sum(dim=-1)
        eye = torch.eye(C, dtype=arr.dtype, device=arr.device)
        loaded = arr + (float(delta) * tr / max(C, 1)).unsqueeze(-1).unsqueeze(-1) * eye
        return hermitian(loaded)

    arr = hermitian(_to_numpy(R))
    C = arr.shape[-1]
    tr = np.trace(arr, axis1=-2, axis2=-1).real
    loaded = arr + (float(delta) * tr / max(C, 1))[:, None, None] * np.eye(C, dtype=arr.dtype)[None]
    return hermitian(loaded)


def mvdr_weights(
    Rnn: np.ndarray | "torch.Tensor",
    d: np.ndarray | "torch.Tensor",
    ref_ch: int = 0,
) -> np.ndarray | "torch.Tensor":
    if _is_torch(Rnn) and _is_torch(d):
        F, C, _ = Rnn.shape
        u = torch.linalg.solve(Rnn, d.unsqueeze(-1)).squeeze(-1)
        denom = torch.sum(d.conj() * u, dim=-1)
        w = u / torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom)).unsqueeze(-1)
        fallback = torch.zeros((F, C), dtype=Rnn.dtype, device=Rnn.device)
        fallback[:, int(ref_ch)] = 1.0 + 0.0j
        return torch.where((denom.abs() > 1e-8).unsqueeze(-1), w, fallback)

    R = _to_numpy(Rnn)
    dv = _to_numpy(d)
    F, C, _ = R.shape
    w = np.zeros((F, C), dtype=np.complex64)
    for f in range(F):
        u = np.linalg.solve(R[f], dv[f])
        denom = np.vdot(dv[f], u)
        if abs(denom) < 1e-8:
            w[f] = 0.0
            w[f, int(ref_ch)] = 1.0 + 0.0j
        else:
            w[f] = u / denom
    return w


def lcmv_weights(
    Rnn: np.ndarray | "torch.Tensor", D: np.ndarray | "torch.Tensor", g: np.ndarray | "torch.Tensor"
) -> np.ndarray | "torch.Tensor":
    if _is_torch(Rnn) and _is_torch(D) and _is_torch(g):
        F, C, K = D.shape
        RinD = torch.linalg.solve(Rnn, D)
        gram = D.conj().transpose(-1, -2) @ RinD
        gv = g.reshape(1, K, 1).expand(F, K, 1)
        q = torch.linalg.solve(gram, gv)
        return (RinD @ q).squeeze(-1)

    R = _to_numpy(Rnn)
    Dn = _to_numpy(D)
    gn = _to_numpy(g).reshape(-1, 1)
    F, C, _ = Dn.shape
    w = np.zeros((F, C), dtype=np.complex64)
    for f in range(F):
        RinD = np.linalg.solve(R[f], Dn[f])
        gram = np.conjugate(Dn[f]).T @ RinD
        q = np.linalg.solve(gram, gn)
        w[f] = (RinD @ q).reshape(-1)
    return w


def apply_beamformer(X: np.ndarray | "torch.Tensor", w: np.ndarray | "torch.Tensor") -> np.ndarray | "torch.Tensor":
    if _is_torch(X) and _is_torch(w):
        return torch.einsum("fc,ftc->ft", w.conj(), X)
    return np.einsum("fc,ftc->ft", np.conjugate(_to_numpy(w)), _to_numpy(X))
