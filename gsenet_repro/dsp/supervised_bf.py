from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gsenet_repro.dsp.stft import istft as np_istft
from gsenet_repro.dsp.stft import stft as np_stft
from gsenet_repro.dsp.beamform import apply_beamformer, diag_load, hermitian, lcmv_weights, mvdr_weights
from gsenet_repro.dsp.rtf_lib import (
    build_doa_rtf_library as _build_doa_rtf_library,
    get_d_from_lib,
    load_rtf_lib,
    parse_doas_from_filename,
    save_rtf_lib,
)

try:  # pragma: no cover - optional torch path
    import torch
    from gsenet_repro.dsp.torch_stft import istft as torch_istft
    from gsenet_repro.dsp.torch_stft import stft as torch_stft
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    torch_istft = None  # type: ignore[assignment]
    torch_stft = None  # type: ignore[assignment]


def _is_torch(x: Any) -> bool:
    return torch is not None and torch.is_tensor(x)


def _to_numpy(x: np.ndarray | "torch.Tensor") -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def stft_4ch(wav: np.ndarray | "torch.Tensor", stft_cfg: dict[str, Any]) -> np.ndarray | "torch.Tensor":
    x = _to_numpy(wav).astype(np.float32)
    if x.ndim != 2 or x.shape[0] != 4:
        raise ValueError(f"Expected wav shape (4,T), got {x.shape}")

    if torch_stft is not None and _is_torch(wav):
        Xct = torch_stft(
            wav.to(dtype=torch.float32),
            n_fft=int(stft_cfg["n_fft"]),
            win_length=int(stft_cfg["win_length"]),
            hop_length=int(stft_cfg["hop_length"]),
            window=str(stft_cfg.get("window", "hann")),
            center=bool(stft_cfg.get("center", False)),
        )
        return Xct.permute(1, 2, 0).contiguous().to(torch.complex64)

    return np.stack(
        [
            np_stft(
                x[c],
                n_fft=int(stft_cfg["n_fft"]),
                win_length=int(stft_cfg["win_length"]),
                hop_length=int(stft_cfg["hop_length"]),
                window=str(stft_cfg.get("window", "hann")),
                center=bool(stft_cfg.get("center", False)),
            )
            for c in range(4)
        ],
        axis=-1,
    ).astype(np.complex64)


def estimate_rnn_from_noise(Xn: np.ndarray | "torch.Tensor", delta: float = 1e-2) -> np.ndarray | "torch.Tensor":
    X = _to_numpy(Xn)
    if X.ndim != 3:
        raise ValueError("Xn must be (F,T,C)")
    F, T, C = X.shape
    if T == 0:
        R = np.tile(np.eye(C, dtype=np.complex64)[None], (F, 1, 1))
    else:
        R = np.mean(X[:, :, :, None] * np.conjugate(X[:, :, None, :]), axis=1).astype(np.complex64)
    R = diag_load(hermitian(R), delta=float(delta))
    if _is_torch(Xn):
        return torch.as_tensor(R, dtype=Xn.dtype, device=Xn.device)
    return np.asarray(R, dtype=np.complex64)


def estimate_rtf_from_clean_ev(
    Xs: np.ndarray | "torch.Tensor", ref_ch: int = 0, eps: float = 1e-8
) -> np.ndarray | "torch.Tensor":
    X = _to_numpy(Xs)
    if X.ndim != 3:
        raise ValueError("Xs must be (F,T,C)")
    F, T, C = X.shape
    if T == 0:
        d = np.zeros((F, C), dtype=np.complex64)
        d[:, ref_ch] = 1.0 + 0.0j
        return torch.as_tensor(d, dtype=Xs.dtype, device=Xs.device) if _is_torch(Xs) else d

    Rs = np.mean(X[:, :, :, None] * np.conjugate(X[:, :, None, :]), axis=1).astype(np.complex64)
    Rs = np.asarray(hermitian(Rs), dtype=np.complex64)
    d = np.zeros((F, C), dtype=np.complex64)
    prev = np.zeros((C,), dtype=np.complex64)
    prev[ref_ch] = 1.0 + 0.0j
    for f in range(F):
        eigvals, eigvecs = np.linalg.eigh(Rs[f])
        cur = eigvecs[:, int(np.argmax(eigvals))]
        ref = cur[ref_ch]
        if abs(ref) <= eps:
            cur = prev
            ref = cur[ref_ch]
        d[f] = cur / ref
        prev = d[f]
    d[:, ref_ch] = 1.0 + 0.0j
    if _is_torch(Xs):
        return torch.as_tensor(d, dtype=Xs.dtype, device=Xs.device)
    return d


def build_doa_rtf_library(
    train_root: str | Path,
    binsize_deg: int = 1,
    stft_cfg: dict[str, Any] | None = None,
    ref_ch: int = 0,
    artifact_dir: str | Path = "artifacts",
    sample_rate: int = 16000,
) -> dict[str, Any]:
    if stft_cfg is None:
        stft_cfg = {"n_fft": 256, "win_length": 256, "hop_length": 128, "window": "hann", "center": False}
    lib = _build_doa_rtf_library(train_root=train_root, binsize_deg=binsize_deg, stft_cfg=stft_cfg, ref_ch=ref_ch)
    lib["sample_rate"] = int(lib.get("sample_rate", sample_rate))
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"rtf_lib_oppo_binsize{binsize_deg}.npz"
    save_rtf_lib(save_path, lib)
    lib["path"] = str(save_path)
    return lib


def beamform_sample(
    clean4: np.ndarray | "torch.Tensor",
    noise4: np.ndarray | "torch.Tensor",
    noisy4: np.ndarray | "torch.Tensor",
    src_doas: list[int],
    stft_cfg: dict[str, Any],
    rtf_lib: dict[str, Any] | None = None,
    ref_ch: int = 0,
    delta: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    Xs = stft_4ch(clean4, stft_cfg)
    Xn = stft_4ch(noise4, stft_cfg)
    Xy = stft_4ch(noisy4, stft_cfg)
    Rnn = estimate_rnn_from_noise(Xn, delta=delta)

    constraint_error = 0.0
    if len(src_doas) == 1:
        d = estimate_rtf_from_clean_ev(Xs, ref_ch=ref_ch)
        w = mvdr_weights(Rnn, d, ref_ch=ref_ch)
        proj = np.sum(np.conjugate(_to_numpy(w)) * _to_numpy(d), axis=1)
        constraint_error = float(np.max(np.abs(proj - 1.0)))
    elif len(src_doas) == 2 and rtf_lib is not None:
        d1 = get_d_from_lib(rtf_lib, src_doas[0])
        d2 = get_d_from_lib(rtf_lib, src_doas[1])
        D = np.stack([d1, d2], axis=-1)
        w = lcmv_weights(Rnn, D, np.array([1.0, 1.0], dtype=np.float32))
    else:
        d = estimate_rtf_from_clean_ev(Xs, ref_ch=ref_ch)
        w = mvdr_weights(Rnn, d, ref_ch=ref_ch)
        proj = np.sum(np.conjugate(_to_numpy(w)) * _to_numpy(d), axis=1)
        constraint_error = float(np.max(np.abs(proj - 1.0)))

    Y = _to_numpy(apply_beamformer(Xy, w))
    n_fft = int(stft_cfg["n_fft"])
    hop = int(stft_cfg["hop_length"])
    win = int(stft_cfg["win_length"])
    y0 = np_istft(Y, n_fft=n_fft, win_length=win, hop_length=hop, window="hann", center=False, length=_to_numpy(noisy4).shape[1]).astype(np.float32)
    y1 = _to_numpy(noisy4)[ref_ch].astype(np.float32)
    min_len = min(y0.shape[0], y1.shape[0])
    y0 = y0[:min_len]
    y1 = y1[:min_len]
    debug = {
        "constraint_error": float(constraint_error),
        "rms_y0": float(np.sqrt(np.mean(np.square(y0)) + 1e-12)),
        "rms_y1": float(np.sqrt(np.mean(np.square(y1)) + 1e-12)),
    }
    return y0, y1, debug
