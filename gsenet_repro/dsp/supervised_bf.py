from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

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


def parse_doas_from_filename(name: str) -> tuple[list[int], list[int]]:
    stem = Path(name).name
    src_match = re.search(r"src\[\s*([^\]]*?)\s*\]", stem)
    int_match = re.search(r"int\[\s*([^\]]*?)\s*\]", stem)
    if src_match is None or int_match is None:
        raise ValueError(f"Unable to parse src/int DOAs from filename: {name}")

    def _parse_list(raw: str) -> list[int]:
        if not raw.strip():
            return []
        vals = [int(token.strip()) for token in raw.split(",") if token.strip()]
        for value in vals:
            if value < 0 or value > 359:
                raise ValueError(f"DOA out of range [0,359]: {value} in {name}")
        return sorted(vals)

    return _parse_list(src_match.group(1)), _parse_list(int_match.group(1))


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

    n_fft = int(stft_cfg["n_fft"])
    hop = int(stft_cfg["hop_length"])
    win = int(stft_cfg["win_length"])
    noverlap = win - hop
    X = []
    for c in range(4):
        _, _, Z = signal.stft(
            x[c],
            fs=1.0,
            window="hann",
            nperseg=win,
            noverlap=noverlap,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )
        X.append(Z.astype(np.complex64))
    return np.stack(X, axis=-1)


def estimate_rnn_from_noise(Xn: np.ndarray | "torch.Tensor", delta: float = 1e-2) -> np.ndarray | "torch.Tensor":
    X = _to_numpy(Xn)
    if X.ndim != 3:
        raise ValueError("Xn must be (F,T,C)")
    F, T, C = X.shape
    if T == 0:
        R = np.tile(np.eye(C, dtype=np.complex64)[None], (F, 1, 1))
    else:
        R = np.mean(X[:, :, :, None] * np.conjugate(X[:, :, None, :]), axis=1).astype(np.complex64)
    R = 0.5 * (R + np.swapaxes(np.conjugate(R), -1, -2))
    tr = np.trace(R, axis1=-2, axis2=-1).real
    R = R + (float(delta) * tr / max(C, 1))[:, None, None] * np.eye(C, dtype=np.complex64)[None]
    R = 0.5 * (R + np.swapaxes(np.conjugate(R), -1, -2))
    if _is_torch(Xn):
        return torch.as_tensor(R, dtype=Xn.dtype, device=Xn.device)
    return R


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
    Rs = 0.5 * (Rs + np.swapaxes(np.conjugate(Rs), -1, -2))
    d = np.zeros((F, C), dtype=np.complex64)
    prev = np.zeros((C,), dtype=np.complex64)
    prev[ref_ch] = 1.0 + 0.0j
    for f in range(F):
        eigvals, eigvecs = np.linalg.eigh(Rs[f])
        cur = eigvecs[:, int(np.argmax(eigvals))]
        ref = cur[ref_ch]
        if abs(ref) <= eps:
            if f + 1 < F:
                eigvals2, eigvecs2 = np.linalg.eigh(Rs[f + 1])
                nxt = eigvecs2[:, int(np.argmax(eigvals2))]
                if abs(nxt[ref_ch]) > eps:
                    cur = nxt
                    ref = cur[ref_ch]
                else:
                    cur = prev
                    ref = cur[ref_ch]
            else:
                cur = prev
                ref = cur[ref_ch]
        d[f] = cur / ref
        prev = d[f]
    d[:, ref_ch] = 1.0 + 0.0j
    if _is_torch(Xs):
        return torch.as_tensor(d, dtype=Xs.dtype, device=Xs.device)
    return d


def _doa_to_bin(doa: int, binsize_deg: int) -> int:
    doa_wrapped = int(doa) % 360
    return int(round(doa_wrapped / binsize_deg) * binsize_deg) % 360


def build_doa_rtf_library(
    train_root: str | Path,
    binsize_deg: int = 1,
    stft_cfg: dict[str, Any] | None = None,
    ref_ch: int = 0,
    artifact_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    if stft_cfg is None:
        stft_cfg = {"n_fft": 256, "win_length": 256, "hop_length": 128, "window": "hann", "center": False}
    import soundfile as sf

    root = Path(train_root)
    accum: dict[int, list[np.ndarray]] = {}
    for clean_wav in sorted(root.glob("*/clean/*.wav")):
        src_doas, _ = parse_doas_from_filename(clean_wav.name)
        if len(src_doas) != 1:
            continue
        wav, sr = sf.read(str(clean_wav), always_2d=True, dtype="float32")
        if sr != 16000:
            raise ValueError(f"Expected sr=16000, got {sr} for {clean_wav}")
        d = estimate_rtf_from_clean_ev(stft_4ch(wav.T, stft_cfg), ref_ch=ref_ch)
        d_np = _to_numpy(d).astype(np.complex64)
        accum.setdefault(_doa_to_bin(src_doas[0], binsize_deg), []).append(d_np)

    if not accum:
        raise ValueError(f"No single-target clean samples found under: {root}")
    doa_bins = sorted(accum.keys())
    d_stack = np.stack([np.stack(accum[b], axis=0).mean(axis=0) for b in doa_bins], axis=0)
    d_stack[:, :, ref_ch] = np.complex64(1.0 + 0.0j)

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"rtf_lib_oppo_binsize{binsize_deg}.npz"
    np.savez(save_path, doa_bins=np.asarray(doa_bins, np.int32), d_mean=d_stack.astype(np.complex64), binsize_deg=np.int32(binsize_deg), ref_ch=np.int32(ref_ch))
    return {"doa_bins": np.asarray(doa_bins, np.int32), "d_mean": d_stack.astype(np.complex64), "binsize_deg": int(binsize_deg), "ref_ch": int(ref_ch), "path": str(save_path)}


def get_d_from_lib(rtf_lib: dict[str, Any], doa: int) -> np.ndarray:
    binsize = int(rtf_lib["binsize_deg"])
    doa_bins = np.asarray(rtf_lib["doa_bins"], dtype=np.int32)
    d_mean = np.asarray(rtf_lib["d_mean"], dtype=np.complex64)
    target_bin = _doa_to_bin(doa, binsize)
    idx = int(np.argmin(np.abs(((doa_bins - target_bin + 180) % 360) - 180)))
    d = d_mean[idx].copy()
    d[:, int(rtf_lib.get("ref_ch", 0))] = np.complex64(1.0 + 0.0j)
    return d


def mvdr_weights(Rnn: np.ndarray | "torch.Tensor", d: np.ndarray | "torch.Tensor") -> np.ndarray | "torch.Tensor":
    R = _to_numpy(Rnn)
    dv = _to_numpy(d)
    F, C, _ = R.shape
    w = np.zeros((F, C), dtype=np.complex64)
    for f in range(F):
        u = np.linalg.solve(R[f], dv[f])
        denom = np.vdot(dv[f], u)
        w[f] = np.array([1, 0, 0, 0], dtype=np.complex64) if abs(denom) < 1e-8 else (u / denom)
    if _is_torch(Rnn):
        return torch.as_tensor(w, dtype=Rnn.dtype, device=Rnn.device)
    return w


def lcmv_weights(Rnn: np.ndarray | "torch.Tensor", D: np.ndarray | "torch.Tensor", g: np.ndarray | "torch.Tensor") -> np.ndarray | "torch.Tensor":
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
    if _is_torch(Rnn):
        return torch.as_tensor(w, dtype=Rnn.dtype, device=Rnn.device)
    return w


def apply_beamformer(X: np.ndarray | "torch.Tensor", w: np.ndarray | "torch.Tensor") -> np.ndarray | "torch.Tensor":
    if _is_torch(X) and _is_torch(w):
        return torch.einsum("fc,ftc->ft", w.conj(), X)
    return np.einsum("fc,ftc->ft", np.conjugate(_to_numpy(w)), _to_numpy(X))


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
        w = mvdr_weights(Rnn, d)
        proj = np.sum(np.conjugate(_to_numpy(w)) * _to_numpy(d), axis=1)
        constraint_error = float(np.max(np.abs(proj - 1.0)))
    elif len(src_doas) == 2 and rtf_lib is not None:
        d1 = get_d_from_lib(rtf_lib, src_doas[0])
        d2 = get_d_from_lib(rtf_lib, src_doas[1])
        D = np.stack([d1, d2], axis=-1)
        w = lcmv_weights(Rnn, D, np.array([1.0, 1.0], dtype=np.float32))
    else:
        # Fallback: clean EV favors dominant target when two sources are present.
        d = estimate_rtf_from_clean_ev(Xs, ref_ch=ref_ch)
        w = mvdr_weights(Rnn, d)
        proj = np.sum(np.conjugate(_to_numpy(w)) * _to_numpy(d), axis=1)
        constraint_error = float(np.max(np.abs(proj - 1.0)))

    Y = _to_numpy(apply_beamformer(Xy, w))
    n_fft = int(stft_cfg["n_fft"])
    hop = int(stft_cfg["hop_length"])
    win = int(stft_cfg["win_length"])
    if torch_istft is not None and _is_torch(Y):
        y0 = torch_istft(Y, n_fft=n_fft, win_length=win, hop_length=hop, window="hann", center=False, length=_to_numpy(noisy4).shape[1]).detach().cpu().numpy().astype(np.float32)
    else:
        _, y0 = signal.istft(Y, fs=1.0, window="hann", nperseg=win, noverlap=win - hop, nfft=n_fft, input_onesided=True, boundary=False)
        y0 = y0[: _to_numpy(noisy4).shape[1]].astype(np.float32)
    y1 = _to_numpy(noisy4)[ref_ch].astype(np.float32)
    min_len = min(y0.shape[0], y1.shape[0])
    y0 = y0[:min_len]
    y1 = y1[:min_len]
    debug = {"constraint_error": float(constraint_error), "rms_y0": float(np.sqrt(np.mean(np.square(y0)) + 1e-12)), "rms_y1": float(np.sqrt(np.mean(np.square(y1)) + 1e-12))}
    return y0, y1, debug


def load_rtf_lib(path: str | Path) -> dict[str, Any]:
    data = np.load(path)
    return {"doa_bins": data["doa_bins"], "d_mean": data["d_mean"], "binsize_deg": int(data["binsize_deg"]), "ref_ch": int(data["ref_ch"]), "path": str(path)}
