from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from gsenet_repro.dsp.stft import stft


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


def _doa_to_bin(doa: int, binsize_deg: int) -> int:
    doa_wrapped = int(doa) % 360
    return int(round(doa_wrapped / binsize_deg) * binsize_deg) % 360


def _stft_4ch_np(wav4: np.ndarray, stft_cfg: dict[str, Any]) -> np.ndarray:
    if wav4.ndim != 2 or wav4.shape[0] != 4:
        raise ValueError(f"Expected wav shape (4,T), got {wav4.shape}")
    return np.stack(
        [
            stft(
                wav4[ch],
                n_fft=int(stft_cfg["n_fft"]),
                win_length=int(stft_cfg["win_length"]),
                hop_length=int(stft_cfg["hop_length"]),
                window=str(stft_cfg.get("window", "hann")),
                center=bool(stft_cfg.get("center", False)),
            )
            for ch in range(4)
        ],
        axis=-1,
    ).astype(np.complex64)


def _estimate_rtf_from_clean_ev(Xs: np.ndarray, ref_ch: int = 0, eps: float = 1e-8) -> np.ndarray:
    if Xs.ndim != 3:
        raise ValueError("Xs must be (F,T,C)")
    F, T, C = Xs.shape
    if T == 0:
        d = np.zeros((F, C), dtype=np.complex64)
        d[:, ref_ch] = 1.0 + 0.0j
        return d

    Rs = np.mean(Xs[:, :, :, None] * np.conjugate(Xs[:, :, None, :]), axis=1).astype(np.complex64)
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
    return d


def build_doa_rtf_library(
    train_root: str | Path,
    binsize_deg: int,
    stft_cfg: dict[str, Any],
    ref_ch: int = 0,
) -> dict[str, Any]:
    root = Path(train_root)
    accum: dict[int, list[np.ndarray]] = {}
    for clean_wav in sorted(root.glob("*/clean/*.wav")):
        src_doas, _ = parse_doas_from_filename(clean_wav.name)
        if len(src_doas) != 1:
            continue
        wav, sr = sf.read(str(clean_wav), always_2d=True, dtype="float32")
        if sr != 16000:
            raise ValueError(f"Expected sr=16000, got {sr} for {clean_wav}")
        d = _estimate_rtf_from_clean_ev(_stft_4ch_np(wav.T, stft_cfg), ref_ch=ref_ch)
        d_np = np.asarray(d, dtype=np.complex64)
        accum.setdefault(_doa_to_bin(src_doas[0], binsize_deg), []).append(d_np)

    if not accum:
        raise ValueError(f"No single-target clean samples found under: {root}")
    doa_bins = sorted(accum.keys())
    d_stack = np.stack([np.stack(accum[b], axis=0).mean(axis=0) for b in doa_bins], axis=0)
    d_stack[:, :, ref_ch] = np.complex64(1.0 + 0.0j)
    return {
        "doa_bins": np.asarray(doa_bins, np.int32),
        "d_mean": d_stack.astype(np.complex64),
        "binsize_deg": int(binsize_deg),
        "ref_ch": int(ref_ch),
    }


def load_rtf_lib(path: str | Path) -> dict[str, Any]:
    data = np.load(path)
    return {
        "doa_bins": data["doa_bins"],
        "d_mean": data["d_mean"],
        "binsize_deg": int(data["binsize_deg"]),
        "ref_ch": int(data["ref_ch"]),
        "path": str(path),
    }


def get_d_from_lib(rtf_lib: dict[str, Any], doa_deg: int) -> np.ndarray:
    binsize = int(rtf_lib["binsize_deg"])
    doa_bins = np.asarray(rtf_lib["doa_bins"], dtype=np.int32)
    d_mean = np.asarray(rtf_lib["d_mean"], dtype=np.complex64)
    target_bin = _doa_to_bin(doa_deg, binsize)
    idx = int(np.argmin(np.abs(((doa_bins - target_bin + 180) % 360) - 180)))
    d = d_mean[idx].copy()
    d[:, int(rtf_lib.get("ref_ch", 0))] = np.complex64(1.0 + 0.0j)
    return d
