from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from gsenet_repro.dsp.stft import stft

REQUIRED_METADATA_FIELDS = (
    "sample_rate",
    "n_fft",
    "win_length",
    "hop_length",
    "window",
    "center",
    "num_mics",
    "ref_ch",
    "binsize_deg",
)


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


def save_rtf_lib(path: str | Path, lib: dict[str, Any]) -> None:
    path = Path(path)
    payload: dict[str, Any] = {
        "doa_bins": np.asarray(lib["doa_bins"], np.int32),
        "d_mean": np.asarray(lib["d_mean"], np.complex64),
    }
    for key in REQUIRED_METADATA_FIELDS:
        if key not in lib:
            raise ValueError(f"Missing required rtf_lib metadata: {key}")
    payload["sample_rate"] = np.int32(lib["sample_rate"])
    payload["n_fft"] = np.int32(lib["n_fft"])
    payload["win_length"] = np.int32(lib["win_length"])
    payload["hop_length"] = np.int32(lib["hop_length"])
    payload["window"] = np.asarray(str(lib["window"]))
    payload["center"] = np.bool_(lib["center"])
    payload["num_mics"] = np.int32(lib["num_mics"])
    payload["ref_ch"] = np.int32(lib["ref_ch"])
    payload["binsize_deg"] = np.int32(lib["binsize_deg"])
    np.savez(path, **payload)


def build_doa_rtf_library(
    train_root: str | Path,
    binsize_deg: int,
    stft_cfg: dict[str, Any],
    ref_ch: int = 0,
) -> dict[str, Any]:
    root = Path(train_root)
    accum: dict[int, list[np.ndarray]] = {}
    sample_rate: int | None = None
    for clean_wav in sorted(root.glob("*/clean/*.wav")):
        src_doas, _ = parse_doas_from_filename(clean_wav.name)
        if len(src_doas) != 1:
            continue
        wav, sr = sf.read(str(clean_wav), always_2d=True, dtype="float32")
        if sample_rate is None:
            sample_rate = int(sr)
        elif int(sr) != sample_rate:
            raise ValueError(f"Inconsistent sample rates in {root}: got {sr} and {sample_rate}")
        d = _estimate_rtf_from_clean_ev(_stft_4ch_np(wav.T, stft_cfg), ref_ch=ref_ch)
        d_np = np.asarray(d, dtype=np.complex64)
        accum.setdefault(_doa_to_bin(src_doas[0], binsize_deg), []).append(d_np)

    if not accum:
        raise ValueError(f"No single-target clean samples found under: {root}")
    if sample_rate is None:
        raise ValueError(f"Failed to determine sample rate under: {root}")
    doa_bins = sorted(accum.keys())
    d_stack = np.stack([np.stack(accum[b], axis=0).mean(axis=0) for b in doa_bins], axis=0)
    d_stack[:, :, ref_ch] = np.complex64(1.0 + 0.0j)
    return {
        "doa_bins": np.asarray(doa_bins, np.int32),
        "d_mean": d_stack.astype(np.complex64),
        "sample_rate": int(sample_rate),
        "n_fft": int(stft_cfg["n_fft"]),
        "win_length": int(stft_cfg["win_length"]),
        "hop_length": int(stft_cfg["hop_length"]),
        "window": str(stft_cfg.get("window", "hann")),
        "center": bool(stft_cfg.get("center", False)),
        "num_mics": int(d_stack.shape[-1]),
        "binsize_deg": int(binsize_deg),
        "ref_ch": int(ref_ch),
    }


def load_rtf_lib(path: str | Path) -> dict[str, Any]:
    data = np.load(path)
    payload: dict[str, Any] = {
        "doa_bins": data["doa_bins"],
        "d_mean": data["d_mean"],
        "path": str(path),
    }
    missing = [key for key in REQUIRED_METADATA_FIELDS if key not in data]
    if missing:
        warnings.warn(
            f"rtf_lib missing metadata {missing}; please rebuild before online streaming.",
            RuntimeWarning,
            stacklevel=2,
        )
        payload["missing_metadata"] = tuple(missing)
        for key in REQUIRED_METADATA_FIELDS:
            payload.setdefault(key, None)
        return payload

    payload.update(
        {
            "sample_rate": int(data["sample_rate"]),
            "n_fft": int(data["n_fft"]),
            "win_length": int(data["win_length"]),
            "hop_length": int(data["hop_length"]),
            "window": str(data["window"].item() if np.asarray(data["window"]).shape == () else data["window"]),
            "center": bool(data["center"]),
            "num_mics": int(data["num_mics"]),
            "ref_ch": int(data["ref_ch"]),
            "binsize_deg": int(data["binsize_deg"]),
            "missing_metadata": tuple(),
        }
    )
    return payload


def get_d_from_lib(rtf_lib: dict[str, Any], doa_deg: int) -> np.ndarray:
    binsize = int(rtf_lib["binsize_deg"])
    doa_bins = np.asarray(rtf_lib["doa_bins"], dtype=np.int32)
    d_mean = np.asarray(rtf_lib["d_mean"], dtype=np.complex64)
    target_bin = _doa_to_bin(doa_deg, binsize)
    idx = int(np.argmin(np.abs(((doa_bins - target_bin + 180) % 360) - 180)))
    d = d_mean[idx].copy()
    d[:, int(rtf_lib.get("ref_ch", 0))] = np.complex64(1.0 + 0.0j)
    return d
