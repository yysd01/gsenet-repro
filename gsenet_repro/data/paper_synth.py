from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class PaperParams:
    """Sampled parameters aligned with Table 1 (GSENet row)."""

    gn_lin: float
    gi_lin: float
    pi: float
    alpha_lin: float
    beta_lin: float
    gn_db: float
    gi_db: float
    alpha_db: float
    beta_db: float
    global_gain: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "gn_lin": self.gn_lin,
            "gi_lin": self.gi_lin,
            "pi": self.pi,
            "alpha_lin": self.alpha_lin,
            "beta_lin": self.beta_lin,
            "gn_db": self.gn_db,
            "gi_db": self.gi_db,
            "alpha_db": self.alpha_db,
            "beta_db": self.beta_db,
            "global_gain": self.global_gain,
        }


def db_to_lin(db: np.ndarray | float) -> np.ndarray | float:
    """Convert decibel scale to linear amplitude (20*log10)."""
    return 10 ** (np.asarray(db) / 20.0)


def lin_to_db(lin: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    """Convert linear amplitude to decibel scale (20*log10)."""
    lin_safe = np.maximum(np.asarray(lin), eps)
    return 20.0 * np.log10(lin_safe)


def sample_paper_params(
    rng: np.random.Generator,
    variant: str = "gsenet",
    global_gain: float = 1.0,
) -> PaperParams:
    """Sample gain parameters following Table 1 (GSENet row).

    Reference: arXiv:2303.07486v1, Table 1 (GSENet) and Section 2.1.
    """
    if variant != "gsenet":
        raise ValueError(f"Unsupported variant '{variant}'. Only 'gsenet' is implemented.")

    gn_db = float(rng.normal(loc=-5.0, scale=10.0))
    gi_db = float(rng.normal(loc=-3.0, scale=3.0))
    alpha_db = float(max(rng.normal(loc=0.0, scale=3.0), -4.0))
    beta_db = float(max(rng.normal(loc=4.0, scale=6.0), 4.0))
    pi = float(rng.random() < 0.4)

    return PaperParams(
        gn_lin=float(db_to_lin(gn_db)),
        gi_lin=float(db_to_lin(gi_db)),
        pi=pi,
        alpha_lin=float(db_to_lin(alpha_db)),
        beta_lin=float(db_to_lin(beta_db)),
        gn_db=gn_db,
        gi_db=gi_db,
        alpha_db=alpha_db,
        beta_db=beta_db,
        global_gain=float(global_gain),
    )


def normalize_rms(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize waveform or RIR by RMS (power normalization approximation)."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    rms = np.sqrt(np.mean(x**2))
    return (x / (rms + eps)).astype(np.float32)


def _make_rir(
    rng: np.random.Generator,
    length: int,
    direct_delay: int,
    early_reflections: int,
    max_early_delay: int,
    tail_decay: float,
    tail_level: float,
) -> np.ndarray:
    rir = np.zeros(length, dtype=np.float32)
    direct_delay = int(np.clip(direct_delay, 0, length - 1))
    rir[direct_delay] = 1.0

    for idx in range(early_reflections):
        delay = direct_delay + int(rng.integers(1, max_early_delay + 1))
        if delay >= length:
            break
        amp = rng.uniform(0.1, 0.5) * (0.7**idx)
        sign = -1.0 if rng.random() < 0.5 else 1.0
        rir[delay] += sign * amp

    tail_start = min(length - 1, direct_delay + max_early_delay)
    if tail_start < length - 1:
        t = np.arange(tail_start, length)
        decay = np.exp(-tail_decay * (t - tail_start))
        noise = rng.normal(scale=tail_level, size=t.shape)
        rir[tail_start:] += (decay * noise).astype(np.float32)

    return rir


def generate_rir_3src_2mic(
    rng: np.random.Generator,
    rir_length: int = 1024,
    fs: int = 16000,
    max_direct_delay_diff: int = 4,
    early_reflections: int = 3,
    max_early_delay: int = 80,
    tail_decay: float = 0.02,
    tail_level: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate lightweight RIRs for 3 sources and 2 close microphones.

    Reference: arXiv:2303.07486v1, Section 2.1.
    The two receivers are placed close enough so their direct-path delay
    difference stays below ``max_direct_delay_diff`` samples.
    """
    _ = fs  # placeholder for future physical parameterization

    rir = np.zeros((3, 2, rir_length), dtype=np.float32)
    direct_delays = np.zeros((3, 2), dtype=np.int32)

    for src_idx in range(3):
        base_delay = int(rng.integers(20, 80))
        for mic_idx in range(2):
            offset = int(rng.integers(-max_direct_delay_diff, max_direct_delay_diff + 1))
            delay = max(0, base_delay + offset)
            direct_delays[src_idx, mic_idx] = delay
            rir[src_idx, mic_idx] = _make_rir(
                rng,
                length=rir_length,
                direct_delay=delay,
                early_reflections=early_reflections,
                max_early_delay=max_early_delay,
                tail_decay=tail_decay,
                tail_level=tail_level,
            )

    rir = np.stack(
        [normalize_rms(rir[src, mic]) for src in range(3) for mic in range(2)],
        axis=0,
    ).reshape(3, 2, rir_length)

    max_delay = int(np.max(direct_delays))
    anechoic_length = max_delay + 1
    rir_anechoic = np.zeros((3, 2, anechoic_length), dtype=np.float32)
    for src_idx in range(3):
        for mic_idx in range(2):
            delay = direct_delays[src_idx, mic_idx]
            rir_anechoic[src_idx, mic_idx, delay] = 1.0
            rir_anechoic[src_idx, mic_idx] = normalize_rms(rir_anechoic[src_idx, mic_idx])

    return rir.astype(np.float32), rir_anechoic.astype(np.float32)


def generate_rir_3src_3mic(
    rng: np.random.Generator,
    rir_length: int = 1024,
    fs: int = 16000,
    max_direct_delay_diff: int = 4,
    early_reflections: int = 3,
    max_early_delay: int = 80,
    tail_decay: float = 0.02,
    tail_level: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate lightweight RIRs for 3 sources and 3 close microphones."""
    _ = fs

    rir = np.zeros((3, 3, rir_length), dtype=np.float32)
    direct_delays = np.zeros((3, 3), dtype=np.int32)

    for src_idx in range(3):
        base_delay = int(rng.integers(20, 80))
        for mic_idx in range(3):
            offset = int(rng.integers(-max_direct_delay_diff, max_direct_delay_diff + 1))
            delay = max(0, base_delay + offset)
            direct_delays[src_idx, mic_idx] = delay
            rir[src_idx, mic_idx] = _make_rir(
                rng,
                length=rir_length,
                direct_delay=delay,
                early_reflections=early_reflections,
                max_early_delay=max_early_delay,
                tail_decay=tail_decay,
                tail_level=tail_level,
            )

    rir = np.stack(
        [normalize_rms(rir[src, mic]) for src in range(3) for mic in range(3)],
        axis=0,
    ).reshape(3, 3, rir_length)

    max_delay = int(np.max(direct_delays))
    anechoic_length = max_delay + 1
    rir_anechoic = np.zeros((3, 3, anechoic_length), dtype=np.float32)
    for src_idx in range(3):
        for mic_idx in range(3):
            delay = direct_delays[src_idx, mic_idx]
            rir_anechoic[src_idx, mic_idx, delay] = 1.0
            rir_anechoic[src_idx, mic_idx] = normalize_rms(rir_anechoic[src_idx, mic_idx])

    return rir.astype(np.float32), rir_anechoic.astype(np.float32)


def _normalize_params(params: Optional[PaperParams | Dict[str, Any]]) -> PaperParams:
    if params is None:
        raise ValueError("params must be provided for synthesis.")
    if isinstance(params, PaperParams):
        return params
    required = {"gn_lin", "gi_lin", "pi", "alpha_lin", "beta_lin"}
    missing = required.difference(params.keys())
    if missing:
        raise ValueError(f"params is missing keys: {sorted(missing)}")
    gn_lin = float(params["gn_lin"])
    gi_lin = float(params["gi_lin"])
    pi = float(params["pi"])
    alpha_lin = float(params["alpha_lin"])
    beta_lin = float(params["beta_lin"])
    return PaperParams(
        gn_lin=gn_lin,
        gi_lin=gi_lin,
        pi=pi,
        alpha_lin=alpha_lin,
        beta_lin=beta_lin,
        gn_db=float(params.get("gn_db", lin_to_db(gn_lin))),
        gi_db=float(params.get("gi_db", lin_to_db(gi_lin))),
        alpha_db=float(params.get("alpha_db", lin_to_db(alpha_lin))),
        beta_db=float(params.get("beta_db", lin_to_db(beta_lin))),
        global_gain=float(params.get("global_gain", 1.0)),
    )


def _fftconvolve_truncate(signal: np.ndarray, kernel: np.ndarray, length: int) -> np.ndarray:
    out = fftconvolve(signal, kernel, mode="full")
    return out[:length].astype(np.float32)


def synthesize_y0_y1_yt(
    s: np.ndarray,
    n: np.ndarray,
    i: np.ndarray,
    rir: np.ndarray,
    rir_anechoic: np.ndarray,
    params: PaperParams | Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize y0/y1/yt as defined in Section 2.1 of the paper.

    y0 = s * r(0,0) + gn * n * r(1,0) + pi * gi * i * r(2,0)
    y1 = s * r(0,1) + alpha * gn * n * r(1,1) + beta * pi * gi * i * r(2,1)
    yt = s * r_anechoic(0,0) (anechoic only keeps the strongest path)
    """
    params = _normalize_params(params)

    s = normalize_rms(np.asarray(s, dtype=np.float32))
    n = normalize_rms(np.asarray(n, dtype=np.float32))
    i = normalize_rms(np.asarray(i, dtype=np.float32))

    rir = np.asarray(rir, dtype=np.float32)
    rir_anechoic = np.asarray(rir_anechoic, dtype=np.float32)
    if rir.shape[:2] != (3, 2):
        raise ValueError("rir must have shape (3, 2, L)")
    if rir_anechoic.shape[:2] != (3, 2):
        raise ValueError("rir_anechoic must have shape (3, 2, L_anechoic)")

    length = s.shape[0]

    rir_norm = np.stack(
        [normalize_rms(rir[src, mic]) for src in range(3) for mic in range(2)],
        axis=0,
    ).reshape(3, 2, -1)
    rir_anechoic_norm = np.stack(
        [normalize_rms(rir_anechoic[src, mic]) for src in range(3) for mic in range(2)],
        axis=0,
    ).reshape(3, 2, -1)

    y0 = _fftconvolve_truncate(s, rir_norm[0, 0], length)
    y0 += params.gn_lin * _fftconvolve_truncate(n, rir_norm[1, 0], length)
    y0 += params.pi * params.gi_lin * _fftconvolve_truncate(i, rir_norm[2, 0], length)

    y1 = _fftconvolve_truncate(s, rir_norm[0, 1], length)
    y1 += params.alpha_lin * params.gn_lin * _fftconvolve_truncate(n, rir_norm[1, 1], length)
    y1 += params.beta_lin * params.pi * params.gi_lin * _fftconvolve_truncate(i, rir_norm[2, 1], length)

    anechoic = rir_anechoic_norm[0, 0]
    k_star = int(np.argmax(np.abs(anechoic)))
    h_main = np.zeros_like(anechoic)
    h_main[k_star] = anechoic[k_star]
    yt = _fftconvolve_truncate(s, h_main, length)

    if params.global_gain != 1.0:
        y0 *= params.global_gain
        y1 *= params.global_gain
        yt *= params.global_gain

    return y0.astype(np.float32), y1.astype(np.float32), yt.astype(np.float32)


def synthesize_y0_y1_y2_yt(
    s: np.ndarray,
    n: np.ndarray,
    i: np.ndarray,
    rir: np.ndarray,
    rir_anechoic: np.ndarray,
    params: PaperParams | Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize y0/y1/y2/yt for 3-microphone mixtures."""
    params = _normalize_params(params)

    s = normalize_rms(np.asarray(s, dtype=np.float32))
    n = normalize_rms(np.asarray(n, dtype=np.float32))
    i = normalize_rms(np.asarray(i, dtype=np.float32))

    rir = np.asarray(rir, dtype=np.float32)
    rir_anechoic = np.asarray(rir_anechoic, dtype=np.float32)
    if rir.shape[:2] != (3, 3):
        raise ValueError("rir must have shape (3, 3, L)")
    if rir_anechoic.shape[:2] != (3, 3):
        raise ValueError("rir_anechoic must have shape (3, 3, L_anechoic)")

    length = s.shape[0]

    rir_norm = np.stack(
        [normalize_rms(rir[src, mic]) for src in range(3) for mic in range(3)],
        axis=0,
    ).reshape(3, 3, -1)
    rir_anechoic_norm = np.stack(
        [normalize_rms(rir_anechoic[src, mic]) for src in range(3) for mic in range(3)],
        axis=0,
    ).reshape(3, 3, -1)

    y0 = _fftconvolve_truncate(s, rir_norm[0, 0], length)
    y0 += params.gn_lin * _fftconvolve_truncate(n, rir_norm[1, 0], length)
    y0 += params.pi * params.gi_lin * _fftconvolve_truncate(i, rir_norm[2, 0], length)

    y1 = _fftconvolve_truncate(s, rir_norm[0, 1], length)
    y1 += params.alpha_lin * params.gn_lin * _fftconvolve_truncate(n, rir_norm[1, 1], length)
    y1 += params.beta_lin * params.pi * params.gi_lin * _fftconvolve_truncate(i, rir_norm[2, 1], length)

    y2 = _fftconvolve_truncate(s, rir_norm[0, 2], length)
    y2 += params.alpha_lin * params.gn_lin * _fftconvolve_truncate(n, rir_norm[1, 2], length)
    y2 += params.beta_lin * params.pi * params.gi_lin * _fftconvolve_truncate(i, rir_norm[2, 2], length)

    anechoic = rir_anechoic_norm[0, 0]
    k_star = int(np.argmax(np.abs(anechoic)))
    h_main = np.zeros_like(anechoic)
    h_main[k_star] = anechoic[k_star]
    yt = _fftconvolve_truncate(s, h_main, length)

    if params.global_gain != 1.0:
        y0 *= params.global_gain
        y1 *= params.global_gain
        y2 *= params.global_gain
        yt *= params.global_gain

    return (
        y0.astype(np.float32),
        y1.astype(np.float32),
        y2.astype(np.float32),
        yt.astype(np.float32),
    )
