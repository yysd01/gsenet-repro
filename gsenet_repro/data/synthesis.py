from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class Gains:
    gn: float
    gi: float
    alpha: float
    beta: float
    pi: float
    gn_db: float
    gi_db: float
    alpha_db: float
    beta_db: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "gn": self.gn,
            "gi": self.gi,
            "alpha": self.alpha,
            "beta": self.beta,
            "pi": self.pi,
            "gn_db": self.gn_db,
            "gi_db": self.gi_db,
            "alpha_db": self.alpha_db,
            "beta_db": self.beta_db,
        }


def db_to_amp(db: np.ndarray | float) -> np.ndarray | float:
    return 10 ** (np.asarray(db) / 20.0)


def amp_to_db(amp: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    amp_safe = np.maximum(np.asarray(amp), eps)
    return 20.0 * np.log10(amp_safe)


def sample_gains(rng: np.random.Generator) -> Gains:
    gn_db = rng.normal(loc=-5.0, scale=10.0)
    gi_db = rng.normal(loc=-3.0, scale=3.0)
    alpha_db = max(rng.normal(loc=0.0, scale=3.0), -4.0)
    beta_db = max(rng.normal(loc=4.0, scale=6.0), 4.0)
    pi = float(rng.random() < 0.4)
    return Gains(
        gn=float(db_to_amp(gn_db)),
        gi=float(db_to_amp(gi_db)),
        alpha=float(db_to_amp(alpha_db)),
        beta=float(db_to_amp(beta_db)),
        pi=pi,
        gn_db=float(gn_db),
        gi_db=float(gi_db),
        alpha_db=float(alpha_db),
        beta_db=float(beta_db),
    )


def _normalize_gains(gains: Optional[Gains | Dict[str, Any]]) -> Gains:
    if gains is None:
        raise ValueError("gains must be provided for normalization.")
    if isinstance(gains, Gains):
        return gains
    required = {"gn", "gi", "alpha", "beta", "pi"}
    missing = required.difference(gains.keys())
    if missing:
        raise ValueError(f"gains is missing keys: {sorted(missing)}")
    gn = float(gains["gn"])
    gi = float(gains["gi"])
    alpha = float(gains["alpha"])
    beta = float(gains["beta"])
    pi = float(gains["pi"])
    return Gains(
        gn=gn,
        gi=gi,
        alpha=alpha,
        beta=beta,
        pi=pi,
        gn_db=float(gains.get("gn_db", amp_to_db(gn))),
        gi_db=float(gains.get("gi_db", amp_to_db(gi))),
        alpha_db=float(gains.get("alpha_db", amp_to_db(alpha))),
        beta_db=float(gains.get("beta_db", amp_to_db(beta))),
    )


def _fftconvolve_truncate(signal: np.ndarray, kernel: np.ndarray, length: int) -> np.ndarray:
    out = fftconvolve(signal, kernel, mode="full")
    return out[:length].astype(np.float32)


def _extract_anechoic_target(rir_anechoic: np.ndarray) -> np.ndarray:
    if rir_anechoic.ndim == 1:
        return rir_anechoic
    if rir_anechoic.ndim == 2:
        return rir_anechoic[0]
    return rir_anechoic[0, 0]


def synthesize_pair(
    s: np.ndarray,
    n: np.ndarray,
    i: np.ndarray,
    rir: np.ndarray,
    rir_anechoic: np.ndarray,
    rng: np.random.Generator,
    gains: Optional[Gains | Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if gains is None:
        gains = sample_gains(rng)
    gains = _normalize_gains(gains)

    s = np.asarray(s, dtype=np.float32)
    n = np.asarray(n, dtype=np.float32)
    i = np.asarray(i, dtype=np.float32)
    length = len(s)

    y0 = _fftconvolve_truncate(s, rir[0, 0], length)
    y0 += gains.gn * _fftconvolve_truncate(n, rir[1, 0], length)
    y0 += gains.pi * gains.gi * _fftconvolve_truncate(i, rir[2, 0], length)

    y1 = _fftconvolve_truncate(s, rir[0, 1], length)
    y1 += gains.alpha * gains.gn * _fftconvolve_truncate(n, rir[1, 1], length)
    y1 += gains.beta * gains.pi * gains.gi * _fftconvolve_truncate(i, rir[2, 1], length)

    anechoic = _extract_anechoic_target(rir_anechoic)
    k_star = int(np.argmax(np.abs(anechoic)))
    h_main = np.zeros_like(anechoic)
    h_main[k_star] = anechoic[k_star]
    yt = _fftconvolve_truncate(s, h_main, length)

    meta = {
        "length_out": length,
        "length_rule": "truncate_to_source",
        "k_star": k_star,
        "gains": gains.as_dict(),
    }
    return y0.astype(np.float32), y1.astype(np.float32), yt.astype(np.float32), meta
