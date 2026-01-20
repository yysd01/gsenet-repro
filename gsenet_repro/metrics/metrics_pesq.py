"""Optional PESQ wrapper."""
from __future__ import annotations

import importlib.util
from typing import Optional

import numpy as np


def _load_pesq() -> Optional[object]:
    if importlib.util.find_spec("pesq") is None:
        return None
    from pesq import pesq as pesq_fn

    return pesq_fn


def pesq_available() -> bool:
    return _load_pesq() is not None


def pesq_score(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 16000,
    mode: str = "wb",
) -> float:
    """Compute PESQ score (optional dependency)."""
    pesq_fn = _load_pesq()
    if pesq_fn is None:
        raise RuntimeError("pesq package is not installed")
    ref = np.asarray(reference, dtype=np.float32)
    est = np.asarray(estimate, dtype=np.float32)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have the same shape")
    if ref.ndim != 1:
        raise ValueError("reference and estimate must be 1D arrays")
    return float(pesq_fn(sample_rate, ref, est, mode))
