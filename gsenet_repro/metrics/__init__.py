"""Metrics helpers."""

import importlib.util

from .metrics_pesq import pesq_available, pesq_score

__all__ = ["pesq_available", "pesq_score"]

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    from .metrics_torch import batch_sisdr, si_snr_db, sisdr, snr_db

    __all__ += ["batch_sisdr", "si_snr_db", "sisdr", "snr_db"]
