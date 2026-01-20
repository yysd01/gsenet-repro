from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from gsenet_repro.metrics.metrics_pesq import pesq_score


@pytest.mark.skipif(importlib.util.find_spec("pesq") is None, reason="pesq not installed")
def test_pesq_range() -> None:
    sample_rate = 16000
    t = np.linspace(0, 0.25, int(0.25 * sample_rate), endpoint=False)
    ref = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    est = ref + 0.01 * np.random.RandomState(0).randn(ref.size).astype(np.float32)
    score = pesq_score(ref, est, sample_rate=sample_rate)
    assert 0.0 <= score <= 5.0
