"""Core utilities for GSENet reproduction scaffolding."""
from __future__ import annotations

import numpy as np


def fft_roundtrip(signal: np.ndarray) -> np.ndarray:
    """Perform an FFT/iFFT round-trip to validate DSP plumbing."""
    spectrum = np.fft.fft(signal)
    return np.fft.ifft(spectrum)


__all__ = ["fft_roundtrip"]
