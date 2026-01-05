from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)

from gsenet_repro.dsp import MODEL_STFT, istft, stft


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    rng = np.random.default_rng(0)
    fs = 16000
    x = rng.standard_normal(fs, dtype=np.float32)
    X = stft(x, **MODEL_STFT, center=True)
    x_rec = istft(X, **MODEL_STFT, length=x.shape[0], center=True)
    error = rmse(x, x_rec)
    print(f"RMSE: {error:.6f}")
    assert error < 1e-3, f"RMSE too high: {error}"
    print("OK")


if __name__ == "__main__":
    main()
