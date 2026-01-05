import numpy as np

from gsenet_repro.dsp import MODEL_STFT, istft, stft


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_roundtrip_random() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000, dtype=np.float32)
    X = stft(x, **MODEL_STFT, center=True)
    x_rec = istft(X, **MODEL_STFT, length=x.shape[0], center=True)
    error = rmse(x, x_rec)
    assert error < 1e-3


def test_roundtrip_sine() -> None:
    fs = 16000
    t = np.arange(fs, dtype=np.float32) / fs
    x = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    X = stft(x, **MODEL_STFT, center=True)
    x_rec = istft(X, **MODEL_STFT, length=x.shape[0], center=True)
    error = rmse(x, x_rec)
    assert error < 1e-3
