import numpy as np

from gsenet_repro.losses import stft_reconstruction_loss


def test_loss_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8000, dtype=np.float32)
    loss = stft_reconstruction_loss(x, x)
    assert loss < 1e-6


def test_loss_positive() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8000, dtype=np.float32)
    noise = 0.01 * rng.standard_normal(8000, dtype=np.float32)
    y = x + noise
    loss = stft_reconstruction_loss(y, x)
    assert loss > 0.0


def test_loss_batch() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 4000), dtype=np.float32)
    y = x + 0.02 * rng.standard_normal((2, 4000), dtype=np.float32)
    loss = stft_reconstruction_loss(y, x)
    assert loss > 0.0
