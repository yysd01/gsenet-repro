import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.dsp import MODEL_STFT
from gsenet_repro.dsp.torch_stft import istft, stft


def test_torch_roundtrip() -> None:
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(16000, generator=rng, dtype=torch.float32)
    X = stft(x, **MODEL_STFT, center=True)
    x_rec = istft(X, **MODEL_STFT, length=x.shape[0], center=True)
    torch.testing.assert_close(x_rec, x, rtol=1e-3, atol=2e-4)
