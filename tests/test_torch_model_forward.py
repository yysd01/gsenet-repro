import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.models.gsenet_torch import GSENetTorch


def test_torch_model_forward_shape_and_dtype() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    batch = 2
    length = 16000
    y0 = torch.from_numpy(rng.normal(size=(batch, length)).astype(np.float32))
    y1 = torch.from_numpy(rng.normal(size=(batch, length)).astype(np.float32))

    model = GSENetTorch()
    y_hat = model(y0, y1)

    assert y_hat.shape == y0.shape
    assert y_hat.dtype == torch.float32
    assert torch.isfinite(y_hat).all()
