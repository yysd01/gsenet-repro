import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.models.gsenet_torch import GSENetTorch


def test_torch_model_causality() -> None:
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    batch = 1
    length = 24000

    y0 = torch.from_numpy(rng.normal(size=(batch, length)).astype(np.float32))
    y1 = torch.from_numpy(rng.normal(size=(batch, length)).astype(np.float32))
    y0_trunc = y0.clone()
    y1_trunc = y1.clone()
    cutoff = int(length * 2 / 3)
    y0_trunc[:, cutoff:] = 0.0
    y1_trunc[:, cutoff:] = 0.0

    model = GSENetTorch()
    model.eval()
    with torch.no_grad():
        y_hat_full = model(y0, y1)
        y_hat_trunc = model(y0_trunc, y1_trunc)

    # Compare early region well before the truncated area to avoid window overlap effects.
    compare_end = int(length * 0.4)
    diff = y_hat_full[:, :compare_end] - y_hat_trunc[:, :compare_end]
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    assert rmse < 1e-4
