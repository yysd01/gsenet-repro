import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.models.gsenet_torch import MinimalGSENet


def test_torch_model_forward_backward() -> None:
    torch.manual_seed(0)
    model = MinimalGSENet()
    y0 = torch.randn(2, 512)
    y1 = torch.randn(2, 512)

    y_hat = model(y0, y1)
    assert y_hat.shape == y0.shape

    loss = y_hat.mean()
    loss.backward()

    assert any(param.grad is not None for param in model.parameters())


def test_torch_model_shape_mismatch() -> None:
    model = MinimalGSENet()
    y0 = torch.randn(2, 512)
    y1 = torch.randn(2, 256)

    with pytest.raises(ValueError):
        _ = model(y0, y1)
