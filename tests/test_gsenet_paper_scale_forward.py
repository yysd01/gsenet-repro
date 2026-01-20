import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.models.gsenet_paper_torch import GSENetPaperScale


def test_gsenet_paper_scale_forward_shape() -> None:
    torch.manual_seed(0)
    model = GSENetPaperScale()
    y0 = torch.randn(2, 16000)
    y1 = torch.randn(2, 16000)

    y_hat = model(y0, y1)
    assert y_hat.shape == y0.shape
    assert torch.isfinite(y_hat).all()


def test_gsenet_paper_scale_causal_prefix() -> None:
    torch.manual_seed(1)
    model = GSENetPaperScale()
    model.eval()
    length = 16000
    change_idx = 12000

    y0 = torch.randn(2, length)
    y1 = torch.randn(2, length)
    y0_changed = y0.clone()
    y1_changed = y1.clone()
    y0_changed[:, change_idx:] += 0.5
    y1_changed[:, change_idx:] -= 0.25

    with torch.no_grad():
        y_hat = model(y0, y1)
        y_hat_changed = model(y0_changed, y1_changed)

    win_length = int(model.stft_params["win_length"])
    compare_end = change_idx - win_length
    if compare_end <= 0:
        pytest.skip("Safe prefix is empty for the selected change index.")

    torch.testing.assert_close(
        y_hat[:, :compare_end],
        y_hat_changed[:, :compare_end],
        rtol=1e-4,
        atol=1e-4,
    )
