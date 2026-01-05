import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.losses.stft_loss_torch import stft_magnitude_loss


def test_torch_loss_zero_for_identical_signals() -> None:
    torch.manual_seed(0)
    y = torch.randn(2, 1024)
    loss = stft_magnitude_loss(y, y)
    assert loss.item() < 1e-6


def test_torch_loss_increases_with_noise() -> None:
    torch.manual_seed(1)
    y = torch.randn(2, 1024)
    noisy = y + 0.1 * torch.randn_like(y)

    loss_clean = stft_magnitude_loss(y, y)
    loss_noisy = stft_magnitude_loss(noisy, y)

    assert loss_noisy > loss_clean + 1e-4
