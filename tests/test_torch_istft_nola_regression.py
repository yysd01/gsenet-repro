import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.dsp.torch_stft import istft, stft


@pytest.mark.parametrize("batch", [None, 2])
def test_istft_center_false_nola_regression(batch: int | None) -> None:
    rng = torch.Generator().manual_seed(0)
    length = 1600
    n_fft = 320
    win_length = 320
    hop_length = 160
    if batch is None:
        x = torch.randn(length, generator=rng, dtype=torch.float32)
    else:
        x = torch.randn(batch, length, generator=rng, dtype=torch.float32)

    X = stft(
        x,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=False,
    )
    x_hat = istft(
        X,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=False,
        length=length,
    )

    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()
    mse = torch.mean((x_hat - x) ** 2).item()
    assert mse < 1e-4
