import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.dsp.torch_stft import istft


def test_istft_center_false_tail_not_near_zero() -> None:
    torch.manual_seed(0)

    batch = 2
    n_fft = 320
    freq_bins = n_fft // 2 + 1
    frames = 19

    real = torch.randn(batch, freq_bins, frames)
    imag = torch.randn(batch, freq_bins, frames)
    X = real + 1j * imag

    y = istft(
        X,
        n_fft=n_fft,
        win_length=320,
        hop_length=160,
        center=False,
        length=3200,
    )

    front = torch.mean(torch.abs(y[:, :1280]))
    back = torch.mean(torch.abs(y[:, 1280:2880]))

    assert (back / front).item() > 0.05
