import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.dsp.torch_stft import istft, stft


def test_torch_stft_safe_prefix_causality() -> None:
    n_fft = 256
    win_length = 256
    hop_length = 128
    change_idx = 1024

    x = torch.zeros(2048, dtype=torch.float32)
    x[256:768] = 0.5
    x_changed = x.clone()
    x_changed[change_idx:] += 0.25

    X = stft(x, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    X_changed = stft(
        x_changed, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False
    )
    y = istft(X, n_fft=n_fft, win_length=win_length, hop_length=hop_length, length=x.shape[0])
    y_changed = istft(
        X_changed, n_fft=n_fft, win_length=win_length, hop_length=hop_length, length=x.shape[0]
    )

    compare_end = change_idx - win_length
    if compare_end <= 0:
        pytest.skip("Safe prefix is empty for the selected change index.")

    torch.testing.assert_close(
        y[:compare_end],
        y_changed[:compare_end],
        rtol=1e-4,
        atol=1e-4,
    )
