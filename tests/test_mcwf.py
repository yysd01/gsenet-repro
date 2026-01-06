import importlib.util

import numpy as np
import pytest

from gsenet_repro.dsp.mcwf import mcwf


def _windowed_power(input_stft: np.ndarray, window_len: int = 4) -> np.ndarray:
    power = np.abs(input_stft).astype(np.float32) ** 2
    power_pad = np.pad(power, ((0, 0), (0, 0), (window_len - 1, 0), (0, 0)))
    cumulative = np.cumsum(power_pad, axis=2, dtype=np.float32)
    cumulative = np.pad(cumulative, ((0, 0), (0, 0), (1, 0), (0, 0)), mode="constant")
    window_sum = cumulative[:, :, window_len:, :] - cumulative[:, :, :-window_len, :]
    return window_sum / float(window_len)


def test_mcwf_shape_and_non_negative():
    rng = np.random.default_rng(0)
    input_stft = rng.normal(size=(2, 4, 6, 3)) + 1j * rng.normal(size=(2, 4, 6, 3))
    output = mcwf(
        input_stft,
        stft_win_length=320,
        stft_hop_size=160,
        noise_pow=0.1,
        signal_pow=1.0,
    )

    assert output.shape == input_stft.shape
    assert np.all(output >= 0.0)


def test_mcwf_gain_changes_with_snr():
    input_stft = np.ones((1, 2, 5, 3), dtype=np.complex64)
    low_noise = mcwf(
        input_stft,
        stft_win_length=320,
        stft_hop_size=160,
        noise_pow=0.1,
        signal_pow=1.0,
    )
    high_noise = mcwf(
        input_stft,
        stft_win_length=320,
        stft_hop_size=160,
        noise_pow=2.0,
        signal_pow=1.0,
    )

    assert np.mean(low_noise) > np.mean(high_noise)


def test_mcwf_output_tracks_input_power():
    rng = np.random.default_rng(1)
    input_stft = rng.normal(size=(1, 3, 8, 3)) + 1j * rng.normal(size=(1, 3, 8, 3))
    output = mcwf(
        input_stft,
        stft_win_length=320,
        stft_hop_size=160,
        noise_pow=0.2,
        signal_pow=1.5,
    )

    expected_power = _windowed_power(input_stft)
    gain = 1.5 / (1.5 + 0.2)

    assert np.allclose(output, expected_power * gain, atol=1e-6)


def test_mcwf_gain_controls_gsenet_frontend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")

    import torch

    from gsenet_repro.models.gsenet_torch import MinimalGSENet
    from gsenet_repro.dsp.torch_stft import torch_stft

    torch.manual_seed(0)

    model = MinimalGSENet(mcwf_gain_min=0.0, mcwf_gain_max=1.0, mcwf_noise_scale=1.0)

    y = torch.randn(2, 512)
    X0 = torch_stft(y, **model.stft_params, center=False)
    X1 = torch_stft(y * 0.9, **model.stft_params, center=False)
    X2 = torch_stft(y * 1.1, **model.stft_params, center=False)

    X0_low, X1_low = model._apply_mcwf(X0, X1, X2, noise_level=torch.tensor([0.5, 0.5]))
    X0_high, X1_high = model._apply_mcwf(X0, X1, X2, noise_level=torch.tensor([2.0, 2.0]))

    assert torch.mean(torch.abs(X0_low)) > torch.mean(torch.abs(X0_high))
    assert torch.mean(torch.abs(X1_low)) > torch.mean(torch.abs(X1_high))
