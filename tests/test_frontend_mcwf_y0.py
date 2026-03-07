import numpy as np

from gsenet_repro.pipeline.mcwf_frontend import mcwf_make_y0


def test_mcwf_frontend_output_shape() -> None:
    rng = np.random.default_rng(0)
    x_mics = rng.normal(size=(4, 512)).astype(np.float32)
    y0 = mcwf_make_y0(x_mics, stft_params={"n_fft": 320, "win_length": 320, "hop_length": 160})
    assert y0.shape == (512,)
    assert np.isfinite(y0).all()


def test_mcwf_frontend_batch_shape() -> None:
    rng = np.random.default_rng(1)
    x_mics = rng.normal(size=(2, 4, 400)).astype(np.float32)
    y0 = mcwf_make_y0(x_mics, stft_params={"n_fft": 320, "win_length": 320, "hop_length": 160})
    assert y0.shape == (2, 400)
    assert np.isfinite(y0).all()
