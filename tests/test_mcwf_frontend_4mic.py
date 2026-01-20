import numpy as np

from gsenet_repro.pipeline.mcwf_frontend import mcwf_make_y0


def test_mcwf_frontend_4mic_shape() -> None:
    rng = np.random.default_rng(123)
    x_mics = rng.normal(size=(4, 1024)).astype(np.float32)
    y0 = mcwf_make_y0(x_mics, stft_params={"n_fft": 320, "win_length": 320, "hop_length": 160})
    assert y0.shape == (1024,)
    assert np.isfinite(y0).all()
