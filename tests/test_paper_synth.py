import numpy as np

from gsenet_repro.data.paper_synth import (
    add_background_noise,
    generate_background_noise,
    generate_rir_3src_2mic,
    sample_paper_params,
    synthesize_y0_y1_yt,
)


def test_sample_paper_params_constraints() -> None:
    rng = np.random.default_rng(0)
    params = sample_paper_params(rng)
    assert params.alpha_db >= -4.0
    assert params.beta_db >= 4.0
    assert params.pi in (0.0, 1.0)
    assert isinstance(params.gn_lin, float)
    assert isinstance(params.gi_lin, float)


def test_rir_close_mic_direct_path_delay() -> None:
    rng = np.random.default_rng(1)
    rir, _ = generate_rir_3src_2mic(rng, max_direct_delay_diff=4)
    for src_idx in range(rir.shape[0]):
        delay0 = int(np.argmax(np.abs(rir[src_idx, 0])))
        delay1 = int(np.argmax(np.abs(rir[src_idx, 1])))
        assert abs(delay0 - delay1) <= 5
        for mic_idx in range(rir.shape[1]):
            delay = int(np.argmax(np.abs(rir[src_idx, mic_idx])))
            early_segment = rir[src_idx, mic_idx, delay + 1 : delay + 700]
            assert np.sum(np.abs(early_segment) > 0.05) >= 2


def test_synthesize_outputs() -> None:
    rng = np.random.default_rng(2)
    length = 8000
    s = rng.normal(size=length).astype(np.float32)
    n = rng.normal(size=length).astype(np.float32)
    i = rng.normal(size=length).astype(np.float32)
    rir, rir_anechoic = generate_rir_3src_2mic(rng)
    params = sample_paper_params(rng)
    y0, y1, yt = synthesize_y0_y1_yt(s, n, i, rir, rir_anechoic, params)
    assert y0.shape == s.shape
    assert y1.shape == s.shape
    assert yt.shape == s.shape
    assert np.isfinite(y0).all()
    assert np.isfinite(y1).all()
    assert np.isfinite(yt).all()


def test_background_noise_mixing() -> None:
    rng = np.random.default_rng(3)
    length = 4000
    s = rng.normal(size=length).astype(np.float32)
    n = rng.normal(size=length).astype(np.float32)
    i = rng.normal(size=length).astype(np.float32)
    rir, rir_anechoic = generate_rir_3src_2mic(rng)
    params = sample_paper_params(rng)
    y0, y1, _ = synthesize_y0_y1_yt(s, n, i, rir, rir_anechoic, params)

    noise_mics, _ = generate_background_noise(rng, length=length, fs=16000, num_mics=2)
    y_mics, _ = add_background_noise(np.stack([y0, y1], axis=0), noise_mics, snr_db=5.0)
    assert np.mean(y_mics[0] ** 2) > np.mean(y0**2)
    assert np.mean(y_mics[1] ** 2) > np.mean(y1**2)
