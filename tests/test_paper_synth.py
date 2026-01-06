import numpy as np

from gsenet_repro.data.paper_synth import (
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
