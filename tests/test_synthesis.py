import numpy as np

from gsenet_repro.data.synthesis import Gains, sample_gains, synthesize_pair


def test_truncation_bounds() -> None:
    rng = np.random.default_rng(0)
    alpha_db = []
    beta_db = []
    for _ in range(5000):
        gains = sample_gains(rng)
        alpha_db.append(gains.alpha_db)
        beta_db.append(gains.beta_db)
    assert min(alpha_db) >= -4.0 - 1e-6
    assert min(beta_db) >= 4.0 - 1e-6


def test_reproducibility() -> None:
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    s = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    n = np.linspace(1.0, -1.0, 64, dtype=np.float32)
    i = np.zeros_like(s)
    rir = np.zeros((3, 2, 8), dtype=np.float32)
    rir[:, :, 0] = 1.0
    rir_anechoic = rir.copy()

    y0_a, y1_a, yt_a, meta_a = synthesize_pair(s, n, i, rir, rir_anechoic, rng1)
    y0_b, y1_b, yt_b, meta_b = synthesize_pair(s, n, i, rir, rir_anechoic, rng2)

    assert np.allclose(y0_a, y0_b)
    assert np.allclose(y1_a, y1_b)
    assert np.allclose(yt_a, yt_b)
    assert meta_a == meta_b


def test_contrast_exists_deterministic() -> None:
    rng = np.random.default_rng(0)
    length = 128
    n = rng.normal(size=length).astype(np.float32)
    i = rng.normal(size=length).astype(np.float32)
    zero = np.zeros_like(n)
    rir = np.zeros((3, 2, 4), dtype=np.float32)
    rir[:, :, 0] = 1.0
    rir_anechoic = rir.copy()

    gains = Gains(
        gn=1.0,
        gi=1.0,
        alpha=2.0,
        beta=2.0,
        pi=1.0,
        gn_db=0.0,
        gi_db=0.0,
        alpha_db=6.020599913279624,
        beta_db=6.020599913279624,
    )

    y0_noise, y1_noise, _, _ = synthesize_pair(
        s=zero,
        n=n,
        i=zero,
        rir=rir,
        rir_anechoic=rir_anechoic,
        rng=rng,
        gains=gains,
    )
    y0_int, y1_int, _, _ = synthesize_pair(
        s=zero,
        n=zero,
        i=i,
        rir=rir,
        rir_anechoic=rir_anechoic,
        rng=rng,
        gains=gains,
    )

    noise_ratio = np.mean(y1_noise**2) / (np.mean(y0_noise**2) + 1e-12)
    int_ratio = np.mean(y1_int**2) / (np.mean(y0_int**2) + 1e-12)

    assert noise_ratio > 1.5
    assert int_ratio > 1.5
