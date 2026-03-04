import numpy as np

from gsenet_repro.dsp.mvdr import apply_beamformer, estimate_rnn, estimate_rtf, mvdr_weights


def _make_x(F=16, T=32, C=4, seed=0):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((F, T, C)).astype(np.float32)
    imag = rng.standard_normal((F, T, C)).astype(np.float32)
    return real + 1j * imag


def test_rnn_hermitian() -> None:
    X = _make_x()
    gate = np.ones(X.shape[1], dtype=np.float32)
    R = estimate_rnn(X, gate, smoothing=0.9)
    assert np.allclose(R, np.swapaxes(R.conj(), -1, -2), atol=1e-5)


def test_mvdr_distortionless_constraint() -> None:
    X = _make_x()
    gate = np.ones(X.shape[1], dtype=np.float32)
    rnn = estimate_rnn(X, gate, smoothing=0.9)
    d = estimate_rtf(X, gate, ref_ch=1, smoothing=0.1)
    w = mvdr_weights(rnn, d, diag_load=1e-2)
    val = np.sum(np.conjugate(w) * d, axis=-1)
    assert np.allclose(val, 1.0, atol=2e-2)


def test_apply_beamformer_shape() -> None:
    X = _make_x()
    gate = np.ones(X.shape[1], dtype=np.float32)
    w = mvdr_weights(estimate_rnn(X, gate), estimate_rtf(X, gate, ref_ch=0))
    Y = apply_beamformer(w, X)
    assert Y.shape == X.shape[:2]
    assert np.isfinite(Y.real).all()


def test_rnn_smoothing_is_decay() -> None:
    X = _make_x(F=8, T=1, C=4, seed=1)
    gate = np.ones(1, dtype=np.float32)
    R = estimate_rnn(X, gate, smoothing=0.96)
    xxh = X[:, 0, :][:, :, None] * np.conjugate(X[:, 0, :][:, None, :])
    I = np.tile(np.eye(X.shape[-1], dtype=np.complex64)[None, :, :], (X.shape[0], 1, 1))
    rel = np.linalg.norm(R - I) / (np.linalg.norm(xxh - I) + 1e-8)
    assert rel < 0.2
