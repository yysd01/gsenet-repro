import numpy as np

import gsenet_repro


def test_import_and_fft_roundtrip():
    signal = np.random.randn(256)
    reconstructed = gsenet_repro.fft_roundtrip(signal)
    assert np.allclose(reconstructed, signal)
