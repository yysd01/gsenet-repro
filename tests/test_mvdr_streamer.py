from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from gsenet_repro.streaming.mvdr_streamer import MVDRStreamer


def test_mvdr_streamer_shape_and_finite() -> None:
    torch.manual_seed(0)
    streamer = MVDRStreamer(num_mics=4, center=False)

    F = streamer.n_fft // 2 + 1
    d = np.zeros((1, F, 4), dtype=np.complex64)
    d[:, :, 0] = 1.0 + 0.0j
    d[:, :, 1] = 0.8 + 0.1j
    d[:, :, 2] = 0.6 - 0.2j
    d[:, :, 3] = 0.3 + 0.05j
    streamer.rtf_lib = {
        "doa_bins": np.array([0], dtype=np.int32),
        "d_mean": d,
        "binsize_deg": 1,
        "ref_ch": 0,
    }
    streamer.set_target_doa(0)

    chunk = 1280
    outs = []
    for _ in range(10):
        x = torch.randn(4, chunk)
        y = streamer.process(x)
        assert y.shape == (chunk,)
        assert torch.isfinite(y).all()
        outs.append(y)

    y_all = torch.cat(outs)
    assert y_all.shape == (10 * chunk,)


def test_target_like_higher_for_coherent_signal() -> None:
    torch.manual_seed(0)
    streamer = MVDRStreamer(num_mics=4, center=False)
    F = streamer.n_fft // 2 + 1
    d = np.zeros((1, F, 4), dtype=np.complex64)
    d[:, :, 0] = 1.0 + 0.0j
    streamer.rtf_lib = {
        "doa_bins": np.array([0], dtype=np.int32),
        "d_mean": d,
        "binsize_deg": 1,
        "ref_ch": 0,
    }
    streamer.set_target_doa(0)

    coherent = torch.randn(1, 1280)
    coherent4 = coherent.repeat(4, 1)
    _ = streamer.process(coherent4)
    score_coh = float(streamer.last_target_like)

    incoherent4 = torch.randn(4, 1280)
    _ = streamer.process(incoherent4)
    score_incoh = float(streamer.last_target_like)

    assert score_coh >= score_incoh
