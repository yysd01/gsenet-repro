from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.streaming.mvdr_streamer import MVDRStreamer


def _dummy_rtf_lib(streamer: MVDRStreamer) -> dict[str, object]:
    F = streamer.n_fft // 2 + 1
    d = np.ones((1, F, streamer.num_mics), dtype=np.complex64)
    return {
        "doa_bins": np.array([0], dtype=np.int32),
        "d_mean": d,
        "binsize_deg": 1,
        "ref_ch": streamer.ref_ch,
        "sample_rate": streamer.sample_rate,
        "n_fft": streamer.n_fft,
        "win_length": streamer.win_length,
        "hop_length": streamer.hop_length,
        "window": streamer.window,
        "center": streamer.center,
        "num_mics": streamer.num_mics,
        "missing_metadata": tuple(),
    }


def test_single_stream_only() -> None:
    streamer = MVDRStreamer(num_mics=4, center=False)
    with pytest.raises(ValueError, match="single-stream"):
        streamer.process(torch.randn(2, 4, 128))


def test_streamer_no_nan() -> None:
    torch.manual_seed(0)
    streamer = MVDRStreamer(num_mics=4, center=False)
    streamer.rtf_lib = _dummy_rtf_lib(streamer)
    streamer.set_target_doa(0)

    chunk = 640
    outs = []
    for _ in range(8):
        x = torch.randn(4, chunk)
        y = streamer.process(x)
        assert y.shape == (chunk,)
        assert torch.isfinite(y).all()
        outs.append(y)

    y_all = torch.cat(outs)
    assert y_all.shape == (8 * chunk,)


def test_target_like_higher_for_coherent_signal() -> None:
    torch.manual_seed(0)
    streamer = MVDRStreamer(num_mics=4, center=False)
    streamer.rtf_lib = _dummy_rtf_lib(streamer)
    streamer.set_target_doa(0)

    coherent = torch.randn(1, 1280)
    coherent4 = coherent.repeat(4, 1)
    _ = streamer.process(coherent4)
    score_coh = float(streamer.last_target_like)

    incoherent4 = torch.randn(4, 1280)
    _ = streamer.process(incoherent4)
    score_incoh = float(streamer.last_target_like)

    assert score_coh >= score_incoh


def test_from_config_stft_mismatch_rejected() -> None:
    cfg = {
        "data": {"sample_rate": 16000, "num_mics": 4, "ref_mic_index": 1},
        "stft_model": {"n_fft": 320, "win_length": 320, "hop_length": 160},
        "stft_streaming": {"n_fft": 512, "win_length": 320, "hop_length": 160},
        "frontend": {"ref_ch": 1},
    }
    with pytest.raises(ValueError, match="stft_streaming mismatch"):
        MVDRStreamer.from_config(cfg)
