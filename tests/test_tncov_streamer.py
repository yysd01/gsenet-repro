from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.streaming.tncov_streamer import TraceNormCovStreamer


def test_tncov_output_length_and_finite() -> None:
    torch.manual_seed(0)
    streamer = TraceNormCovStreamer(num_mics=4, center=False)

    hop = streamer.hop_length
    total_len = 4096
    x = torch.randn(4, total_len)

    outs = []
    for start in range(0, total_len, hop):
        y = streamer.process(x[:, start : start + hop])
        assert y.shape == (hop,)
        assert torch.isfinite(y).all()
        outs.append(y)

    y_all = torch.cat(outs)[:total_len]
    assert y_all.shape[-1] == total_len
    assert torch.isfinite(y_all).all()


def test_trace_denominator_or_fallback_is_valid() -> None:
    streamer = TraceNormCovStreamer(num_mics=4, center=False)

    hop = streamer.hop_length
    for _ in range(32):
        x = 1e-5 * torch.randn(4, hop)
        y = streamer.process(x)
        assert torch.isfinite(y).all()

    assert streamer.last_trace_den is not None
    den = streamer.last_trace_den
    finite = torch.isfinite(den)
    valid = finite & (den > streamer.eps_trace)
    if not torch.all(valid):
        assert streamer.last_fallback_ratio > 0.0
