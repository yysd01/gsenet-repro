from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from gsenet_repro.pipeline.mcwf_frontend import mcwf_make_y0
from gsenet_repro.streaming.mcwf_streamer import MCWFStreamer


def test_mcwf_streamer_matches_block() -> None:
    torch.manual_seed(0)
    batch = 1
    channels = 3
    length = 1600
    hop = 160

    x = torch.randn(batch, channels, length)
    streamer = MCWFStreamer(
        sample_rate=16000,
        n_fft=320,
        win_length=320,
        hop_length=hop,
        causal_frames=4,
    )

    outputs = []
    for idx in range(0, length, hop):
        chunk = x[:, :, idx : idx + hop]
        outputs.append(streamer.process(chunk))
    y_stream = torch.cat(outputs, dim=-1)

    y_block = mcwf_make_y0(x, stft_params={"n_fft": 320, "win_length": 320, "hop_length": hop})
    delay = streamer.algorithmic_delay_samples
    y_stream_aligned = y_stream[:, delay : delay + y_block.shape[-1]]
    y_block_aligned = y_block[:, : y_stream_aligned.shape[-1]]

    diff = y_stream_aligned - y_block_aligned
    assert torch.isfinite(diff).all()
    mse = torch.mean(diff**2).item()
    assert mse < 1e-3
