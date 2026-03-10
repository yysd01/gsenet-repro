from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.streaming.mcwf_streamer import MCWFStreamer


def test_mcwf_streamer_matches_block() -> None:
    torch.manual_seed(0)
    batch = 1
    channels = 4
    length = 1600
    hop = 160

    x = torch.randn(batch, channels, length)
    streamer = MCWFStreamer(
        sample_rate=16000,
        n_fft=320,
        win_length=320,
        hop_length=hop,
        causal_frames=4,
        num_mics=channels,
    )

    outputs = []
    for idx in range(0, length, hop):
        chunk = x[:, :, idx : idx + hop]
        outputs.append(streamer.process(chunk))
    y_stream = torch.cat(outputs, dim=-1)

    streamer_block = MCWFStreamer(
        sample_rate=16000,
        n_fft=320,
        win_length=320,
        hop_length=hop,
        causal_frames=4,
        num_mics=channels,
    )
    y_block = streamer_block.process(x)

    delay = streamer.algorithmic_delay_samples
    y_stream_aligned = y_stream[:, delay:]
    y_block_aligned = y_block[:, delay:]
    compare_len = min(y_stream_aligned.shape[-1], y_block_aligned.shape[-1])
    y_stream_aligned = y_stream_aligned[:, :compare_len]
    y_block_aligned = y_block_aligned[:, :compare_len]

    diff = y_stream_aligned - y_block_aligned
    assert torch.isfinite(diff).all()
    mse = torch.mean(diff**2).item()
    assert mse < 1e-3
