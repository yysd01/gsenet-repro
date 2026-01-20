from __future__ import annotations

import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    print("torch not installed. Install requirements-torch.txt to run this script.", file=sys.stderr)
    raise SystemExit(1)

import torch

from gsenet_repro.pipeline.mcwf_frontend import mcwf_make_y0
from gsenet_repro.streaming.mcwf_streamer import MCWFStreamer


def main() -> None:
    torch.manual_seed(0)
    batch = 1
    channels = 4
    length = 3200
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
    y_block = mcwf_make_y0(
        x,
        stft_params={"n_fft": 320, "win_length": 320, "hop_length": hop},
        causal_frames=4,
    )
    delay = streamer.algorithmic_delay_samples
    y_stream_aligned = y_stream[:, delay : delay + y_block.shape[-1]]
    y_block_aligned = y_block[:, : y_stream_aligned.shape[-1]]
    diff = y_stream_aligned - y_block_aligned
    mse = torch.mean(diff**2).item()
    max_abs = torch.max(diff.abs()).item()

    print(f"MCWFStreamer delay: {delay} samples")
    print(f"stream_out_len={y_stream.shape[-1]} block_len={y_block.shape[-1]}")
    print(f"mse={mse:.6e} max_abs={max_abs:.6e}")


if __name__ == "__main__":
    main()
