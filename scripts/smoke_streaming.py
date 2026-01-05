from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit("torch is required to run smoke_streaming.py") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from gsenet_repro.dsp import MODEL_STFT
from gsenet_repro.models.gsenet_torch import MinimalGSENet
from gsenet_repro.streaming.gsenet_streamer import GSENetStreamer


def _load_or_create_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    artifacts_path = Path("artifacts") / "dummy_batch.npz"
    if artifacts_path.exists():
        data = np.load(artifacts_path)
        return data["y0"], data["y1"], data["yt"]

    rng = np.random.default_rng(0)
    batch = 2
    length = 8000
    y0 = rng.normal(scale=0.2, size=(batch, length)).astype(np.float32)
    y1 = rng.normal(scale=0.2, size=(batch, length)).astype(np.float32)
    yt = 0.6 * y0 + 0.4 * y1
    return y0, y1, yt.astype(np.float32)


def _stream_inference(
    streamer: GSENetStreamer, y0: torch.Tensor, y1: torch.Tensor
) -> torch.Tensor:
    outputs = []
    length = y0.shape[1]
    chunk_size = streamer.chunk_size
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        y0_chunk = y0[:, start:end]
        y1_chunk = y1[:, start:end]
        if end - start < chunk_size:
            pad_len = chunk_size - (end - start)
            y0_chunk = torch.nn.functional.pad(y0_chunk, (0, pad_len))
            y1_chunk = torch.nn.functional.pad(y1_chunk, (0, pad_len))
        out = streamer.process(y0_chunk, y1_chunk)
        outputs.append(out)
    y_stream = torch.cat(outputs, dim=-1)
    return y_stream[:, :length]


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    y0_np, y1_np, yt_np = _load_or_create_batch()
    y0 = torch.tensor(y0_np, dtype=torch.float32)
    y1 = torch.tensor(y1_np, dtype=torch.float32)
    yt = torch.tensor(yt_np, dtype=torch.float32)

    model = MinimalGSENet()
    model.eval()
    with torch.no_grad():
        y_offline = model(y0, y1)

    streamer = GSENetStreamer(
        model,
        chunk_size=1600,
        lookback=4096,
        algorithmic_delay=MODEL_STFT["win_length"],
    )
    y_stream = _stream_inference(streamer, y0, y1)

    delay = streamer.algorithmic_delay
    y_stream_aligned = y_stream[:, delay:]
    y_offline_aligned = y_offline[:, delay:]

    max_abs_error = torch.max(torch.abs(y_stream_aligned - y_offline_aligned)).item()
    mse = torch.mean((y_stream_aligned - y_offline_aligned) ** 2).item()

    torch.testing.assert_close(
        y_stream_aligned,
        y_offline_aligned,
        rtol=1e-3,
        atol=1e-3,
    )

    print(
        "len={} delay={} max_abs_error={:.6f} mse={:.8f}".format(
            y0.shape[1], delay, max_abs_error, mse
        )
    )
    if torch.isnan(yt).any():
        raise SystemExit("Sanity check failed: yt contains NaNs")


if __name__ == "__main__":
    main()
