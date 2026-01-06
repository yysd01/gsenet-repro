import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.dsp import MODEL_STFT
from gsenet_repro.models.gsenet_torch import MinimalGSENet
from gsenet_repro.streaming.gsenet_streamer import GSENetStreamer


def _make_batch(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(0)
    t = np.arange(length, dtype=np.float32) / 16000.0
    tone = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    y0 = rng.normal(scale=0.2, size=(batch, length)).astype(np.float32) + tone
    y1 = rng.normal(scale=0.2, size=(batch, length)).astype(np.float32) - tone
    y2 = 0.6 * y0 + 0.4 * y1
    return torch.tensor(y0), torch.tensor(y1), torch.tensor(y2)


def _stream_inference(
    streamer: GSENetStreamer, y0: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor
) -> torch.Tensor:
    outputs = []
    length = y0.shape[1]
    chunk_size = streamer.chunk_size
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        y0_chunk = y0[:, start:end]
        y1_chunk = y1[:, start:end]
        y2_chunk = y2[:, start:end]
        if end - start < chunk_size:
            pad_len = chunk_size - (end - start)
            y0_chunk = torch.nn.functional.pad(y0_chunk, (0, pad_len))
            y1_chunk = torch.nn.functional.pad(y1_chunk, (0, pad_len))
            y2_chunk = torch.nn.functional.pad(y2_chunk, (0, pad_len))
        outputs.append(streamer.process(y0_chunk, y1_chunk, y2_chunk))
    y_stream = torch.cat(outputs, dim=-1)
    return y_stream[:, :length]


def test_streaming_matches_offline() -> None:
    torch.manual_seed(0)
    batch = 2
    length = 8000
    chunk_size = 1600
    lookback = 4096

    y0, y1, y2 = _make_batch(batch, length)
    model = MinimalGSENet()

    with torch.no_grad():
        y_offline = model(y0, y1, y2)

    streamer = GSENetStreamer(
        model,
        chunk_size=chunk_size,
        lookback=lookback,
        algorithmic_delay=MODEL_STFT["win_length"],
    )
    y_stream = _stream_inference(streamer, y0, y1, y2)

    delay = streamer.algorithmic_delay
    torch.testing.assert_close(
        y_stream[:, delay:],
        y_offline[:, delay:],
        rtol=1e-3,
        atol=1e-3,
    )


def test_streaming_safe_prefix() -> None:
    torch.manual_seed(1)
    batch = 2
    length = 12000
    chunk_size = 1600
    lookback = 4096
    change_idx = 9000

    y0, y1, y2 = _make_batch(batch, length)
    y0_changed = y0.clone()
    y1_changed = y1.clone()
    y2_changed = y2.clone()
    y0_changed[:, change_idx:] += 0.3
    y1_changed[:, change_idx:] -= 0.2
    y2_changed[:, change_idx:] += 0.1

    model = MinimalGSENet()
    streamer = GSENetStreamer(
        model,
        chunk_size=chunk_size,
        lookback=lookback,
        algorithmic_delay=MODEL_STFT["win_length"],
    )

    y_stream = _stream_inference(streamer, y0, y1, y2)
    streamer.reset()
    y_stream_changed = _stream_inference(streamer, y0_changed, y1_changed, y2_changed)

    safe_margin = lookback + MODEL_STFT["win_length"]
    compare_end = change_idx - safe_margin
    if compare_end <= 0:
        pytest.skip("Safe prefix is empty for the selected change index.")

    torch.testing.assert_close(
        y_stream[:, :compare_end],
        y_stream_changed[:, :compare_end],
        rtol=1e-3,
        atol=1e-3,
    )
