import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from gsenet_repro.data.paper_dataset import PaperLikeDataset
from gsenet_repro.data.paper_synth import generate_rir_3src_4mic


def test_paper_like_dataset_shapes() -> None:
    dataset = PaperLikeDataset(
        sample_rate=8000,
        segment_seconds=0.5,
        seed=0,
        num_samples=4,
    )
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))

    y0 = batch["y0"]
    y1 = batch["y1"]
    yt = batch["yt"]
    x_mics = batch["x_mics"]

    assert y0.ndim == 2
    assert y1.ndim == 2
    assert yt.ndim == 2
    assert y0.shape == y1.shape == yt.shape
    assert x_mics.shape[0] == y0.shape[0]
    assert x_mics.shape[1] == 4
    assert x_mics.shape[2] == y0.shape[1]
    assert torch.isfinite(y0).all()
    assert torch.isfinite(y1).all()
    assert torch.isfinite(yt).all()


def test_rir_close_mic_direct_path_delay_4mic() -> None:
    rng = np.random.default_rng(4)
    rir, _ = generate_rir_3src_4mic(rng, max_direct_delay_diff=4)
    for src_idx in range(rir.shape[0]):
        delays = []
        for mic_idx in range(rir.shape[1]):
            delays.append(int(np.argmax(np.abs(rir[src_idx, mic_idx]))))
        assert max(delays) - min(delays) <= 4
