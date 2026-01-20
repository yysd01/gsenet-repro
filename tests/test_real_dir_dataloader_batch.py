from __future__ import annotations

import importlib.util

import pytest
if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    import torch
    from torch.utils.data import DataLoader
else:  # pragma: no cover
    torch = None
    DataLoader = None

from gsenet_repro.data.real_fourmic_dir_dataset import RealFourMicDirDataset
from scripts.make_dummy_real_dir_dataset import make_dummy_real_dir_dataset


torch_available = torch is not None


@pytest.mark.skipif(not torch_available, reason="torch not available")
def test_real_dir_dataloader_batch(tmp_path) -> None:
    assert torch is not None
    from gsenet_repro.dsp import MODEL_STFT
    from gsenet_repro.models.gsenet_torch import MinimalGSENet

    root = tmp_path / "dummy_real_dir"
    make_dummy_real_dir_dataset(root)

    dataset = RealFourMicDirDataset(
        root=root,
        split="train",
        sample_rate=16000,
        segment_seconds=1.0,
        num_mics=4,
        random_crop=True,
    )
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))
    x_mics = batch["x_mics"]
    assert x_mics.shape[0] == 2
    assert x_mics.shape[1] == 4

    y1 = batch["y1"]
    y2 = x_mics[:, 2]
    y0 = y1
    model = MinimalGSENet(stft_params=MODEL_STFT)
    y_hat = model(y0, y1, y2)
    assert y_hat.shape == y1.shape
