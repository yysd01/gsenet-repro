from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from gsenet_repro.data.real_dataset import RealMultichannelDataset


def _make_dummy_manifest() -> Path:
    subprocess.run([sys.executable, "scripts/_legacy/make_dummy_real_manifest.py"], check=True)
    return Path("artifacts") / "dummy_real_dataset" / "manifest.csv"


def test_real_dataset_manifest_slice() -> None:
    manifest_path = _make_dummy_manifest()
    dataset = RealMultichannelDataset(
        manifest_path=str(manifest_path),
        sample_rate=16000,
        segment_seconds=0.5,
        num_mics=4,
        ref_mic_index=0,
        use_mcwf=False,
    )
    sample = dataset[0]
    x_mics = sample["x_mics"]
    yt = sample["yt"]
    assert x_mics.shape[0] == 4
    assert x_mics.shape[1] == yt.shape[0]
    assert np.isfinite(x_mics).all()
    assert np.isfinite(yt).all()


def test_real_dataset_manifest_dataloader() -> None:
    if RealMultichannelDataset.__mro__[1].__name__ == "object":
        return
    from torch.utils.data import DataLoader

    manifest_path = Path("artifacts") / "dummy_real_dataset" / "manifest.csv"
    if not manifest_path.exists():
        manifest_path = _make_dummy_manifest()
    dataset = RealMultichannelDataset(
        manifest_path=str(manifest_path),
        sample_rate=16000,
        segment_seconds=0.25,
        num_mics=4,
        ref_mic_index=0,
        use_mcwf=False,
    )
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))
    assert batch["x_mics"].shape[1] == 4
    assert batch["x_mics"].shape[0] == 2
    import torch

    assert torch.isfinite(batch["x_mics"]).all()
