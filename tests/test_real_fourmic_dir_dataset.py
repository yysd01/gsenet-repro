from __future__ import annotations

import numpy as np

from gsenet_repro.data.real_fourmic_dir_dataset import RealFourMicDirDataset
from scripts.make_dummy_real_dir_dataset import make_dummy_real_dir_dataset


def test_real_fourmic_dir_dataset_shapes(tmp_path) -> None:
    root = tmp_path / "dummy_real_dir"
    make_dummy_real_dir_dataset(root)

    sample_rate = 16000
    segment_seconds = 1.0
    target_frames = int(sample_rate * segment_seconds)

    dataset = RealFourMicDirDataset(
        root=root,
        split="train",
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        num_mics=4,
        random_crop=True,
    )

    assert len(dataset) == 3

    sample = dataset[0]
    x_mics = sample["x_mics"]
    yt = sample["yt"]
    assert x_mics.shape == (4, target_frames)
    assert yt.shape == (target_frames,)
    assert x_mics.dtype == np.float32
    assert yt.dtype == np.float32
    assert np.isfinite(x_mics).all()
    assert np.isfinite(yt).all()

    padded = None
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item["meta"]["orig_frames"] < target_frames:
            padded = item
            break
    assert padded is not None
    pad_start = int(padded["meta"]["orig_frames"])
    assert np.allclose(padded["y1"][pad_start:], 0.0)

    valid_ds = RealFourMicDirDataset(
        root=root,
        split="valid",
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        num_mics=4,
        random_crop=False,
        fixed_crop="center",
    )
    for idx in range(len(valid_ds)):
        item = valid_ds[idx]
        start_frame = item["meta"]["start_frame"]
        assert start_frame >= 0
        if item["meta"]["orig_frames"] > target_frames:
            expected = (item["meta"]["orig_frames"] - target_frames) // 2
            assert start_frame == expected
