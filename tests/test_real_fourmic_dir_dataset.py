from __future__ import annotations

from pathlib import Path

import numpy as np

from gsenet_repro.data.real_fourmic_dir_dataset import (
    RealFourMicDirDataset,
    canonical_pair_key,
)
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
        clean_ref_mic_index=0,
        clean_is_multichannel=True,
        random_crop=True,
    )

    assert len(dataset) == 3

    sample = dataset[0]
    x_mics = sample["x_mics"]
    y1 = sample["y1"]
    yt = sample["yt"]
    assert x_mics.shape == (4, target_frames)
    assert y1.shape == (target_frames,)
    assert yt.shape == (target_frames,)
    assert x_mics.dtype == np.float32
    assert y1.dtype == np.float32
    assert yt.dtype == np.float32
    assert np.isfinite(x_mics).all()
    assert np.isfinite(y1).all()
    assert np.isfinite(yt).all()
    pair_key = sample["meta"]["pair_key"]
    assert pair_key == canonical_pair_key(
        Path(sample["meta"]["clean_path"]).name, "clean", dataset.pairing_config
    )
    assert pair_key == canonical_pair_key(
        Path(sample["meta"]["mic_path"]).name, "mic", dataset.pairing_config
    )
    assert sample["meta"]["ref_mic_index"] == 0
    assert sample["meta"]["clean_ref_mic_index"] == 0

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
        clean_ref_mic_index=0,
        clean_is_multichannel=True,
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

    dataset_ref0 = RealFourMicDirDataset(
        root=root,
        split="train",
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        num_mics=4,
        clean_ref_mic_index=0,
        clean_is_multichannel=True,
        random_crop=False,
        fixed_crop="start",
    )
    dataset_ref3 = RealFourMicDirDataset(
        root=root,
        split="train",
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        num_mics=4,
        clean_ref_mic_index=3,
        clean_is_multichannel=True,
        random_crop=False,
        fixed_crop="start",
    )
    sample_ref0 = dataset_ref0[0]
    sample_ref3 = dataset_ref3[0]
    diff = sample_ref0["yt"] - sample_ref3["yt"]
    assert np.max(np.abs(diff)) > 0
