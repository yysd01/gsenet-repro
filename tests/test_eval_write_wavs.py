from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import pytest

from gsenet_repro.config import resolve_config
from scripts._legacy.make_dummy_real_dir_dataset import make_dummy_real_dir_dataset


def _write_dummy_checkpoint(run_dir: Path, config: dict) -> None:
    torch = pytest.importorskip("torch")
    from gsenet_repro.models.gsenet_torch import MinimalGSENet

    model = MinimalGSENet(stft_params=config["stft_model"])
    ckpt = {"model_state": model.state_dict(), "config": config}
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)


def test_eval_write_wavs_real_dir(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    root = tmp_path / "dummy_real_dir"
    make_dummy_real_dir_dataset(root)

    run_dir = tmp_path / "run"
    config = resolve_config(
        None,
        overrides={
            "data": {
                "mode": "real_dir",
                "root": str(root),
                "segment_seconds": 0.5,
                "sample_rate": 16000,
                "num_mics": 4,
                "ref_mic_index": 0,
                "clean_ref_mic_index": 0,
                "use_mcwf": 0,
            },
            "model": {"name": "minimal"},
            "metrics": {"enable_pesq": False},
        },
    )
    _write_dummy_checkpoint(run_dir, config)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/test.py",
            "--run_dir",
            str(run_dir),
            "--num_batches",
            "1",
            "--batch_size",
            "2",
            "--write_wavs",
            "--max_wavs",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    wav_dir = run_dir / "artifacts" / "test_wavs" / "test"
    assert wav_dir.exists()
    assert list(wav_dir.glob("*_y1_ref.wav"))
    assert list(wav_dir.glob("*_y0_bf.wav"))
    assert list(wav_dir.glob("*_yhat.wav"))
    assert list(wav_dir.glob("*_yt.wav"))

    per_sample_csv = run_dir / "artifacts" / "per_sample_metrics.csv"
    assert per_sample_csv.exists()
    with per_sample_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows
    for row in rows:
        if row["path_y1"]:
            assert Path(row["path_y1"]).exists()
            assert Path(row["path_y0"]).exists()
            assert Path(row["path_yhat"]).exists()
            assert Path(row["path_yt"]).exists()
