from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


_TRAIN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_paper_like_full.py"
_SPEC = importlib.util.spec_from_file_location("train_paper_like_full", _TRAIN_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("Unable to load train_paper_like_full module")
train = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train)


def test_grad_stats_no_nan() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    x = torch.randn(3, 8)
    target = torch.randn(3, 2)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()

    grad_norm, grad_max = train._grad_stats(model)
    param_norm = train._param_norm(model)

    assert torch.isfinite(torch.tensor(grad_norm))
    assert torch.isfinite(torch.tensor(grad_max))
    assert torch.isfinite(torch.tensor(param_norm))


def test_dump_debug_writes_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    batch = {
        "y0": torch.randn(2, 2000),
        "y1": torch.randn(2, 2000),
        "yt": torch.randn(2, 2000),
    }
    y_hat = torch.randn(2, 2000)
    debug_cfg = {
        "max_items": 2,
        "seconds": 0.05,
        "dir": "debug",
    }
    train._dump_debug(
        run_dir=run_dir,
        step=10,
        batch=batch,
        y_hat=y_hat,
        sample_rate=16000,
        debug_cfg=debug_cfg,
        extra_stats_dict={"loss": 1.23, "step": 10},
    )

    dump_dir = run_dir / "debug" / "step_00010"
    assert (dump_dir / "batch.pt").exists()
    assert (dump_dir / "meta.json").exists()

    wav_or_npy_count = len(list(dump_dir.glob("*.wav"))) + len(list(dump_dir.glob("*.npy")))
    assert wav_or_npy_count >= 4
