from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

if importlib.util.find_spec("torch") is None:
    print(
        "torch not installed. Install requirements-torch.txt to run training.",
        file=sys.stderr,
    )
    raise SystemExit(1)

import torch
from torch.utils.data import DataLoader

from gsenet_repro.data.paper_dataset import PaperLikeDataset
from gsenet_repro.losses.stft_loss_torch import stft_magnitude_loss
from gsenet_repro.metrics.metrics_pesq import pesq_available, pesq_score
from gsenet_repro.metrics.metrics_torch import si_snr_db, sisdr, snr_db
from gsenet_repro.models.gsenet_torch import MinimalGSENet


def _build_run_dir(run_dir: str | None) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "runs" / timestamp


def _write_csv_row(path: Path, row: Dict[str, float | int], header: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _make_eval_batch(
    seed: int,
    batch_size: int,
    sample_rate: int,
    segment_seconds: float,
    use_mcwf: bool,
    model_stft: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    dataset = PaperLikeDataset(
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        seed=seed,
        num_samples=batch_size,
        use_mcwf=use_mcwf,
        stft_params=model_stft,
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    return next(iter(loader))


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Dict[str, object],
) -> None:
    rng_state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "config": config,
        "rng_state": rng_state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-like full training pipeline.")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_seconds", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--ckpt_every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_mcwf", type=int, default=1)
    parser.add_argument("--loss_stft_win", type=int, default=1024)
    parser.add_argument("--loss_stft_hop", type=int, default=256)
    parser.add_argument("--loss_stft_n_fft", type=int, default=1024)
    parser.add_argument("--model_stft_win", type=int, default=320)
    parser.add_argument("--model_stft_hop", type=int, default=160)
    parser.add_argument("--model_stft_n_fft", type=int, default=320)
    args = parser.parse_args()

    run_dir = _build_run_dir(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cuda")
        if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device if args.device != "auto" else "cpu")
    )

    _seed_everything(args.seed)

    loss_stft = {
        "n_fft": args.loss_stft_n_fft,
        "win_length": args.loss_stft_win,
        "hop_length": args.loss_stft_hop,
    }
    model_stft = {
        "n_fft": args.model_stft_n_fft,
        "win_length": args.model_stft_win,
        "hop_length": args.model_stft_hop,
    }

    config = {
        "seed": args.seed,
        "device": str(device),
        "sample_rate": args.sample_rate,
        "segment_seconds": args.segment_seconds,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "log_every": args.log_every,
        "eval_every": args.eval_every,
        "ckpt_every": args.ckpt_every,
        "lr": args.lr,
        "use_mcwf": bool(args.use_mcwf),
        "loss_stft": loss_stft,
        "model_stft": model_stft,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    dataset = PaperLikeDataset(
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        seed=args.seed,
        num_samples=None,
        use_mcwf=bool(args.use_mcwf),
        stft_params=model_stft,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)
    train_iter = iter(loader)

    eval_seed = args.seed + 1234
    eval_batch = _make_eval_batch(
        seed=eval_seed,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        use_mcwf=bool(args.use_mcwf),
        model_stft=model_stft,
    )

    model = MinimalGSENet(stft_params=model_stft).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics_path = run_dir / "metrics.csv"
    eval_path = run_dir / "eval.csv"
    best_metric = None
    has_pesq = pesq_available()
    if not has_pesq:
        print("pesq not installed, skipping PESQ metrics.", file=sys.stderr)

    for step in range(1, args.num_steps + 1):
        batch = next(train_iter)
        batch = _to_device(batch, device)
        y0 = batch["y0"]
        y1 = batch["y1"]
        y2 = batch["y2"]
        yt = batch["yt"]

        model.train()
        optimizer.zero_grad()
        y_hat = model(y0, y1, y2)
        loss = stft_magnitude_loss(y_hat, yt, stft_params=loss_stft)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            snr_in = snr_db(yt, y0).mean().item()
            snr_out = snr_db(yt, y_hat).mean().item()
            sisnr_in = si_snr_db(yt, y0).mean().item()
            sisnr_out = si_snr_db(yt, y_hat).mean().item()
            sisdr_in = sisdr(yt, y0).mean().item()
            sisdr_out = sisdr(yt, y_hat).mean().item()

        snr_impr = snr_out - snr_in
        sisnr_impr = sisnr_out - sisnr_in
        sisdr_impr = sisdr_out - sisdr_in

        _write_csv_row(
            metrics_path,
            {
                "step": step,
                "loss": float(loss.item()),
                "snr_in": snr_in,
                "snr_out": snr_out,
                "snr_impr": snr_impr,
                "sisnr_in": sisnr_in,
                "sisnr_out": sisnr_out,
                "sisnr_impr": sisnr_impr,
                "sisdr_in": sisdr_in,
                "sisdr_out": sisdr_out,
                "sisdr_impr": sisdr_impr,
                "pesq_in": float("nan"),
                "pesq_out": float("nan"),
                "pesq_impr": float("nan"),
            },
            header=[
                "step",
                "loss",
                "snr_in",
                "snr_out",
                "snr_impr",
                "sisnr_in",
                "sisnr_out",
                "sisnr_impr",
                "sisdr_in",
                "sisdr_out",
                "sisdr_impr",
                "pesq_in",
                "pesq_out",
                "pesq_impr",
            ],
        )

        if step % args.log_every == 0:
            print(
                f"step={step:05d} loss={loss.item():.6f} "
                f"snr_impr={snr_impr:.2f} sisnr_impr={sisnr_impr:.2f}"
            )

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_batch_device = _to_device(eval_batch, device)
                y_hat_eval = model(
                    eval_batch_device["y0"],
                    eval_batch_device["y1"],
                    eval_batch_device["y2"],
                )
                eval_loss = stft_magnitude_loss(
                    y_hat_eval, eval_batch_device["yt"], stft_params=loss_stft
                )
                eval_snr_in = snr_db(
                    eval_batch_device["yt"], eval_batch_device["y0"]
                ).mean().item()
                eval_snr_out = snr_db(
                    eval_batch_device["yt"], y_hat_eval
                ).mean().item()
                eval_sisnr_in = si_snr_db(
                    eval_batch_device["yt"], eval_batch_device["y0"]
                ).mean().item()
                eval_sisnr_out = si_snr_db(
                    eval_batch_device["yt"], y_hat_eval
                ).mean().item()
                eval_sisdr_in = sisdr(
                    eval_batch_device["yt"], eval_batch_device["y0"]
                ).mean().item()
                eval_sisdr_out = sisdr(
                    eval_batch_device["yt"], y_hat_eval
                ).mean().item()
            eval_snr_impr = eval_snr_out - eval_snr_in
            eval_sisnr_impr = eval_sisnr_out - eval_sisnr_in
            eval_sisdr_impr = eval_sisdr_out - eval_sisdr_in
            eval_pesq_in = float("nan")
            eval_pesq_out = float("nan")
            eval_pesq_impr = float("nan")
            if has_pesq:
                yt_np = eval_batch_device["yt"].cpu().numpy()
                y0_np = eval_batch_device["y0"].cpu().numpy()
                yhat_np = y_hat_eval.cpu().numpy()
                pesq_in_vals = []
                pesq_out_vals = []
                for idx in range(yt_np.shape[0]):
                    pesq_in_vals.append(pesq_score(yt_np[idx], y0_np[idx], args.sample_rate))
                    pesq_out_vals.append(pesq_score(yt_np[idx], yhat_np[idx], args.sample_rate))
                eval_pesq_in = float(np.mean(pesq_in_vals))
                eval_pesq_out = float(np.mean(pesq_out_vals))
                eval_pesq_impr = eval_pesq_out - eval_pesq_in

            _write_csv_row(
                eval_path,
                {
                    "step": step,
                    "loss": float(eval_loss.item()),
                    "snr_in": eval_snr_in,
                    "snr_out": eval_snr_out,
                    "snr_impr": eval_snr_impr,
                    "sisnr_in": eval_sisnr_in,
                    "sisnr_out": eval_sisnr_out,
                    "sisnr_impr": eval_sisnr_impr,
                    "sisdr_in": eval_sisdr_in,
                    "sisdr_out": eval_sisdr_out,
                    "sisdr_impr": eval_sisdr_impr,
                    "pesq_in": eval_pesq_in,
                    "pesq_out": eval_pesq_out,
                    "pesq_impr": eval_pesq_impr,
                },
                header=[
                    "step",
                    "loss",
                    "snr_in",
                    "snr_out",
                    "snr_impr",
                    "sisnr_in",
                    "sisnr_out",
                    "sisnr_impr",
                    "sisdr_in",
                    "sisdr_out",
                    "sisdr_impr",
                    "pesq_in",
                    "pesq_out",
                    "pesq_impr",
                ],
            )

            metric = eval_sisnr_impr
            if best_metric is None or metric > best_metric:
                best_metric = metric
                _save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    model,
                    optimizer,
                    step,
                    config,
                )

        if step % args.ckpt_every == 0:
            _save_checkpoint(
                run_dir / "checkpoints" / f"step_{step:05d}.pt",
                model,
                optimizer,
                step,
                config,
            )

    _save_checkpoint(
        run_dir / "checkpoints" / f"step_{args.num_steps:05d}.pt",
        model,
        optimizer,
        args.num_steps,
        config,
    )


if __name__ == "__main__":
    main()
