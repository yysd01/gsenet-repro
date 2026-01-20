from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
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

from gsenet_repro.config import resolve_config, resolve_run_dir, save_resolved_config
from gsenet_repro.data.paper_dataset import PaperLikeDataset
from gsenet_repro.data.real_fourmic_dir_dataset import RealFourMicDirDataset
from gsenet_repro.data.real_dataset import RealMultichannelDataset
from gsenet_repro.pipeline.mcwf_frontend import mcwf_make_y0
from gsenet_repro.losses.stft_loss_torch import stft_magnitude_loss
from gsenet_repro.metrics.metrics_pesq import pesq_available, pesq_score
from gsenet_repro.metrics.metrics_torch import si_snr_db, sisdr, snr_db
from gsenet_repro.models.gsenet_torch import MinimalGSENet

try:  # pragma: no cover - optional tqdm
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _build_run_dir(run_dir: str | None) -> Path:
    return resolve_run_dir(run_dir)


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
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _make_eval_batch(
    data_config: Dict[str, object],
    batch_size: int,
    stft_params: Dict[str, int],
    seed: int,
) -> Dict[str, torch.Tensor]:
    if data_config["mode"] == "real":
        dataset = RealMultichannelDataset(
            manifest_path=data_config.get("manifest_path"),
            root_dir=data_config.get("root_dir"),
            sample_rate=data_config["sample_rate"],
            segment_seconds=data_config["segment_seconds"],
            num_mics=data_config["num_mics"],
            ref_mic_index=data_config["ref_mic_index"],
            use_mcwf=bool(data_config["use_mcwf"]),
            stft_params=stft_params,
            causal_frames=data_config["mcwf_causal_frames"],
            seed=seed,
        )
    elif data_config["mode"] == "real_dir":
        dataset = RealFourMicDirDataset(
            root=data_config["root"],
            split="valid",
            sample_rate=data_config["sample_rate"],
            segment_seconds=data_config["segment_seconds"],
            num_mics=data_config["num_mics"],
            ref_mic_index=data_config["ref_mic_index"],
            random_crop=False,
            eval_full_length=bool(data_config.get("eval_full_length", False)),
            fixed_crop=data_config.get("fixed_crop", "center"),
            resample=bool(data_config.get("resample", True)),
            cache_metadata=bool(data_config.get("cache_metadata", True)),
        )
        _log_real_dir_dataset(dataset, "valid", data_config)
    else:
        dataset = PaperLikeDataset(
            sample_rate=data_config["sample_rate"],
            segment_seconds=data_config["segment_seconds"],
            seed=seed,
            num_samples=batch_size,
            use_mcwf=bool(data_config["use_mcwf"]),
            ref_mic=data_config["ref_mic_index"],
            num_mics=data_config["num_mics"],
            stft_params=stft_params,
            causal_frames=data_config["mcwf_causal_frames"],
        )
    loader = DataLoader(dataset, batch_size=batch_size)
    return next(iter(loader))


def _log_real_dir_dataset(dataset: RealFourMicDirDataset, split: str, data_config: Dict[str, object]) -> None:
    resample = bool(data_config.get("resample", True))
    print(
        "RealFourMicDirDataset split={split} root={root} samples={samples} "
        "segment_seconds={segment_seconds} sample_rate={sample_rate} num_mics={num_mics} "
        "resample={resample} use_mcwf={use_mcwf}".format(
            split=split,
            root=data_config.get("root"),
            samples=len(dataset),
            segment_seconds=data_config["segment_seconds"],
            sample_rate=data_config["sample_rate"],
            num_mics=data_config["num_mics"],
            resample=resample,
            use_mcwf=bool(data_config["use_mcwf"]),
        )
    )


def _prepare_batch_for_model(
    batch: Dict[str, torch.Tensor],
    data_config: Dict[str, object],
    stft_params: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    if "x_mics" not in batch:
        return batch
    x_mics = batch["x_mics"]
    y1 = batch["y1"]
    yt = batch["yt"]
    if bool(data_config["use_mcwf"]):
        y0 = mcwf_make_y0(x_mics, stft_params=stft_params, causal_frames=data_config["mcwf_causal_frames"])
    else:
        y0 = y1
    mic_index = min(2, x_mics.shape[1] - 1)
    y2 = x_mics[:, mic_index]
    prepared = dict(batch)
    prepared.update({"y0": y0, "y1": y1, "y2": y2, "yt": yt})
    return prepared


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


def _make_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    kwargs = {}
    if num_workers > 0:
        kwargs["num_workers"] = num_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, batch_size=batch_size, **kwargs)


def _log_eval_metrics(
    step: int,
    eval_loss: torch.Tensor,
    eval_batch_device: Dict[str, torch.Tensor],
    y_hat_eval: torch.Tensor,
    sample_rate: int,
    has_pesq: bool,
) -> Dict[str, float]:
    eval_snr_in = snr_db(eval_batch_device["yt"], eval_batch_device["y0"]).mean().item()
    eval_snr_out = snr_db(eval_batch_device["yt"], y_hat_eval).mean().item()
    eval_sisnr_in = si_snr_db(eval_batch_device["yt"], eval_batch_device["y0"]).mean().item()
    eval_sisnr_out = si_snr_db(eval_batch_device["yt"], y_hat_eval).mean().item()
    eval_sisdr_in = sisdr(eval_batch_device["yt"], eval_batch_device["y0"]).mean().item()
    eval_sisdr_out = sisdr(eval_batch_device["yt"], y_hat_eval).mean().item()
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
            pesq_in_vals.append(pesq_score(yt_np[idx], y0_np[idx], sample_rate))
            pesq_out_vals.append(pesq_score(yt_np[idx], yhat_np[idx], sample_rate))
        eval_pesq_in = float(np.mean(pesq_in_vals))
        eval_pesq_out = float(np.mean(pesq_out_vals))
        eval_pesq_impr = eval_pesq_out - eval_pesq_in
    return {
        "step": step,
        "loss": float(eval_loss.item()),
        "snr_in": eval_snr_in,
        "snr_out": eval_snr_out,
        "snr_impr": eval_snr_out - eval_snr_in,
        "sisnr_in": eval_sisnr_in,
        "sisnr_out": eval_sisnr_out,
        "sisnr_impr": eval_sisnr_out - eval_sisnr_in,
        "sisdr_in": eval_sisdr_in,
        "sisdr_out": eval_sisdr_out,
        "sisdr_impr": eval_sisdr_out - eval_sisdr_in,
        "pesq_in": eval_pesq_in,
        "pesq_out": eval_pesq_out,
        "pesq_impr": eval_pesq_impr,
    }


def train_with_config(config: Dict[str, object]) -> None:
    run_config = config["run"]
    data_config = config["data"]
    train_config = config["train"]
    loss_stft = config["stft_loss"]
    model_stft = config["stft_model"]

    run_dir = _build_run_dir(run_config.get("run_dir"))
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(run_dir, config)

    device = (
        torch.device("cuda")
        if run_config["device"] == "auto" and torch.cuda.is_available()
        else torch.device(run_config["device"] if run_config["device"] != "auto" else "cpu")
    )

    _seed_everything(run_config["seed"])

    if data_config["mode"] == "real":
        dataset = RealMultichannelDataset(
            manifest_path=data_config.get("manifest_path"),
            root_dir=data_config.get("root_dir"),
            sample_rate=data_config["sample_rate"],
            segment_seconds=data_config["segment_seconds"],
            num_mics=data_config["num_mics"],
            ref_mic_index=data_config["ref_mic_index"],
            use_mcwf=bool(data_config["use_mcwf"]),
            stft_params=model_stft,
            causal_frames=data_config["mcwf_causal_frames"],
            seed=run_config["seed"],
        )
    elif data_config["mode"] == "real_dir":
        dataset = RealFourMicDirDataset(
            root=data_config["root"],
            split="train",
            sample_rate=data_config["sample_rate"],
            segment_seconds=data_config["segment_seconds"],
            num_mics=data_config["num_mics"],
            ref_mic_index=data_config["ref_mic_index"],
            random_crop=True,
            eval_full_length=False,
            fixed_crop=data_config.get("fixed_crop", "center"),
            resample=bool(data_config.get("resample", True)),
            cache_metadata=bool(data_config.get("cache_metadata", True)),
        )
        _log_real_dir_dataset(dataset, "train", data_config)
    else:
        dataset = PaperLikeDataset(
            sample_rate=data_config["sample_rate"],
            segment_seconds=data_config["segment_seconds"],
            seed=run_config["seed"],
            num_samples=None,
            use_mcwf=bool(data_config["use_mcwf"]),
            ref_mic=data_config["ref_mic_index"],
            num_mics=data_config["num_mics"],
            stft_params=model_stft,
            causal_frames=data_config["mcwf_causal_frames"],
        )

    loader = _make_dataloader(
        dataset,
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"],
        prefetch_factor=train_config["prefetch_factor"],
    )
    train_iter = iter(loader)

    eval_seed = run_config["seed"] + 1234
    eval_batch = _make_eval_batch(
        data_config=data_config,
        batch_size=train_config["batch_size"],
        stft_params=model_stft,
        seed=eval_seed,
    )

    model = MinimalGSENet(stft_params=model_stft).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

    metrics_path = run_dir / "metrics.csv"
    eval_path = run_dir / "eval.csv"
    best_metric = None
    has_pesq = pesq_available() if config["metrics"]["enable_pesq"] else False
    if not has_pesq:
        print("pesq not installed, skipping PESQ metrics.", file=sys.stderr)

    log_header = [
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
    ]

    progress_iter = range(1, train_config["num_steps"] + 1)
    progress = tqdm(progress_iter, total=train_config["num_steps"]) if tqdm else progress_iter
    last_log_time = time.time()

    for step in progress:
        batch = next(train_iter)
        batch = _prepare_batch_for_model(_to_device(batch, device), data_config, model_stft)
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
            header=log_header,
        )

        if tqdm:
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                snr_impr=f"{snr_impr:.2f}",
                sisnr_impr=f"{sisnr_impr:.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        elif step % train_config["log_every"] == 0:
            elapsed = time.time() - last_log_time
            last_log_time = time.time()
            print(
                f"step={step:05d} loss={loss.item():.6f} "
                f"snr_impr={snr_impr:.2f} sisnr_impr={sisnr_impr:.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"iter_s={train_config['log_every'] / max(elapsed, 1e-6):.2f}"
            )

        if step % train_config["eval_every"] == 0:
            model.eval()
            with torch.no_grad():
                eval_batch_device = _prepare_batch_for_model(
                    _to_device(eval_batch, device), data_config, model_stft
                )
                y_hat_eval = model(
                    eval_batch_device["y0"],
                    eval_batch_device["y1"],
                    eval_batch_device["y2"],
                )
                eval_loss = stft_magnitude_loss(
                    y_hat_eval, eval_batch_device["yt"], stft_params=loss_stft
                )
            eval_row = _log_eval_metrics(
                step=step,
                eval_loss=eval_loss,
                eval_batch_device=eval_batch_device,
                y_hat_eval=y_hat_eval,
                sample_rate=data_config["sample_rate"],
                has_pesq=has_pesq,
            )
            _write_csv_row(eval_path, eval_row, header=log_header)

            print(
                "eval step={step:05d} loss={loss:.6f} snr_impr={snr_impr:.2f} "
                "sisnr_impr={sisnr_impr:.2f} sisdr_impr={sisdr_impr:.2f} pesq_impr={pesq}".format(
                    step=step,
                    loss=eval_row["loss"],
                    snr_impr=eval_row["snr_impr"],
                    sisnr_impr=eval_row["sisnr_impr"],
                    sisdr_impr=eval_row["sisdr_impr"],
                    pesq=f"{eval_row['pesq_impr']:.2f}" if has_pesq else "NA",
                )
            )

            metric = eval_row["sisnr_impr"]
            if best_metric is None or metric > best_metric:
                best_metric = metric
                _save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    model,
                    optimizer,
                    step,
                    config,
                )

        if step % train_config["ckpt_every"] == 0:
            _save_checkpoint(
                run_dir / "checkpoints" / f"step_{step:05d}.pt",
                model,
                optimizer,
                step,
                config,
            )

    _save_checkpoint(
        run_dir / "checkpoints" / f"step_{train_config['num_steps']:05d}.pt",
        model,
        optimizer,
        train_config["num_steps"],
        config,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-like full training pipeline.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=None)
    parser.add_argument("--segment_seconds", type=float, default=None)
    parser.add_argument("--num_mics", type=int, default=None)
    parser.add_argument("--ref_mic_index", type=int, default=None)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--use_mcwf", type=int, default=None)
    parser.add_argument("--mcwf_causal_frames", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--ckpt_every", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--data_mode", type=str, default=None)
    parser.add_argument("--eval_full_length", type=int, default=None)
    parser.add_argument("--fixed_crop", type=str, default=None)
    parser.add_argument("--resample", type=int, default=None)
    parser.add_argument("--cache_metadata", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    overrides: Dict[str, Dict[str, object]] = {}

    run_overrides: Dict[str, object] = {}
    if args.run_dir is not None:
        run_overrides["run_dir"] = args.run_dir
    if args.seed is not None:
        run_overrides["seed"] = args.seed
    if args.device is not None:
        run_overrides["device"] = args.device
    if run_overrides:
        overrides["run"] = run_overrides

    data_overrides: Dict[str, object] = {}
    if args.sample_rate is not None:
        data_overrides["sample_rate"] = args.sample_rate
    if args.segment_seconds is not None:
        data_overrides["segment_seconds"] = args.segment_seconds
    if args.num_mics is not None:
        data_overrides["num_mics"] = args.num_mics
    if args.ref_mic_index is not None:
        data_overrides["ref_mic_index"] = args.ref_mic_index
    if args.manifest_path is not None:
        data_overrides["manifest_path"] = args.manifest_path
    if args.root_dir is not None:
        data_overrides["root_dir"] = args.root_dir
    if args.root is not None:
        data_overrides["root"] = args.root
    if args.use_mcwf is not None:
        data_overrides["use_mcwf"] = args.use_mcwf
    if args.mcwf_causal_frames is not None:
        data_overrides["mcwf_causal_frames"] = args.mcwf_causal_frames
    if args.data_mode is not None:
        data_overrides["mode"] = args.data_mode
    if args.eval_full_length is not None:
        data_overrides["eval_full_length"] = bool(args.eval_full_length)
    if args.fixed_crop is not None:
        data_overrides["fixed_crop"] = args.fixed_crop
    if args.resample is not None:
        data_overrides["resample"] = bool(args.resample)
    if args.cache_metadata is not None:
        data_overrides["cache_metadata"] = bool(args.cache_metadata)
    if data_overrides:
        overrides["data"] = data_overrides

    train_overrides: Dict[str, object] = {}
    if args.batch_size is not None:
        train_overrides["batch_size"] = args.batch_size
    if args.num_steps is not None:
        train_overrides["num_steps"] = args.num_steps
    if args.log_every is not None:
        train_overrides["log_every"] = args.log_every
    if args.eval_every is not None:
        train_overrides["eval_every"] = args.eval_every
    if args.ckpt_every is not None:
        train_overrides["ckpt_every"] = args.ckpt_every
    if args.lr is not None:
        train_overrides["lr"] = args.lr
    if args.num_workers is not None:
        train_overrides["num_workers"] = args.num_workers
    if args.prefetch_factor is not None:
        train_overrides["prefetch_factor"] = args.prefetch_factor
    if train_overrides:
        overrides["train"] = train_overrides

    config = resolve_config(args.config, overrides=overrides)
    train_with_config(config)


if __name__ == "__main__":
    main()
