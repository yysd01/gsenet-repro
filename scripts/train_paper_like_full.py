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

try:  # pragma: no cover - optional soundfile
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

from gsenet_repro.config import resolve_config, resolve_run_dir, save_resolved_config
from gsenet_repro.data.oppo_triplet_dataset import OppoPrecomputedY0Dataset
from gsenet_repro.data.paper_dataset import PaperLikeDataset
from gsenet_repro.data.real_dataset import RealMultichannelDataset
from gsenet_repro.data.real_fourmic_dir_dataset import RealFourMicDirDataset
from gsenet_repro.losses.stft_loss_torch import stft_magnitude_loss
from gsenet_repro.metrics.metrics_pesq import pesq_available, pesq_score
from gsenet_repro.metrics.metrics_torch import si_snr_db, sisdr, snr_db
from gsenet_repro.models.gsenet_paper_torch import GSENetPaperScale
from gsenet_repro.models.gsenet_torch import MinimalGSENet
from gsenet_repro.pipeline.mcwf_frontend import mcwf_make_y0

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


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _format_stft_params(stft_params: Dict[str, object]) -> str:
    return "n_fft={n_fft} win_length={win_length} hop_length={hop_length} window={window} center={center}".format(
        n_fft=stft_params.get("n_fft"),
        win_length=stft_params.get("win_length"),
        hop_length=stft_params.get("hop_length"),
        window=stft_params.get("window", "hann"),
        center=stft_params.get("center", False),
    )


def _dataset_size(dataset: torch.utils.data.Dataset) -> str:
    if hasattr(dataset, "__len__"):
        try:
            return str(len(dataset))
        except TypeError:
            return "unknown"
    return "unknown"


def _build_model(config: Dict[str, object]) -> tuple[torch.nn.Module, str]:
    model_config = config.get("model", {})
    model_name = str(model_config.get("name", "gsenet_paper_scale"))
    if model_name == "gsenet_paper_scale":
        model = GSENetPaperScale(
            stft_params=config["stft_model"],
            leaky_relu_slope=float(model_config.get("leaky_relu_slope", 0.3)),
            encoder_blocks=model_config.get("encoder_blocks"),
            decoder_blocks=model_config.get("decoder_blocks"),
            stem_channels=int(model_config.get("stem_channels", 16)),
            head_channels=int(model_config.get("head_channels", 2)),
            remove_dc=bool(model_config.get("remove_dc", False)),
        )
    elif model_name == "minimal":
        model = MinimalGSENet(stft_params=config["stft_model"])
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model, model_name


def _forward_model(
    model: torch.nn.Module,
    model_name: str,
    y0: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor | None,
) -> torch.Tensor:
    if model_name == "gsenet_paper_scale":
        return model(y0, y1)
    if model_name == "minimal":
        if y2 is None:
            raise ValueError("MinimalGSENet requires y2 input.")
        return model(y0, y1, y2)
    raise ValueError(f"Unknown model name: {model_name}")


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
    pairing_config: Dict[str, object] | None = None,
) -> tuple[Dict[str, torch.Tensor], torch.utils.data.Dataset]:
    if data_config.get("dataset_type") == "oppo_triplet":
        dataset = OppoPrecomputedY0Dataset(
            dataset_root=data_config["dataset_root"],
            precomputed_y0_root=data_config["precomputed_y0_root"],
            split="train",
            case_filter=data_config.get("case_filter"),
            sample_rate=data_config["sample_rate"],
            ref_mic_index=data_config.get("ref_mic_index", 0),
        )
    elif data_config["mode"] == "real":
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
            mic_positions=data_config.get("mic_positions"),
        )
    elif data_config["mode"] == "real_dir":
        dataset = RealFourMicDirDataset(
            root=data_config["root"],
            split="valid",
            sample_rate=data_config["sample_rate"],
            segment_seconds=data_config["segment_seconds"],
            num_mics=data_config["num_mics"],
            ref_mic_index=data_config["ref_mic_index"],
            clean_ref_mic_index=data_config.get("clean_ref_mic_index", 0),
            clean_is_multichannel=bool(data_config.get("clean_is_multichannel", True)),
            random_crop=False,
            eval_full_length=bool(data_config.get("eval_full_length", False)),
            fixed_crop=data_config.get("fixed_crop", "center"),
            resample=bool(data_config.get("resample", True)),
            cache_metadata=bool(data_config.get("cache_metadata", True)),
            pairing_config=pairing_config,
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
            mic_positions=data_config.get("mic_positions"),
        )
    loader = DataLoader(dataset, batch_size=batch_size)
    return next(iter(loader)), dataset


def _log_real_dir_dataset(
    dataset: RealFourMicDirDataset, split: str, data_config: Dict[str, object]
) -> None:
    resample = bool(data_config.get("resample", True))
    print(
        "RealFourMicDirDataset split={split} root={root} samples={samples} "
        "segment_seconds={segment_seconds} sample_rate={sample_rate} num_mics={num_mics} "
        "resample={resample} use_mcwf={use_mcwf} ref_mic_index={ref_mic_index} "
        "clean_ref_mic_index={clean_ref_mic_index}".format(
            split=split,
            root=data_config.get("root"),
            samples=len(dataset),
            segment_seconds=data_config["segment_seconds"],
            sample_rate=data_config["sample_rate"],
            num_mics=data_config["num_mics"],
            resample=resample,
            use_mcwf=bool(data_config["use_mcwf"]),
            ref_mic_index=data_config.get("ref_mic_index", 0),
            clean_ref_mic_index=data_config.get("clean_ref_mic_index", 0),
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
        y0 = mcwf_make_y0(
            x_mics,
            stft_params=stft_params,
            causal_frames=data_config["mcwf_causal_frames"],
            ref_ch=int(data_config["ref_mic_index"]),
            sample_rate=int(data_config["sample_rate"]),
            mic_positions=data_config.get("mic_positions"),
        )
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


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.detach().float() ** 2)).item())


def _clip_ratio(x: torch.Tensor, threshold: float = 0.999) -> float:
    return float((x.detach().abs() > threshold).float().mean().item())


def _grad_stats(model: torch.nn.Module) -> tuple[float, float]:
    sum_sq = 0.0
    grad_max = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        sum_sq += float(torch.sum(grad * grad).item())
        if grad.numel() > 0:
            grad_max = max(grad_max, float(grad.abs().max().item()))
    return float(np.sqrt(sum_sq)), grad_max


def _param_norm(model: torch.nn.Module) -> float:
    sum_sq = 0.0
    for param in model.parameters():
        data = param.detach().float()
        sum_sq += float(torch.sum(data * data).item())
    return float(np.sqrt(sum_sq))


def _compute_metric_summary(yt: torch.Tensor, y0: torch.Tensor, y1: torch.Tensor, y_hat: torch.Tensor) -> Dict[str, float]:
    snr_in = snr_db(yt, y0).mean().item()
    snr_out = snr_db(yt, y_hat).mean().item()
    snr_y1 = snr_db(yt, y1).mean().item()
    sisnr_in = si_snr_db(yt, y0).mean().item()
    sisnr_out = si_snr_db(yt, y_hat).mean().item()
    sisnr_y1 = si_snr_db(yt, y1).mean().item()
    sisdr_in = sisdr(yt, y0).mean().item()
    sisdr_out = sisdr(yt, y_hat).mean().item()
    sisdr_y1 = sisdr(yt, y1).mean().item()
    return {
        "snr_in": snr_in,
        "snr_out": snr_out,
        "snr_y1": snr_y1,
        "snr_impr": snr_out - snr_in,
        "delta_snr_y0_vs_y1": snr_in - snr_y1,
        "delta_snr_yhat_vs_y0": snr_out - snr_in,
        "delta_snr_yhat_vs_y1": snr_out - snr_y1,
        "sisnr_in": sisnr_in,
        "sisnr_out": sisnr_out,
        "sisnr_y1": sisnr_y1,
        "sisnr_impr": sisnr_out - sisnr_in,
        "delta_sisnr_y0_vs_y1": sisnr_in - sisnr_y1,
        "delta_sisnr_yhat_vs_y0": sisnr_out - sisnr_in,
        "delta_sisnr_yhat_vs_y1": sisnr_out - sisnr_y1,
        "sisdr_in": sisdr_in,
        "sisdr_out": sisdr_out,
        "sisdr_y1": sisdr_y1,
        "sisdr_impr": sisdr_out - sisdr_in,
        "delta_sisdr_y0_vs_y1": sisdr_in - sisdr_y1,
        "delta_sisdr_yhat_vs_y0": sisdr_out - sisdr_in,
        "delta_sisdr_yhat_vs_y1": sisdr_out - sisdr_y1,
    }


def _dump_debug(
    run_dir: Path,
    step: int,
    batch: Dict[str, torch.Tensor],
    y_hat: torch.Tensor,
    sample_rate: int,
    debug_cfg: Dict[str, object],
    extra_stats_dict: Dict[str, object],
) -> None:
    debug_dir_name = str(debug_cfg.get("dir", "debug"))
    dump_dir = run_dir / debug_dir_name / f"step_{step:05d}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    max_items = max(1, int(debug_cfg.get("max_items", 2)))
    seconds = float(debug_cfg.get("seconds", 3.0))
    clip_samples = max(1, int(round(sample_rate * seconds)))

    tensors = {
        "y0": batch["y0"],
        "y1": batch["y1"],
        "yt": batch["yt"],
        "yhat": y_hat,
    }

    saved_tensors: Dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        clipped = value.detach().float().cpu()[:max_items, :clip_samples]
        saved_tensors[name] = clipped
        for idx in range(clipped.shape[0]):
            item_audio = clipped[idx].numpy()
            stem = dump_dir / f"{name}_{idx:02d}"
            if sf is not None:
                sf.write(stem.with_suffix(".wav"), item_audio, sample_rate)
            else:
                np.save(stem.with_suffix(".npy"), item_audio)

    torch.save(saved_tensors, dump_dir / "batch.pt")
    with (dump_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(extra_stats_dict, handle, indent=2, sort_keys=True)


def _log_eval_metrics(
    step: int,
    eval_loss: torch.Tensor,
    eval_batch_device: Dict[str, torch.Tensor],
    y_hat_eval: torch.Tensor,
    sample_rate: int,
    has_pesq: bool,
) -> Dict[str, float]:
    summary = _compute_metric_summary(
        eval_batch_device["yt"], eval_batch_device["y0"], eval_batch_device["y1"], y_hat_eval
    )
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
    y0_rms = _rms(eval_batch_device["y0"])
    y1_rms = _rms(eval_batch_device["y1"])
    yt_rms = _rms(eval_batch_device["yt"])
    yhat_rms = _rms(y_hat_eval)
    return {
        "step": step,
        "loss": float(eval_loss.item()),
        **summary,
        "grad_norm_l2": float("nan"),
        "grad_max_abs": float("nan"),
        "param_norm_l2": float("nan"),
        "y0_rms": y0_rms,
        "y1_rms": y1_rms,
        "yt_rms": yt_rms,
        "yhat_rms": yhat_rms,
        "rms_ratio_yhat_to_yt": yhat_rms / (yt_rms + 1e-8),
        "clip_ratio_yhat": _clip_ratio(y_hat_eval),
        "pesq_in": eval_pesq_in,
        "pesq_out": eval_pesq_out,
        "pesq_impr": eval_pesq_impr,
    }


def train_with_config(config: Dict[str, object], config_path: str | None = None) -> None:
    run_config = config["run"]
    data_config = config["data"]
    pairing_config = config.get("pairing")
    train_config = config["train"]
    debug_cfg = {
        "enable": False,
        "dump_every": 0,
        "dump_on_nan": True,
        "max_items": 2,
        "seconds": 3.0,
        "dir": "debug",
    }
    debug_cfg.update(config.get("debug", {}))
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
    print(
        "data_ref_mic_index={ref_mic_index} data_clean_ref_mic_index={clean_ref_mic_index}".format(
            ref_mic_index=data_config.get("ref_mic_index", 0),
            clean_ref_mic_index=data_config.get("clean_ref_mic_index", 0),
        )
    )

    if data_config.get("dataset_type") == "oppo_triplet":
        dataset = OppoPrecomputedY0Dataset(
            dataset_root=data_config["dataset_root"],
            precomputed_y0_root=data_config["precomputed_y0_root"],
            split="train",
            case_filter=data_config.get("case_filter"),
            sample_rate=data_config["sample_rate"],
            ref_mic_index=data_config.get("ref_mic_index", 0),
        )
    elif data_config["mode"] == "real":
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
            mic_positions=data_config.get("mic_positions"),
        )
    elif data_config["mode"] == "real_dir":
        dataset = RealFourMicDirDataset(
            root=data_config["root"],
            split="train",
            sample_rate=data_config["sample_rate"],
            segment_seconds=data_config["segment_seconds"],
            num_mics=data_config["num_mics"],
            ref_mic_index=data_config["ref_mic_index"],
            clean_ref_mic_index=data_config.get("clean_ref_mic_index", 0),
            clean_is_multichannel=bool(data_config.get("clean_is_multichannel", True)),
            random_crop=True,
            eval_full_length=False,
            fixed_crop=data_config.get("fixed_crop", "center"),
            resample=bool(data_config.get("resample", True)),
            cache_metadata=bool(data_config.get("cache_metadata", True)),
            pairing_config=pairing_config,
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
            mic_positions=data_config.get("mic_positions"),
        )

    loader = _make_dataloader(
        dataset,
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"],
        prefetch_factor=train_config["prefetch_factor"],
    )
    train_iter = iter(loader)

    eval_seed = run_config["seed"] + 1234
    eval_batch, eval_dataset = _make_eval_batch(
        data_config=data_config,
        batch_size=train_config["batch_size"],
        stft_params=model_stft,
        seed=eval_seed,
        pairing_config=pairing_config,
    )

    model, model_name = _build_model(config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

    print(
        "config_path={config_path} data_mode={mode} split=train samples={samples} "
        "batch_size={batch_size} num_steps={num_steps} lr={lr}".format(
            config_path=config_path or "default",
            mode=data_config["mode"],
            samples=_dataset_size(dataset),
            batch_size=train_config["batch_size"],
            num_steps=train_config["num_steps"],
            lr=train_config["lr"],
        )
    )
    print("eval split=valid samples={samples}".format(samples=_dataset_size(eval_dataset)))
    print(
        "model={name} params={params:.3f}M".format(
            name=model_name, params=_count_parameters(model) / 1e6
        )
    )
    print("model_stft " + _format_stft_params(model_stft))
    print("loss_stft " + _format_stft_params(loss_stft))

    sanity_batch = _prepare_batch_for_model(_to_device(eval_batch, device), data_config, model_stft)
    with torch.no_grad():
        sanity_delta_sisdr = (
            sisdr(sanity_batch["yt"], sanity_batch["y0"]).mean().item()
            - sisdr(sanity_batch["yt"], sanity_batch["y1"]).mean().item()
        )
        print(
            "sanity valid_batch_len={length} sample_rate={sample_rate} "
            "y0_rms={y0_rms:.5f} y1_rms={y1_rms:.5f} yt_rms={yt_rms:.5f} "
            "delta_sisdr_y0_vs_y1={delta_sisdr:.3f}".format(
                length=sanity_batch["yt"].shape[-1],
                sample_rate=data_config["sample_rate"],
                y0_rms=_rms(sanity_batch["y0"]),
                y1_rms=_rms(sanity_batch["y1"]),
                yt_rms=_rms(sanity_batch["yt"]),
                delta_sisdr=sanity_delta_sisdr,
            )
        )
        if sanity_delta_sisdr <= 0:
            print(
                "WARNING sanity delta_sisdr_y0_vs_y1 <= 0. "
                "MCWF y0 may not outperform y1 on this fixed eval batch."
            )

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
        "snr_y1",
        "snr_impr",
        "delta_snr_y0_vs_y1",
        "delta_snr_yhat_vs_y0",
        "delta_snr_yhat_vs_y1",
        "sisnr_in",
        "sisnr_out",
        "sisnr_y1",
        "sisnr_impr",
        "delta_sisnr_y0_vs_y1",
        "delta_sisnr_yhat_vs_y0",
        "delta_sisnr_yhat_vs_y1",
        "sisdr_in",
        "sisdr_out",
        "sisdr_y1",
        "sisdr_impr",
        "delta_sisdr_y0_vs_y1",
        "delta_sisdr_yhat_vs_y0",
        "delta_sisdr_yhat_vs_y1",
        "grad_norm_l2",
        "grad_max_abs",
        "param_norm_l2",
        "y0_rms",
        "y1_rms",
        "yt_rms",
        "yhat_rms",
        "rms_ratio_yhat_to_yt",
        "clip_ratio_yhat",
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
        y_hat = _forward_model(model, model_name, y0, y1, y2)
        loss = stft_magnitude_loss(y_hat, yt, stft_params=loss_stft)
        loss.backward()
        grad_norm_l2, grad_max_abs = _grad_stats(model)
        param_norm_l2 = _param_norm(model)

        loss_finite = bool(torch.isfinite(loss).all().item())
        yhat_finite = bool(torch.isfinite(y_hat).all().item())
        grad_finite = bool(np.isfinite(grad_norm_l2)) and bool(np.isfinite(grad_max_abs))
        if not (loss_finite and yhat_finite and grad_finite):
            debug_meta = {
                "step": step,
                "error": "non_finite_detected_before_optimizer_step",
                "loss": float(loss.item()) if torch.isfinite(loss).all() else "non_finite",
                "loss_is_finite": loss_finite,
                "yhat_is_finite": yhat_finite,
                "grad_is_finite": grad_finite,
                "grad_norm_l2": grad_norm_l2,
                "grad_max_abs": grad_max_abs,
                "sample_rate": data_config["sample_rate"],
                "ref_mic_index": data_config.get("ref_mic_index", 0),
                "model_stft": model_stft,
                "loss_stft": loss_stft,
            }
            if bool(debug_cfg.get("dump_on_nan", True)):
                _dump_debug(run_dir, step, batch, y_hat, data_config["sample_rate"], debug_cfg, debug_meta)
            raise RuntimeError(f"Non-finite training state at step {step}: {debug_meta}")

        optimizer.step()

        with torch.no_grad():
            metrics_summary = _compute_metric_summary(yt, y0, y1, y_hat)
            y0_rms = _rms(y0)
            y1_rms = _rms(y1)
            yt_rms = _rms(yt)
            yhat_rms = _rms(y_hat)
            rms_ratio_yhat_to_yt = yhat_rms / (yt_rms + 1e-8)
            clip_ratio_yhat = _clip_ratio(y_hat)

        dump_extra = {
            "step": step,
            "loss": float(loss.item()),
            **metrics_summary,
            "grad_norm_l2": grad_norm_l2,
            "grad_max_abs": grad_max_abs,
            "param_norm_l2": param_norm_l2,
            "y0_rms": y0_rms,
            "y1_rms": y1_rms,
            "yt_rms": yt_rms,
            "yhat_rms": yhat_rms,
            "rms_ratio_yhat_to_yt": rms_ratio_yhat_to_yt,
            "clip_ratio_yhat": clip_ratio_yhat,
            "sample_rate": data_config["sample_rate"],
            "ref_mic_index": data_config.get("ref_mic_index", 0),
            "model_stft": model_stft,
            "loss_stft": loss_stft,
        }
        if bool(debug_cfg.get("enable", False)) and int(debug_cfg.get("dump_every", 0)) > 0:
            if step % int(debug_cfg["dump_every"]) == 0:
                _dump_debug(run_dir, step, batch, y_hat, data_config["sample_rate"], debug_cfg, dump_extra)

        _write_csv_row(
            metrics_path,
            {
                "step": step,
                "loss": float(loss.item()),
                **metrics_summary,
                "grad_norm_l2": grad_norm_l2,
                "grad_max_abs": grad_max_abs,
                "param_norm_l2": param_norm_l2,
                "y0_rms": y0_rms,
                "y1_rms": y1_rms,
                "yt_rms": yt_rms,
                "yhat_rms": yhat_rms,
                "rms_ratio_yhat_to_yt": rms_ratio_yhat_to_yt,
                "clip_ratio_yhat": clip_ratio_yhat,
                "pesq_in": float("nan"),
                "pesq_out": float("nan"),
                "pesq_impr": float("nan"),
            },
            header=log_header,
        )

        if tqdm:
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                delta_sisnr_yhat_vs_y0=f"{metrics_summary['delta_sisnr_yhat_vs_y0']:.2f}",
                delta_sisnr_yhat_vs_y1=f"{metrics_summary['delta_sisnr_yhat_vs_y1']:.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        elif step % train_config["log_every"] == 0:
            elapsed = time.time() - last_log_time
            last_log_time = time.time()
            print(
                f"step={step:05d} loss={loss.item():.6f} "
                f"delta_sisnr_yhat_vs_y0={metrics_summary['delta_sisnr_yhat_vs_y0']:.2f} "
                f"delta_sisnr_yhat_vs_y1={metrics_summary['delta_sisnr_yhat_vs_y1']:.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"iter_s={train_config['log_every'] / max(elapsed, 1e-6):.2f}"
            )

        if step % train_config["eval_every"] == 0:
            model.eval()
            with torch.no_grad():
                eval_batch_device = _prepare_batch_for_model(
                    _to_device(eval_batch, device), data_config, model_stft
                )
                y_hat_eval = _forward_model(
                    model,
                    model_name,
                    eval_batch_device["y0"],
                    eval_batch_device["y1"],
                    eval_batch_device.get("y2"),
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
    parser.add_argument("--debug_enable", type=int, default=None)
    parser.add_argument("--debug_dump_every", type=int, default=None)
    parser.add_argument("--debug_dump_on_nan", type=int, default=None)
    parser.add_argument("--debug_max_items", type=int, default=None)
    parser.add_argument("--debug_seconds", type=float, default=None)
    parser.add_argument("--debug_dir", type=str, default=None)
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

    debug_overrides: Dict[str, object] = {}
    if args.debug_enable is not None:
        debug_overrides["enable"] = bool(args.debug_enable)
    if args.debug_dump_every is not None:
        debug_overrides["dump_every"] = args.debug_dump_every
    if args.debug_dump_on_nan is not None:
        debug_overrides["dump_on_nan"] = bool(args.debug_dump_on_nan)
    if args.debug_max_items is not None:
        debug_overrides["max_items"] = args.debug_max_items
    if args.debug_seconds is not None:
        debug_overrides["seconds"] = args.debug_seconds
    if args.debug_dir is not None:
        debug_overrides["dir"] = args.debug_dir
    if debug_overrides:
        overrides["debug"] = debug_overrides

    config = resolve_config(args.config, overrides=overrides)
    train_with_config(config, config_path=args.config)


if __name__ == "__main__":
    main()
