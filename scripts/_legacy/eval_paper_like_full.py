# LEGACY / internal demo
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

if importlib.util.find_spec("torch") is None:
    print(
        "torch not installed. Install requirements-torch.txt to run evaluation.",
        file=sys.stderr,
    )
    raise SystemExit(1)

import torch
from torch.utils.data import DataLoader

from gsenet_repro.config import resolve_config
from gsenet_repro.data.paper_dataset import PaperLikeDataset
from gsenet_repro.data.real_dataset import RealMultichannelDataset
from gsenet_repro.data.real_fourmic_dir_dataset import RealFourMicDirDataset
from gsenet_repro.io.audio import safe_write_wav
from gsenet_repro.losses.stft_loss_torch import stft_magnitude_loss
from gsenet_repro.metrics.metrics_pesq import pesq_available, pesq_score
from gsenet_repro.metrics.metrics_torch import si_snr_db, sisdr, snr_db
from gsenet_repro.models.gsenet_paper_torch import GSENetPaperScale
from gsenet_repro.models.gsenet_torch import MinimalGSENet
from gsenet_repro.pipeline.mcwf_frontend import mcwf_make_y0


def _normalize_audio(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    max_val = np.max(np.abs(x)) + 1e-8
    return (x / max_val * peak).astype(np.float32)


def _load_config_from_ckpt(ckpt: dict) -> dict:
    raw_config = ckpt.get("config", {})
    if isinstance(raw_config, dict) and "data" in raw_config:
        return raw_config
    return resolve_config(None, overrides={"data": raw_config})


def _make_dataset(config: dict, num_samples: int, seed: int) -> torch.utils.data.Dataset:
    data_config = config["data"]
    pairing_config = config.get("pairing")
    if data_config["mode"] == "real":
        return RealMultichannelDataset(
            manifest_path=data_config.get("manifest_path"),
            root_dir=data_config.get("root_dir"),
            sample_rate=data_config["sample_rate"],
            segment_seconds=data_config["segment_seconds"],
            num_mics=data_config["num_mics"],
            ref_mic_index=data_config["ref_mic_index"],
            use_mcwf=bool(data_config["use_mcwf"]),
            stft_params=config["stft_model"],
            causal_frames=data_config["mcwf_causal_frames"],
            seed=seed,
            mic_positions=data_config.get("mic_positions"),
        )
    if data_config["mode"] == "real_dir":
        return RealFourMicDirDataset(
            root=data_config["root"],
            split="test",
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
    return PaperLikeDataset(
        sample_rate=data_config["sample_rate"],
        segment_seconds=data_config["segment_seconds"],
        seed=seed,
        num_samples=num_samples,
        use_mcwf=bool(data_config["use_mcwf"]),
        ref_mic=data_config["ref_mic_index"],
        num_mics=data_config["num_mics"],
        stft_params=config["stft_model"],
        causal_frames=data_config["mcwf_causal_frames"],
        mic_positions=data_config.get("mic_positions"),
    )


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _format_stft_params(stft_params: dict) -> str:
    return "n_fft={n_fft} win_length={win_length} hop_length={hop_length} window={window} center={center}".format(
        n_fft=stft_params.get("n_fft"),
        win_length=stft_params.get("win_length"),
        hop_length=stft_params.get("hop_length"),
        window=stft_params.get("window", "hann"),
        center=stft_params.get("center", False),
    )


def _build_model(config: dict) -> tuple[torch.nn.Module, str]:
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


def _sanitize_key(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "sample"


def _get_meta_list(meta: object, batch_size: int) -> list[dict[str, object] | None]:
    if isinstance(meta, list):
        return meta
    if isinstance(meta, dict):
        expanded = []
        for idx in range(batch_size):
            item: dict[str, object] = {}
            for key, value in meta.items():
                if isinstance(value, (list, tuple)) and len(value) > idx:
                    item[key] = value[idx]
                else:
                    item[key] = value
            expanded.append(item)
        return expanded
    return [None] * batch_size


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GSENet model.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--write_wavs", action="store_true", help="Write per-sample wav outputs.")
    parser.add_argument("--max_wavs", type=int, default=20, help="Max samples to export.")
    parser.add_argument("--wav_dir", type=str, default=None, help="Directory for exported wavs.")
    parser.add_argument(
        "--wav_norm",
        type=str,
        default="peak",
        choices=("peak", "none"),
        help="Peak normalize audio before writing.",
    )
    parser.add_argument(
        "--save_mic_4ch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write multi-channel mic wav for each sample.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = resolve_config(args.config, overrides=_load_config_from_ckpt(ckpt))

    sample_rate = int(config["data"]["sample_rate"])
    model_stft = config["stft_model"]
    loss_stft = config["stft_loss"]

    out_dir = (
        Path(args.out_dir) if args.out_dir is not None else ckpt_path.parents[1] / "eval_outputs"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = ckpt_path.parents[1]
    wav_root = (
        Path(args.wav_dir) if args.wav_dir is not None else run_dir / "artifacts" / "test_wavs"
    )
    wav_split_dir = wav_root / "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_name = _build_model(config)
    model = model.to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(
        "model={name} params={params:.3f}M".format(
            name=model_name, params=_count_parameters(model) / 1e6
        )
    )
    print("model_stft " + _format_stft_params(model_stft))
    print("loss_stft " + _format_stft_params(loss_stft))

    eval_seed = int(config["run"]["seed"]) + 2468
    total_samples = args.num_batches * args.batch_size
    dataset = _make_dataset(config, num_samples=total_samples, seed=eval_seed)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    loss_y1_vals = []
    loss_y0_vals = []
    loss_yhat_vals = []
    snr_y1_vals = []
    snr_y0_vals = []
    snr_yhat_vals = []
    sisnr_y1_vals = []
    sisnr_y0_vals = []
    sisnr_yhat_vals = []
    sisdr_y1_vals = []
    sisdr_y0_vals = []
    sisdr_yhat_vals = []
    pesq_y1_vals = []
    pesq_y0_vals = []
    pesq_yhat_vals = []
    has_pesq = pesq_available() if config["metrics"]["enable_pesq"] else False
    if not has_pesq:
        print("pesq not installed, skipping PESQ metrics.", file=sys.stderr)

    audio_samples = []
    per_sample_rows: list[dict[str, object]] = []
    wavs_written = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break
            if "x_mics" in batch:
                x_mics = batch["x_mics"].to(device)
                y1 = batch["y1"].to(device)
                yt = batch["yt"].to(device)
                if bool(config["data"]["use_mcwf"]):
                    y0 = mcwf_make_y0(
                        x_mics,
                        stft_params=config["stft_model"],
                        causal_frames=config["data"]["mcwf_causal_frames"],
                        ref_ch=int(config["data"]["ref_mic_index"]),
                        sample_rate=int(config["data"]["sample_rate"]),
                        mic_positions=config["data"].get("mic_positions"),
                    )
                else:
                    y0 = y1
                y2 = x_mics[:, min(2, x_mics.shape[1] - 1)]
            else:
                y0 = batch["y0"].to(device)
                y1 = batch["y1"].to(device)
                y2 = batch["y2"].to(device)
                yt = batch["yt"].to(device)
                x_mics = batch.get("x_mics")
            y_hat = _forward_model(model, model_name, y0, y1, y2)

            loss_y1 = stft_magnitude_loss(y1, yt, stft_params=loss_stft)
            loss_y0 = stft_magnitude_loss(y0, yt, stft_params=loss_stft)
            loss_yhat = stft_magnitude_loss(y_hat, yt, stft_params=loss_stft)
            loss_y1_vals.append(float(loss_y1.item()))
            loss_y0_vals.append(float(loss_y0.item()))
            loss_yhat_vals.append(float(loss_yhat.item()))
            snr_y1_batch = snr_db(yt, y1).cpu().numpy()
            snr_y0_batch = snr_db(yt, y0).cpu().numpy()
            snr_yhat_batch = snr_db(yt, y_hat).cpu().numpy()
            sisnr_y1_batch = si_snr_db(yt, y1).cpu().numpy()
            sisnr_y0_batch = si_snr_db(yt, y0).cpu().numpy()
            sisnr_yhat_batch = si_snr_db(yt, y_hat).cpu().numpy()
            sisdr_y1_batch = sisdr(yt, y1).cpu().numpy()
            sisdr_y0_batch = sisdr(yt, y0).cpu().numpy()
            sisdr_yhat_batch = sisdr(yt, y_hat).cpu().numpy()
            snr_y1_vals.append(float(np.mean(snr_y1_batch)))
            snr_y0_vals.append(float(np.mean(snr_y0_batch)))
            snr_yhat_vals.append(float(np.mean(snr_yhat_batch)))
            sisnr_y1_vals.append(float(np.mean(sisnr_y1_batch)))
            sisnr_y0_vals.append(float(np.mean(sisnr_y0_batch)))
            sisnr_yhat_vals.append(float(np.mean(sisnr_yhat_batch)))
            sisdr_y1_vals.append(float(np.mean(sisdr_y1_batch)))
            sisdr_y0_vals.append(float(np.mean(sisdr_y0_batch)))
            sisdr_yhat_vals.append(float(np.mean(sisdr_yhat_batch)))

            pesq_y1_batch = []
            pesq_y0_batch = []
            pesq_yhat_batch = []
            for idx in range(yt.shape[0]):
                if has_pesq:
                    ref = yt[idx].cpu().numpy()
                    pesq_y1 = pesq_score(ref, y1[idx].cpu().numpy(), sample_rate)
                    pesq_y0 = pesq_score(ref, y0[idx].cpu().numpy(), sample_rate)
                    pesq_yhat = pesq_score(ref, y_hat[idx].cpu().numpy(), sample_rate)
                    pesq_y1_vals.append(pesq_y1)
                    pesq_y0_vals.append(pesq_y0)
                    pesq_yhat_vals.append(pesq_yhat)
                else:
                    pesq_y1 = float("nan")
                    pesq_y0 = float("nan")
                    pesq_yhat = float("nan")
                pesq_y1_batch.append(pesq_y1)
                pesq_y0_batch.append(pesq_y0)
                pesq_yhat_batch.append(pesq_yhat)

            meta_list = _get_meta_list(batch.get("meta"), int(yt.shape[0]))
            for idx in range(int(yt.shape[0])):
                meta = meta_list[idx]
                if meta and isinstance(meta, dict):
                    raw_key = meta.get("pair_key") or meta.get("basename")
                else:
                    raw_key = None
                if raw_key is None:
                    raw_key = f"sample_{batch_idx * args.batch_size + idx:05d}"
                sample_key = _sanitize_key(str(raw_key))
                write_paths = {}
                if args.write_wavs and wavs_written < args.max_wavs:
                    y1_np = y1[idx].detach().cpu().numpy()
                    y0_np = y0[idx].detach().cpu().numpy()
                    yhat_np = y_hat[idx].detach().cpu().numpy()
                    yt_np = yt[idx].detach().cpu().numpy()
                    write_paths = {
                        "path_y1": wav_split_dir / f"{sample_key}_y1_ref.wav",
                        "path_y0": wav_split_dir / f"{sample_key}_y0_bf.wav",
                        "path_yhat": wav_split_dir / f"{sample_key}_yhat.wav",
                        "path_yt": wav_split_dir / f"{sample_key}_yt.wav",
                    }
                    safe_write_wav(write_paths["path_y1"], y1_np, sample_rate, norm=args.wav_norm)
                    safe_write_wav(write_paths["path_y0"], y0_np, sample_rate, norm=args.wav_norm)
                    safe_write_wav(
                        write_paths["path_yhat"], yhat_np, sample_rate, norm=args.wav_norm
                    )
                    safe_write_wav(write_paths["path_yt"], yt_np, sample_rate, norm=args.wav_norm)
                    if args.save_mic_4ch and x_mics is not None:
                        mic_np = x_mics[idx].detach().cpu().numpy()
                        mic_path = wav_split_dir / f"{sample_key}_mic4ch.wav"
                        safe_write_wav(mic_path, mic_np, sample_rate, norm=args.wav_norm)
                        write_paths["path_mic4ch"] = mic_path
                    wavs_written += 1

                row = {
                    "sample_key": sample_key,
                    "snr_y1": float(snr_y1_batch[idx]),
                    "snr_y0": float(snr_y0_batch[idx]),
                    "snr_yhat": float(snr_yhat_batch[idx]),
                    "sisnr_y1": float(sisnr_y1_batch[idx]),
                    "sisnr_y0": float(sisnr_y0_batch[idx]),
                    "sisnr_yhat": float(sisnr_yhat_batch[idx]),
                    "sisdr_y1": float(sisdr_y1_batch[idx]),
                    "sisdr_y0": float(sisdr_y0_batch[idx]),
                    "sisdr_yhat": float(sisdr_yhat_batch[idx]),
                    "pesq_y1": float(pesq_y1_batch[idx]),
                    "pesq_y0": float(pesq_y0_batch[idx]),
                    "pesq_yhat": float(pesq_yhat_batch[idx]),
                    "delta_snr_y0_vs_y1": float(snr_y0_batch[idx] - snr_y1_batch[idx]),
                    "delta_snr_yhat_vs_y1": float(snr_yhat_batch[idx] - snr_y1_batch[idx]),
                    "delta_snr_yhat_vs_y0": float(snr_yhat_batch[idx] - snr_y0_batch[idx]),
                    "delta_sisnr_y0_vs_y1": float(sisnr_y0_batch[idx] - sisnr_y1_batch[idx]),
                    "delta_sisnr_yhat_vs_y1": float(sisnr_yhat_batch[idx] - sisnr_y1_batch[idx]),
                    "delta_sisnr_yhat_vs_y0": float(sisnr_yhat_batch[idx] - sisnr_y0_batch[idx]),
                    "delta_sisdr_y0_vs_y1": float(sisdr_y0_batch[idx] - sisdr_y1_batch[idx]),
                    "delta_sisdr_yhat_vs_y1": float(sisdr_yhat_batch[idx] - sisdr_y1_batch[idx]),
                    "delta_sisdr_yhat_vs_y0": float(sisdr_yhat_batch[idx] - sisdr_y0_batch[idx]),
                    "delta_pesq_y0_vs_y1": float(pesq_y0_batch[idx] - pesq_y1_batch[idx]),
                    "delta_pesq_yhat_vs_y1": float(pesq_yhat_batch[idx] - pesq_y1_batch[idx]),
                    "delta_pesq_yhat_vs_y0": float(pesq_yhat_batch[idx] - pesq_y0_batch[idx]),
                    "path_y1": "",
                    "path_y0": "",
                    "path_yhat": "",
                    "path_yt": "",
                }
                if args.save_mic_4ch:
                    row["path_mic4ch"] = ""
                if write_paths:
                    row["path_y1"] = str(write_paths["path_y1"])
                    row["path_y0"] = str(write_paths["path_y0"])
                    row["path_yhat"] = str(write_paths["path_yhat"])
                    row["path_yt"] = str(write_paths["path_yt"])
                    if "path_mic4ch" in write_paths:
                        row["path_mic4ch"] = str(write_paths["path_mic4ch"])
                per_sample_rows.append(row)

            if len(audio_samples) < 3:
                for idx in range(y0.shape[0]):
                    if len(audio_samples) >= 3:
                        break
                    audio_samples.append(
                        {
                            "y1": y1[idx].cpu().numpy(),
                            "y0": y0[idx].cpu().numpy(),
                            "yt": yt[idx].cpu().numpy(),
                            "y_hat": y_hat[idx].cpu().numpy(),
                        }
                    )

    snr_y1_mean = float(np.mean(snr_y1_vals))
    snr_y0_mean = float(np.mean(snr_y0_vals))
    snr_yhat_mean = float(np.mean(snr_yhat_vals))
    sisnr_y1_mean = float(np.mean(sisnr_y1_vals))
    sisnr_y0_mean = float(np.mean(sisnr_y0_vals))
    sisnr_yhat_mean = float(np.mean(sisnr_yhat_vals))
    sisdr_y1_mean = float(np.mean(sisdr_y1_vals))
    sisdr_y0_mean = float(np.mean(sisdr_y0_vals))
    sisdr_yhat_mean = float(np.mean(sisdr_yhat_vals))
    pesq_y1_mean = float(np.mean(pesq_y1_vals)) if has_pesq else float("nan")
    pesq_y0_mean = float(np.mean(pesq_y0_vals)) if has_pesq else float("nan")
    pesq_yhat_mean = float(np.mean(pesq_yhat_vals)) if has_pesq else float("nan")

    summary = {
        "loss_stft_y1_mean": float(np.mean(loss_y1_vals)),
        "loss_stft_y0_mean": float(np.mean(loss_y0_vals)),
        "loss_stft_yhat_mean": float(np.mean(loss_yhat_vals)),
        "snr_y1_mean": snr_y1_mean,
        "snr_y0_mean": snr_y0_mean,
        "snr_yhat_mean": snr_yhat_mean,
        "sisnr_y1_mean": sisnr_y1_mean,
        "sisnr_y0_mean": sisnr_y0_mean,
        "sisnr_yhat_mean": sisnr_yhat_mean,
        "sisdr_y1_mean": sisdr_y1_mean,
        "sisdr_y0_mean": sisdr_y0_mean,
        "sisdr_yhat_mean": sisdr_yhat_mean,
        "pesq_y1_mean": pesq_y1_mean,
        "pesq_y0_mean": pesq_y0_mean,
        "pesq_yhat_mean": pesq_yhat_mean,
        "delta_snr_y0_vs_y1": snr_y0_mean - snr_y1_mean,
        "delta_snr_yhat_vs_y1": snr_yhat_mean - snr_y1_mean,
        "delta_snr_yhat_vs_y0": snr_yhat_mean - snr_y0_mean,
        "delta_sisnr_y0_vs_y1": sisnr_y0_mean - sisnr_y1_mean,
        "delta_sisnr_yhat_vs_y1": sisnr_yhat_mean - sisnr_y1_mean,
        "delta_sisnr_yhat_vs_y0": sisnr_yhat_mean - sisnr_y0_mean,
        "delta_sisdr_y0_vs_y1": sisdr_y0_mean - sisdr_y1_mean,
        "delta_sisdr_yhat_vs_y1": sisdr_yhat_mean - sisdr_y1_mean,
        "delta_sisdr_yhat_vs_y0": sisdr_yhat_mean - sisdr_y0_mean,
        "delta_pesq_y0_vs_y1": pesq_y0_mean - pesq_y1_mean,
        "delta_pesq_yhat_vs_y1": pesq_yhat_mean - pesq_y1_mean,
        "delta_pesq_yhat_vs_y0": pesq_yhat_mean - pesq_y0_mean,
    }

    print(
        "eval summary loss={loss:.6f} snr_impr={snr_impr:.2f} sisnr_impr={sisnr_impr:.2f} "
        "sisdr_impr={sisdr_impr:.2f} pesq_impr={pesq_impr}".format(
            loss=summary["loss_stft_yhat_mean"],
            snr_impr=summary["delta_snr_yhat_vs_y0"],
            sisnr_impr=summary["delta_sisnr_yhat_vs_y0"],
            sisdr_impr=summary["delta_sisdr_yhat_vs_y0"],
            pesq_impr=f"{summary['delta_pesq_yhat_vs_y0']:.2f}" if has_pesq else "NA",
        )
    )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "summary.csv").open("w", newline="") as handle:
        handle.write(",".join(summary.keys()) + "\n")
        handle.write(",".join(f"{summary[key]:.6f}" for key in summary.keys()) + "\n")

    if per_sample_rows:
        per_sample_path = out_dir / "per_sample_metrics.csv"
        per_sample_fields = list(per_sample_rows[0].keys())
        with per_sample_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=per_sample_fields)
            writer.writeheader()
            writer.writerows(per_sample_rows)

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(audio_samples):
        for name, audio in sample.items():
            audio_path = audio_dir / f"sample_{idx:02d}_{name}.wav"
            audio_norm = _normalize_audio(audio, peak=0.99)
            sf.write(str(audio_path), audio_norm, samplerate=sample_rate)


if __name__ == "__main__":
    main()
