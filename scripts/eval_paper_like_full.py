from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
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
from gsenet_repro.data.real_fourmic_dir_dataset import RealFourMicDirDataset
from gsenet_repro.data.real_dataset import RealMultichannelDataset
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
    )


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _format_stft_params(stft_params: dict) -> str:
    return (
        "n_fft={n_fft} win_length={win_length} hop_length={hop_length} window={window} center={center}".format(
            n_fft=stft_params.get("n_fft"),
            win_length=stft_params.get("win_length"),
            hop_length=stft_params.get("hop_length"),
            window=stft_params.get("window", "hann"),
            center=stft_params.get("center", False),
        )
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
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = resolve_config(args.config, overrides=_load_config_from_ckpt(ckpt))

    sample_rate = int(config["data"]["sample_rate"])
    model_stft = config["stft_model"]
    loss_stft = config["stft_loss"]

    out_dir = Path(args.out_dir) if args.out_dir is not None else ckpt_path.parents[1] / "eval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

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
                    )
                else:
                    y0 = y1
                y2 = x_mics[:, min(2, x_mics.shape[1] - 1)]
            else:
                y0 = batch["y0"].to(device)
                y1 = batch["y1"].to(device)
                y2 = batch["y2"].to(device)
                yt = batch["yt"].to(device)
            y_hat = _forward_model(model, model_name, y0, y1, y2)

            loss_y1 = stft_magnitude_loss(y1, yt, stft_params=loss_stft)
            loss_y0 = stft_magnitude_loss(y0, yt, stft_params=loss_stft)
            loss_yhat = stft_magnitude_loss(y_hat, yt, stft_params=loss_stft)
            loss_y1_vals.append(float(loss_y1.item()))
            loss_y0_vals.append(float(loss_y0.item()))
            loss_yhat_vals.append(float(loss_yhat.item()))
            snr_y1_vals.append(float(snr_db(yt, y1).mean().item()))
            snr_y0_vals.append(float(snr_db(yt, y0).mean().item()))
            snr_yhat_vals.append(float(snr_db(yt, y_hat).mean().item()))
            sisnr_y1_vals.append(float(si_snr_db(yt, y1).mean().item()))
            sisnr_y0_vals.append(float(si_snr_db(yt, y0).mean().item()))
            sisnr_yhat_vals.append(float(si_snr_db(yt, y_hat).mean().item()))
            sisdr_y1_vals.append(float(sisdr(yt, y1).mean().item()))
            sisdr_y0_vals.append(float(sisdr(yt, y0).mean().item()))
            sisdr_yhat_vals.append(float(sisdr(yt, y_hat).mean().item()))
            if has_pesq:
                for idx in range(yt.shape[0]):
                    ref = yt[idx].cpu().numpy()
                    pesq_y1_vals.append(pesq_score(ref, y1[idx].cpu().numpy(), sample_rate))
                    pesq_y0_vals.append(pesq_score(ref, y0[idx].cpu().numpy(), sample_rate))
                    pesq_yhat_vals.append(pesq_score(ref, y_hat[idx].cpu().numpy(), sample_rate))

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

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(audio_samples):
        for name, audio in sample.items():
            audio_path = audio_dir / f"sample_{idx:02d}_{name}.wav"
            audio_norm = _normalize_audio(audio, peak=0.99)
            sf.write(str(audio_path), audio_norm, samplerate=sample_rate)


if __name__ == "__main__":
    main()
