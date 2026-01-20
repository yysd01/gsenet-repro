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

from gsenet_repro.data.paper_dataset import PaperLikeDataset
from gsenet_repro.losses.stft_loss_torch import stft_magnitude_loss
from gsenet_repro.metrics.metrics_pesq import pesq_available, pesq_score
from gsenet_repro.metrics.metrics_torch import si_snr_db, sisdr, snr_db
from gsenet_repro.models.gsenet_torch import MinimalGSENet


def _normalize_audio(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    max_val = np.max(np.abs(x)) + 1e-8
    return (x / max_val * peak).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper-like GSENet model.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})
    sample_rate = int(config.get("sample_rate", 16000))
    segment_seconds = float(config.get("segment_seconds", 1.0))
    model_stft = config.get("model_stft", {"n_fft": 320, "win_length": 320, "hop_length": 160})
    loss_stft = config.get("loss_stft", {"n_fft": 1024, "win_length": 1024, "hop_length": 256})
    use_mcwf = bool(config.get("use_mcwf", True))

    out_dir = Path(args.out_dir) if args.out_dir is not None else ckpt_path.parents[1] / "eval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MinimalGSENet(stft_params=model_stft).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eval_seed = int(config.get("seed", 0)) + 2468
    dataset = PaperLikeDataset(
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        seed=eval_seed,
        num_samples=args.num_batches * args.batch_size,
        use_mcwf=use_mcwf,
        stft_params=model_stft,
    )
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
    has_pesq = pesq_available()
    if not has_pesq:
        print("pesq not installed, skipping PESQ metrics.", file=sys.stderr)

    audio_samples = []
    with torch.no_grad():
        for batch in loader:
            y0 = batch["y0"].to(device)
            y1 = batch["y1"].to(device)
            y2 = batch["y2"].to(device)
            yt = batch["yt"].to(device)
            y_hat = model(y0, y1, y2)

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
