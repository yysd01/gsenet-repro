# LEGACY / internal demo
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    print("torch not installed. Install requirements-torch.txt to run stats.", file=sys.stderr)
    raise SystemExit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

import torch

from gsenet_repro.config import resolve_config
from gsenet_repro.models.gsenet_paper_torch import GSENetPaperScale


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Print GSENetPaperScale stats.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/real_dataset_paper_scale.toml",
        help="Config path (default: configs/real_dataset_paper_scale.toml)",
    )
    args = parser.parse_args()

    config = resolve_config(args.config)
    model_config = config.get("model", {})

    model = GSENetPaperScale(
        stft_params=config["stft_model"],
        leaky_relu_slope=float(model_config.get("leaky_relu_slope", 0.3)),
        encoder_blocks=model_config.get("encoder_blocks"),
        decoder_blocks=model_config.get("decoder_blocks"),
        stem_channels=int(model_config.get("stem_channels", 16)),
        head_channels=int(model_config.get("head_channels", 2)),
        remove_dc=bool(model_config.get("remove_dc", False)),
    )

    sample_rate = int(config["data"]["sample_rate"])
    y0 = torch.randn(1, sample_rate)
    y1 = torch.randn(1, sample_rate)

    with torch.no_grad():
        y_hat = model(y0, y1)

    print("model_name=gsenet_paper_scale")
    print("params={:.3f}M".format(_count_parameters(model) / 1e6))
    print("input_shape y0={} y1={} output_shape={}".format(y0.shape, y1.shape, y_hat.shape))
    print("model_stft " + _format_stft_params(config["stft_model"]))
    print("loss_stft " + _format_stft_params(config["stft_loss"]))


if __name__ == "__main__":
    main()
