from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from gsenet_repro.data.paper_synth import (
    generate_rir_3src_3mic,
    sample_paper_params,
    synthesize_y0_y1_y2_yt,
)


def _load_or_create_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    artifacts_path = Path("artifacts") / "paper_batch.npz"
    if artifacts_path.exists():
        data = np.load(artifacts_path)
        if "y2" in data:
            return data["y0"], data["y1"], data["y2"], data["yt"], data["noise_level"]

    rng = np.random.default_rng(0)
    batch = 2
    fs = 16000
    length = fs

    y0_list = []
    y1_list = []
    y2_list = []
    yt_list = []
    noise_level_list = []

    for _ in range(batch):
        s = rng.normal(size=length).astype(np.float32)
        n = rng.normal(scale=0.3, size=length).astype(np.float32)
        i = rng.normal(scale=0.2, size=length).astype(np.float32)
        rir, rir_anechoic = generate_rir_3src_3mic(rng)
        params = sample_paper_params(rng)
        y0, y1, y2, yt = synthesize_y0_y1_y2_yt(s, n, i, rir, rir_anechoic, params)
        y0_list.append(y0)
        y1_list.append(y1)
        y2_list.append(y2)
        yt_list.append(yt)
        noise_pow = np.mean(n**2) + np.mean(i**2)
        signal_pow = np.mean(s**2) + 1e-8
        noise_level_list.append(np.float32(noise_pow / signal_pow))

    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        artifacts_path,
        y0=np.stack(y0_list),
        y1=np.stack(y1_list),
        y2=np.stack(y2_list),
        yt=np.stack(yt_list),
        noise_level=np.stack(noise_level_list),
    )
    return (
        np.stack(y0_list),
        np.stack(y1_list),
        np.stack(y2_list),
        np.stack(yt_list),
        np.stack(noise_level_list),
    )


def _snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    noise = reference - estimate
    ref_pow = np.mean(reference**2) + 1e-12
    noise_pow = np.mean(noise**2) + 1e-12
    return float(10.0 * np.log10(ref_pow / noise_pow))


def main() -> None:
    if importlib.util.find_spec("torch") is None:
        print("torch not installed; skipping smoke_train_paper_like")
        return

    import torch

    from gsenet_repro.losses.stft_loss_torch import stft_magnitude_loss
    from gsenet_repro.models.gsenet_torch import MinimalGSENet

    torch.manual_seed(0)
    np.random.seed(0)

    y0_np, y1_np, y2_np, yt_np, noise_level_np = _load_or_create_batch()
    y0 = torch.tensor(y0_np, dtype=torch.float32)
    y1 = torch.tensor(y1_np, dtype=torch.float32)
    y2 = torch.tensor(y2_np, dtype=torch.float32)
    yt = torch.tensor(yt_np, dtype=torch.float32)
    noise_level = torch.tensor(noise_level_np, dtype=torch.float32)

    model = MinimalGSENet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    stft_params = {"n_fft": 1024, "win_length": 1024, "hop_length": 256}

    model.train()
    with torch.no_grad():
        initial_loss = stft_magnitude_loss(
            model(y0, y1, y2, noise_level=noise_level),
            yt,
            stft_params=stft_params,
        ).item()
    print(f"initial_loss={initial_loss:.6f}")

    for _ in range(40):
        optimizer.zero_grad()
        y_hat = model(y0, y1, y2, noise_level=noise_level)
        loss = stft_magnitude_loss(y_hat, yt, stft_params=stft_params)
        reg = 0.1 * torch.mean(torch.abs(torch.stft(
            y_hat,
            n_fft=stft_params["n_fft"],
            hop_length=stft_params["hop_length"],
            win_length=stft_params["win_length"],
            window=torch.hann_window(stft_params["win_length"], device=y_hat.device),
            center=False,
            return_complex=True,
        )))
        total_loss = loss + reg
        total_loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        final_loss = stft_magnitude_loss(
            model(y0, y1, y2, noise_level=noise_level),
            yt,
            stft_params=stft_params,
        ).item()
    print(f"final_loss={final_loss:.6f}")

    if final_loss > initial_loss * 0.99:
        raise SystemExit("final loss did not improve by at least 1%")

    with torch.no_grad():
        y_hat = model(y0, y1, y2, noise_level=noise_level).cpu().numpy()
    snr_in = _snr_db(yt_np, y0_np)
    snr_out = _snr_db(yt_np, y_hat)
    snr_improve = snr_out - snr_in
    print(f"snr_in={snr_in:.2f} snr_out={snr_out:.2f} snr_improve={snr_improve:.2f}")
    if snr_improve < 0.5:
        raise SystemExit("SNR improvement was not significant")


if __name__ == "__main__":
    main()
