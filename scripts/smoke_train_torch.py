from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, lfilter
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from gsenet_repro.data.rir import generate_dummy_rir
from gsenet_repro.data.synthesis import synthesize_pair
from gsenet_repro.losses.stft_loss import stft_reconstruction_loss
from gsenet_repro.models.gsenet_torch import GSENetTorch


def _bandpass_noise(
    rng: np.random.Generator,
    length: int,
    fs: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    b, a = butter(4, [low_hz / (fs / 2), high_hz / (fs / 2)], btype="band")
    noise = rng.normal(size=length)
    filtered = lfilter(b, a, noise)
    filtered = filtered / (np.max(np.abs(filtered)) + 1e-8)
    return filtered.astype(np.float32)


def make_batch(
    rng: np.random.Generator,
    batch: int,
    length: int,
    fs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y0_list = []
    y1_list = []
    yt_list = []

    for _ in range(batch):
        s = _bandpass_noise(rng, length, fs, 300.0, 3400.0)
        n = rng.normal(scale=0.3, size=length).astype(np.float32)
        i = _bandpass_noise(rng, length, fs, 500.0, 4500.0)
        rir, rir_anechoic = generate_dummy_rir(rng)
        y0, y1, yt, _ = synthesize_pair(
            s=s,
            n=n,
            i=i,
            rir=rir,
            rir_anechoic=rir_anechoic,
            rng=rng,
        )
        y0_list.append(y0)
        y1_list.append(y1)
        yt_list.append(yt)

    return (
        np.stack(y0_list).astype(np.float32),
        np.stack(y1_list).astype(np.float32),
        np.stack(yt_list).astype(np.float32),
    )


def main() -> None:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    batch = 2
    fs = 16000
    length = fs

    y0, y1, yt = make_batch(rng, batch, length, fs)

    device = torch.device("cpu")
    y0_t = torch.from_numpy(y0).to(device)
    y1_t = torch.from_numpy(y1).to(device)
    yt_t = torch.from_numpy(yt).to(device)

    model = GSENetTorch().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step in range(8):
        optimizer.zero_grad()
        y_hat = model(y0_t, y1_t)
        loss = stft_reconstruction_loss(y_hat, yt_t)
        loss.backward()
        optimizer.step()
        print(f"step={step} loss={loss.item():.6f}")

    print("OK")


if __name__ == "__main__":
    main()
