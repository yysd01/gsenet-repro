from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from gsenet_repro.data.paper_synth import (
    generate_rir_3src_2mic,
    sample_paper_params,
    synthesize_y0_y1_yt,
)


def _load_or_create_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    artifacts_path = Path("artifacts") / "paper_batch.npz"
    if artifacts_path.exists():
        data = np.load(artifacts_path)
        return data["y0"], data["y1"], data["yt"]

    rng = np.random.default_rng(0)
    batch = 2
    fs = 16000
    length = fs

    y0_list = []
    y1_list = []
    yt_list = []

    for _ in range(batch):
        s = rng.normal(size=length).astype(np.float32)
        n = rng.normal(scale=0.3, size=length).astype(np.float32)
        i = rng.normal(scale=0.2, size=length).astype(np.float32)
        rir, rir_anechoic = generate_rir_3src_2mic(rng)
        params = sample_paper_params(rng)
        y0, y1, yt = synthesize_y0_y1_yt(s, n, i, rir, rir_anechoic, params)
        y0_list.append(y0)
        y1_list.append(y1)
        yt_list.append(yt)

    return np.stack(y0_list), np.stack(y1_list), np.stack(yt_list)


def main() -> None:
    if importlib.util.find_spec("torch") is None:
        print("torch not installed; skipping smoke_train_paper_like")
        return

    import torch

    from gsenet_repro.losses.stft_loss_torch import stft_magnitude_loss
    from gsenet_repro.models.gsenet_torch import MinimalGSENet

    torch.manual_seed(0)
    np.random.seed(0)

    y0_np, y1_np, yt_np = _load_or_create_batch()
    y0 = torch.tensor(y0_np, dtype=torch.float32)
    y1 = torch.tensor(y1_np, dtype=torch.float32)
    yt = torch.tensor(yt_np, dtype=torch.float32)

    model = MinimalGSENet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    stft_params = {"n_fft": 1024, "win_length": 1024, "hop_length": 256}

    model.train()
    with torch.no_grad():
        initial_loss = stft_magnitude_loss(model(y0, y1), yt, stft_params=stft_params).item()
    print(f"initial_loss={initial_loss:.6f}")

    for _ in range(40):
        optimizer.zero_grad()
        y_hat = model(y0, y1)
        loss = stft_magnitude_loss(y_hat, yt, stft_params=stft_params)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = stft_magnitude_loss(model(y0, y1), yt, stft_params=stft_params).item()
    print(f"final_loss={final_loss:.6f}")

    if final_loss > initial_loss * 0.99:
        raise SystemExit("final loss did not improve by at least 1%")


if __name__ == "__main__":
    main()
