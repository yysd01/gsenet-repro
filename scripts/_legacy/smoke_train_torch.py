# LEGACY / internal demo
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from gsenet_repro.losses.stft_loss_torch import stft_magnitude_loss
from gsenet_repro.models.gsenet_torch import MinimalGSENet


def _load_or_create_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    artifacts_path = Path("artifacts") / "dummy_batch.npz"
    if artifacts_path.exists():
        data = np.load(artifacts_path)
        if "y2" in data:
            return data["y0"], data["y1"], data["y2"], data["yt"]

    rng = np.random.default_rng(0)
    batch = 2
    length = 2048
    y0 = rng.normal(scale=0.2, size=(batch, length)).astype(np.float32)
    y1 = rng.normal(scale=0.2, size=(batch, length)).astype(np.float32)
    y2 = 0.7 * y0 + 0.3 * y1
    yt = 0.6 * y0 + 0.4 * y1
    return y0, y1, y2.astype(np.float32), yt.astype(np.float32)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    y0_np, y1_np, y2_np, yt_np = _load_or_create_batch()
    y0 = torch.tensor(y0_np, dtype=torch.float32)
    y1 = torch.tensor(y1_np, dtype=torch.float32)
    y2 = torch.tensor(y2_np, dtype=torch.float32)
    yt = torch.tensor(yt_np, dtype=torch.float32)

    model = MinimalGSENet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    with torch.no_grad():
        initial_loss = stft_magnitude_loss(model(y0, y1, y2), yt).item()
    print(f"initial_loss={initial_loss:.6f}")

    for _ in range(20):
        optimizer.zero_grad()
        y_hat = model(y0, y1, y2)
        loss = stft_magnitude_loss(y_hat, yt)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = stft_magnitude_loss(model(y0, y1, y2), yt).item()
    print(f"final_loss={final_loss:.6f}")

    if final_loss > initial_loss:
        raise SystemExit("final loss did not improve")


if __name__ == "__main__":
    main()
