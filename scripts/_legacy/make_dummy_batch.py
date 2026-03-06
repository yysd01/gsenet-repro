# LEGACY / internal demo
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, lfilter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from gsenet_repro.data.rir import generate_dummy_rir
from gsenet_repro.data.synthesis import synthesize_pair


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


def main() -> None:
    rng = np.random.default_rng(0)
    batch = 8
    fs = 16000
    length = fs

    y0_list = []
    y1_list = []
    yt_list = []
    meta_list = []
    alpha_db_list = []
    beta_db_list = []
    pi_list = []
    noise_ratio_list = []

    for _ in range(batch):
        s = _bandpass_noise(rng, length, fs, 300.0, 3400.0)
        n = rng.normal(scale=0.3, size=length).astype(np.float32)
        i = _bandpass_noise(rng, length, fs, 500.0, 4500.0)

        rir, rir_anechoic = generate_dummy_rir(rng)
        y0, y1, yt, meta = synthesize_pair(
            s=s,
            n=n,
            i=i,
            rir=rir,
            rir_anechoic=rir_anechoic,
            rng=rng,
        )
        gains = meta["gains"]

        zero = np.zeros_like(s)
        y0_noise, y1_noise, _, _ = synthesize_pair(
            s=zero,
            n=n,
            i=zero,
            rir=rir,
            rir_anechoic=rir_anechoic,
            rng=rng,
            gains=gains,
        )
        energy_ratio = float(np.mean(y1_noise**2) / (np.mean(y0_noise**2) + 1e-12))

        y0_list.append(y0)
        y1_list.append(y1)
        yt_list.append(yt)
        meta_list.append(meta)
        alpha_db_list.append(gains["alpha_db"])
        beta_db_list.append(gains["beta_db"])
        pi_list.append(gains["pi"])
        noise_ratio_list.append(energy_ratio)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "dummy_batch.npz"

    np.savez(
        out_path,
        y0=np.stack(y0_list),
        y1=np.stack(y1_list),
        yt=np.stack(yt_list),
        meta=json.dumps(meta_list),
    )

    alpha_db = np.array(alpha_db_list)
    beta_db = np.array(beta_db_list)
    pi = np.array(pi_list)
    noise_ratio = np.array(noise_ratio_list)

    summary = (
        "alpha_db(min/mean/max)=({:.2f}/{:.2f}/{:.2f}) "
        "beta_db(min/mean/max)=({:.2f}/{:.2f}/{:.2f}) "
        "pi_ratio={:.2f} noise_ratio_mean={:.2f}"
    ).format(
        alpha_db.min(),
        alpha_db.mean(),
        alpha_db.max(),
        beta_db.min(),
        beta_db.mean(),
        beta_db.max(),
        pi.mean(),
        noise_ratio.mean(),
    )
    print(summary)


if __name__ == "__main__":
    main()
