# LEGACY / internal demo
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from gsenet_repro.data.paper_synth import (
    generate_noise_mix,
    generate_rir_3src_4mic,
    sample_paper_params,
    synthesize_y0_y1_y2_y3_yt,
)


def _synthetic_source(rng: np.random.Generator, length: int, fs: int) -> np.ndarray:
    t = np.arange(length) / fs
    n_tones = 4
    freqs = rng.uniform(120.0, 2800.0, size=n_tones)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_tones)
    amps = rng.uniform(0.2, 1.0, size=n_tones)

    signal = np.zeros_like(t)
    for freq, phase, amp in zip(freqs, phases, amps):
        signal += amp * np.sin(2 * np.pi * freq * t + phase)

    envelope = np.hanning(length * 2)[length:]
    signal = signal * envelope
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    return signal.astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(0)
    batch = 2
    fs = 16000
    length = fs

    y0_list = []
    y1_list = []
    y2_list = []
    y3_list = []
    yt_list = []
    meta_list = []
    noise_level_list = []

    for idx in range(batch):
        s = _synthetic_source(rng, length, fs)
        n, n_meta = generate_noise_mix(rng, length, fs)
        i, i_meta = generate_noise_mix(rng, length, fs, noise_types=("speech", "babble", "pink"))

        rir, rir_anechoic = generate_rir_3src_4mic(rng)
        params = sample_paper_params(rng, variant="gsenet")
        background_config = {
            "rng": rng,
            "fs": fs,
            "snr_db_range": (-2.0, 12.0),
            "noise_types": ("white", "pink", "speech", "babble"),
            "metadata": [],
        }
        y0, y1, y2, y3, yt = synthesize_y0_y1_y2_y3_yt(
            s, n, i, rir, rir_anechoic, params, background_config=background_config
        )

        y0_list.append(y0)
        y1_list.append(y1)
        y2_list.append(y2)
        y3_list.append(y3)
        yt_list.append(yt)
        background_meta = background_config["metadata"][0] if background_config["metadata"] else {}
        meta = params.as_dict()
        meta.update(
            {"noise_meta": n_meta, "interf_meta": i_meta, "background_meta": background_meta}
        )
        meta_list.append(meta)
        noise_pow = np.mean(n**2) + np.mean(i**2)
        signal_pow = np.mean(s**2) + 1e-8
        noise_level_list.append(float(noise_pow / signal_pow))

        print(
            "sample[{idx}] gn_db={gn_db:.2f} gi_db={gi_db:.2f} "
            "alpha_db={alpha_db:.2f} beta_db={beta_db:.2f} pi={pi:.0f}".format(
                idx=idx,
                gn_db=params.gn_db,
                gi_db=params.gi_db,
                alpha_db=params.alpha_db,
                beta_db=params.beta_db,
                pi=params.pi,
            )
        )

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "paper_batch.npz"

    np.savez(
        out_path,
        y0=np.stack(y0_list),
        y1=np.stack(y1_list),
        y2=np.stack(y2_list),
        y3=np.stack(y3_list),
        yt=np.stack(yt_list),
        noise_level=np.array(noise_level_list, dtype=np.float32),
        meta=json.dumps(meta_list),
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
