# LEGACY / internal demo
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from gsenet_repro.data.paper_synth import (
    generate_noise_mix,
    generate_rir_3src_4mic,
    sample_paper_params,
    synthesize_y_mics_yt,
)


def main() -> None:
    rng = np.random.default_rng(0)
    sample_rate = 16000
    segment_seconds = 1.0
    length = int(sample_rate * segment_seconds)
    num_samples = 4

    out_root = Path("artifacts") / "dummy_real_dataset"
    mic_dirs = [out_root / "mics" / f"mic{idx}" for idx in range(4)]
    target_dir = out_root / "target"
    for mic_dir in mic_dirs:
        mic_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.csv"
    rows = []

    for idx in range(num_samples):
        s = rng.normal(size=length).astype(np.float32)
        n, _ = generate_noise_mix(rng, length, sample_rate)
        i, _ = generate_noise_mix(
            rng, length, sample_rate, noise_types=("speech", "babble", "pink")
        )
        rir, rir_anechoic = generate_rir_3src_4mic(rng)
        params = sample_paper_params(rng, global_gain=0.8)
        y_mics, yt = synthesize_y_mics_yt(
            s,
            n,
            i,
            rir,
            rir_anechoic,
            params,
            num_mics=4,
            background_config={"rng": rng, "fs": sample_rate},
        )

        stem = f"sample_{idx:03d}.wav"
        mic_paths = []
        for mic_idx, mic_dir in enumerate(mic_dirs):
            path = mic_dir / stem
            sf.write(str(path), y_mics[mic_idx], sample_rate)
            mic_paths.append(path)
        target_path = target_dir / stem
        sf.write(str(target_path), yt, sample_rate)

        row = {f"mic{mic_idx}": str(mic_paths[mic_idx]) for mic_idx in range(4)}
        row["target"] = str(target_path)
        row["sr"] = str(sample_rate)
        rows.append(row)

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mic0", "mic1", "mic2", "mic3", "target", "sr"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote dummy dataset manifest to {manifest_path}")


if __name__ == "__main__":
    main()
