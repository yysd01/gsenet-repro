from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


def _make_signal(duration_s: float, sample_rate: int, freq: float) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_dummy_real_dir_dataset(root: Path, sample_rate: int = 16000) -> None:
    if root.exists():
        shutil.rmtree(root)
    splits = ["train", "valid", "test"]
    durations = [0.6, 1.2, 2.5]
    root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        clean_dir = root / split / "clean"
        mic_dir = root / split / "mic"
        clean_dir.mkdir(parents=True, exist_ok=True)
        mic_dir.mkdir(parents=True, exist_ok=True)

        for idx, duration in enumerate(durations, start=1):
            clean = _make_signal(duration, sample_rate, freq=220.0 + idx * 30.0)
            noise = np.random.normal(scale=0.02, size=(clean.shape[0], 4)).astype(np.float32)
            mic_stack = np.stack(
                [
                    clean * 1.0,
                    clean * 0.9,
                    clean * 1.1,
                    clean * 0.8,
                ],
                axis=1,
            )
            mic = mic_stack + noise
            name = f"{idx:04d}.wav"
            sf.write(str(clean_dir / name), clean, samplerate=sample_rate)
            sf.write(str(mic_dir / name), mic, samplerate=sample_rate)

    print(f"Dummy dataset created at: {root}")
    for split in splits:
        clean_files = list((root / split / "clean").glob("*.wav"))
        mic_files = list((root / split / "mic").glob("*.wav"))
        print(f"{split}: clean={len(clean_files)} mic={len(mic_files)}")

    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root)
            print(rel)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dummy real dir dataset.")
    parser.add_argument(
        "--root",
        type=str,
        default="artifacts/dummy_real_dir_dataset",
        help="Output root directory",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    make_dummy_real_dir_dataset(Path(args.root), sample_rate=args.sample_rate)


if __name__ == "__main__":
    main()
