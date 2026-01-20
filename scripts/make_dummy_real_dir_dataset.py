from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from gsenet_repro.data.real_fourmic_dir_dataset import RealFourMicDirDataset


def _make_signal(duration_s: float, sample_rate: int, freq: float) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_dummy_real_dir_dataset(root: Path, sample_rate: int = 16000) -> None:
    if root.exists():
        shutil.rmtree(root)
    splits = ["train", "valid", "test"]
    durations = [0.6, 1.2, 2.5]
    date_tag = "20251112"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)

    for split in splits:
        clean_dir = root / split / "clean"
        mic_dir = root / split / "mic"
        clean_dir.mkdir(parents=True, exist_ok=True)
        mic_dir.mkdir(parents=True, exist_ok=True)

        for idx, duration in enumerate(durations, start=1):
            clean = _make_signal(duration, sample_rate, freq=220.0 + idx * 30.0)
            noise = rng.normal(scale=0.02, size=(clean.shape[0], 4)).astype(np.float32)
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
            clean_stack = np.stack(
                [
                    clean * 1.0,
                    clean * 0.99,
                    clean * 1.01,
                    clean * 0.98,
                ],
                axis=1,
            )
            core = f"{idx}-1_src30-int90-p257-367_doa0"
            clean_name = f"clean_{core}_data.wav"
            mic_name = f"mic_{core}_{date_tag}.wav"
            sf.write(str(clean_dir / clean_name), clean_stack, samplerate=sample_rate)
            sf.write(str(mic_dir / mic_name), mic, samplerate=sample_rate)

    print(f"Dummy dataset created at: {root}")
    for split in splits:
        clean_files = list((root / split / "clean").glob("*.wav"))
        mic_files = list((root / split / "mic").glob("*.wav"))
        print(f"{split}: clean={len(clean_files)} mic={len(mic_files)}")

    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root)
            print(rel)

    ref_mic_index = 0
    clean_ref_mic_index = 0
    for split in splits:
        dataset = RealFourMicDirDataset(
            root=root,
            split=split,
            sample_rate=sample_rate,
            segment_seconds=1.0,
            num_mics=4,
            ref_mic_index=ref_mic_index,
            clean_ref_mic_index=clean_ref_mic_index,
            clean_is_multichannel=True,
            random_crop=False,
            cache_metadata=False,
        )
        print(
            "paired_samples={paired} ref_mic_index={ref_mic_index} clean_ref_mic_index={clean_ref_mic_index}".format(
                paired=len(dataset),
                ref_mic_index=ref_mic_index,
                clean_ref_mic_index=clean_ref_mic_index,
            )
        )


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
