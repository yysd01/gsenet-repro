from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from gsenet_repro.dsp import MODEL_STFT

try:  # pragma: no cover - optional torch dependency
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object


def _read_audio(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr).astype(np.float32)
    return audio.astype(np.float32)


def _resolve_manifest_entries(
    manifest_path: Optional[Path],
    root_dir: Optional[Path],
    num_mics: int,
) -> List[Dict[str, str]]:
    if manifest_path is not None:
        return _load_manifest(manifest_path, num_mics=num_mics)
    if root_dir is None:
        raise ValueError("Either manifest_path or root_dir must be provided")
    return _scan_root_dir(root_dir, num_mics=num_mics)


def _load_manifest(path: Path, num_mics: int) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    entries: List[Dict[str, str]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entries.append({str(k): str(v) for k, v in row.items()})
    else:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                entries.append({str(k): str(v) for k, v in row.items()})

    required = [f"mic{idx}" for idx in range(num_mics)] + ["target"]
    for idx, row in enumerate(entries):
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Manifest row {idx} missing keys: {missing}")
    return entries


def _scan_root_dir(root_dir: Path, num_mics: int) -> List[Dict[str, str]]:
    mic_dirs = [root_dir / "mics" / f"mic{idx}" for idx in range(num_mics)]
    target_dir = root_dir / "target"
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    for mic_dir in mic_dirs:
        if not mic_dir.exists():
            raise FileNotFoundError(f"Mic directory not found: {mic_dir}")

    mic_sets = [{path.name for path in mic_dir.glob("*.wav")} for mic_dir in mic_dirs]
    target_set = {path.name for path in target_dir.glob("*.wav")}
    common = set.intersection(*mic_sets, target_set)
    if not common:
        raise ValueError("No matching filenames found across mic/target directories")
    if any(mic_set != common for mic_set in mic_sets) or target_set != common:
        raise ValueError(
            "Filename mismatch across mic/target directories. Use a manifest file instead."
        )

    entries = []
    for filename in sorted(common):
        row = {f"mic{idx}": str(mic_dirs[idx] / filename) for idx in range(num_mics)}
        row["target"] = str(target_dir / filename)
        entries.append(row)
    return entries


def _slice_aligned(
    arrays: Sequence[np.ndarray],
    length: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    min_len = min(array.shape[0] for array in arrays)
    if min_len < length:
        pad_width = length - min_len
        arrays = [np.pad(arr[:min_len], (0, pad_width)) for arr in arrays]
        min_len = length
    max_start = max(min_len - length, 0)
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    end = start + length
    return [arr[start:end] for arr in arrays]


class RealMultichannelDataset(Dataset):
    """Dataset for real multi-mic recordings as raw waveform training inputs."""

    def __init__(
        self,
        manifest_path: str | None = None,
        root_dir: str | None = None,
        sample_rate: int = 16000,
        segment_seconds: float = 1.0,
        num_mics: int = 4,
        ref_mic_index: int = 0,
        use_mcwf: bool = True,
        include_legacy_targets: bool = False,
        stft_params: Optional[Dict[str, int]] = None,
        causal_frames: int = 4,
        seed: int = 0,
        mic_positions: list[list[float]] | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.segment_length = int(round(self.sample_rate * self.segment_seconds))
        self.num_mics = int(num_mics)
        if self.num_mics < 2:
            raise ValueError("num_mics must be >= 2")
        self.ref_mic_index = int(ref_mic_index)
        if not 0 <= self.ref_mic_index < self.num_mics:
            raise ValueError("ref_mic_index out of range")
        self.use_mcwf = bool(use_mcwf)  # compatibility only; frontend now computes y0
        self.include_legacy_targets = bool(include_legacy_targets)
        self.stft_params = dict(stft_params) if stft_params is not None else dict(MODEL_STFT)
        self.causal_frames = int(causal_frames)
        self.seed = int(seed)
        self.mic_positions = mic_positions

        manifest = Path(manifest_path) if manifest_path else None
        root = Path(root_dir) if root_dir else None
        self.entries = _resolve_manifest_entries(manifest, root, num_mics=self.num_mics)
        if not self.entries:
            raise ValueError("No entries found for dataset")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray | "torch.Tensor"]:
        entry = self.entries[idx]
        rng = np.random.default_rng(self.seed + idx)

        mic_paths = [Path(entry[f"mic{mic_idx}"]) for mic_idx in range(self.num_mics)]
        target_path = Path(entry["target"])
        mics = [_read_audio(path, self.sample_rate) for path in mic_paths]
        target = _read_audio(target_path, self.sample_rate)

        sliced = _slice_aligned([*mics, target], self.segment_length, rng)
        mic_slices = sliced[: self.num_mics]
        target_slice = sliced[-1]

        x_mics = np.stack(mic_slices, axis=0).astype(np.float32)
        y1 = x_mics[self.ref_mic_index]

        sample: Dict[str, np.ndarray | "torch.Tensor"] = {
            "x_mics": x_mics.astype(np.float32),
            "y1": y1.astype(np.float32),
            "yt": target_slice.astype(np.float32),
        }
        if self.include_legacy_targets:
            y2 = x_mics[min(2, self.num_mics - 1)]
            sample["y2"] = y2.astype(np.float32)

        if torch is not None:
            return {key: torch.tensor(value, dtype=torch.float32) for key, value in sample.items()}
        return sample
