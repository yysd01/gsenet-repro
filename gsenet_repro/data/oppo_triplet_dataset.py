from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass

from gsenet_repro.dsp.supervised_bf import parse_doas_from_filename


class OppoTripletDataset(Dataset):
    """Load synchronized 4-channel clean/noise/noisy wav triplets.

    Directory layout:
      dataset_root/<split>/<case>/{clean,noise,noisy}/*.wav

    __getitem__ returns:
      clean/noise/noisy: float32 arrays (C,T), C=4
      src_doas/int_doas: list[int]
      case: str
      utt_id: filename
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        case_filter: list[str] | None = None,
        sample_rate: int = 16000,
    ) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be one of {'train','valid','test'}")
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.sample_rate = int(sample_rate)

        split_root = self.dataset_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Split root not found: {split_root}")

        selected_cases = set(case_filter) if case_filter is not None else None
        all_cases = sorted([p.name for p in split_root.iterdir() if p.is_dir()])
        cases = [case for case in all_cases if selected_cases is None or case in selected_cases]
        self.samples: list[dict[str, Path | str]] = []
        for case in cases:
            case_root = split_root / case
            clean_dir = case_root / "clean"
            noise_dir = case_root / "noise"
            noisy_dir = case_root / "noisy"
            for required in (clean_dir, noise_dir, noisy_dir):
                if not required.exists():
                    raise FileNotFoundError(f"Missing required directory: {required}")

            for noisy_path in sorted(noisy_dir.glob("*.wav")):
                clean_path = clean_dir / noisy_path.name
                noise_path = noise_dir / noisy_path.name
                if not clean_path.exists() or not noise_path.exists():
                    raise FileNotFoundError(
                        f"Missing pair for {noisy_path.name} in case={case} split={split}"
                    )
                self.samples.append(
                    {
                        "case": case,
                        "utt_id": noisy_path.name,
                        "clean_path": clean_path,
                        "noise_path": noise_path,
                        "noisy_path": noisy_path,
                    }
                )

        if not self.samples:
            raise ValueError(f"No triplet samples found in {split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_4ch(self, wav_path: Path) -> np.ndarray:
        wav, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
        if sr != self.sample_rate:
            raise ValueError(f"Expected sample_rate={self.sample_rate}, got {sr} for {wav_path}")
        x = wav.T.astype(np.float32)
        if x.shape[0] != 4:
            raise ValueError(f"Expected 4 channels, got {x.shape[0]} for {wav_path}")
        return x

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        clean = self._read_4ch(Path(sample["clean_path"]))
        noise = self._read_4ch(Path(sample["noise_path"]))
        noisy = self._read_4ch(Path(sample["noisy_path"]))

        min_len = min(clean.shape[1], noise.shape[1], noisy.shape[1])
        if clean.shape[1] != noise.shape[1] or noise.shape[1] != noisy.shape[1]:
            warnings.warn(
                f"Length mismatch for {sample['utt_id']}; truncating to {min_len} samples.",
                RuntimeWarning,
            )
        clean = clean[:, :min_len]
        noise = noise[:, :min_len]
        noisy = noisy[:, :min_len]

        src_doas, int_doas = parse_doas_from_filename(str(sample["utt_id"]))
        return {
            "clean": clean,
            "noise": noise,
            "noisy": noisy,
            "src_doas": src_doas,
            "int_doas": int_doas,
            "case": sample["case"],
            "utt_id": sample["utt_id"],
        }


class OppoPrecomputedY0Dataset(Dataset):
    """Training/eval dataset for Oppo triplets with precomputed y0.

    Returns torch tensors: y0, y1, yt (shape T) and x_mics (4,T) for compatibility.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        precomputed_y0_root: str | Path,
        split: str,
        case_filter: list[str] | None = None,
        sample_rate: int = 16000,
        ref_mic_index: int = 0,
    ) -> None:
        self.triplet = OppoTripletDataset(
            dataset_root=dataset_root,
            split=split,
            case_filter=case_filter,
            sample_rate=sample_rate,
        )
        self.precomputed_y0_root = Path(precomputed_y0_root)
        self.ref_mic_index = int(ref_mic_index)

    def __len__(self) -> int:
        return len(self.triplet)

    def __getitem__(self, index: int) -> dict[str, "torch.Tensor"]:
        sample = self.triplet[index]
        clean = np.asarray(sample["clean"], dtype=np.float32)
        noisy = np.asarray(sample["noisy"], dtype=np.float32)
        case = str(sample["case"])
        utt_id = str(sample["utt_id"])

        y0_path = self.precomputed_y0_root / self.triplet.split / case / "y0" / utt_id
        if not y0_path.exists():
            raise FileNotFoundError(f"Precomputed y0 not found: {y0_path}")
        y0_wav, sr = sf.read(str(y0_path), always_2d=True, dtype="float32")
        if sr != self.triplet.sample_rate:
            raise ValueError(f"Expected sample_rate={self.triplet.sample_rate}, got {sr} for {y0_path}")
        y0 = y0_wav[:, 0].astype(np.float32)

        y1 = noisy[self.ref_mic_index]
        yt = clean[self.ref_mic_index]
        min_len = min(y0.shape[0], y1.shape[0], yt.shape[0])

        if torch is None:
            raise RuntimeError("torch is required for OppoPrecomputedY0Dataset")
        return {
            "y0": torch.tensor(y0[:min_len], dtype=torch.float32),
            "y1": torch.tensor(y1[:min_len], dtype=torch.float32),
            "yt": torch.tensor(yt[:min_len], dtype=torch.float32),
            "x_mics": torch.tensor(noisy[:, :min_len], dtype=torch.float32),
        }
