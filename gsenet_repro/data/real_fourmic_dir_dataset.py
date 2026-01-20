from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:  # pragma: no cover - optional torch dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class _AudioInfo:
    sample_rate: int
    frames: int
    channels: int


def _load_audio_info(path: Path) -> _AudioInfo:
    info = sf.info(str(path))
    return _AudioInfo(sample_rate=int(info.samplerate), frames=int(info.frames), channels=int(info.channels))


def _ensure_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    if signal.shape[0] == target_length:
        return signal
    if signal.shape[0] < target_length:
        pad_width = target_length - signal.shape[0]
        return np.pad(signal, (0, pad_width))
    return signal[:target_length]


def _ensure_length_2d(signal: np.ndarray, target_length: int) -> np.ndarray:
    if signal.shape[0] == target_length:
        return signal
    if signal.shape[0] < target_length:
        pad_width = target_length - signal.shape[0]
        return np.pad(signal, ((0, pad_width), (0, 0)))
    return signal[:target_length]


class RealFourMicDirDataset:
    """Dataset for real 4-mic recordings stored as clean/mic directory pairs."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        sample_rate: int = 16000,
        segment_seconds: float = 1.0,
        num_mics: int = 4,
        ref_mic_index: int = 0,
        random_crop: bool = False,
        eval_full_length: bool = False,
        fixed_crop: str = "center",
        resample: bool = True,
        cache_metadata: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        if self.split not in {"train", "valid", "test"}:
            raise ValueError(f"split must be train/valid/test, got {self.split}")
        self.sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.segment_frames_target = int(round(self.sample_rate * self.segment_seconds))
        self.num_mics = int(num_mics)
        if self.num_mics <= 0:
            raise ValueError("num_mics must be positive")
        self.ref_mic_index = int(ref_mic_index)
        if not 0 <= self.ref_mic_index < self.num_mics:
            raise ValueError("ref_mic_index out of range")
        self.random_crop = bool(random_crop)
        self.eval_full_length = bool(eval_full_length)
        self.fixed_crop = fixed_crop
        if self.fixed_crop not in {"center", "start"}:
            raise ValueError("fixed_crop must be 'center' or 'start'")
        self.resample = bool(resample)
        self.cache_metadata = bool(cache_metadata)
        self._rng = np.random.default_rng()

        clean_dir = self.root / self.split / "clean"
        mic_dir = self.root / self.split / "mic"
        if not clean_dir.exists():
            raise FileNotFoundError(f"Clean directory not found: {clean_dir}")
        if not mic_dir.exists():
            raise FileNotFoundError(f"Mic directory not found: {mic_dir}")

        mic_paths = sorted(mic_dir.glob("*.wav"))
        entries: List[Dict[str, object]] = []
        missing_clean = 0
        for mic_path in mic_paths:
            clean_path = clean_dir / mic_path.name
            if not clean_path.exists():
                missing_clean += 1
                continue
            entry: Dict[str, object] = {
                "mic_path": mic_path,
                "clean_path": clean_path,
                "mic_info": None,
                "clean_info": None,
            }
            if self.cache_metadata:
                mic_info = _load_audio_info(mic_path)
                if mic_info.channels != self.num_mics:
                    raise ValueError(
                        f"Expected {self.num_mics} channels for {mic_path}, got {mic_info.channels}"
                    )
                clean_info = _load_audio_info(clean_path)
                if clean_info.channels > 1:
                    warnings.warn(
                        f"Clean audio {clean_path} has {clean_info.channels} channels; using channel 0.",
                        RuntimeWarning,
                    )
                entry["mic_info"] = mic_info
                entry["clean_info"] = clean_info
            entries.append(entry)

        if missing_clean > 0:
            print(
                f"RealFourMicDirDataset[{self.split}]: skipped {missing_clean} mic files without clean pairs."
            )
        if not entries:
            raise ValueError(f"No paired samples found under {self.root / self.split}")
        print(
            f"RealFourMicDirDataset[{self.split}]: paired {len(entries)} samples "
            f"(missing clean: {missing_clean})."
        )
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def _load_pair_info(self, entry: Dict[str, object]) -> tuple[_AudioInfo, _AudioInfo]:
        mic_info = entry.get("mic_info")
        clean_info = entry.get("clean_info")
        if isinstance(mic_info, _AudioInfo) and isinstance(clean_info, _AudioInfo):
            return mic_info, clean_info
        mic_path = Path(entry["mic_path"])
        clean_path = Path(entry["clean_path"])
        mic_info = _load_audio_info(mic_path)
        if mic_info.channels != self.num_mics:
            raise ValueError(
                f"Expected {self.num_mics} channels for {mic_path}, got {mic_info.channels}"
            )
        clean_info = _load_audio_info(clean_path)
        if clean_info.channels > 1:
            warnings.warn(
                f"Clean audio {clean_path} has {clean_info.channels} channels; using channel 0.",
                RuntimeWarning,
            )
        return mic_info, clean_info

    def _pick_start_frame(self, total_frames: int, segment_frames: int) -> int:
        if total_frames <= segment_frames:
            return 0
        if self.random_crop:
            max_start = total_frames - segment_frames
            return int(self._rng.integers(0, max_start + 1))
        if self.fixed_crop == "center":
            return max((total_frames - segment_frames) // 2, 0)
        return 0

    def __getitem__(self, idx: int) -> Dict[str, object]:
        entry = self.entries[idx]
        mic_path = Path(entry["mic_path"])
        clean_path = Path(entry["clean_path"])
        mic_info, clean_info = self._load_pair_info(entry)
        if mic_info.sample_rate != clean_info.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: {mic_path}={mic_info.sample_rate}, "
                f"{clean_path}={clean_info.sample_rate}"
            )

        total_frames = min(mic_info.frames, clean_info.frames)
        if mic_info.frames != clean_info.frames:
            warnings.warn(
                f"Frame mismatch between {mic_path} ({mic_info.frames}) and {clean_path} "
                f"({clean_info.frames}); using min length {total_frames}.",
                RuntimeWarning,
            )

        if self.eval_full_length:
            segment_frames_orig = total_frames
        else:
            segment_frames_orig = int(round(self.segment_seconds * mic_info.sample_rate))

        start_frame = self._pick_start_frame(total_frames, segment_frames_orig)
        frames_to_read = min(segment_frames_orig, total_frames - start_frame)

        with sf.SoundFile(str(mic_path)) as mic_file:
            mic_file.seek(start_frame)
            mic_audio = mic_file.read(frames=frames_to_read, dtype="float32", always_2d=True)
        with sf.SoundFile(str(clean_path)) as clean_file:
            clean_file.seek(start_frame)
            clean_audio = clean_file.read(frames=frames_to_read, dtype="float32", always_2d=True)

        if mic_audio.shape[1] != self.num_mics:
            raise ValueError(
                f"Expected {self.num_mics} channels for {mic_path}, got {mic_audio.shape[1]}"
            )

        if clean_audio.shape[1] > 1:
            clean_audio = clean_audio[:, 0:1]

        if not self.eval_full_length:
            mic_audio = _ensure_length_2d(mic_audio, segment_frames_orig)
            clean_audio = _ensure_length_2d(clean_audio, segment_frames_orig)

        if self.resample and mic_info.sample_rate != self.sample_rate:
            mic_audio = resample_poly(
                mic_audio, self.sample_rate, mic_info.sample_rate, axis=0
            ).astype(np.float32)
            clean_audio = resample_poly(
                clean_audio, self.sample_rate, mic_info.sample_rate, axis=0
            ).astype(np.float32)
            if not self.eval_full_length:
                mic_audio = _ensure_length_2d(mic_audio, self.segment_frames_target)
                clean_audio = _ensure_length_2d(clean_audio, self.segment_frames_target)

        if self.eval_full_length:
            segment_frames_target = mic_audio.shape[0]
        else:
            segment_frames_target = self.segment_frames_target
            mic_audio = _ensure_length_2d(mic_audio, segment_frames_target)
            clean_audio = _ensure_length_2d(clean_audio, segment_frames_target)

        x_mics = mic_audio.T.astype(np.float32)
        yt = clean_audio[:, 0].astype(np.float32)
        y1 = x_mics[self.ref_mic_index]

        meta = {
            "basename": mic_path.name,
            "mic_path": str(mic_path),
            "clean_path": str(clean_path),
            "orig_sr": mic_info.sample_rate,
            "orig_frames": total_frames,
            "start_frame": int(start_frame),
            "segment_frames": int(segment_frames_target),
            "resampled": bool(self.resample and mic_info.sample_rate != self.sample_rate),
            "fixed_crop": self.fixed_crop,
            "eval_full_length": self.eval_full_length,
            "clean_channels": clean_info.channels,
            "mic_channels": mic_info.channels,
        }

        sample = {
            "x_mics": x_mics,
            "y1": y1,
            "yt": yt,
            "meta": meta,
        }

        if torch is not None and torch.is_tensor(x_mics):  # pragma: no cover - safety
            return sample
        return sample
