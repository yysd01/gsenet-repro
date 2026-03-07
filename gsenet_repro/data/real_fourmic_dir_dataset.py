from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
    return _AudioInfo(
        sample_rate=int(info.samplerate), frames=int(info.frames), channels=int(info.channels)
    )


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


DEFAULT_PAIRING_CONFIG: Dict[str, object] = {
    "clean_prefix": "clean_",
    "mic_prefix": "mic_",
    "drop_last_underscore_segment": True,
    "strict_pairing": False,
}


def canonical_pair_key(filename: str, kind: str, cfg: Dict[str, object]) -> str:
    stem = Path(filename).stem
    clean_prefix = str(cfg.get("clean_prefix", DEFAULT_PAIRING_CONFIG["clean_prefix"]))
    mic_prefix = str(cfg.get("mic_prefix", DEFAULT_PAIRING_CONFIG["mic_prefix"]))
    prefix = ""
    if kind == "clean":
        prefix = clean_prefix
    elif kind == "mic":
        prefix = mic_prefix
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix) :]
    if bool(cfg.get("drop_last_underscore_segment", True)) and "_" in stem:
        stem = stem.rsplit("_", 1)[0]
    return stem


def _normalize_pairing_config(pairing_config: Dict[str, object] | None) -> Dict[str, object]:
    normalized = dict(DEFAULT_PAIRING_CONFIG)
    if pairing_config:
        for key in DEFAULT_PAIRING_CONFIG:
            if key in pairing_config:
                normalized[key] = pairing_config[key]
    return normalized


def _build_key_map(
    paths: List[Path],
    kind: str,
    cfg: Dict[str, object],
) -> Dict[str, List[Path]]:
    key_map: Dict[str, List[Path]] = {}
    for path in paths:
        key = canonical_pair_key(path.name, kind, cfg)
        key_map.setdefault(key, []).append(path)
    for key in key_map:
        key_map[key] = sorted(key_map[key])
    return key_map


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
        clean_ref_mic_index: int = 0,
        clean_is_multichannel: bool = True,
        random_crop: bool = False,
        eval_full_length: bool = False,
        fixed_crop: str = "center",
        resample: bool = True,
        cache_metadata: bool = True,
        pairing_config: Dict[str, object] | None = None,
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
        self.clean_ref_mic_index = int(clean_ref_mic_index)
        self.clean_is_multichannel = bool(clean_is_multichannel)
        if self.clean_is_multichannel:
            if not 0 <= self.clean_ref_mic_index < self.num_mics:
                raise ValueError("clean_ref_mic_index out of range")
        self.random_crop = bool(random_crop)
        self.eval_full_length = bool(eval_full_length)
        self.fixed_crop = fixed_crop
        if self.fixed_crop not in {"center", "start"}:
            raise ValueError("fixed_crop must be 'center' or 'start'")
        self.resample = bool(resample)
        self.cache_metadata = bool(cache_metadata)
        self.pairing_config = _normalize_pairing_config(pairing_config)
        self._rng = np.random.default_rng()

        clean_dir = self.root / self.split / "clean"
        mic_dir = self.root / self.split / "mic"
        if not clean_dir.exists():
            raise FileNotFoundError(f"Clean directory not found: {clean_dir}")
        if not mic_dir.exists():
            raise FileNotFoundError(f"Mic directory not found: {mic_dir}")

        clean_paths = sorted(clean_dir.glob("*.wav"))
        mic_paths = sorted(mic_dir.glob("*.wav"))
        clean_map = _build_key_map(clean_paths, "clean", self.pairing_config)
        mic_map = _build_key_map(mic_paths, "mic", self.pairing_config)
        clean_total = len(clean_paths)
        mic_total = len(mic_paths)
        clean_keys = set(clean_map.keys())
        mic_keys = set(mic_map.keys())
        paired_keys = sorted(clean_keys & mic_keys)
        missing_clean = sum(len(mic_map[key]) for key in mic_keys - clean_keys)
        missing_mic = sum(len(clean_map[key]) for key in clean_keys - mic_keys)
        duplicate_key_clean = sum(1 for paths in clean_map.values() if len(paths) > 1)
        duplicate_key_mic = sum(1 for paths in mic_map.values() if len(paths) > 1)
        strict_pairing = bool(self.pairing_config.get("strict_pairing", False))

        entries: List[Dict[str, object]] = []
        for key in paired_keys:
            clean_candidates = clean_map[key]
            mic_candidates = mic_map[key]
            if len(clean_candidates) > 1:
                message = (
                    f"Pairing key '{key}' has {len(clean_candidates)} clean files; "
                    f"using {clean_candidates[0].name} and ignoring {len(clean_candidates) - 1}."
                )
                if strict_pairing:
                    raise ValueError(message)
                warnings.warn(message, RuntimeWarning)
            if len(mic_candidates) > 1:
                message = (
                    f"Pairing key '{key}' has {len(mic_candidates)} mic files; "
                    f"using {mic_candidates[0].name} and ignoring {len(mic_candidates) - 1}."
                )
                if strict_pairing:
                    raise ValueError(message)
                warnings.warn(message, RuntimeWarning)
            clean_path = clean_candidates[0]
            mic_path = mic_candidates[0]
            entry: Dict[str, object] = {
                "mic_path": mic_path,
                "clean_path": clean_path,
                "mic_info": None,
                "clean_info": None,
                "pair_key": key,
            }
            if self.cache_metadata:
                mic_info = _load_audio_info(mic_path)
                if mic_info.channels != self.num_mics:
                    raise ValueError(
                        f"Expected {self.num_mics} channels for {mic_path}, got {mic_info.channels}"
                    )
                clean_info = _load_audio_info(clean_path)
                if self.clean_is_multichannel:
                    if clean_info.channels <= self.clean_ref_mic_index:
                        raise ValueError(
                            "Clean audio {path} has {channels} channels, cannot select "
                            "clean_ref_mic_index={index}.".format(
                                path=clean_path,
                                channels=clean_info.channels,
                                index=self.clean_ref_mic_index,
                            )
                        )
                elif clean_info.channels > 1:
                    warnings.warn(
                        f"Clean audio {clean_path} has {clean_info.channels} channels; using channel 0.",
                        RuntimeWarning,
                    )
                entry["mic_info"] = mic_info
                entry["clean_info"] = clean_info
            entries.append(entry)

        if missing_clean > 0 or missing_mic > 0:
            warnings.warn(
                "RealFourMicDirDataset[{split}]: missing pairs "
                "(missing_clean={missing_clean}, missing_mic={missing_mic}).".format(
                    split=self.split,
                    missing_clean=missing_clean,
                    missing_mic=missing_mic,
                ),
                RuntimeWarning,
            )
        if not entries:
            raise ValueError(f"No paired samples found under {self.root / self.split}")
        print(
            "RealFourMicDirDataset[{split}]: clean_total={clean_total} mic_total={mic_total} "
            "paired={paired} missing_clean={missing_clean} missing_mic={missing_mic} "
            "duplicate_key_clean={duplicate_key_clean} duplicate_key_mic={duplicate_key_mic}.".format(
                split=self.split,
                clean_total=clean_total,
                mic_total=mic_total,
                paired=len(entries),
                missing_clean=missing_clean,
                missing_mic=missing_mic,
                duplicate_key_clean=duplicate_key_clean,
                duplicate_key_mic=duplicate_key_mic,
            )
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
        if self.clean_is_multichannel:
            if clean_info.channels <= self.clean_ref_mic_index:
                raise ValueError(
                    "Clean audio {path} has {channels} channels, cannot select "
                    "clean_ref_mic_index={index}.".format(
                        path=clean_path,
                        channels=clean_info.channels,
                        index=self.clean_ref_mic_index,
                    )
                )
        elif clean_info.channels > 1:
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

        if self.clean_is_multichannel:
            if clean_audio.shape[1] <= self.clean_ref_mic_index:
                raise ValueError(
                    "Clean audio {path} has {channels} channels, cannot select "
                    "clean_ref_mic_index={index}.".format(
                        path=clean_path,
                        channels=clean_audio.shape[1],
                        index=self.clean_ref_mic_index,
                    )
                )
            clean_audio_mono = clean_audio[:, self.clean_ref_mic_index]
        else:
            if clean_audio.shape[1] > 1:
                warnings.warn(
                    f"Clean audio {clean_path} has {clean_audio.shape[1]} channels; using channel 0.",
                    RuntimeWarning,
                )
            clean_audio_mono = clean_audio[:, 0]

        if not self.eval_full_length:
            mic_audio = _ensure_length_2d(mic_audio, segment_frames_orig)
            clean_audio_mono = _ensure_length(clean_audio_mono, segment_frames_orig)

        if self.resample and mic_info.sample_rate != self.sample_rate:
            mic_audio = resample_poly(
                mic_audio, self.sample_rate, mic_info.sample_rate, axis=0
            ).astype(np.float32)
            clean_audio = resample_poly(
                clean_audio_mono, self.sample_rate, mic_info.sample_rate, axis=0
            ).astype(np.float32)
            if not self.eval_full_length:
                mic_audio = _ensure_length_2d(mic_audio, self.segment_frames_target)
                clean_audio = _ensure_length(clean_audio, self.segment_frames_target)
            clean_audio_mono = clean_audio
        else:
            clean_audio = clean_audio_mono

        if self.eval_full_length:
            segment_frames_target = mic_audio.shape[0]
        else:
            segment_frames_target = self.segment_frames_target
            mic_audio = _ensure_length_2d(mic_audio, segment_frames_target)
            clean_audio = _ensure_length(clean_audio, segment_frames_target)

        x_mics = mic_audio.T.astype(np.float32)
        yt = clean_audio.astype(np.float32)
        y1 = x_mics[self.ref_mic_index]

        meta = {
            "basename": mic_path.name,
            "mic_path": str(mic_path),
            "clean_path": str(clean_path),
            "pair_key": entry.get("pair_key"),
            "orig_sr": mic_info.sample_rate,
            "orig_frames": total_frames,
            "start_frame": int(start_frame),
            "segment_frames": int(segment_frames_target),
            "resampled": bool(self.resample and mic_info.sample_rate != self.sample_rate),
            "fixed_crop": self.fixed_crop,
            "eval_full_length": self.eval_full_length,
            "clean_channels": clean_info.channels,
            "mic_channels": mic_info.channels,
            "ref_mic_index": self.ref_mic_index,
            "clean_ref_mic_index": self.clean_ref_mic_index,
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
