from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def _to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio


def _maybe_normalize(audio: np.ndarray, norm: str) -> np.ndarray:
    if norm == "none":
        return audio
    absmax = float(np.max(np.abs(audio))) if audio.size else 0.0
    if absmax > 1.0:
        audio = audio * (0.98 / (absmax + 1e-8))
    return audio


def safe_write_wav(
    path: str | Path, audio: np.ndarray, sample_rate: int, norm: str = "peak"
) -> None:
    if norm not in {"peak", "none"}:
        raise ValueError(f"norm must be 'peak' or 'none', got {norm}")
    wav_path = Path(path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio)
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T
    audio = _to_float32(audio)
    audio = _maybe_normalize(audio, norm=norm)
    sf.write(str(wav_path), audio, samplerate=int(sample_rate), subtype="FLOAT")
