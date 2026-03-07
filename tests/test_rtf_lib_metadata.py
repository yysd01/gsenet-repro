from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.dsp.rtf_lib import load_rtf_lib, save_rtf_lib
from gsenet_repro.streaming.mvdr_streamer import MVDRStreamer


def _make_lib(sr: int = 16000, n_fft: int = 256, num_mics: int = 4) -> dict[str, object]:
    F = n_fft // 2 + 1
    d = np.zeros((1, F, num_mics), dtype=np.complex64)
    d[:, :, 0] = 1.0 + 0.0j
    return {
        "doa_bins": np.array([0], dtype=np.int32),
        "d_mean": d,
        "sample_rate": sr,
        "n_fft": n_fft,
        "win_length": n_fft,
        "hop_length": n_fft // 2,
        "window": "hann",
        "center": False,
        "num_mics": num_mics,
        "ref_ch": 0,
        "binsize_deg": 1,
    }


def test_rtf_lib_metadata_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "rtf_lib.npz"
    src = _make_lib(sr=22050, n_fft=512)
    save_rtf_lib(path, src)
    loaded = load_rtf_lib(path)

    for key in ("sample_rate", "n_fft", "win_length", "hop_length", "window", "center", "num_mics", "ref_ch", "binsize_deg"):
        assert loaded[key] == src[key]


def test_rtf_lib_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "rtf_lib_bad.npz"
    save_rtf_lib(path, _make_lib(sr=8000, n_fft=512))

    streamer = MVDRStreamer(sample_rate=16000, n_fft=256, center=False, num_mics=4)
    with pytest.raises(ValueError, match="请用相同 STFT 参数重新 build rtf_lib"):
        streamer.load_rtf_lib(path)
