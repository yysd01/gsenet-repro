from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from gsenet_repro.data.oppo_triplet_dataset import OppoTripletDataset
from gsenet_repro.dsp.supervised_bf import beamform_sample, parse_doas_from_filename


def test_parse_doas_from_filename() -> None:
    src, intr = parse_doas_from_filename("src[1, 19]_int[97, 289]_p232_226.wav")
    assert src == [1, 19]
    assert intr == [97, 289]


def test_oppo_triplet_and_beamform(tmp_path: Path) -> None:
    root = tmp_path / "oppo"
    base = root / "valid" / "1-1"
    for sub in ("clean", "noise", "noisy"):
        (base / sub).mkdir(parents=True, exist_ok=True)

    sr = 16000
    t = np.arange(0, 512, dtype=np.float32)
    clean_mono = np.sin(2 * np.pi * 300 * t / sr).astype(np.float32)
    noise_mono = 0.03 * np.random.default_rng(0).standard_normal(size=t.shape[0]).astype(np.float32)
    clean = np.stack([clean_mono, clean_mono, clean_mono, clean_mono], axis=1)
    noise = np.stack([noise_mono, noise_mono, noise_mono, noise_mono], axis=1)
    noisy = clean + noise

    fname = "src[1]_int[61]_p257_261.wav"
    sf.write(base / "clean" / fname, clean, sr)
    sf.write(base / "noise" / fname, noise, sr)
    sf.write(base / "noisy" / fname, noisy, sr)

    ds = OppoTripletDataset(dataset_root=root, split="valid")
    item = ds[0]
    y0, y1, dbg = beamform_sample(
        clean4=item["clean"],
        noise4=item["noise"],
        noisy4=item["noisy"],
        src_doas=item["src_doas"],
        stft_cfg={
            "n_fft": 256,
            "win_length": 256,
            "hop_length": 128,
            "window": "hann",
            "center": False,
        },
    )
    assert y0.ndim == 1 and y1.ndim == 1
    assert y0.shape[0] == y1.shape[0] == item["clean"].shape[1]
    assert dbg["rms_y0"] > 0
