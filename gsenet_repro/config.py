from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "run": {
        "seed": 0,
        "device": "auto",
        "run_dir": None,
    },
    "data": {
        "mode": "paper_like",
        "sample_rate": 16000,
        "segment_seconds": 1.0,
        "num_mics": 4,
        "ref_mic_index": 1,
        "clean_ref_mic_index": 0,
        "clean_is_multichannel": True,
        "manifest_path": None,
        "root_dir": None,
        "root": None,
        "split": "train",
        "random_crop": False,
        "eval_full_length": False,
        "fixed_crop": "center",
        "resample": True,
        "cache_metadata": True,
        "use_mcwf": 1,
        "mcwf_causal_frames": 4,
        "mic_positions": None,
        "dataset_type": None,
        "dataset_root": None,
        "precomputed_y0_root": None,
        "use_precomputed_y0": False,
        "case_filter": None,
    },
    "stft_model": {
        "n_fft": 320,
        "win_length": 320,
        "hop_length": 160,
        "window": "hann",
        "center": False,
    },
    "stft_loss": {
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
    },
    "model": {
        "name": "gsenet_paper_scale",
        "leaky_relu_slope": 0.3,
        "stem_channels": 16,
        "head_channels": 2,
        "remove_dc": False,
        "encoder_blocks": [
            {"cin": 16, "cout": 32, "stime": 1, "sfreq": 2, "dtime": False},
            {"cin": 32, "cout": 48, "stime": 2, "sfreq": 2, "dtime": False},
            {"cin": 48, "cout": 48, "stime": 1, "sfreq": 2, "dtime": True},
            {"cin": 48, "cout": 96, "stime": 1, "sfreq": 2, "dtime": True},
            {"cin": 96, "cout": 96, "stime": 1, "sfreq": 2, "dtime": True},
        ],
        "decoder_blocks": [
            {"cin": 96, "cout": 96, "stime": 1, "sfreq": 2, "dtime": True},
            {"cin": 96, "cout": 48, "stime": 1, "sfreq": 2, "dtime": True},
            {"cin": 48, "cout": 48, "stime": 1, "sfreq": 2, "dtime": True},
            {"cin": 48, "cout": 32, "stime": 2, "sfreq": 2, "dtime": False},
            {"cin": 32, "cout": 16, "stime": 1, "sfreq": 2, "dtime": False},
        ],
    },
    "train": {
        "batch_size": 4,
        "num_steps": 2000,
        "lr": 1e-4,
        "log_every": 10,
        "eval_every": 200,
        "ckpt_every": 500,
        "num_workers": 0,
        "prefetch_factor": 2,
    },
    "metrics": {
        "enable_pesq": True,
        "enable_sisdr": True,
        "enable_sisnr": True,
        "enable_snr": True,
    },
    "pairing": {
        "clean_prefix": "clean_",
        "mic_prefix": "mic_",
        "drop_last_underscore_segment": True,
        "strict_pairing": False,
    },
}


def _deep_merge(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_toml(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data


def resolve_config(
    config_path: str | Path | None, overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config = _deep_merge(config, load_toml(config_path))
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def resolve_run_dir(run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "runs" / timestamp


def save_resolved_config(run_dir: Path, config: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "config_resolved.json"
    resolved_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
