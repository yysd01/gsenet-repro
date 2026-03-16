from __future__ import annotations

import warnings
from typing import Dict

import numpy as np

from gsenet_repro.pipeline.frontend import DEFAULT_MIC_POSITIONS, make_y0_from_frontend
from gsenet_repro.pipeline.gates import gates_from_probs


def mcwf_make_y0(
    x_mics,
    stft_params: Dict[str, int] | None,
    causal_frames: int = 4,
    ref_ch: int = 1,
    diag_load: float = 1e-2,
    sample_rate: int | None = None,
    mic_positions: np.ndarray | None = None,
):
    del causal_frames
    warnings.warn(
        "mcwf_make_y0 is deprecated; use make_y0_from_frontend with frontend.type",
        DeprecationWarning,
        stacklevel=2,
    )
    frontend_cfg = {
        "type": "mvdr",
        "gate_mode": "sector",
        "ref_ch": int(ref_ch),
        "diag_load_v": float(diag_load),
        "mic_positions": DEFAULT_MIC_POSITIONS if mic_positions is None else mic_positions,
    }
    data_cfg = {"sample_rate": 16000 if sample_rate is None else int(sample_rate), "ref_mic_index": int(ref_ch)}
    return make_y0_from_frontend(x_mics, frontend_cfg=frontend_cfg, stft_cfg=stft_params, data_cfg=data_cfg)
