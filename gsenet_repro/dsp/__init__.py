"""DSP utilities."""

import importlib.util

from .mcwf import mcwf
from .mvdr import apply_beamformer, estimate_rnn, estimate_rtf, mvdr_weights
from .stft import LOSS_STFT, MODEL_STFT, istft, stft
from .supervised_bf import (
    apply_beamformer as supervised_apply_beamformer,
)
from .supervised_bf import (
    beamform_sample,
    build_doa_rtf_library,
    estimate_rnn_from_noise,
    estimate_rtf_from_clean_ev,
    lcmv_weights,
    parse_doas_from_filename,
    stft_4ch,
)
from .supervised_bf import (
    mvdr_weights as supervised_mvdr_weights,
)

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    from .mcwf_torch import windowed_power as mcwf_windowed_power
    from .torch_stft import istft as torch_istft
    from .torch_stft import stft as torch_stft
else:  # pragma: no cover
    torch_stft = None
    torch_istft = None
    mcwf_windowed_power = None

__all__ = [
    "LOSS_STFT",
    "MODEL_STFT",
    "mcwf",
    "stft",
    "estimate_rnn",
    "estimate_rtf",
    "mvdr_weights",
    "apply_beamformer",
    "istft",
    "torch_stft",
    "torch_istft",
    "mcwf_windowed_power",
    "parse_doas_from_filename",
    "stft_4ch",
    "estimate_rnn_from_noise",
    "estimate_rtf_from_clean_ev",
    "supervised_mvdr_weights",
    "lcmv_weights",
    "supervised_apply_beamformer",
    "build_doa_rtf_library",
    "beamform_sample",
]
