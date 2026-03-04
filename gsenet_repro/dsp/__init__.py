"""DSP utilities."""
import importlib.util

from .mcwf import mcwf
from .mvdr import apply_beamformer, estimate_rnn, estimate_rtf, mvdr_weights
from .stft import LOSS_STFT, MODEL_STFT, istft, stft

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    from .torch_stft import istft as torch_istft
    from .torch_stft import stft as torch_stft
    from .mcwf_torch import windowed_power as mcwf_windowed_power
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
]
