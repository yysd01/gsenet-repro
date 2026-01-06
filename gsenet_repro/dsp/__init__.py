"""DSP utilities."""
import importlib.util

from .mcwf import mcwf
from .stft import LOSS_STFT, MODEL_STFT, istft, stft

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    from .torch_stft import istft as torch_istft
    from .torch_stft import stft as torch_stft
else:  # pragma: no cover
    torch_stft = None
    torch_istft = None

__all__ = [
    "LOSS_STFT",
    "MODEL_STFT",
    "mcwf",
    "stft",
    "istft",
    "torch_stft",
    "torch_istft",
]
