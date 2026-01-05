"""DSP utilities."""
from .stft import LOSS_STFT, MODEL_STFT, istft, stft

try:  # torch is optional for numpy-only workflows.
    from .torch_stft import torch_istft, torch_stft
except ModuleNotFoundError:  # pragma: no cover
    torch_stft = None
    torch_istft = None

__all__ = ["LOSS_STFT", "MODEL_STFT", "stft", "istft"]
if torch_stft is not None and torch_istft is not None:
    __all__ += ["torch_stft", "torch_istft"]
