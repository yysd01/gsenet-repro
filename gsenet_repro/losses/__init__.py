"""Loss functions."""

import importlib.util

from .stft_loss import stft_reconstruction_loss

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    from .stft_loss_torch import stft_magnitude_loss
else:  # pragma: no cover
    stft_magnitude_loss = None

__all__ = ["stft_reconstruction_loss", "stft_magnitude_loss"]
