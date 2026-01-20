"""Model definitions."""
import importlib.util

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    from .gsenet_torch import MinimalGSENet
    from .gsenet_paper_torch import GSENetPaperScale
else:  # pragma: no cover
    MinimalGSENet = None
    GSENetPaperScale = None

__all__ = ["MinimalGSENet", "GSENetPaperScale"]
