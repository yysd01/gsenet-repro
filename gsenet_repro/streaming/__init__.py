"""Streaming utilities for GSENet reproduction."""

import importlib.util

from .gsenet_streamer import GSENetStreamer

__all__ = ["GSENetStreamer"]

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    from .mcwf_streamer import MCWFStreamer

    __all__.append("MCWFStreamer")
