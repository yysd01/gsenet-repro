"""Streaming utilities for GSENet reproduction."""

import importlib.util

from .gsenet_streamer import GSENetStreamer

__all__ = ["GSENetStreamer"]

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    from .mcwf_streamer import MCWFStreamer  # noqa: F401
    from .mvdr_streamer import MVDRStreamer  # noqa: F401
    from .tncov_streamer import TraceNormCovStreamer  # noqa: F401

    __all__.extend(["MCWFStreamer", "MVDRStreamer", "TraceNormCovStreamer"])
