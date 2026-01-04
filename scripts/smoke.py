from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gsenet_repro  # noqa: E402


def main() -> None:
    signal = np.random.randn(128)
    _ = gsenet_repro.fft_roundtrip(signal)
    print("OK")


if __name__ == "__main__":
    main()
