from __future__ import annotations

from typing import Tuple

import numpy as np


def _make_rir_taps(
    rng: np.random.Generator,
    length: int,
    base_delay: int,
    reflections: int,
) -> np.ndarray:
    taps = np.zeros(length, dtype=np.float32)
    direct_amp = rng.uniform(0.6, 1.0)
    base_delay = int(np.clip(base_delay, 0, length - 1))
    taps[base_delay] = direct_amp
    for idx in range(reflections):
        delay = base_delay + rng.integers(5, min(80, length - base_delay))
        delay = int(np.clip(delay, 0, length - 1))
        decay = np.exp(-0.25 * (idx + 1))
        amp = direct_amp * decay * rng.uniform(0.05, 0.4)
        taps[delay] += amp * rng.choice([-1.0, 1.0])
    return taps


def generate_dummy_rir(
    rng: np.random.Generator,
    L: int = 1024,
    two_receivers_close: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    rir = np.zeros((3, 2, L), dtype=np.float32)
    for src_idx in range(3):
        base_delay = rng.integers(10, 80)
        for recv_idx in range(2):
            if two_receivers_close:
                offset = rng.integers(-1, 2)
                delay = base_delay + offset + recv_idx * 0
            else:
                delay = base_delay + recv_idx * rng.integers(3, 10)
            reflections = rng.integers(3, 7)
            rir[src_idx, recv_idx] = _make_rir_taps(rng, L, delay, reflections)

    rir_anechoic = np.zeros_like(rir)
    for src_idx in range(3):
        for recv_idx in range(2):
            taps = rir[src_idx, recv_idx]
            k_star = int(np.argmax(np.abs(taps)))
            rir_anechoic[src_idx, recv_idx, k_star] = taps[k_star]

    return rir, rir_anechoic
