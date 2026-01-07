from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve, firwin, lfilter


@dataclass(frozen=True)
class PaperParams:
    """Sampled parameters aligned with Table 1 (GSENet row)."""

    gn_lin: float
    gi_lin: float
    pi: float
    alpha_lin: float
    beta_lin: float
    gn_db: float
    gi_db: float
    alpha_db: float
    beta_db: float
    global_gain: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "gn_lin": self.gn_lin,
            "gi_lin": self.gi_lin,
            "pi": self.pi,
            "alpha_lin": self.alpha_lin,
            "beta_lin": self.beta_lin,
            "gn_db": self.gn_db,
            "gi_db": self.gi_db,
            "alpha_db": self.alpha_db,
            "beta_db": self.beta_db,
            "global_gain": self.global_gain,
        }


def db_to_lin(db: np.ndarray | float) -> np.ndarray | float:
    """Convert decibel scale to linear amplitude (20*log10)."""
    return 10 ** (np.asarray(db) / 20.0)


def lin_to_db(lin: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    """Convert linear amplitude to decibel scale (20*log10)."""
    lin_safe = np.maximum(np.asarray(lin), eps)
    return 20.0 * np.log10(lin_safe)


def sample_paper_params(
    rng: np.random.Generator,
    variant: str = "gsenet",
    global_gain: float = 1.0,
) -> PaperParams:
    """Sample gain parameters following Table 1 (GSENet row).

    Reference: arXiv:2303.07486v1, Table 1 (GSENet) and Section 2.1.
    """
    if variant != "gsenet":
        raise ValueError(f"Unsupported variant '{variant}'. Only 'gsenet' is implemented.")

    gn_db = float(rng.normal(loc=-5.0, scale=10.0))
    gi_db = float(rng.normal(loc=-3.0, scale=3.0))
    alpha_db = float(max(rng.normal(loc=0.0, scale=3.0), -4.0))
    beta_db = float(max(rng.normal(loc=4.0, scale=6.0), 4.0))
    pi = float(rng.random() < 0.4)

    return PaperParams(
        gn_lin=float(db_to_lin(gn_db)),
        gi_lin=float(db_to_lin(gi_db)),
        pi=pi,
        alpha_lin=float(db_to_lin(alpha_db)),
        beta_lin=float(db_to_lin(beta_db)),
        gn_db=gn_db,
        gi_db=gi_db,
        alpha_db=alpha_db,
        beta_db=beta_db,
        global_gain=float(global_gain),
    )


def normalize_rms(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize waveform or RIR by RMS (power normalization approximation)."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    rms = np.sqrt(np.mean(x**2))
    return (x / (rms + eps)).astype(np.float32)


def _smooth_envelope(rng: np.random.Generator, length: int, fs: int) -> np.ndarray:
    kernel_len = max(3, int(0.04 * fs))
    if kernel_len % 2 == 0:
        kernel_len += 1
    kernel = np.hanning(kernel_len).astype(np.float32)
    kernel /= np.sum(kernel)
    noise = rng.normal(size=length).astype(np.float32)
    smooth = np.convolve(noise, kernel, mode="same")
    smooth = np.abs(smooth)
    smooth = smooth / (np.max(smooth) + 1e-8)
    return smooth.astype(np.float32)


def _pink_noise(rng: np.random.Generator, length: int, fs: int) -> np.ndarray:
    white = rng.normal(size=length).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, d=1.0 / fs)
    scale = np.where(freqs == 0, 0.0, 1.0 / np.sqrt(freqs))
    spectrum *= scale.astype(np.float32)
    pink = np.fft.irfft(spectrum, n=length)
    return normalize_rms(pink)


def _band_limited_noise(
    rng: np.random.Generator,
    length: int,
    fs: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    taps = firwin(numtaps=255, cutoff=[low_hz, high_hz], pass_zero=False, fs=fs)
    noise = rng.normal(size=length).astype(np.float32)
    filtered = lfilter(taps, 1.0, noise).astype(np.float32)
    return normalize_rms(filtered)


def _speech_like_noise(rng: np.random.Generator, length: int, fs: int) -> np.ndarray:
    speech_band = _band_limited_noise(rng, length, fs, low_hz=200.0, high_hz=3400.0)
    envelope = _smooth_envelope(rng, length, fs)
    modulated = speech_band * (0.2 + 0.8 * envelope)
    return normalize_rms(modulated)


def generate_noise(
    rng: np.random.Generator,
    length: int,
    fs: int,
    noise_type: str,
) -> np.ndarray:
    """Generate a noise waveform with a specific spectral profile."""
    noise_type = noise_type.lower()
    if noise_type == "white":
        return normalize_rms(rng.normal(size=length).astype(np.float32))
    if noise_type == "pink":
        return _pink_noise(rng, length, fs)
    if noise_type == "speech":
        return _speech_like_noise(rng, length, fs)
    if noise_type == "babble":
        babble = np.zeros(length, dtype=np.float32)
        for _ in range(3):
            babble += _speech_like_noise(rng, length, fs)
        return normalize_rms(babble)
    raise ValueError(f"Unsupported noise_type '{noise_type}'")


def generate_noise_mix(
    rng: np.random.Generator,
    length: int,
    fs: int,
    noise_types: Optional[Tuple[str, ...]] = None,
    max_components: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a mixture of multiple noise types."""
    if noise_types is None:
        noise_types = ("white", "pink", "speech", "babble")

    n_components = int(rng.integers(1, max_components + 1))
    selected = list(rng.choice(noise_types, size=n_components, replace=True))
    weights = rng.uniform(0.2, 1.0, size=n_components)
    weights = weights / np.sum(weights)

    mix = np.zeros(length, dtype=np.float32)
    for weight, noise_type in zip(weights, selected):
        mix += weight * generate_noise(rng, length, fs, noise_type)

    return normalize_rms(mix), {"types": selected, "weights": weights.tolist()}


def generate_background_noise(
    rng: np.random.Generator,
    length: int,
    fs: int,
    num_mics: int = 2,
    noise_types: Optional[Tuple[str, ...]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate background noise for multiple microphones with small spatial variation."""
    base_noise, meta = generate_noise_mix(rng, length, fs, noise_types=noise_types)

    noise_mics = np.zeros((num_mics, length), dtype=np.float32)
    delays = []
    gains = []
    for mic_idx in range(num_mics):
        delay = int(rng.integers(0, 4))
        gain = float(rng.uniform(0.7, 1.3))
        delayed = np.pad(base_noise, (delay, 0), mode="constant")[:length]
        noise_mics[mic_idx] = normalize_rms(delayed) * gain
        delays.append(delay)
        gains.append(gain)

    meta.update({"mic_delays": delays, "mic_gains": gains})
    return noise_mics.astype(np.float32), meta


def add_background_noise(
    y_mics: np.ndarray,
    noise_mics: np.ndarray,
    snr_db: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mix background noise into multi-microphone signals at a target SNR."""
    y_mics = np.asarray(y_mics, dtype=np.float32)
    noise_mics = np.asarray(noise_mics, dtype=np.float32)
    if y_mics.shape != noise_mics.shape:
        raise ValueError("y_mics and noise_mics must have the same shape")

    target_ratio = 10 ** (snr_db / 20.0)
    scaled_noise = np.zeros_like(noise_mics)
    for mic_idx in range(y_mics.shape[0]):
        signal_rms = np.sqrt(np.mean(y_mics[mic_idx] ** 2) + 1e-8)
        noise_rms = np.sqrt(np.mean(noise_mics[mic_idx] ** 2) + 1e-8)
        scale = signal_rms / (noise_rms * target_ratio + 1e-8)
        scaled_noise[mic_idx] = noise_mics[mic_idx] * scale

    return (y_mics + scaled_noise).astype(np.float32), scaled_noise.astype(np.float32)


def _make_rir(
    rng: np.random.Generator,
    length: int,
    direct_delay: int,
    early_reflections: int,
    max_early_delay: int,
    tail_decay: float,
    tail_level: float,
) -> np.ndarray:
    rir = np.zeros(length, dtype=np.float32)
    direct_delay = int(np.clip(direct_delay, 0, length - 1))
    rir[direct_delay] = 1.0

    for idx in range(early_reflections):
        delay = direct_delay + int(rng.integers(1, max_early_delay + 1))
        if delay >= length:
            break
        amp = rng.uniform(0.12, 0.45) * (0.7**idx)
        sign = -1.0 if rng.random() < 0.5 else 1.0
        rir[delay] += sign * amp

    tail_start = min(length - 1, direct_delay + max_early_delay)
    if tail_start < length - 1:
        t = np.arange(tail_start, length)
        decay = np.exp(-tail_decay * (t - tail_start))
        noise = rng.normal(scale=tail_level, size=t.shape)
        rir[tail_start:] += (decay * noise).astype(np.float32)

    return rir


def _sample_room_acoustics(rng: np.random.Generator, fs: int) -> Dict[str, float]:
    rt60 = float(rng.uniform(0.2, 0.8))
    tail_decay = float(np.log(1000.0) / (rt60 * fs))
    tail_level = float(rng.uniform(0.01, 0.05))
    max_early_delay = int(rng.integers(int(0.01 * fs), int(0.04 * fs)))
    early_reflections = int(rng.integers(3, 7))
    return {
        "tail_decay": tail_decay,
        "tail_level": tail_level,
        "max_early_delay": max_early_delay,
        "early_reflections": early_reflections,
    }


def generate_rir_3src_2mic(
    rng: np.random.Generator,
    rir_length: int = 1024,
    fs: int = 16000,
    max_direct_delay_diff: int = 4,
    early_reflections: Optional[int] = None,
    max_early_delay: Optional[int] = None,
    tail_decay: Optional[float] = None,
    tail_level: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate lightweight RIRs for 3 sources and 2 close microphones.

    Reference: arXiv:2303.07486v1, Section 2.1.
    The two receivers are placed close enough so their direct-path delay
    difference stays below ``max_direct_delay_diff`` samples.
    """
    room_params = _sample_room_acoustics(rng, fs)
    early_reflections = early_reflections or int(room_params["early_reflections"])
    max_early_delay = max_early_delay or int(room_params["max_early_delay"])
    tail_decay = tail_decay or float(room_params["tail_decay"])
    tail_level = tail_level or float(room_params["tail_level"])

    rir = np.zeros((3, 2, rir_length), dtype=np.float32)
    direct_delays = np.zeros((3, 2), dtype=np.int32)

    for src_idx in range(3):
        base_delay = int(rng.integers(int(0.004 * fs), int(0.015 * fs)))
        offsets = [0, int(rng.integers(-max_direct_delay_diff, max_direct_delay_diff + 1))]
        for mic_idx in range(2):
            delay = max(0, base_delay + offsets[mic_idx])
            direct_delays[src_idx, mic_idx] = delay
            rir[src_idx, mic_idx] = _make_rir(
                rng,
                length=rir_length,
                direct_delay=delay,
                early_reflections=early_reflections,
                max_early_delay=max_early_delay,
                tail_decay=tail_decay,
                tail_level=tail_level,
            )

    rir = np.stack(
        [normalize_rms(rir[src, mic]) for src in range(3) for mic in range(2)],
        axis=0,
    ).reshape(3, 2, rir_length)

    max_delay = int(np.max(direct_delays))
    anechoic_length = max_delay + 1
    rir_anechoic = np.zeros((3, 2, anechoic_length), dtype=np.float32)
    for src_idx in range(3):
        for mic_idx in range(2):
            delay = direct_delays[src_idx, mic_idx]
            rir_anechoic[src_idx, mic_idx, delay] = 1.0
            rir_anechoic[src_idx, mic_idx] = normalize_rms(rir_anechoic[src_idx, mic_idx])

    return rir.astype(np.float32), rir_anechoic.astype(np.float32)


def generate_rir_3src_3mic(
    rng: np.random.Generator,
    rir_length: int = 1024,
    fs: int = 16000,
    max_direct_delay_diff: int = 4,
    early_reflections: Optional[int] = None,
    max_early_delay: Optional[int] = None,
    tail_decay: Optional[float] = None,
    tail_level: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate lightweight RIRs for 3 sources and 3 close microphones."""
    room_params = _sample_room_acoustics(rng, fs)
    early_reflections = early_reflections or int(room_params["early_reflections"])
    max_early_delay = max_early_delay or int(room_params["max_early_delay"])
    tail_decay = tail_decay or float(room_params["tail_decay"])
    tail_level = tail_level or float(room_params["tail_level"])

    rir = np.zeros((3, 3, rir_length), dtype=np.float32)
    direct_delays = np.zeros((3, 3), dtype=np.int32)

    for src_idx in range(3):
        base_delay = int(rng.integers(int(0.004 * fs), int(0.015 * fs)))
        for mic_idx in range(3):
            offset = int(rng.integers(-max_direct_delay_diff, max_direct_delay_diff + 1))
            delay = max(0, base_delay + offset)
            direct_delays[src_idx, mic_idx] = delay
            rir[src_idx, mic_idx] = _make_rir(
                rng,
                length=rir_length,
                direct_delay=delay,
                early_reflections=early_reflections,
                max_early_delay=max_early_delay,
                tail_decay=tail_decay,
                tail_level=tail_level,
            )

    rir = np.stack(
        [normalize_rms(rir[src, mic]) for src in range(3) for mic in range(3)],
        axis=0,
    ).reshape(3, 3, rir_length)

    max_delay = int(np.max(direct_delays))
    anechoic_length = max_delay + 1
    rir_anechoic = np.zeros((3, 3, anechoic_length), dtype=np.float32)
    for src_idx in range(3):
        for mic_idx in range(3):
            delay = direct_delays[src_idx, mic_idx]
            rir_anechoic[src_idx, mic_idx, delay] = 1.0
            rir_anechoic[src_idx, mic_idx] = normalize_rms(rir_anechoic[src_idx, mic_idx])

    return rir.astype(np.float32), rir_anechoic.astype(np.float32)


def _normalize_params(params: Optional[PaperParams | Dict[str, Any]]) -> PaperParams:
    if params is None:
        raise ValueError("params must be provided for synthesis.")
    if isinstance(params, PaperParams):
        return params
    required = {"gn_lin", "gi_lin", "pi", "alpha_lin", "beta_lin"}
    missing = required.difference(params.keys())
    if missing:
        raise ValueError(f"params is missing keys: {sorted(missing)}")
    gn_lin = float(params["gn_lin"])
    gi_lin = float(params["gi_lin"])
    pi = float(params["pi"])
    alpha_lin = float(params["alpha_lin"])
    beta_lin = float(params["beta_lin"])
    return PaperParams(
        gn_lin=gn_lin,
        gi_lin=gi_lin,
        pi=pi,
        alpha_lin=alpha_lin,
        beta_lin=beta_lin,
        gn_db=float(params.get("gn_db", lin_to_db(gn_lin))),
        gi_db=float(params.get("gi_db", lin_to_db(gi_lin))),
        alpha_db=float(params.get("alpha_db", lin_to_db(alpha_lin))),
        beta_db=float(params.get("beta_db", lin_to_db(beta_lin))),
        global_gain=float(params.get("global_gain", 1.0)),
    )


def _fftconvolve_truncate(signal: np.ndarray, kernel: np.ndarray, length: int) -> np.ndarray:
    out = fftconvolve(signal, kernel, mode="full")
    return out[:length].astype(np.float32)


def synthesize_y0_y1_yt(
    s: np.ndarray,
    n: np.ndarray,
    i: np.ndarray,
    rir: np.ndarray,
    rir_anechoic: np.ndarray,
    params: PaperParams | Dict[str, Any],
    background_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize y0/y1/yt as defined in Section 2.1 of the paper.

    y0 = s * r(0,0) + gn * n * r(1,0) + pi * gi * i * r(2,0)
    y1 = s * r(0,1) + alpha * gn * n * r(1,1) + beta * pi * gi * i * r(2,1)
    yt = s * r_anechoic(0,0) (anechoic only keeps the strongest path)
    """
    params = _normalize_params(params)

    s = normalize_rms(np.asarray(s, dtype=np.float32))
    n = normalize_rms(np.asarray(n, dtype=np.float32))
    i = normalize_rms(np.asarray(i, dtype=np.float32))

    rir = np.asarray(rir, dtype=np.float32)
    rir_anechoic = np.asarray(rir_anechoic, dtype=np.float32)
    if rir.shape[:2] != (3, 2):
        raise ValueError("rir must have shape (3, 2, L)")
    if rir_anechoic.shape[:2] != (3, 2):
        raise ValueError("rir_anechoic must have shape (3, 2, L_anechoic)")

    length = s.shape[0]

    rir_norm = np.stack(
        [normalize_rms(rir[src, mic]) for src in range(3) for mic in range(2)],
        axis=0,
    ).reshape(3, 2, -1)
    rir_anechoic_norm = np.stack(
        [normalize_rms(rir_anechoic[src, mic]) for src in range(3) for mic in range(2)],
        axis=0,
    ).reshape(3, 2, -1)

    y0 = _fftconvolve_truncate(s, rir_norm[0, 0], length)
    y0 += params.gn_lin * _fftconvolve_truncate(n, rir_norm[1, 0], length)
    y0 += params.pi * params.gi_lin * _fftconvolve_truncate(i, rir_norm[2, 0], length)

    y1 = _fftconvolve_truncate(s, rir_norm[0, 1], length)
    y1 += params.alpha_lin * params.gn_lin * _fftconvolve_truncate(n, rir_norm[1, 1], length)
    y1 += params.beta_lin * params.pi * params.gi_lin * _fftconvolve_truncate(i, rir_norm[2, 1], length)

    anechoic = rir_anechoic_norm[0, 0]
    k_star = int(np.argmax(np.abs(anechoic)))
    h_main = np.zeros_like(anechoic)
    h_main[k_star] = anechoic[k_star]
    yt = _fftconvolve_truncate(s, h_main, length)

    if background_config is not None:
        rng = background_config.get("rng")
        if rng is None:
            raise ValueError("background_config requires a numpy Generator as 'rng'")
        fs = int(background_config.get("fs", 16000))
        noise_types = background_config.get("noise_types")
        snr_db_range = background_config.get("snr_db_range", (-2.0, 12.0))
        y_mics = np.stack([y0, y1], axis=0)
        noise_mics, noise_meta = generate_background_noise(
            rng,
            length=length,
            fs=fs,
            num_mics=2,
            noise_types=noise_types,
        )
        snr_db = float(rng.uniform(*snr_db_range))
        y_mics, _ = add_background_noise(y_mics, noise_mics, snr_db=snr_db)
        y0, y1 = y_mics[0], y_mics[1]
        if "metadata" in background_config:
            background_config["metadata"].append({**noise_meta, "snr_db": snr_db})

    if params.global_gain != 1.0:
        y0 *= params.global_gain
        y1 *= params.global_gain
        yt *= params.global_gain

    return y0.astype(np.float32), y1.astype(np.float32), yt.astype(np.float32)


def synthesize_y0_y1_y2_yt(
    s: np.ndarray,
    n: np.ndarray,
    i: np.ndarray,
    rir: np.ndarray,
    rir_anechoic: np.ndarray,
    params: PaperParams | Dict[str, Any],
    background_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize y0/y1/y2/yt for 3-microphone mixtures."""
    params = _normalize_params(params)

    s = normalize_rms(np.asarray(s, dtype=np.float32))
    n = normalize_rms(np.asarray(n, dtype=np.float32))
    i = normalize_rms(np.asarray(i, dtype=np.float32))

    rir = np.asarray(rir, dtype=np.float32)
    rir_anechoic = np.asarray(rir_anechoic, dtype=np.float32)
    if rir.shape[:2] != (3, 3):
        raise ValueError("rir must have shape (3, 3, L)")
    if rir_anechoic.shape[:2] != (3, 3):
        raise ValueError("rir_anechoic must have shape (3, 3, L_anechoic)")

    length = s.shape[0]

    rir_norm = np.stack(
        [normalize_rms(rir[src, mic]) for src in range(3) for mic in range(3)],
        axis=0,
    ).reshape(3, 3, -1)
    rir_anechoic_norm = np.stack(
        [normalize_rms(rir_anechoic[src, mic]) for src in range(3) for mic in range(3)],
        axis=0,
    ).reshape(3, 3, -1)

    y0 = _fftconvolve_truncate(s, rir_norm[0, 0], length)
    y0 += params.gn_lin * _fftconvolve_truncate(n, rir_norm[1, 0], length)
    y0 += params.pi * params.gi_lin * _fftconvolve_truncate(i, rir_norm[2, 0], length)

    y1 = _fftconvolve_truncate(s, rir_norm[0, 1], length)
    y1 += params.alpha_lin * params.gn_lin * _fftconvolve_truncate(n, rir_norm[1, 1], length)
    y1 += params.beta_lin * params.pi * params.gi_lin * _fftconvolve_truncate(i, rir_norm[2, 1], length)

    y2 = _fftconvolve_truncate(s, rir_norm[0, 2], length)
    y2 += params.alpha_lin * params.gn_lin * _fftconvolve_truncate(n, rir_norm[1, 2], length)
    y2 += params.beta_lin * params.pi * params.gi_lin * _fftconvolve_truncate(i, rir_norm[2, 2], length)

    anechoic = rir_anechoic_norm[0, 0]
    k_star = int(np.argmax(np.abs(anechoic)))
    h_main = np.zeros_like(anechoic)
    h_main[k_star] = anechoic[k_star]
    yt = _fftconvolve_truncate(s, h_main, length)

    if background_config is not None:
        rng = background_config.get("rng")
        if rng is None:
            raise ValueError("background_config requires a numpy Generator as 'rng'")
        fs = int(background_config.get("fs", 16000))
        noise_types = background_config.get("noise_types")
        snr_db_range = background_config.get("snr_db_range", (-2.0, 12.0))
        y_mics = np.stack([y0, y1, y2], axis=0)
        noise_mics, noise_meta = generate_background_noise(
            rng,
            length=length,
            fs=fs,
            num_mics=3,
            noise_types=noise_types,
        )
        snr_db = float(rng.uniform(*snr_db_range))
        y_mics, _ = add_background_noise(y_mics, noise_mics, snr_db=snr_db)
        y0, y1, y2 = y_mics[0], y_mics[1], y_mics[2]
        if "metadata" in background_config:
            background_config["metadata"].append({**noise_meta, "snr_db": snr_db})

    if params.global_gain != 1.0:
        y0 *= params.global_gain
        y1 *= params.global_gain
        y2 *= params.global_gain
        yt *= params.global_gain

    return (
        y0.astype(np.float32),
        y1.astype(np.float32),
        y2.astype(np.float32),
        yt.astype(np.float32),
    )
