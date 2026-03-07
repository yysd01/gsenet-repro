import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from gsenet_repro.dsp.mvdr import apply_beamformer, estimate_rnn, mvdr_weights


def snr_db(clean: np.ndarray, est: np.ndarray) -> float:
    num = np.mean(np.abs(clean) ** 2)
    den = np.mean(np.abs(clean - est) ** 2) + 1e-8
    return float(10.0 * np.log10((num + 1e-8) / den))


def main() -> None:
    rng = np.random.default_rng(0)
    F, T, C = 96, 240, 4
    ref = 1

    freqs = np.linspace(0.0, 1.0, F, dtype=np.float32)
    tau_tar = np.array([0.0, 2e-5, -1e-5, 1e-5], dtype=np.float32)
    tau_int = np.array([0.0, -7e-5, 6e-5, -5e-5], dtype=np.float32)
    d_tar = np.exp(-1j * 2 * np.pi * freqs[:, None] * tau_tar[None, :] * 8000.0).astype(
        np.complex64
    )
    d_int = np.exp(-1j * 2 * np.pi * freqs[:, None] * tau_int[None, :] * 8000.0).astype(
        np.complex64
    )
    d_tar /= d_tar[:, [ref]]

    s = (rng.standard_normal((F, T)) + 1j * rng.standard_normal((F, T))).astype(np.complex64)
    i = (rng.standard_normal((F, T)) + 1j * rng.standard_normal((F, T))).astype(np.complex64)
    noise = 0.15 * (rng.standard_normal((F, T, C)) + 1j * rng.standard_normal((F, T, C))).astype(
        np.complex64
    )

    s[:, : T // 3] = 0.0
    X = s[:, :, None] * d_tar[:, None, :] + 1.2 * i[:, :, None] * d_int[:, None, :] + noise

    noise_gate = np.zeros(T, dtype=np.float32)
    noise_gate[: T // 3] = 1.0
    noise_gate[T // 3 :] = 0.5
    rnn = estimate_rnn(X, noise_gate, smoothing=0.96)
    w = mvdr_weights(rnn, d_tar, diag_load=1e-2)
    Y = apply_beamformer(w, X)

    y_ref = X[:, :, ref]
    clean_ref = s
    gain = snr_db(clean_ref, Y) - snr_db(clean_ref, y_ref)
    print(f"mvdr_vs_singlemic_snr_gain={gain:.3f} dB")
    if gain <= 0.0:
        raise SystemExit("MVDR did not beat single mic baseline")


if __name__ == "__main__":
    main()
