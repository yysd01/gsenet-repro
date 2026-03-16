from __future__ import annotations

import warnings

import numpy as np
import pytest

from gsenet_repro.config import resolve_config
from gsenet_repro.dsp.trace_norm import diag_load_np, hermitian_np, trace_norm_weights
from gsenet_repro.pipeline.frontend import make_y0_from_frontend


def test_trace_norm_weights_matches_formula() -> None:
    rng = np.random.default_rng(0)
    F, C = 16, 4
    Z1 = rng.normal(size=(F, C, C)) + 1j * rng.normal(size=(F, C, C))
    Z2 = rng.normal(size=(F, C, C)) + 1j * rng.normal(size=(F, C, C))
    phi_v = hermitian_np(np.matmul(Z1, np.swapaxes(np.conjugate(Z1), -1, -2))).astype(np.complex64)
    phi_y = hermitian_np(np.matmul(Z2, np.swapaxes(np.conjugate(Z2), -1, -2))).astype(np.complex64) + phi_v
    phi_x = hermitian_np(phi_y - phi_v)

    w = trace_norm_weights(phi_v, phi_x, ref_ch=2, diag_load_v=1e-2, diag_load_x=1e-3, eps_trace=1e-6, psd_project=False)

    pv = diag_load_np(hermitian_np(phi_v), 1e-2)
    px = diag_load_np(hermitian_np(phi_x), 1e-3)
    A = np.linalg.solve(pv, px)
    den = np.clip(A.diagonal(axis1=-2, axis2=-1).sum(axis=-1).real, 1e-6, None)
    w_ref = A[..., 2] / den[:, None]
    assert np.allclose(w, w_ref.astype(np.complex64), atol=1e-5, rtol=1e-4)


def test_trace_norm_frontend_output_shape_and_finite() -> None:
    rng = np.random.default_rng(1)
    stft_cfg = {"n_fft": 320, "win_length": 320, "hop_length": 160}
    frontend_cfg = {"type": "trace_norm", "gate_mode": "vad", "ref_ch": 1}
    data_cfg = {"sample_rate": 16000, "ref_mic_index": 1}

    x = rng.normal(size=(4, 1024)).astype(np.float32)
    y = make_y0_from_frontend(x, frontend_cfg, stft_cfg, data_cfg)
    assert y.shape == (1024,)
    assert np.isfinite(y).all()

    xb = rng.normal(size=(2, 4, 1024)).astype(np.float32)
    yb = make_y0_from_frontend(xb, frontend_cfg, stft_cfg, data_cfg)
    assert yb.shape == (2, 1024)
    assert np.isfinite(yb).all()


def test_trace_norm_frontend_ref_channel_fallback() -> None:
    F, C = 8, 4
    phi_v = np.zeros((F, C, C), dtype=np.complex64)
    phi_x = np.zeros((F, C, C), dtype=np.complex64)
    w = trace_norm_weights(phi_v, phi_x, ref_ch=3, eps_trace=1e-6)
    expected = np.zeros((F, C), dtype=np.complex64)
    expected[:, 3] = 1.0 + 0.0j
    assert np.allclose(w, expected)


def test_streaming_trace_norm_ref_and_stft_consistency() -> None:
    pytest.importorskip("torch")
    from gsenet_repro.streaming.tncov_streamer import TraceNormCovStreamer
    cfg = {
        "data": {"sample_rate": 16000, "num_mics": 4, "ref_mic_index": 2},
        "stft_model": {"n_fft": 320, "win_length": 320, "hop_length": 160},
        "frontend": {"ref_ch": 2, "type": "trace_norm"},
    }
    s = TraceNormCovStreamer.from_config(cfg)
    assert s.ref_ch == 2
    assert s.n_fft == 320
    assert s.win_length == 320
    assert s.hop_length == 160


def test_streaming_trace_norm_stft_mismatch_rejected() -> None:
    pytest.importorskip("torch")
    from gsenet_repro.streaming.tncov_streamer import TraceNormCovStreamer

    cfg = {
        "data": {"sample_rate": 16000, "num_mics": 4, "ref_mic_index": 1},
        "stft_model": {"n_fft": 320, "win_length": 320, "hop_length": 160},
        "stft_streaming": {"n_fft": 512, "win_length": 320, "hop_length": 160},
        "frontend": {"type": "trace_norm", "ref_ch": 1},
    }
    with pytest.raises(ValueError, match="stft_streaming mismatch"):
        TraceNormCovStreamer.from_config(cfg)


def test_legacy_use_mcwf_config_is_mapped() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cfg = resolve_config(None, {"data": {"use_mcwf": 0}})
    assert cfg["frontend"]["type"] == "none"
    assert any("deprecated" in str(w.message) for w in rec)


@pytest.mark.filterwarnings("ignore:data.use_mcwf is deprecated and mapped to frontend.type:DeprecationWarning")
def test_prepare_batch_uses_frontend_type() -> None:
    torch = pytest.importorskip("torch")
    from scripts.train_paper_like_full import _prepare_batch_for_model

    rng = np.random.default_rng(2)
    x = torch.from_numpy(rng.normal(size=(2, 4, 640)).astype(np.float32))
    y1 = x[:, 1]
    yt = x[:, 0]
    batch = {"x_mics": x, "y1": y1, "yt": yt}
    data_cfg = {"sample_rate": 16000, "ref_mic_index": 1, "mic_positions": None, "use_mcwf": 1}
    stft_cfg = {"n_fft": 320, "win_length": 320, "hop_length": 160}
    frontend_cfg = {"type": "trace_norm", "gate_mode": "vad", "ref_ch": 1}

    prepared = _prepare_batch_for_model(batch, data_cfg, stft_cfg, frontend_cfg, "gsenet_paper_scale")
    assert "y0" in prepared
    assert "y2" not in prepared
    assert prepared["y0"].shape == (2, 640)
