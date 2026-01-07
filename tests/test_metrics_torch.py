import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.metrics.metrics_torch import si_snr_db, snr_db


def test_snr_db_matches_identical_signal() -> None:
    ref = torch.ones(2, 160)
    est = ref.clone()
    snr = snr_db(ref, est)
    assert torch.isfinite(snr).all()
    assert torch.all(snr > 30.0)


def test_snr_improves_with_less_noise() -> None:
    torch.manual_seed(0)
    ref = torch.randn(2, 200)
    noisy = ref + 0.5 * torch.randn(2, 200)
    denoised = ref + 0.1 * torch.randn(2, 200)
    snr_in = snr_db(ref, noisy)
    snr_out = snr_db(ref, denoised)
    assert torch.mean(snr_out - snr_in) > 0.0


def test_si_snr_improves_with_less_noise() -> None:
    torch.manual_seed(1)
    ref = torch.randn(1, 240)
    noisy = ref + 0.6 * torch.randn(1, 240)
    denoised = ref + 0.2 * torch.randn(1, 240)
    sisnr_in = si_snr_db(ref, noisy)
    sisnr_out = si_snr_db(ref, denoised)
    assert torch.mean(sisnr_out - sisnr_in) > 0.0
