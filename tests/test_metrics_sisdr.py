from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gsenet_repro.metrics.metrics_torch import batch_sisdr, sisdr


def test_sisdr_prefers_clean_signal() -> None:
    torch.manual_seed(0)
    reference = torch.randn(2, 1600)
    noisy = reference + 0.5 * torch.randn_like(reference)

    clean_score = sisdr(reference, reference).mean().item()
    noisy_score = sisdr(reference, noisy).mean().item()

    assert clean_score > noisy_score


def test_batch_sisdr_shape() -> None:
    torch.manual_seed(1)
    reference = torch.randn(3, 800)
    estimate = reference + 0.1 * torch.randn_like(reference)
    scores = batch_sisdr(reference, estimate)
    assert scores.shape == (3,)
