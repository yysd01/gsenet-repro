from __future__ import annotations

import torch


def snr_db(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute SNR in dB per sample (batch-aware)."""
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    if reference.ndim == 1:
        reference = reference[None, :]
        estimate = estimate[None, :]
    noise = reference - estimate
    ref_pow = torch.mean(reference**2, dim=-1) + eps
    noise_pow = torch.mean(noise**2, dim=-1) + eps
    return 10.0 * torch.log10(ref_pow / noise_pow)


def si_snr_db(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute SI-SNR in dB per sample (batch-aware)."""
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    if reference.ndim == 1:
        reference = reference[None, :]
        estimate = estimate[None, :]
    ref = reference - torch.mean(reference, dim=-1, keepdim=True)
    est = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref
    denom = torch.sum(ref**2, dim=-1, keepdim=True) + eps
    proj = proj / denom
    noise = est - proj
    ratio = torch.sum(proj**2, dim=-1) / (torch.sum(noise**2, dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def sisdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute SI-SDR in dB per sample (batch-aware)."""
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    if reference.ndim == 1:
        reference = reference[None, :]
        estimate = estimate[None, :]
    ref = reference - torch.mean(reference, dim=-1, keepdim=True)
    est = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref
    denom = torch.sum(ref**2, dim=-1, keepdim=True) + eps
    proj = proj / denom
    noise = est - proj
    ratio = torch.sum(proj**2, dim=-1) / (torch.sum(noise**2, dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def batch_sisdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batch SI-SDR that always returns a 1D tensor of shape (B,)."""
    sisdr_vals = sisdr(reference, estimate, eps=eps)
    if sisdr_vals.ndim != 1:
        raise RuntimeError("sisdr must return a 1D tensor")
    return sisdr_vals
