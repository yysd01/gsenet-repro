from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:  # pragma: no cover
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("stream_tncov requires torch installed") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gsenet_repro.streaming.tncov_streamer import TraceNormCovStreamer


def _str2bool(v: str) -> bool:
    value = str(v).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream trace-normalized covariance beamforming for 4ch wav input."
    )
    parser.add_argument("--wav4ch", type=str, required=True)
    parser.add_argument("--out", type=str, default="y0_tncov.wav")
    parser.add_argument("--chunk-size", type=int, default=1280)
    parser.add_argument("--mode", type=str, default="trace_norm")
    parser.add_argument("--gate-mode", type=str, default="vad", choices=("vad", "sector", "coherence"))
    parser.add_argument("--rtf-lib", type=str, default=None)
    parser.add_argument("--doa", type=int, default=None)
    parser.add_argument("--alpha-y", type=float, default=0.92)
    parser.add_argument("--alpha-v", type=float, default=0.98)
    parser.add_argument("--diag-load-v", type=float, default=1e-2)
    parser.add_argument("--diag-load-x", type=float, default=1e-3)
    parser.add_argument("--eps-trace", type=float, default=1e-6)
    parser.add_argument("--psd-project", type=_str2bool, default=False)
    parser.add_argument("--coh-fmin", type=float, default=200.0)
    parser.add_argument("--coh-fmax", type=float, default=5000.0)
    parser.add_argument("--coh-t0", type=float, default=0.15)
    parser.add_argument("--coh-t1", type=float, default=0.35)
    parser.add_argument("--vad-db-thresh", type=float, default=-35.0)
    parser.add_argument("--vad-smooth", type=float, default=6.0)
    parser.add_argument("--ref-ch", type=int, default=1)
    parser.add_argument("--log-interval-frames", type=int, default=100)
    args = parser.parse_args()

    if args.mode not in {"trace_norm", "tncov"}:
        raise SystemExit("--mode only supports 'trace_norm' (alias: tncov)")

    wav, sr = sf.read(args.wav4ch, always_2d=True, dtype="float32")
    if wav.shape[1] != 4:
        raise SystemExit(f"expected 4 channels, got {wav.shape[1]}")
    x = torch.from_numpy(wav.T)

    streamer = TraceNormCovStreamer(
        sample_rate=sr,
        num_mics=4,
        center=False,
        ref_ch=args.ref_ch,
        alpha_y=args.alpha_y,
        alpha_v=args.alpha_v,
        diag_load_v=args.diag_load_v,
        diag_load_x=args.diag_load_x,
        eps_trace=args.eps_trace,
        psd_project=args.psd_project,
        coh_fmin_hz=args.coh_fmin,
        coh_fmax_hz=args.coh_fmax,
        coh_t0=args.coh_t0,
        coh_t1=args.coh_t1,
        vad_db_thresh=args.vad_db_thresh,
        vad_smooth=args.vad_smooth,
        log_interval_frames=args.log_interval_frames,
        gate_mode=args.gate_mode,
    )

    if args.rtf_lib:
        streamer.load_rtf_lib(args.rtf_lib)
        if args.doa is not None:
            streamer.set_target_doa(args.doa)

    hop = streamer.hop_length
    chunk_size = max(args.chunk_size, hop)

    outputs = []
    for start in range(0, x.shape[-1], chunk_size):
        chunk = x[:, start : start + chunk_size]
        valid_len = min(chunk_size, x.shape[-1] - start)
        if chunk.shape[-1] < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[-1]))

        chunk_out_parts = []
        for sub_start in range(0, chunk.shape[-1], hop):
            sub = chunk[:, sub_start : sub_start + hop]
            y = streamer.process(sub, target_doa=args.doa)
            chunk_out_parts.append(y)
        chunk_out = torch.cat(chunk_out_parts, dim=-1)[:valid_len]
        outputs.append(chunk_out)

    y0 = torch.cat(outputs, dim=-1)[: x.shape[-1]].detach().cpu().numpy().astype(np.float32)
    sf.write(args.out, y0, sr)
    print(f"Wrote {args.out} (len={y0.shape[0]}, sr={sr})")


if __name__ == "__main__":
    main()
