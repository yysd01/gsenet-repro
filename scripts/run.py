from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_BY_COMMAND = {
    "stft": REPO_ROOT / "scripts" / "smoke_stft.py",
    "mvdr": REPO_ROOT / "scripts" / "smoke_mvdr.py",
    "train": REPO_ROOT / "scripts" / "train.py",
    "test": REPO_ROOT / "scripts" / "test.py",
    "report": REPO_ROOT / "scripts" / "report_paper_like_full.py",
    "stream-mvdr": REPO_ROOT / "scripts" / "stream_mvdr.py",
    "stream-tncov": REPO_ROOT / "scripts" / "stream_tncov.py",
}


def _normalize_extra(extra: list[str]) -> list[str]:
    if extra and extra[0] == "--":
        return extra[1:]
    return extra


def _run_passthrough(command: str, extra_args: list[str]) -> None:
    script = SCRIPT_BY_COMMAND[command]
    subprocess.run([sys.executable, str(script), *_normalize_extra(extra_args)], check=True)


def _load_wav(path: Path, target_sr: int, max_mics: int):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    x = wav.T  # (C, T)
    if x.shape[0] < 2:
        raise ValueError(f"Expected >=2 channels, got {x.shape[0]} from {path}")
    if x.shape[0] > max_mics:
        print(f"WARNING: input has {x.shape[0]} channels; truncating to first {max_mics}")
        x = x[:max_mics]

    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz")
        x = np.stack([resample_poly(ch, target_sr, sr).astype(np.float32) for ch in x], axis=0)
    return x.astype(np.float32), target_sr


def _find_dummy_mic_wav(root: Path) -> Path:
    import soundfile as sf

    candidates = sorted(root.glob("*/mic/*.wav"))
    if not candidates:
        raise FileNotFoundError(f"No dummy mic wav files found under {root}")

    for wav in candidates:
        data, _ = sf.read(str(wav), always_2d=True, dtype="float32")
        if data.shape[1] >= 4:
            return wav
    return candidates[0]


def _prepare_dummy_input(target_sr: int):
    from scripts._legacy.make_dummy_real_dir_dataset import make_dummy_real_dir_dataset

    dummy_root = REPO_ROOT / "artifacts" / "_dummy_real_dir"
    if not dummy_root.exists():
        print(f"Dummy dataset not found, generating at {dummy_root}")
        make_dummy_real_dir_dataset(dummy_root, sample_rate=target_sr)
    else:
        print(f"Reusing dummy dataset at {dummy_root}")

    wav_path = _find_dummy_mic_wav(dummy_root)
    x_mics, sr = _load_wav(wav_path, target_sr=target_sr, max_mics=999)
    return x_mics, sr, wav_path


def _resolve_mic_positions(cfg: dict, override_json: str | None, channels: int):
    import numpy as np

    from gsenet_repro.pipeline.mcwf_frontend import DEFAULT_MIC_POSITIONS

    if override_json is not None:
        positions = np.asarray(json.loads(override_json), dtype=np.float32)
    elif cfg["data"].get("mic_positions") is not None:
        positions = np.asarray(cfg["data"]["mic_positions"], dtype=np.float32)
    else:
        positions = DEFAULT_MIC_POSITIONS

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("mic_positions must have shape (C, 3) in config or --mic-positions-json")
    if positions.shape[0] < channels:
        raise ValueError(
            f"mic_positions provides {positions.shape[0]} channels but input has {channels}"
        )
    return positions[:channels]


def _diag_gates(args: argparse.Namespace) -> None:
    import numpy as np
    import soundfile as sf

    from gsenet_repro.config import resolve_config
    from gsenet_repro.pipeline.mcwf_frontend import _estimate_gates, mcwf_make_y0

    cfg = resolve_config(args.config)
    sample_rate = int(args.sample_rate or cfg["data"]["sample_rate"])
    num_mics = int(cfg["data"].get("num_mics", 4))
    ref_ch = int(args.ref_ch if args.ref_ch is not None else cfg["data"].get("ref_mic_index", 1))

    if args.wav is not None:
        input_path = Path(args.wav)
        if not input_path.exists():
            raise SystemExit(f"Input wav does not exist: {input_path}")
        x_mics, sample_rate = _load_wav(input_path, target_sr=sample_rate, max_mics=num_mics)
        source_msg = str(input_path)
    else:
        x_mics, sample_rate, wav_path = _prepare_dummy_input(target_sr=sample_rate)
        if x_mics.shape[0] > num_mics:
            print(
                f"WARNING: dummy wav has {x_mics.shape[0]} channels; truncating to first {num_mics}"
            )
            x_mics = x_mics[:num_mics]
        source_msg = f"dummy:{wav_path}"

    max_samples = int(max(1.0, args.seconds) * sample_rate)
    x_mics = x_mics[:, :max_samples]

    stft_cfg = cfg.get("stft_model", {})
    hop_length = int(stft_cfg.get("hop_length", 160))
    win_length = int(stft_cfg.get("win_length", 320))
    mic_positions = _resolve_mic_positions(cfg, args.mic_positions_json, channels=x_mics.shape[0])

    target_gate, noise_gate = _estimate_gates(
        x_mics,
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        mic_positions=mic_positions,
        ref_ch=ref_ch,
    )

    on_target = target_gate > 0.5
    off_target = ~on_target

    noise_on_target = float(np.mean(noise_gate[on_target])) if np.any(on_target) else 0.0
    noise_off_target = float(np.mean(noise_gate[off_target])) if np.any(off_target) else 0.0

    print(f"input_source={source_msg}")
    print(f"frames={target_gate.shape[0]}")
    print(f"target_gate_mean={float(np.mean(target_gate)):.6f}")
    print(f"noise_gate_mean={float(np.mean(noise_gate)):.6f}")
    print(f"noise_gate_on_target_mean={noise_on_target:.6f}")
    print(f"noise_gate_off_target_mean={noise_off_target:.6f}")
    if not np.any(on_target):
        print("WARNING: no target frames detected (target_gate > 0.5).")
    if noise_on_target > 0.01:
        print(
            "WARNING: noise_gate_on_target_mean > 0.01; R_nn may be contaminated and MVDR can be unstable."
        )

    head = max(0, int(args.print_head))
    print(f"first_{head}_frames=(target_gate, noise_gate)")
    for idx in range(min(head, target_gate.shape[0])):
        print(f"  t={idx}: ({target_gate[idx]:.3f}, {noise_gate[idx]:.3f})")

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    y1 = x_mics[ref_ch]
    y0 = mcwf_make_y0(
        x_mics,
        stft_params=stft_cfg,
        ref_ch=ref_ch,
        sample_rate=sample_rate,
        mic_positions=mic_positions,
    )

    np.savez(
        export_dir / "gates.npz",
        target_gate=target_gate.astype(np.float32),
        noise_gate=noise_gate.astype(np.float32),
        sample_rate=np.int32(sample_rate),
        ref_ch=np.int32(ref_ch),
        hop_length=np.int32(hop_length),
        win_length=np.int32(win_length),
        mic_positions=mic_positions.astype(np.float32),
    )
    sf.write(str(export_dir / "y0.wav"), y0.astype(np.float32), sample_rate)
    sf.write(str(export_dir / "y1.wav"), y1.astype(np.float32), sample_rate)
    print(f"Exported diagnostics to {export_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for smoke checks, training/eval wrappers, and gate diagnostics."
    )
    subparsers = parser.add_subparsers(dest="command")

    for cmd, help_text in (
        ("stft", "Run STFT/iSTFT smoke test (underlying script: scripts/smoke_stft.py)."),
        ("mvdr", "Run MVDR smoke test (underlying script: scripts/smoke_mvdr.py)."),
        ("train", "Run full training wrapper (underlying script: scripts/train.py)."),
        ("test", "Run evaluation wrapper (underlying script: scripts/test.py)."),
        (
            "report",
            "Generate paper-like report (underlying script: scripts/report_paper_like_full.py).",
        ),
        (
            "stream-mvdr",
            "Run online MVDR streamer over a 4ch wav (underlying script: scripts/stream_mvdr.py).",
        ),
        (
            "stream-tncov",
            "Run online trace-normalized covariance streamer over a 4ch wav (underlying script: scripts/stream_tncov.py).",
        ),
    ):
        sub = subparsers.add_parser(cmd, help=help_text, description=help_text)
        sub.add_argument(
            "extra_args",
            nargs=argparse.REMAINDER,
            help="Extra args forwarded to underlying script.",
        )

    diag = subparsers.add_parser(
        "diag-gates",
        help="Estimate target/noise gates on 4ch wav (or dummy data), print stats, and export gates + y0/y1.",
        description="Run gate diagnostics for MVDR front-end stability checks.",
    )
    diag.add_argument(
        "--wav",
        type=str,
        default=None,
        help="Path to input multi-channel wav. If omitted, use dummy data.",
    )
    diag.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional TOML config path resolved via resolve_config().",
    )
    diag.add_argument(
        "--ref-ch", type=int, default=None, help="Override reference microphone channel index."
    )
    diag.add_argument(
        "--sample-rate", type=int, default=None, help="Override sample rate used for diagnostics."
    )
    diag.add_argument(
        "--mic-positions-json",
        type=str,
        default=None,
        help='Override mic positions as JSON string, e.g. "[[0,0,0],[0.04,0,0],[0.01,0.035,0],[-0.03,0.01,0]]".',
    )
    diag.add_argument(
        "--export-dir",
        type=str,
        default="artifacts/gate_diag",
        help="Output directory for gates.npz, y0.wav, y1.wav.",
    )
    diag.add_argument(
        "--seconds", type=float, default=5.0, help="Only process the first N seconds."
    )
    diag.add_argument(
        "--print-head",
        type=int,
        default=10,
        help="Print first N frames of (target_gate, noise_gate).",
    )

    prep = subparsers.add_parser(
        "prep-oppo-y0",
        help="Offline supervised MVDR/LCMV preprocessing for Oppo triplet dataset.",
        description="Generate y0 (and y1_ref0/debug) from clean/noise/noisy 4ch triplets.",
    )
    prep.add_argument("--dataset-root", type=str, required=True)
    prep.add_argument("--split", type=str, required=True, choices=["train", "valid", "test"])
    prep.add_argument("--out-root", type=str, required=True)
    prep.add_argument("--binsize", type=int, default=1)
    prep.add_argument("--delta", type=float, default=1e-2)
    prep.add_argument("--num-workers", type=int, default=0)

    return parser


def _print_short_help(parser: argparse.ArgumentParser) -> None:
    parser.print_help()
    print("\nCommon examples:")
    print(
        "  python scripts/run.py diag-gates --wav path/to/4ch.wav --config configs/paper_like_4mic.toml"
    )
    print(
        "  python scripts/run.py prep-oppo-y0 --dataset-root /path/to/oppo --split train --out-root artifacts/oppo_y0"
    )
    print("  python scripts/run.py train -- --config configs/paper_like_4mic.toml --num_steps 2000")
    print("  python scripts/run.py test -- --run_dir artifacts/runs/<run_id>")
    print(
        "  python scripts/run.py stream-mvdr -- --wav4ch path/to/4ch.wav --rtf-lib artifacts/oppo_y0/rtf_lib_oppo_binsize1.npz --doa 0"
    )
    print(
        "  python scripts/run.py stream-mvdr -- --wav4ch path/to/4ch.wav --rtf-lib artifacts/oppo_y0/rtf_lib_oppo_binsize1.npz --doa 0 --mode lcmv --lcmv-span-deg 30 --lcmv-k 3"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        _print_short_help(parser)
        return

    if args.command in SCRIPT_BY_COMMAND:
        _run_passthrough(args.command, args.extra_args)
        return

    if args.command == "diag-gates":
        try:
            _diag_gates(args)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        return

    if args.command == "prep-oppo-y0":
        from scripts.prep_oppo_supervised_y0 import run_prep

        run_prep(
            dataset_root=args.dataset_root,
            split=args.split,
            out_root=args.out_root,
            binsize=args.binsize,
            delta=args.delta,
            num_workers=args.num_workers,
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
