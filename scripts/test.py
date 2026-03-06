from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

if importlib.util.find_spec("torch") is None:
    print("torch not installed. Install requirements-torch.txt to run evaluation.", file=sys.stderr)
    raise SystemExit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

_EVAL_PATH = Path(__file__).resolve().parent / "_legacy" / "eval_paper_like_full.py"
_EVAL_SPEC = importlib.util.spec_from_file_location("eval_paper_like_full", _EVAL_PATH)
if _EVAL_SPEC is None or _EVAL_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("Unable to load eval_paper_like_full module")
_eval_module = importlib.util.module_from_spec(_EVAL_SPEC)
_EVAL_SPEC.loader.exec_module(_eval_module)


def _resolve_checkpoint(run_dir: Path, ckpt_path: Path | None) -> Path:
    if ckpt_path is not None:
        return ckpt_path
    best = run_dir / "checkpoints" / "best.pt"
    if best.exists():
        return best
    ckpts = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def _write_summary_table(summary: dict[str, float], report_path: Path) -> None:
    lines = [
        "# Evaluation Report",
        "",
        "## Summary Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation entrypoint.")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--write_wavs", action="store_true", help="Write per-sample wav outputs.")
    parser.add_argument("--max_wavs", type=int, default=20, help="Max samples to export.")
    parser.add_argument("--wav_dir", type=str, default=None, help="Directory for exported wavs.")
    parser.add_argument(
        "--wav_norm",
        type=str,
        default="peak",
        choices=("peak", "none"),
        help="Peak normalize audio before writing.",
    )
    parser.add_argument(
        "--save_mic_4ch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write multi-channel mic wav for each sample.",
    )
    args, unknown = parser.parse_known_args()

    if args.run_dir is None and args.ckpt_path is None:
        raise SystemExit("Either --run_dir or --ckpt_path must be provided")

    run_dir = Path(args.run_dir) if args.run_dir is not None else Path(args.ckpt_path).parents[1]
    ckpt_path = _resolve_checkpoint(run_dir, Path(args.ckpt_path) if args.ckpt_path else None)

    artifacts_dir = run_dir / "artifacts"
    out_dir = artifacts_dir / "eval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "_legacy/eval_paper_like_full.py",
        "--ckpt_path",
        str(ckpt_path),
        "--out_dir",
        str(out_dir),
        "--num_batches",
        str(args.num_batches),
        "--batch_size",
        str(args.batch_size),
    ]
    if args.config is not None:
        argv.extend(["--config", args.config])
    if args.write_wavs:
        argv.append("--write_wavs")
    if args.max_wavs is not None:
        argv.extend(["--max_wavs", str(args.max_wavs)])
    if args.wav_dir is not None:
        argv.extend(["--wav_dir", str(args.wav_dir)])
    if args.wav_norm is not None:
        argv.extend(["--wav_norm", args.wav_norm])
    if args.save_mic_4ch:
        argv.append("--save_mic_4ch")
    else:
        argv.append("--no-save-mic-4ch")
    argv.extend(unknown)
    sys.argv = argv
    _eval_module.main()

    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit("summary.json not found after eval")
    summary = json.loads(summary_path.read_text())

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (artifacts_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(summary.keys())
        writer.writerow([summary[key] for key in summary.keys()])

    per_sample_path = out_dir / "per_sample_metrics.csv"
    if per_sample_path.exists():
        (artifacts_dir / "per_sample_metrics.csv").write_text(
            per_sample_path.read_text(), encoding="utf-8"
        )

    report_path = artifacts_dir / "REPORT.md"
    _write_summary_table(summary, report_path)


if __name__ == "__main__":
    main()
