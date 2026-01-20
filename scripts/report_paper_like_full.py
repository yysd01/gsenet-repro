from __future__ import annotations

import argparse
import csv
import json
import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _best_eval_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("sisnr_impr", "-inf")))


def _last_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    return rows[-1] if rows else None


def _maybe_make_figures(run_dir: Path, metrics_rows: list[dict[str, str]]) -> list[str]:
    if importlib.util.find_spec("matplotlib") is None:
        return []
    import matplotlib.pyplot as plt

    if not metrics_rows:
        return []

    steps = [int(row["step"]) for row in metrics_rows]
    loss = [float(row["loss"]) for row in metrics_rows]
    snr_impr = [float(row["snr_impr"]) for row in metrics_rows]
    sisnr_impr = [float(row["sisnr_impr"]) for row in metrics_rows]
    sisdr_impr = (
        [float(row["sisdr_impr"]) for row in metrics_rows]
        if "sisdr_impr" in metrics_rows[0]
        else []
    )

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    plt.figure()
    plt.plot(steps, loss)
    plt.xlabel("step")
    plt.ylabel("loss")
    loss_path = figures_dir / "loss_curve.png"
    plt.savefig(loss_path)
    plt.close()
    paths.append(str(loss_path.relative_to(run_dir)))

    plt.figure()
    plt.plot(steps, snr_impr)
    plt.xlabel("step")
    plt.ylabel("snr_impr")
    snr_path = figures_dir / "snr_improvement_curve.png"
    plt.savefig(snr_path)
    plt.close()
    paths.append(str(snr_path.relative_to(run_dir)))

    plt.figure()
    plt.plot(steps, sisnr_impr)
    plt.xlabel("step")
    plt.ylabel("sisnr_impr")
    sisnr_path = figures_dir / "sisnr_improvement_curve.png"
    plt.savefig(sisnr_path)
    plt.close()
    paths.append(str(sisnr_path.relative_to(run_dir)))

    if sisdr_impr:
        plt.figure()
        plt.plot(steps, sisdr_impr)
        plt.xlabel("step")
        plt.ylabel("sisdr_impr")
        sisdr_path = figures_dir / "sisdr_improvement_curve.png"
        plt.savefig(sisdr_path)
        plt.close()
        paths.append(str(sisdr_path.relative_to(run_dir)))

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-like training report.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config_resolved.json"
    metrics_path = run_dir / "metrics.csv"
    eval_path = run_dir / "eval.csv"

    if args.config is not None:
        config = json.loads(Path(args.config).read_text())
    else:
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
    metrics_rows = _read_csv(metrics_path)
    eval_rows = _read_csv(eval_path)

    best_eval = _best_eval_row(eval_rows)
    final_metrics = _last_row(metrics_rows)

    fig_paths = _maybe_make_figures(run_dir, metrics_rows)

    audio_dir = run_dir / "eval_outputs" / "audio"
    audio_paths = sorted(str(path.relative_to(run_dir)) for path in audio_dir.glob("*.wav"))

    report_lines = [
        "# Full training report (paper-like)",
        "",
        "## Configuration",
        "```json",
        json.dumps(config, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Metrics",
    ]

    if best_eval:
        report_lines.extend(
            [
                "",
                "### Best eval batch",
                "",
                "| metric | value |",
                "| --- | --- |",
                f"| step | {best_eval.get('step', '')} |",
                f"| loss | {float(best_eval.get('loss', 0.0)):.6f} |",
                f"| snr_impr | {float(best_eval.get('snr_impr', 0.0)):.4f} |",
                f"| sisnr_impr | {float(best_eval.get('sisnr_impr', 0.0)):.4f} |",
                f"| sisdr_impr | {float(best_eval.get('sisdr_impr', 0.0)):.4f} |",
                f"| pesq_impr | {float(best_eval.get('pesq_impr', 0.0)):.4f} |",
            ]
        )

    if final_metrics:
        report_lines.extend(
            [
                "",
                "### Final training step",
                "",
                "| metric | value |",
                "| --- | --- |",
                f"| step | {final_metrics.get('step', '')} |",
                f"| loss | {float(final_metrics.get('loss', 0.0)):.6f} |",
                f"| snr_impr | {float(final_metrics.get('snr_impr', 0.0)):.4f} |",
                f"| sisnr_impr | {float(final_metrics.get('sisnr_impr', 0.0)):.4f} |",
                f"| sisdr_impr | {float(final_metrics.get('sisdr_impr', 0.0)):.4f} |",
                f"| pesq_impr | {float(final_metrics.get('pesq_impr', 0.0)):.4f} |",
            ]
        )

    report_lines.extend(["", "## Audio samples"])
    if audio_paths:
        for path in audio_paths:
            report_lines.append(f"- `{path}`")
    else:
        report_lines.append("- (No audio samples found. Run eval_paper_like_full.py.)")

    report_lines.extend(["", "## Figures"])
    if fig_paths:
        for path in fig_paths:
            report_lines.append(f"![{Path(path).stem}]({path})")
    else:
        report_lines.append(
            "Matplotlib not available. Install requirements-viz.txt and rerun this script."
        )

    report_path = run_dir / "REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
