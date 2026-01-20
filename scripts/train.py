from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

if importlib.util.find_spec("torch") is None:
    print("torch not installed. Install requirements-torch.txt to run training.", file=sys.stderr)
    raise SystemExit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

_TRAIN_PATH = Path(__file__).resolve().parent / "train_paper_like_full.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location("train_paper_like_full", _TRAIN_PATH)
if _TRAIN_SPEC is None or _TRAIN_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("Unable to load train_paper_like_full module")
train_paper_like_full = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(train_paper_like_full)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified training entrypoint.")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use_mcwf", type=int, default=1)
    args, unknown = parser.parse_known_args()

    argv = [
        "train_paper_like_full.py",
        "--num_steps",
        str(args.num_steps),
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
        "--use_mcwf",
        str(args.use_mcwf),
    ]
    if args.run_dir is not None:
        argv.extend(["--run_dir", args.run_dir])
    argv.extend(unknown)

    sys.argv = argv
    train_paper_like_full.main()


if __name__ == "__main__":
    main()
