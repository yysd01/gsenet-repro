from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
    train_paper_like_full.main()


if __name__ == "__main__":
    main()
