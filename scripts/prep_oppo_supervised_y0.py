from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from gsenet_repro.data.oppo_triplet_dataset import OppoTripletDataset
from gsenet_repro.dsp.supervised_bf import (
    beamform_sample,
    build_doa_rtf_library,
    load_rtf_lib,
)

DEFAULT_STFT = {
    "n_fft": 256,
    "win_length": 256,
    "hop_length": 128,
    "window": "hann",
    "center": False,
}


def _percentile_summary(values: list[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=np.float64)
    p = np.percentile(arr, [50, 90, 95, 99, 100])
    return "p50={:.3e}, p90={:.3e}, p95={:.3e}, p99={:.3e}, max={:.3e}".format(*p)


def run_prep(
    dataset_root: str,
    split: str,
    out_root: str,
    binsize: int,
    delta: float,
    num_workers: int,
) -> None:
    del num_workers
    out_root_path = Path(out_root)
    lib_path = out_root_path / f"rtf_lib_oppo_binsize{binsize}.npz"

    if lib_path.exists():
        print(f"Loading existing RTF library: {lib_path}")
        rtf_lib = load_rtf_lib(lib_path)
    else:
        print("Building RTF library from train split single-target clean samples...")
        rtf_lib = build_doa_rtf_library(
            Path(dataset_root) / "train",
            binsize_deg=binsize,
            stft_cfg=DEFAULT_STFT,
            ref_ch=0,
            artifact_dir=out_root_path,
        )
        print(f"Saved RTF library: {rtf_lib['path']}")

    dataset = OppoTripletDataset(dataset_root=dataset_root, split=split)
    single_target = 0
    dual_target = 0
    constraint_errors: list[float] = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        src_doas = list(item["src_doas"])
        int_doas = list(item["int_doas"])
        if len(src_doas) == 1:
            single_target += 1
        elif len(src_doas) == 2:
            dual_target += 1

        y0, y1, debug = beamform_sample(
            clean4=item["clean"],
            noise4=item["noise"],
            noisy4=item["noisy"],
            src_doas=src_doas,
            stft_cfg=DEFAULT_STFT,
            rtf_lib=rtf_lib,
            ref_ch=0,
            delta=delta,
        )
        constraint_errors.append(float(debug["constraint_error"]))

        case = str(item["case"])
        utt_id = str(item["utt_id"])
        y0_dir = out_root_path / split / case / "y0"
        y1_dir = out_root_path / split / case / "y1_ref0"
        dbg_dir = out_root_path / split / case / "debug"
        y0_dir.mkdir(parents=True, exist_ok=True)
        y1_dir.mkdir(parents=True, exist_ok=True)
        dbg_dir.mkdir(parents=True, exist_ok=True)

        sf.write(str(y0_dir / utt_id), y0.astype(np.float32), 16000)
        sf.write(str(y1_dir / utt_id), y1.astype(np.float32), 16000)
        np.savez(
            dbg_dir / f"{Path(utt_id).stem}.npz",
            src_doas=np.asarray(src_doas, dtype=np.int32),
            int_doas=np.asarray(int_doas, dtype=np.int32),
            constraint_error=np.float32(debug["constraint_error"]),
            rms_y0=np.float32(debug["rms_y0"]),
            rms_y1=np.float32(debug["rms_y1"]),
        )

    print("=== Oppo supervised beamforming preprocessing summary ===")
    print(
        f"split={split} samples={len(dataset)} single_target={single_target} dual_target={dual_target}"
    )
    print(f"constraint_error: {_percentile_summary(constraint_errors)}")
    print("output layout:")
    print(f"  {out_root}/{split}/<case>/y0/*.wav")
    print(f"  {out_root}/{split}/<case>/y1_ref0/*.wav")
    print(f"  {out_root}/{split}/<case>/debug/*.npz")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Oppo supervised MVDR/LCMV y0 waveforms offline."
    )
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "valid", "test"])
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--binsize", type=int, default=1)
    parser.add_argument("--delta", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_prep(
        dataset_root=args.dataset_root,
        split=args.split,
        out_root=args.out_root,
        binsize=args.binsize,
        delta=args.delta,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
