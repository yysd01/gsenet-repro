from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

try:  # pragma: no cover
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("stream_mvdr requires torch installed") from exc

from gsenet_repro.streaming.mvdr_streamer import MVDRStreamer


def _load_doa_track(path: Path) -> np.ndarray:
    raw = np.loadtxt(str(path), delimiter=",", ndmin=1)
    return np.asarray(raw, dtype=np.int32).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream MVDR processing for 4ch wav input.")
    parser.add_argument("--wav4ch", type=str, required=True)
    parser.add_argument("--rtf-lib", type=str, required=True)
    parser.add_argument("--doa", type=int, default=None)
    parser.add_argument("--doa-track-csv", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=1280)
    parser.add_argument("--out", type=str, default="y0.wav")
    args = parser.parse_args()

    if args.doa is None and args.doa_track_csv is None:
        raise SystemExit("either --doa or --doa-track-csv must be provided")

    wav, sr = sf.read(args.wav4ch, always_2d=True, dtype="float32")
    if wav.shape[1] != 4:
        raise SystemExit(f"expected 4 channels, got {wav.shape[1]}")
    x = torch.from_numpy(wav.T)

    streamer = MVDRStreamer(sample_rate=sr, num_mics=4, center=False)
    streamer.load_rtf_lib(args.rtf_lib)
    if args.doa is not None:
        streamer.set_target_doa(args.doa)

    doa_track = _load_doa_track(Path(args.doa_track_csv)) if args.doa_track_csv else None
    hop = streamer.hop_length
    chunk_size = max(args.chunk_size, hop)

    outputs = []
    frame_idx = 0
    for start in range(0, x.shape[-1], chunk_size):
        chunk = x[:, start : start + chunk_size]
        if chunk.shape[-1] < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[-1]))
        chunk_out_parts = []
        for sub_start in range(0, chunk.shape[-1], hop):
            sub = chunk[:, sub_start : sub_start + hop]
            doa = None
            if doa_track is not None:
                doa = int(doa_track[min(frame_idx, doa_track.shape[0] - 1)])
            y = streamer.process(sub, target_doa=doa)
            chunk_out_parts.append(y)
            frame_idx += 1
        chunk_out = torch.cat(chunk_out_parts, dim=-1)[: min(chunk_size, x.shape[-1] - start)]
        outputs.append(chunk_out)

    y0 = torch.cat(outputs, dim=-1)[: x.shape[-1]].detach().cpu().numpy().astype(np.float32)
    sf.write(args.out, y0, sr)
    print(f"Wrote {args.out} (len={y0.shape[0]}, sr={sr})")


if __name__ == "__main__":
    main()
