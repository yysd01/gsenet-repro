from __future__ import annotations

import importlib.util
from typing import Dict, Iterator, Optional

import numpy as np

from gsenet_repro.data.paper_synth import (
    generate_noise_mix,
    generate_rir_3src_nmic,
    sample_paper_params,
    synthesize_y_mics_yt,
)
from gsenet_repro.dsp import MODEL_STFT
from gsenet_repro.pipeline.mcwf_frontend import mcwf_make_y0

if importlib.util.find_spec("torch") is not None:  # pragma: no cover
    import torch
    from torch.utils.data import IterableDataset
else:  # pragma: no cover
    torch = None
    IterableDataset = object


class PaperLikeDataset(IterableDataset):
    """On-the-fly paper-like dataset aligned with paper_synth.py (Section 2.1/Table 1)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        segment_seconds: float = 1.0,
        seed: int = 0,
        num_samples: Optional[int] = None,
        use_mcwf: bool = True,
        ref_mic: int = 1,
        num_mics: int = 4,
        stft_params: Optional[Dict[str, int]] = None,
        causal_frames: int = 4,
        snr_db_range: tuple[float, float] = (-4.0, 10.0),
        noise_types: tuple[str, ...] = ("white", "pink", "speech", "babble"),
        global_gain: float = 1.0,
    ) -> None:
        if torch is None:
            raise ImportError(
                "PaperLikeDataset requires torch. Install requirements-torch.txt."
            )
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.length = int(round(self.sample_rate * self.segment_seconds))
        self.seed = int(seed)
        self.num_samples = num_samples
        self.use_mcwf = bool(use_mcwf)  # legacy flag; now selects MVDR frontend
        self.ref_mic = int(ref_mic)
        self.num_mics = int(num_mics)
        if self.num_mics < 2:
            raise ValueError("num_mics must be >= 2")
        if not 0 <= self.ref_mic < self.num_mics:
            raise ValueError("ref_mic must be in range [0, num_mics)")
        self.stft_params = dict(stft_params) if stft_params is not None else dict(MODEL_STFT)
        self.causal_frames = int(causal_frames)
        self.snr_db_range = snr_db_range
        self.noise_types = noise_types
        self.global_gain = float(global_gain)

    def _make_rng(self, worker_id: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + worker_id)

    def _synthesize_sample(self, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        length = self.length
        s = rng.normal(size=length).astype(np.float32)
        n, _ = generate_noise_mix(rng, length, self.sample_rate)
        i, _ = generate_noise_mix(
            rng, length, self.sample_rate, noise_types=("speech", "babble", "pink")
        )
        rir, rir_anechoic = generate_rir_3src_nmic(rng, num_mics=self.num_mics)
        params = sample_paper_params(rng, global_gain=self.global_gain)
        background_config = {
            "rng": rng,
            "fs": self.sample_rate,
            "snr_db_range": self.snr_db_range,
            "noise_types": self.noise_types,
        }
        # Keep synthesis aligned with paper_synth.py (Section 2.1/Table 1).
        y_mics, yt = synthesize_y_mics_yt(
            s,
            n,
            i,
            rir,
            rir_anechoic,
            params,
            num_mics=self.num_mics,
            background_config=background_config,
            target_mic=self.ref_mic,
        )

        x_mics = np.stack(list(y_mics), axis=0).astype(np.float32)
        y1 = x_mics[self.ref_mic]
        if self.use_mcwf:
            y0 = mcwf_make_y0(
                x_mics,
                stft_params=self.stft_params,
                causal_frames=self.causal_frames,
                ref_ch=self.ref_mic,
            )
        else:
            y0 = y1

        y2 = x_mics[min(2, self.num_mics - 1)]
        y3 = x_mics[min(3, self.num_mics - 1)]

        return {
            "y0": torch.tensor(y0, dtype=torch.float32),
            "y1": torch.tensor(y1, dtype=torch.float32),
            "y2": torch.tensor(y2, dtype=torch.float32),
            "y3": torch.tensor(y3, dtype=torch.float32),
            "yt": torch.tensor(yt, dtype=torch.float32),
            "x_mics": torch.tensor(x_mics, dtype=torch.float32),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = self._make_rng(worker_id)
        if self.num_samples is None:
            while True:
                yield self._synthesize_sample(rng)
        else:
            for _ in range(self.num_samples):
                yield self._synthesize_sample(rng)
