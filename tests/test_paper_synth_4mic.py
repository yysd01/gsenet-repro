import numpy as np

from gsenet_repro.data.paper_synth import generate_rir_3src_4mic


def test_rir_4mic_shape_and_direct_path() -> None:
    rng = np.random.default_rng(10)
    rir, _ = generate_rir_3src_4mic(rng, max_direct_delay_diff=4)
    assert rir.shape[0] == 3
    assert rir.shape[1] == 4
    for src_idx in range(rir.shape[0]):
        delays = [int(np.argmax(np.abs(rir[src_idx, mic_idx]))) for mic_idx in range(4)]
        assert max(delays) - min(delays) <= 4
