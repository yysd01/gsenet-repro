import numpy as np

from gsenet_repro.pipeline.mcwf_frontend import gates_from_probs


def test_noise_gate_zero_on_target() -> None:
    p_speech = np.array([1.0, 1.0, 0.2], dtype=np.float32)
    p_tar = np.array([1.0, 0.5, 0.2], dtype=np.float32)
    beta = 0.5
    target_gate, noise_beta = gates_from_probs(
        p_speech,
        p_tar,
        theta_s=0.6,
        theta_t=0.6,
        theta_i=0.4,
        theta_n=0.3,
        beta_speech_interf=beta,
    )

    assert target_gate[0] == 1.0
    assert noise_beta[0] == 0.0

    assert target_gate[1] == 0.0
    assert noise_beta[1] == beta

    assert target_gate[2] == 0.0
    assert noise_beta[2] == 1.0
