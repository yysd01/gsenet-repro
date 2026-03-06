# LEGACY / internal demo
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from gsenet_repro.data.paper_synth import (
    generate_noise_mix,
    generate_rir_3src_4mic,
    sample_paper_params,
    synthesize_y0_y1_y2_y3_yt,
)
from gsenet_repro.eval.metrics import pesq_proxy, snr_db, stoi_proxy
from gsenet_repro.pipeline.mcwf_frontend import mcwf_make_y0


def _generate_dataset(
    rng: np.random.Generator,
    batch: int,
    fs: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y0_list = []
    y1_list = []
    y2_list = []
    yt_list = []
    noise_level_list = []

    for _ in range(batch):
        s = rng.normal(size=length).astype(np.float32)
        n, _ = generate_noise_mix(rng, length, fs)
        i, _ = generate_noise_mix(rng, length, fs, noise_types=("speech", "babble", "pink"))
        rir, rir_anechoic = generate_rir_3src_4mic(rng)
        params = sample_paper_params(rng)
        background_config = {
            "rng": rng,
            "fs": fs,
            "snr_db_range": (-4.0, 10.0),
            "noise_types": ("white", "pink", "speech", "babble"),
        }
        _y0, y1, y2, _y3, yt = synthesize_y0_y1_y2_y3_yt(
            s, n, i, rir, rir_anechoic, params, background_config=background_config, target_mic=1
        )
        y_mics = np.stack([_y0, y1, y2, _y3], axis=0).astype(np.float32)
        y0 = mcwf_make_y0(y_mics, stft_params={"n_fft": 320, "win_length": 320, "hop_length": 160}, ref_ch=1)

        y0_list.append(y0)
        y1_list.append(y1)
        y2_list.append(y2)
        yt_list.append(yt)
        noise_pow = np.mean((y0 - yt) ** 2)
        signal_pow = np.mean(yt**2) + 1e-8
        noise_level_list.append(np.float32(noise_pow / signal_pow))

    return (
        np.stack(y0_list),
        np.stack(y1_list),
        np.stack(y2_list),
        np.stack(yt_list),
        np.stack(noise_level_list),
    )


def main() -> None:
    if importlib.util.find_spec("torch") is None:
        print("torch not installed; skipping smoke_train_paper_like")
        return

    import torch

    from gsenet_repro.losses.stft_loss_torch import stft_magnitude_loss
    from gsenet_repro.models.gsenet_torch import MinimalGSENet

    torch.manual_seed(0)
    np.random.seed(0)

    rng = np.random.default_rng(0)
    fs = 16000
    length = fs
    train = _generate_dataset(rng, batch=6, fs=fs, length=length)
    val = _generate_dataset(rng, batch=2, fs=fs, length=length)
    test = _generate_dataset(rng, batch=2, fs=fs, length=length)

    y0_np, y1_np, y2_np, yt_np, noise_level_np = train
    y0_val, y1_val, y2_val, yt_val, noise_level_val = val
    y0_test, y1_test, y2_test, yt_test, noise_level_test = test
    y0 = torch.tensor(y0_np, dtype=torch.float32)
    y1 = torch.tensor(y1_np, dtype=torch.float32)
    y2 = torch.tensor(y2_np, dtype=torch.float32)
    yt = torch.tensor(yt_np, dtype=torch.float32)
    noise_level = torch.tensor(noise_level_np, dtype=torch.float32)

    model = MinimalGSENet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    stft_params = {"n_fft": 1024, "win_length": 1024, "hop_length": 256}

    y0_val_t = torch.tensor(y0_val, dtype=torch.float32)
    y1_val_t = torch.tensor(y1_val, dtype=torch.float32)
    y2_val_t = torch.tensor(y2_val, dtype=torch.float32)
    yt_val_t = torch.tensor(yt_val, dtype=torch.float32)
    noise_level_val_t = torch.tensor(noise_level_val, dtype=torch.float32)

    log_rows = []
    model.train()
    with torch.no_grad():
        initial_loss = stft_magnitude_loss(
            model(y0, y1, y2, noise_level=noise_level),
            yt,
            stft_params=stft_params,
        ).item()
    print(f"initial_loss={initial_loss:.6f}")

    epochs = 40
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_hat = model(y0, y1, y2, noise_level=noise_level)
        loss = stft_magnitude_loss(y_hat, yt, stft_params=stft_params)
        reg = 0.1 * torch.mean(
            torch.abs(
                torch.stft(
                    y_hat,
                    n_fft=stft_params["n_fft"],
                    hop_length=stft_params["hop_length"],
                    win_length=stft_params["win_length"],
                    window=torch.hann_window(stft_params["win_length"], device=y_hat.device),
                    center=False,
                    return_complex=True,
                )
            )
        )
        total_loss = loss + reg
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_hat = model(y0_val_t, y1_val_t, y2_val_t, noise_level=noise_level_val_t)
                val_loss = stft_magnitude_loss(val_hat, yt_val_t, stft_params=stft_params).item()
                val_hat_np = val_hat.cpu().numpy()
            val_snr = float(np.mean([snr_db(ref, est) for ref, est in zip(yt_val, val_hat_np)]))
            log_rows.append(
                {"epoch": epoch, "train_loss": float(loss.item()), "val_loss": val_loss, "val_snr": val_snr}
            )
            print(
                f"epoch={epoch:03d} train_loss={loss.item():.6f} "
                f"val_loss={val_loss:.6f} val_snr={val_snr:.2f}"
            )
            model.train()

    with torch.no_grad():
        final_loss = stft_magnitude_loss(
            model(y0, y1, y2, noise_level=noise_level),
            yt,
            stft_params=stft_params,
        ).item()
    print(f"final_loss={final_loss:.6f}")

    if final_loss > initial_loss * 0.99:
        raise SystemExit("final loss did not improve by at least 1%")

    with torch.no_grad():
        y_hat = model(y0, y1, y2, noise_level=noise_level).cpu().numpy()
    snr_in = snr_db(yt_np, y0_np)
    snr_out = snr_db(yt_np, y_hat)
    snr_improve = snr_out - snr_in
    print(f"snr_in={snr_in:.2f} snr_out={snr_out:.2f} snr_improve={snr_improve:.2f}")
    if snr_improve < 0.5:
        raise SystemExit("SNR improvement was not significant")

    model.eval()
    with torch.no_grad():
        y_hat_test = model(
            torch.tensor(y0_test, dtype=torch.float32),
            torch.tensor(y1_test, dtype=torch.float32),
            torch.tensor(y2_test, dtype=torch.float32),
            noise_level=torch.tensor(noise_level_test, dtype=torch.float32),
        ).cpu().numpy()

    eval_rows = []
    for idx, (ref, noisy, est) in enumerate(zip(yt_test, y0_test, y_hat_test)):
        snr_before = snr_db(ref, noisy)
        snr_after = snr_db(ref, est)
        eval_rows.append(
            {
                "sample": idx,
                "snr_in": snr_before,
                "snr_out": snr_after,
                "snr_improve": snr_after - snr_before,
                "pesq_proxy": pesq_proxy(ref, est, fs=fs),
                "stoi_proxy": stoi_proxy(ref, est, fs=fs),
            }
        )
        print(
            f"test[{idx}] snr_in={snr_before:.2f} snr_out={snr_after:.2f} "
            f"snr_improve={snr_after - snr_before:.2f} "
            f"pesq_proxy={eval_rows[-1]['pesq_proxy']:.2f} "
            f"stoi_proxy={eval_rows[-1]['stoi_proxy']:.2f}"
        )

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifacts_dir / "gsenet_joint.pt")
    (artifacts_dir / "training_log.json").write_text(
        json.dumps(log_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (artifacts_dir / "eval_results.json").write_text(
        json.dumps(eval_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
