# gsenet-repro

## 复现路线图

- data synthesis
- GSENet
- streaming
- beamformer interface
- evaluation

当前 PR 仅完成复现骨架与自动化检查的搭建。

## 安装依赖

- 基础安装（不含 torch）：

```bash
python -m pip install -r requirements.txt
```

- 可选 torch 支持（离线模型、torch STFT 与相关测试）：

```bash
python -m pip install -r requirements.txt -r requirements-torch.txt
```

未安装 torch 时，torch 相关功能与测试会自动跳过。


## 开发者指南（格式化与提交前检查）

推荐在本地启用 `pre-commit`：

```bash
python -m pip install pre-commit ruff
pre-commit install
pre-commit run --all-files
```

其中会自动执行：

- `ruff check`（基础 lint）
- `ruff format`（统一格式化）
- `tools/sanitize_unicode.py --check`（阻止隐藏/双向 Unicode 字符回归）
- `check-ast`、`end-of-file-fixer`、`trailing-whitespace`

## STFT 参数约定

论文中模型前端与训练损失使用不同的 STFT 参数：

- `MODEL_STFT`: `n_fft=320, win_length=320, hop_length=160`（16 kHz），用于模型前端特征。
- `LOSS_STFT`: `n_fft=1024, win_length=1024, hop_length=256`（16 kHz），用于单尺度 STFT reconstruction loss。

可通过脚本快速验证 STFT/iSTFT roundtrip：

```bash
python scripts/smoke_stft.py
```

## Torch 离线最小版 GSENet

该实现提供基于 torch 的离线 STFT/iSTFT 工具与测试，用于对齐后续的离线模型原型。需要注意的是，STFT 的窗函数会带来算法级延迟，因此严格的 sample-level 因果性比较必须考虑窗口长度（`win_length`）并避开重叠区间。

## Paper-scale GSENet (默认)

默认训练/评测会使用 `GSENetPaperScale`（U-Net 论文规模），输入为 `y1`（reference mic）与 `y0`（MCWF/beamformer 输出）的 complex STFT 拼接（实/虚部共 4 通道），输出为 2 通道 complex STFT 并 iSTFT 还原单通道增强波形。模型前端 STFT 采用 `n_fft=320, win_length=320, hop_length=160`，loss 端 STFT 采用 `n_fft=1024, win_length=1024, hop_length=256`，激活函数为 `leaky-ReLU(0.3)`，所有卷积均为因果卷积（时间维仅左侧 padding）。

如需切回最小模型用于调试，可在配置中设置：

```toml
[model]
name = "minimal"
```

## Streaming

提供 `GSENetStreamer` 以 chunk 形式运行最小版 GSENet。streamer 采用固定的 `algorithmic_delay`（默认 `MODEL_STFT.win_length`）处理 STFT/OLA 带来的算法延迟，因此离线与流式输出在去掉前 `delay` 样本后应一致。可运行脚本验证：

```bash
python scripts/_legacy/smoke_streaming.py
```

该脚本需要安装 torch；若未安装则会自动跳过。

## MCWFStreamer (frame-wise streaming)

新增 `MCWFStreamer` 实现 4-mic 逐帧流式 MCWF（支持 `C>=2`）。每推进一个 hop 形成新帧，仅使用当前帧与过去 3 帧的因果窗统计功率（`causal_frames=4`），更新 Wiener 增益并立即输出该帧对应的 `hop_length` 样本。该实现显式维护 `algorithmic_delay_samples`（默认 `win_length - hop_length`），用于对齐离线 MCWF 输出。可运行：

```bash
python scripts/_legacy/smoke_mcwf_streamer.py
```

通过去掉前导延迟样本，可与离线 MCWF 输出对齐并允许极小数值误差。该实现为论文 MCWF 的可运行简化版，后续可替换更精确的 beamformer 实现。

## MCWF Implementation

新增 STFT 域的简化多通道 Wiener 滤波器（MCWF）接口，面向论文中的多麦克风配置。该接口接收复数 STFT 输入 `(B, F, T, C)`（`C>=2`，默认 4 路麦克风），使用 4 帧因果滑窗统计每个频点的功率谱，并按信号/噪声功率比估计增益，输出与输入形状一致的频域强度谱：

```python
from gsenet_repro.dsp.mcwf import mcwf

output = mcwf(
    input_stft,
    stft_win_length=320,
    stft_hop_size=160,
    noise_pow=0.1,
    signal_pow=1.0,
)
```

该接口作为后续深度网络集成的基础模块，方便在多麦克风 STFT 特征上进行滤波预处理与质量对比。

## Metrics

评测指标包含：`snr`、`sisnr`、`sisdr` 与可选的 `pesq`。`pesq` 依赖 `pesq` 包，可能需要编译器，故作为可选依赖单独提供：

```bash
python -m pip install -r requirements.txt -r requirements-torch.txt
python -m pip install -r requirements-metrics.txt
```

若未安装 `pesq`，评测会跳过并在输出中写入 `NaN`。提升值（improvement）同时提供相对 `y1` 与 `y0` 的对比，例如 `delta_sisdr_yhat_vs_y0`、`delta_pesq_yhat_vs_y1` 等。

## MCWF + GSENet 模型集成

在 `MinimalGSENet` 中加入 MCWF 预处理层，多麦克风输入会先在 STFT 域估计噪声功率并生成增益，再传入 GSENet 卷积层。示例训练/验证流程如下：

```bash
python scripts/run.py train -- --config configs/paper_like_4mic.toml --num_steps 2000
```

`smoke_train_paper_like.py` 会自动生成四麦克风合成数据（包含 RIR、噪声与干扰源），使用 `LOSS_STFT` 参数进行 STFT reconstruction loss，并在训练结束后输出：

- `initial_loss` / `final_loss` 以确认 loss 持续下降
- `snr_in` / `snr_out` / `snr_improve` 以确认 MCWF + GSENet 对噪声与干扰的抑制效果

可通过 `noise_level` 参数控制 MCWF 的增益（噪声越大，增益越低），用于在训练中自适应噪声强度。

## 统一入口（推荐）

为避免 `scripts/` 下入口过多导致使用困惑，推荐统一使用：

```bash
python scripts/run.py <subcommand> ...
```

支持的子命令：

- `stft`：运行 STFT roundtrip 烟雾测试。
- `mvdr`：运行 MVDR 烟雾测试。
- `train`：运行训练入口。
- `test`：运行评测入口。
- `report`：运行报告生成。
- `diag-gates`：运行 gate 诊断并导出 `gates.npz`/`y0.wav`/`y1.wav`。
- `prep-oppo-y0`：对 Oppo 4ch clean/noise/noisy 三元组执行监督式 MVDR/LCMV，离线导出 `y0`。
- `stream-mvdr`：在线 MVDR 流式处理 4ch wav 并输出 `y0.wav`。
- `stream-tncov`：在线 Trace-Normalized Covariance Beamformer 流式处理 4ch wav 并输出 `y0_tncov.wav`。

常用示例：

```bash
python scripts/run.py prep-oppo-y0 --dataset-root /home/yishuoyang/dataset/oppo --split train --out-root artifacts/oppo_y0
python scripts/run.py diag-gates --wav <从 out-root 导出的 noisy wav>
python scripts/run.py train -- --config configs/paper_like_4mic.toml --num_steps 2000
python scripts/run.py test -- --run_dir <...>
python scripts/run.py stream-mvdr -- --wav4ch path/to/4ch.wav --rtf-lib artifacts/oppo_y0/rtf_lib_oppo_binsize1.npz --doa 0
python scripts/run.py stream-tncov -- --wav4ch path/to/4ch.wav --out artifacts/y0_tncov.wav --rtf-lib artifacts/oppo_y0/rtf_lib_oppo_binsize1.npz --doa 0
```

`diag-gates` 会打印并导出关键字段：

- `target_gate_mean`：目标帧占比。
- `noise_gate_mean`：整体噪声门控均值。
- `noise_gate_on_target_mean`：在 `target_gate>0.5` 的帧上 `noise_gate` 的均值。
  - **该值应接近 0**；若偏大，说明目标语音帧被误加到噪声统计，`R_nn` 可能被污染，进而导致 MVDR 不稳定。
- `noise_gate_off_target_mean`：在 `target_gate<=0.5` 的帧上 `noise_gate` 的均值，应明显大于 0。

导出目录默认 `artifacts/gate_diag/`，便于离线排查 gate 行为与听感（`y0` vs `y1`）。

## Legacy / underlying scripts

以下脚本仍保留，作为 `scripts/run.py` 的底层实现，不再作为主要用户入口：

- `scripts/smoke_stft.py`
- `scripts/smoke_mvdr.py`
- `scripts/train.py`
- `scripts/test.py`
- `scripts/report_paper_like_full.py`

## Full training (paper-like)

完整训练/评测/报告入口如下（默认输出在 `artifacts/`，支持 `--config` 指定 TOML 配置）：

```bash
python -m pip install -r requirements.txt -r requirements-torch.txt
python scripts/train.py --config configs/real_dataset_paper_scale.toml --num_steps 2000
python scripts/test.py --run_dir <...>
python scripts/report_paper_like_full.py --run_dir <...>
```

`mcwf_frontend`（保留历史命名）默认已切换为 4 麦频域 MVDR 前端：在 STFT 域估计干扰协方差 `R_nn`、RTF 导向向量 `d(f)`，输出单通道 beamformed `y0`。Gate 使用 VAD + GCC-PHAT 的“前方±60°一致性”判定。

可选数据配置项：`data.mic_positions`（单位米，形状为 `[[x,y,z], ...]`），用于 gate 的 GCC 几何约束；未提供时回退到内置 4 麦示例几何。统一入口 `make_y0_from_frontend(...)` 会使用 `data.sample_rate` 进行时延上限换算，避免固定 16k 带来的门控误判（`mcwf_make_y0` 仅保留兼容层）。

快速查看论文规模模型的参数量与 STFT 配置：

```bash
python scripts/_legacy/print_model_stats.py --config configs/real_dataset_paper_scale.toml
```

## 配置文件（TOML）

训练/评测脚本统一支持 `--config` 读取配置文件（CLI 优先级高于 config，config 高于默认值）。示例：

```bash
python scripts/train.py --config configs/real_dataset_4mic.toml --num_steps 200
```

训练时会在 `run_dir/config_resolved.json` 保存最终生效配置，便于复现。

## 真实数据集读取（4-mic）

提供两种真实数据集读取方式：

- `RealMultichannelDataset`：基于 manifest 的多麦克风数据读取。
- `RealFourMicDirDataset`：基于目录结构的 4-mic + clean 配对读取。

### Real dataset (directory layout)

目录结构示例：

```
dataset_root/
  train/
    clean/  # 可为多通道 wav（默认从 clean_ref_mic_index 取目标）
    mic/    # 4 通道 wav
  valid/
    clean/
    mic/
  test/
    clean/
    mic/
```

同一条样本在 `clean/` 与 `mic/` 下文件名一致（例如 `clean/0001.wav` 对应 `mic/0001.wav`）。

**Filename pairing rule**：默认使用文件名的 canonical key 来配对 clean/mic。规则为：去掉扩展名后，若前缀是 `clean_` 或 `mic_` 则移除；再丢弃最后一个 `_` 之后的尾缀（例如 `clean_1-1_src30-int90-p257-367_doa0_data.wav` 与 `mic_1-1_src30-int90-p257-367_doa0_20251112.wav` 会配对到 key `1-1_src30-int90-p257-367_doa0`）。可在配置文件的 `[pairing]` 中调整 `clean_prefix`/`mic_prefix`/`drop_last_underscore_segment`/`strict_pairing` 等规则。

`ref_mic_index` 用于从 noisy mic 中取 `y1`，`clean_ref_mic_index` 用于从 clean 多通道中取 `yt`（若 `clean_is_multichannel=false` 则兼容单通道 clean）。

快速验收：

```bash
python scripts/_legacy/make_dummy_real_dir_dataset.py
python scripts/train.py --config configs/real_dataset_4mic.toml --num_steps 20 --run_dir artifacts/runs/_demo_real_dir
python scripts/test.py --run_dir artifacts/runs/_demo_real_dir
```

### Manifest dataset (legacy)

提供 `RealMultichannelDataset` 支持 manifest 读取真实 4-mic 数据。可以通过脚本生成 dummy 数据并验证端到端：

```bash
python scripts/_legacy/make_dummy_real_manifest.py
python scripts/train.py --config configs/real_dataset_4mic.toml --num_steps 20
```

可选绘图依赖：

```bash
python -m pip install -r requirements-viz.txt
```

## 合成数据管线（dummy batch）

运行脚本生成样例数据：

```bash
python scripts/_legacy/make_dummy_batch.py
```

输出 `artifacts/dummy_batch.npz`，包含字段：`y0`、`y1`、`yt`、`meta`（JSON 字符串，记录采样到的增益与参数）。

合成方式对齐 GSENet 论文 Table 1 / Section 2.1：`y0/y1` 为多麦混合输入，其中训练默认 `y1` 为参考通道（`ref_mic_index=1`），`yt` 与参考通道语义对齐（anechoic `rir_anechoic[0, ref_mic]` 的主径）；其中 `gn/gi/alpha/beta/pi` 的采样分布与论文一致，并先在 dB 域采样后再转为幅度比例。

## Paper-like synthesis (Section 2.1)

新增 `gsenet_repro/data/paper_synth.py` 用于严格对齐 arXiv:2303.07486v1 第 2.1 节与 Table 1（GSENet 行）的合成定义：

- `y0 = s * r(0,0) + gn * n * r(1,0) + pi * gi * i * r(2,0)`
- `y1 = s * r(0,1) + alpha * gn * n * r(1,1) + beta * pi * gi * i * r(2,1)`
- `yt = s * r_anechoic(0,0)`，anechoic RIR 只保留 strongest path。

其中 `*` 为时域卷积，`s/n/i` 与 `r(k,j)` 会做 RMS 归一化（近似 power normalize）。`pi` 为 Bernoulli 随机变量；`alpha/beta` 用于模拟 beamformer 对噪声/干扰的衰减。两接收器在 `generate_rir_3src_2mic` 中被约束为距离非常近（direct path delay 差 < 5 samples），以避免大的 sample offset。

Table 1 分布（GSENet）：

- `dB(gn) ~ N(-5, 10)`
- `pi ~ Bernoulli(0.4)`
- `dB(gi) ~ N(-3, 3)`
- `dB(alpha) ~ max(N(0, 3), -4)`
- `dB(beta) ~ max(N(4, 6), 4)`

采样在 dB 域进行，最终转回线性幅度比例（`dB(x) = 20*log10(x)`）。当前 RIR 为“轻量 image-method-like”占位实现，包含 direct path、early reflections 与指数衰减尾巴，可在后续替换成更真实的 image method。

生成可复现的小 batch：

```bash
python scripts/make_paper_batch.py
```

输出 `artifacts/paper_batch.npz`，包含 `y0/y1/y2/y3/yt` 与 `noise_level`，其中的 `s/n/i` 当前为可复现的占位合成信号（正弦混合 + 包络），后续可替换为真实语料。模型前端 STFT 参数与训练 loss 参数继续沿用：

- `MODEL_STFT`: `n_fft=320, win_length=320, hop_length=160`（16 kHz）
- `LOSS_STFT`: `n_fft=1024, win_length=1024, hop_length=256`（16 kHz）

## Data Augmentation

合成数据在 `gsenet_repro/data/paper_synth.py` 中扩展了噪声与 RIR 的覆盖范围，确保更贴近真实场景：

- **噪声类型**：白噪声、粉噪声（1/f）、speech-like 噪声以及 babble（多说话人叠加）通过 `generate_noise_mix` 生成，并可组合成背景噪声。
- **背景噪声注入**：`synthesize_y0_y1_yt` / `synthesize_y0_y1_y2_yt` 支持 `background_config`，按随机 SNR 将背景噪声叠加到 `y0/y1(/y2)`，使噪声类型对混合公式产生显著影响。
- **RIR 模拟**：`generate_rir_3src_2mic` / `generate_rir_3src_4mic` 会采样不同的 RT60 与早期反射数量，保证直达路径与多次早期反射，并让尾部衰减符合典型房间特性。

复现扩展数据集流程：

```bash
python scripts/run.py train -- --config configs/paper_like_4mic.toml --num_steps 2000
```

`smoke_train_paper_like.py` 会使用扩展后的合成数据进行联合训练，训练过程中每 5 个 epoch 打印一次 SNR/训练损失/验证损失，并在测试集上输出每个样本的 SNR 提升与音质评分。由于项目保持纯 numpy/scipy 依赖，PESQ/STOI 使用 `gsenet_repro/eval/metrics.py` 中的 proxy 实现，用于回归测试和相对对比。

## Trace-Normalized Covariance Beamformer (streaming)

新增 `TraceNormCovStreamer`（`gsenet_repro/streaming/tncov_streamer.py`）与脚本 `scripts/stream_tncov.py`，沿用 MVDR streamer 的流式 STFT + 单帧 iSTFT + OLA 框架，但将权重替换为 trace-normalized covariance 形式：

```text
w(f) = Φ_v(f)^{-1} Φ_x(f) u_ref / tr( Φ_v(f)^{-1} Φ_x(f) )
Φ_x(f) = Φ_y(f) - Φ_v(f),  u_ref=[1,0,0,0]^T (ref_ch=0)
```

实现细节：

- 不显式求逆，使用 `torch.linalg.solve(Φ_v, Φ_x)`。
- 归一化分母使用 `trace.real`，并用 `eps_trace` 下限截断。
- 分母过小/数值异常时回退到直通参考麦（fallback）。
- `Φ_y/Φ_v` 均为逐帧 EMA 在线更新，`Φ_v` 支持两种门控来源：
  - 提供 `--rtf-lib` + `--doa` 时，基于与 MVDR 一致的 coherence-like score 做 target/noise gate。
  - 未提供导向时，使用能量 VAD logistic 门控（`--vad-db-thresh`, `--vad-smooth`）。
- 稳健性：Hermitian 对称化 + diagonal loading（`diag_load_v`, `diag_load_x`）；可选 `--psd-project` 对 `Φ_x` 做 PSD 投影（会增加特征分解开销，实时场景默认建议关闭）。

与 MVDR/LCMV 的区别：

- MVDR：`w ∝ Φ_v^{-1} d`，保持目标导向向量无失真约束。
- LCMV：在多个线性约束下求最小输出功率。
- TraceNormCov：显式利用 `Φ_x = Φ_y - Φ_v` 的目标统计，并用 trace 归一化整体增益，更偏向统计比值驱动；在目标协方差估计偏差较大时建议提高 `alpha_v`、适度增大 `diag_load_v`。

推荐起始参数（16 kHz, 4ch, n_fft=256）：

- `alpha_y=0.92`, `alpha_v=0.98`
- `diag_load_v=1e-2`, `diag_load_x=1e-3`
- `coh_fmin=200`, `coh_fmax=5000`, `coh_t0=0.15`, `coh_t1=0.35`
- `psd_project=false`（低延迟优先）；若观测到 `Φ_x` 明显不稳定可开启

可运行示例：

```bash
PYTHONPATH="$(pwd)" python scripts/stream_tncov.py \
  --wav4ch /path/to/4ch.wav \
  --out artifacts/y0_tncov.wav \
  --rtf-lib artifacts/rtf_lib_phone_geom_binsize1.npz \
  --doa 31 \
  --alpha-y 0.92 --alpha-v 0.98 \
  --diag-load-v 1e-2 --diag-load-x 1e-3 \
  --coh-fmin 200 --coh-fmax 5000 \
  --coh-t0 0.15 --coh-t1 0.35 \
  --psd-project false
```


## Unified configurable frontend

- 网络主体 `GSENetPaperScale` 未改；改动集中在 `y0` 生成逻辑。
- 新增统一入口 `make_y0_from_frontend(...)`，由 `frontend.type` 选择 `none` / `mvdr` / `trace_norm`。
- `trace_norm` 权重公式采用 `w = solve(Phi_v, Phi_x)[..., ref_ch] / trace.real`，不显式依赖 steering vector。
- gate 与 beamformer 解耦：`frontend.gate_mode` 支持 `sector` / `vad` / `coherence`（仅用于统计量 gate）。
- 旧接口 `mcwf_make_y0` 保留兼容但已 deprecated，并会提示迁移到 `frontend.type`。


## Quick debug run with unified trace_norm frontend

This repository now uses a unified frontend pipeline:

- Datasets provide raw waveforms (`x_mics`, `y1`, `yt`) and **do not generate `y0`**.
- Training/evaluation generates `y0` via `make_y0_from_frontend(...)`.
- `gsenet_paper_scale` consumes `y0 + y1`.
- `minimal` consumes `y0 + y1 + y2` (`y2` is legacy/auxiliary only).

Recommended debug commands:

```bash
# 1) Run tests
python -m pytest -q

# 2) Minimal training smoke (very short; unified trace_norm frontend)
python scripts/train_paper_like_full.py \
  --config configs/paper_like_4mic.toml \
  --num_steps 2 --eval_every 1 --log_every 1 --ckpt_every 2 \
  --batch_size 2 --num_workers 0 \
  --frontend_type trace_norm \
  --run_name debug_trace_norm_smoke

# 3) Minimal evaluation run from that debug checkpoint
python scripts/test.py \
  --run_dir artifacts/runs/debug_trace_norm_smoke \
  --num_batches 1 --batch_size 2 \
  --config configs/paper_like_4mic.toml

# 4) Minimal streaming trace_norm run (requires torch + a 4ch wav)
python scripts/run.py stream-tncov -- \
  --wav path/to/4ch_input.wav \
  --out_wav artifacts/debug_run/stream_tncov.wav \
  --config configs/paper_like_4mic.toml
```
