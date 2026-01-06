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

## Streaming

提供 `GSENetStreamer` 以 chunk 形式运行最小版 GSENet。streamer 采用固定的 `algorithmic_delay`（默认 `MODEL_STFT.win_length`）处理 STFT/OLA 带来的算法延迟，因此离线与流式输出在去掉前 `delay` 样本后应一致。可运行脚本验证：

```bash
python scripts/smoke_streaming.py
```

该脚本需要安装 torch；若未安装则会自动跳过。

## 合成数据管线（dummy batch）

运行脚本生成样例数据：

```bash
python scripts/make_dummy_batch.py
```

输出 `artifacts/dummy_batch.npz`，包含字段：`y0`、`y1`、`yt`、`meta`（JSON 字符串，记录采样到的增益与参数）。

合成方式对齐 GSENet 论文 Table 1 / Section 2.1：`y0` 与 `y1` 分别是两个麦克风的混合信号，`yt` 是用 anechoic RIR 的主径（最大 tap）构造的去混响目标；其中 `gn/gi/alpha/beta/pi` 的采样分布与论文一致，并先在 dB 域采样后再转为幅度比例。

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

输出 `artifacts/paper_batch.npz`，其中的 `s/n/i` 当前为可复现的占位合成信号（正弦混合 + 包络），后续可替换为真实语料。模型前端 STFT 参数与训练 loss 参数继续沿用：

- `MODEL_STFT`: `n_fft=320, win_length=320, hop_length=160`（16 kHz）
- `LOSS_STFT`: `n_fft=1024, win_length=1024, hop_length=256`（16 kHz）
