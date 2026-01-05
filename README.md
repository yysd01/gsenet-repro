# gsenet-repro

## 复现路线图

- data synthesis
- GSENet
- streaming
- beamformer interface
- evaluation

当前 PR 仅完成复现骨架与自动化检查的搭建。

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

## 合成数据管线（dummy batch）

运行脚本生成样例数据：

```bash
python scripts/make_dummy_batch.py
```

输出 `artifacts/dummy_batch.npz`，包含字段：`y0`、`y1`、`yt`、`meta`（JSON 字符串，记录采样到的增益与参数）。

合成方式对齐 GSENet 论文 Table 1 / Section 2.1：`y0` 与 `y1` 分别是两个麦克风的混合信号，`yt` 是用 anechoic RIR 的主径（最大 tap）构造的去混响目标；其中 `gn/gi/alpha/beta/pi` 的采样分布与论文一致，并先在 dB 域采样后再转为幅度比例。
