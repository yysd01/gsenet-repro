# gsenet-repro

## 复现路线图

- data synthesis
- GSENet
- streaming
- beamformer interface
- evaluation

当前 PR 仅完成复现骨架与自动化检查的搭建。

## 合成数据管线（dummy batch）

运行脚本生成样例数据：

```bash
python scripts/make_dummy_batch.py
```

输出 `artifacts/dummy_batch.npz`，包含字段：`y0`、`y1`、`yt`、`meta`（JSON 字符串，记录采样到的增益与参数）。

合成方式对齐 GSENet 论文 Table 1 / Section 2.1：`y0` 与 `y1` 分别是两个麦克风的混合信号，`yt` 是用 anechoic RIR 的主径（最大 tap）构造的去混响目标；其中 `gn/gi/alpha/beta/pi` 的采样分布与论文一致，并先在 dB 域采样后再转为幅度比例。
