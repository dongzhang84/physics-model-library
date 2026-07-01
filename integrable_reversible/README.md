# 可积 / 可逆系统 · Integrable & Reversible Systems

> Physics Model Library 里 🔴「可积系统」这一格的完整研究项目。
> 目标：一篇能进 `cond-mat.dis-nn` / `nlin.SI` 的 research paper。

## 主张（一句话）

**一个真正训练出来、带可积/可逆结构（归纳偏置）的神经网络，在为该结构适配的一类任务上显著优于 Transformer，并额外白送精确可逆与守恒——它靠学习逼近硬编可积的理想上界。**

完整动机、技术方案、benchmark 策略、相关工作见 **[proposal.md](proposal.md)**。

## 现状

- ✅ **前置实验（已完成）**：两个最小 demo，见 [`demos/`](demos/)。
  - `soliton_channel/` — 孤立子无串扰信道（标准一，论文的动机/铺垫）。
  - `box_ball_system/` — 箱球系统 vs Transformer 长度外推（标准二，路线 A 理想上界的预览）。
- 🚧 **主研究（进行中）**：补上缺的一环——一个可学习 + 结构可逆的网络（路线 C）。见下方结构。

## 结构

| 目录 | 内容 | 对应 proposal |
|---|---|---|
| [`proposal.md`](proposal.md) | 研究总纲 | 全篇 |
| [`demos/`](demos/) | 已完成的前置实验（存档） | §1 现状 |
| `data/` | 三类 benchmark 的数据生成器 | §4 |
| `models/` | 那张核心图上的三条线（见下） | §2 |
| `eval/` | 核心图 + 四个诊断指标 | §3 |
| `experiments/` | 训练脚本 + configs（把 data×model×eval 串起来复现结果） | §2 执行步骤 |
| `paper/` | 论文草稿 + figure | — |

**`models/` 三条线**：
- `integrable_exact/` — 路线 A：硬编可积（BBS），理想上界（天花板）。
- `reversible_net/` — 路线 C：可逆 coupling 网络 + 可学碰撞算子 F，**本文主角**。
- `transformer/` — baseline：下界（超训练长度即崩、破坏守恒）。

## 复现前置实验

```bash
# demo 1 — 孤立子信道
cd demos/soliton_channel && pip install numpy matplotlib && python soliton_channel.py

# demo 2 — 箱球系统 vs Transformer（CPU 即可）
cd demos/box_ball_system && pip install torch numpy matplotlib && python box_ball_system.py
```
