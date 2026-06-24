# Catalog 矿脉图：物理模型库清单

这张表是本项目的核心产物——把物理学当成一个可开采的模型库，按"根族"分门别类，标注每族天生擅长的 AI 能力、当前成熟度。**这不是穷尽清单，是一张活地图**，欢迎补充、纠正、认领某一格去开采。

成熟度图例：🟢 已成熟 ｜ 🟡 进展中 ｜ 🔴 几乎空白（潜力区）

| 物理根族 | 物理机制 | 对应的 AI 能力 | 成熟度 | 状态 / 代表方向 |
|---|---|---|---|---|
| Ising / 自旋玻璃 | 能量地形、弛豫到基态 | 联想记忆、去噪、能量模型、生成 | 🟢 | Hopfield、玻尔兹曼机；2024 诺贝尔物理奖 |
| 非平衡热力学 / 扩散 | 加噪—去噪的扩散过程 | 生成（图像、音频、…） | 🟢 | Diffusion / score-based models |
| 对称性 / 群等变 | 不变量与等变性 | 几何数据、分子、图 | 🟢 | 等变网络、几何深度学习 |
| 哈密顿 / 振子 | 沿时间演化的守恒状态 | 长程序列建模 | 🟡 | coRNN、UnICORNN、LinOSS（→ SSM 一脉） |
| 重整化群 / 粗粒化 | 逐尺度积分掉细节 | 多尺度层级、长期记忆 | 🟡 | RG↔深度学习理论；RGMem 等 |
| 张量网络 / 量子多体 | 矩阵乘积态压缩波函数 | 压缩、分类、生成 | 🟡 | MPS/TT 分类与生成（小众） |
| 混沌 / 水库计算 | 混沌边缘的动力系统 | 序列预测（几乎免训练） | 🟡 | Reservoir Computing / ESN（旧但仍有效） |
| **可积系统** | **无穷守恒量、孤立子、精确可逆** | **长度外推、守恒/可逆序列、无串扰信道** | 🔴 | **本仓库 demo（标准一、标准二）** |
| Lindblad / 开放量子系统 | 耗散 + 噪声的演化 | 待探索（含噪推理？） | 🔴 | 空白 |
| KPZ / 表面生长 | 标度律的随机生长 | 待探索（生成？标度结构？） | 🔴 | 空白 |
| Hubbard / Heisenberg | 强关联多体 | 待探索 | 🔴 | 空白 |

> 红色行（🔴）是本项目最看重的开采目标。本仓库先认领了"可积系统"这一格。

---

## 相关工作 / 我们和别人的区别

诚实地图，分三层：

**1. 可积系统 + 深度学习——做的人不少，但方向是反的。**
主流是"用神经网络去解 / 发现可积系统本身"：Lax 对启发的网络（LPNN）、达布变换网络、带守恒律的 PINN（求解或生成孤立子解），以及用网络去构造 / 发现可积系统（学哈密顿量与正则变换、稀疏识别 Lax 算子）。这些全是"物理当目标、AI 当工具"——正是本项目要避开的"用 AI 解物理"方向。我们要的方向（**可积结构当 AI 引擎、在 AI 擂台上判胜负**）基本是空的。

**2. 箱球系统本身——被研究透了，但没人拿它当 AI 模型。**
BBS 在数学物理里很热（孤立子分解、守恒量、RSK、热带几何、概率），也有周期 BBS 做随机数生成、细胞神经网络做孤立子传输线的硬件模拟。但"把 BBS 当序列模型、训练它干 AI 任务、跟 Transformer 比长度泛化"——未查到。本仓库的 demo 形态看起来是新的。

**3. 底层能力结论——别人用别的载体早证过了（必须诚实）。**
"结构化 / 循环模型能长度外推、Transformer 不能"是成熟结论：I-BERT 把位置编码换成循环层以实现对任意长度的外推；状态追踪 / TC⁰ / "Illusion of State" 等理论与实证反复证明 Transformer 在这类任务上的边界。所以本仓库 demo 的新意在**载体和叙事，不在能力**；可积独家贡献的是"精确守恒 + 精确可逆"那部分。

**合起来：** 撞车撞的是零件，没撞的是骨架。可积+DL 在解物理、BBS 在做组合数学、长度泛化在换循环层——零件散落在文献里，但**没有人把它们串成"物理是一个可系统开采的模型库、可积是其中一座没人挖的矿、且去 AI 自己的擂台上证明它"**。护城河是这个组织逻辑。

### 参考（按主题，非穷尽）

- Transformer 表达能力 / 状态追踪：Liu et al. 2023 (Transformers Learn Shortcuts to Automata)；Merrill & Sabharwal 2023 (log-precision Transformers ⊆ TC⁰)；Merrill, Petty & Sabharwal 2024 (The Illusion of State in State-Space Models)。
- 长度 / 归纳外推：I-BERT, arXiv:2006.10220。
- 可积 + 深度学习（方向反的）：Lax Pairs Informed Neural Networks, arXiv:2401.04982；Neural Network Approach to Construction of Classical Integrable Systems, arXiv:2103.00372；SILO (Sparse Identification of Lax Operators), arXiv:2503.00645。
- 箱球系统：Takahashi & Satsuma 1990 (A Soliton Cellular Automaton)；Ferrari et al. 2018 (Soliton Decomposition of the Box-Ball System), arXiv:1806.02798。
- 可逆性压制长程漂移（科学题材近邻）：An Inverse Scattering Inspired Fourier Neural Operator, arXiv:2512.19439。
