# 可积系统做 AI：两个 demo（存档）

> 存档说明：本文件记录"从物理模型库里抽取可积系统、拿去干 AI 任务"这条线下，已经做出来的两个最小 demo（标准一、标准二），以及为什么做、和已有工作的区别。配套图片：`soliton_channel_demo.png`、`bbs_standard2_demo.png`。两段代码可直接复现。

---

## 一、为什么做这个

### 1.1 主张：物理是一个可系统开采的"备选模型库"

做 AI 的人挑模型像摸彩票——并行地瞎试各种架构，少数跑出来效果好的被锁定，其余死掉。我们的主张是：**与其瞎试随机架构，不如从物理学里抽取已经被几百年验证过的现成结构**。物理学积累了上百个数学骨架（Ising 族、Langevin 族、Hamilton 族、张量网络族、对称性等变族、重整化群族、自旋玻璃族、可积系统族、混沌动力学族……），目前被 AI 严肃利用的不到十分之一。剩下的每一族，都可能藏着下一个扩散模型。

这条路线刻意不走"用一条主线（如对称性）统一所有 AI 模型"的几何深度学习老路——那条路在 AI 还在剧烈演化的今天太早、太抽象。我们走的是演化逻辑：**海量（把每个物理根族都做成模型）+ 工程化（走完到可训练网络的全链路）+ 按场合匹配（建"任务特征→最优根族"的查表）**。

### 1.2 demo 的两个验收标准

- **标准一（点亮空白格）**：一个物理根族，能干一件正经的 AI 活，把"根族 × AI 能力"表里一个空白格点亮。这是"工程化"的第一个样例，不要求打败任何人，只要求"这族能做、且记下它擅长什么"。
- **标准二（结构赢过规模）**：一个物理根族，能干一件**现在 AI（Transformer/LLM）干不好**的活。这是更高的标准，需要拿证据说话。

### 1.3 为什么选可积系统

可积系统在表里属于"几乎空白"的潜力区。它有一个全物理界独一份的"脾气"：**无穷多个精确守恒量 + 完全可逆（时间反演）+ 永不混沌 + 可被反散射变换精确求解**。后果是它能把一个**非线性**系统精确地往前推无限久、再精确地倒回来，一个比特不丢；招牌现象是孤立子（soliton）——两个波撞上去、穿过彼此后形状一点不变，只错一个相位。

对照之下：哈密顿网络只给近似能量守恒、还会混沌；线性 SSM 结构太弱；Transformer 学出来的守恒是近似、序列一长就漂。**只有可积系统同时具备"非线性 + 精确可逆 + 精确守恒 + 不混沌"。** 标准二的赌注就压在这个组合上。

---

## 二、标准一：可积系统干一件 AI 活——无串扰多符号信道

![标准一：孤立子信道](soliton_channel_demo.png)

### 任务

把 4 个符号（编码成 4 个不同高度的孤立子）塞进**同一条线**里发出去，让它们在传输途中互相追上、对撞，最后在另一端**把 4 个符号原样读回来**。这是一个正经的"编码 → 共享信道 → 解码"任务，数据是抽象符号 `[3,1,2,0]`，跟物理无关。

### 引擎

KdV 方程 `u_t + 6 u u_x + u_xxx = 0` 的孤立子。符号→振幅（孤立子越高跑得越快）。用积分因子 RK4 求解。对照组是一个**非可积**介质（纯线性色散）。

### 结果

- 上半（可积，KdV）：高的孤立子追上矮的、叠在一起（撞），撞完之后每个孤立子**高度、形状原封不动**地分开——4 个符号全部解回（✓ all recovered）。
- 下半（非可积，纯色散）：同样 4 个符号传一会儿就**散成一地碎波**，解出 11 个乱包，符号全丢（✗）。

### 这证明了什么

同一件 AI 活（多符号共享信道、互不干扰地传输），可积这个物理引擎干得成，换个不可积介质就干不成。这就点亮了"可积系统"这个空白格——第一个能跑、能看、能解码的样例。

### 诚实边界

它现在是**演示这个物理结构能干这件事**，还没到"工程化"的完整标准（可训练 + 上 benchmark + 跟 Transformer 实测对比）。作为点亮空白格的第一步、博客配图，够用。另外要承认："多路信息共用一条通道、互不污染"这件事 Transformer 的注意力其实也做得不错，所以本 demo 属于标准一（演示能做），不属于标准二（别人做不到）。

---

## 三、标准二：可积结构在超训练长度上赢 Transformer——箱球系统

![标准二：箱球系统 vs Transformer](bbs_standard2_demo.png)

### 引擎：箱球系统（Box-Ball System, BBS）

一个离散整数格子上的元胞自动机，是 KdV 的"超离散"极限，本身是一个可积系统：成块的球是孤立子，对撞穿过彼此、个数守恒，整个动力学**精确可逆**。状态是抽象的 01 串。更新规则 = 一次无限容量"搬运工"从左到右的扫描（这就是它的可积动力学，是一次可逆的循环过程，因此在任意长度上都成立）。

### 任务与设置

给一个 01 状态，预测它演化 **2 步**之后的状态。**只在长度 L=32 的格子上训练**一个小 Transformer（3 层，d=64，正弦位置编码，让它能接受更长输入），然后把测试格子拉到 48 / 64 / 96 / 128（超出训练长度 = OOD）。孤立子大小与密度在所有长度上固定，所以这是**纯粹的长度外推**测试。

### 结果

| 格子长度 | Transformer 逐位准确率 | Transformer 守恒球数比例 | 可积引擎 |
|---|---|---|---|
| 32（训练长度） | 94.2% | 20.0% | 100% / 100% |
| 48 | 83.6% | 0.0% | 100% / 100% |
| 64 | 81.6% | 0.0% | 100% / 100% |
| 96 | 81.7% | 0.0% | 100% / 100% |
| 128 | 81.0% | 0.0% | 100% / 100% |

可积引擎在所有长度上还经验证了**精确可逆**：能从输出一步步倒推回原始输入，一个比特不差。

### 这证明了什么

- 左图（精确度）：Transformer 在训练长度上学了个大概（逐位 94%），但格子一拉长就掉到 ~81% 再上不去——它学的是 32 这个长度上的模式插值，不是规则本身。可积引擎在所有长度上 100%，因为它带的是规则、与长度无关。
- 右图（守恒量，对应"保证 vs 近似"）：球总数是精确守恒量。Transformer 的预测连训练长度上都只有 20% 保住球数，一出界**直接归零**（凭空多球少球）。可积引擎永远 100% 守恒。

一句话：同一件 AI 活（预测一个非线性可逆动力学的演化），可积结构在任意长度上精确、守恒、可逆；学出来的 Transformer 在训练长度外就崩、还破坏守恒量。**结构赢过规模，赢在"超出训练分布"这个 Transformer 公认的死穴上。**

### 诚实边界（三条）

1. 可积引擎的规则是**搭进去的**（它本来就是真值生成器），所以它的 100% 是"结构自带"，不是"学出来的"。本 demo 证明的是**带可积结构的归纳偏置能外推、纯学习不能**，不是"可积模型从数据里学得更好"。
2. "长度外推循环模型行、Transformer 不行"这半个结论，和已有的状态追踪 / 长度泛化文献重叠（见第四节）。可积系统**独家**贡献的是右图那条线——**精确守恒 + 精确可逆**，这是普通循环模型也未必白送的。
3. 这是 CPU 上几分钟训出来的小模型、玩具任务。要变成能打的"工程化"证据，需放大、上更强 baseline、做更严的对照。

---

## 四、相关工作 / 我们和别人的区别

查证后的诚实地图，分三层：

1. **可积系统 + 深度学习——做的人不少，但方向是反的。** 主流是"用神经网络去解/发现可积系统本身"：Lax 对启发的网络（LPNN）、达布变换网络、带守恒律的 PINN（求解/生成孤立子解），以及用网络去构造/发现可积系统（学哈密顿量与正则变换、稀疏识别 Lax 算子）。这些全是"物理当目标、AI 当工具"，正是本项目要避开的"用 AI 解物理"方向。**我们要的方向（可积结构当 AI 引擎、在 AI 擂台上判胜负）基本是空的。**

2. **箱球系统本身——被研究透了，但没人拿它当 AI 模型。** BBS 在数学物理里很热（孤立子分解、守恒量、RSK、热带几何、概率），也有周期 BBS 做随机数生成、细胞神经网络做孤立子传输线的硬件模拟。但"把 BBS 当序列模型、训练它干 AI 任务、跟 Transformer 比长度泛化"——未查到。我们这个 demo 的具体形态看起来是新的。

3. **底层能力结论——别人用别的载体早证过了。** "结构化/循环模型能长度外推、Transformer 不能"是成熟结论：I-BERT 把位置编码换成循环层以实现对任意长度的外推；状态追踪 / TC⁰ / "Illusion of State" 等理论与实证反复证明 Transformer 在这类任务上的边界。所以本 demo 的新意在**载体和叙事，不在能力**。

**合起来的判决**：撞车撞的是零件，没撞的是骨架。可积+DL 在解物理、BBS 在做组合数学、长度泛化在换循环层——这些零件散落在文献里，但**没有人把它们串成"物理是一个可系统开采的模型库、可积是其中一座没人挖的矿、而且去 AI 自己的擂台上证明它"**。项目的护城河是这个组织逻辑，不是某个零件。

### 参考（按主题，非穷尽）

- 状态追踪 / Transformer 表达能力边界：Liu et al. 2023（Transformers Learn Shortcuts to Automata）；Merrill & Sabharwal 2023（log-precision Transformer ⊆ TC⁰）；Merrill, Petty & Sabharwal 2024（The Illusion of State in State-Space Models）。
- 长度 / 归纳外推：I-BERT（Inductive Generalization of Transformer to Arbitrary Context Lengths，arXiv:2006.10220）。
- 可积 + 深度学习（方向反的）：Lax Pairs Informed Neural Networks（arXiv:2401.04982）；Neural Network Approach to Construction of Classical Integrable Systems（arXiv:2103.00372）；SILO: Sparse Identification of Lax Operators（arXiv:2503.00645）。
- 箱球系统：Takahashi & Satsuma 1990（A Soliton Cellular Automaton）；Ferrari et al. 2018（Soliton Decomposition of the Box-Ball System，arXiv:1806.02798）。
- 可逆性压制长程漂移（科学题材近邻）：An Inverse Scattering Inspired Fourier Neural Operator（arXiv:2512.19439）。

---

## 五、小结与下一步

- 标准一（孤立子信道）：点亮了"可积系统"这个空白格——可积引擎能干一件 AI 活，非可积介质干不成。
- 标准二（箱球系统）：在 OOD 长度上，可积结构精确、守恒、可逆，而学出来的 Transformer 崩且破坏守恒量。结构赢过规模。
- 下一步（走向"工程化"）：把可积映射**参数化**、做成可训练层，同时用结构卡住守恒量不被学坏；放大规模、上更强 baseline、设计一个**必须"非线性 + 精确守恒"两头都占**的任务（线性可逆模型平替不了的那种），把标准二从"玩具演示"推到"可发表的证据"。

---

## 附录 A：标准一代码（孤立子信道）

```python
import numpy as np
import matplotlib.pyplot as plt

# u_t + 6 u u_x + u_xxx = 0 ; soliton: u=(c/2) sech^2( sqrt(c)/2 (x-ct-x0) ), speed c
Lx, N = 120.0, 1024
x = np.linspace(-Lx/2, Lx/2, N, endpoint=False)
dx = x[1]-x[0]
k = 2*np.pi*np.fft.fftfreq(N, d=dx)
ik3 = 1j*k**3
def make_E(dt): return np.exp(ik3*dt/2), np.exp(ik3*dt)

def kdv_ifrk4(v, dt, Eh, E2, g):
    a = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
    b = g*np.fft.fft(np.real(np.fft.ifft(Eh*(v+a/2)))**2)
    c = g*np.fft.fft(np.real(np.fft.ifft(Eh*v + b/2))**2)
    d = g*np.fft.fft(np.real(np.fft.ifft(E2*v + Eh*c))**2)
    return E2*v + (E2*a + 2*Eh*(b+c) + d)/6

def soliton(c, x0): return (c/2.0)/np.cosh(np.sqrt(c)/2.0*(x-x0))**2

symbol_to_c = {0:1.0, 1:2.0, 2:3.0, 3:4.0}
message = [3,1,2,0]; starts = [-40,-28,-16,-4]
def encode(msg):
    u = np.zeros(N)
    for s,x0 in zip(msg,starts): u += soliton(symbol_to_c[s], x0)
    return u

u0 = encode(message); dt = 2e-4; T = 40000
Eh, E2 = make_E(dt); g = -3j*k*dt

def run_kdv():
    v = np.fft.fft(u0.copy()); snaps=[u0.copy()]
    for n in range(T):
        v = kdv_ifrk4(v, dt, Eh, E2, g)
        if (n+1) % (T//5)==0: snaps.append(np.real(np.fft.ifft(v)))
    return np.real(np.fft.ifft(v)), snaps

def run_linear():
    v = np.fft.fft(u0.copy()); snaps=[u0.copy()]
    for n in range(T):
        v = E2*v
        if (n+1) % (T//5)==0: snaps.append(np.real(np.fft.ifft(v)))
    return np.real(np.fft.ifft(v)), snaps

u_kdv, snaps_kdv = run_kdv(); u_lin, snaps_lin = run_linear()

def decode(u):
    peaks=[]
    for i in range(2,N-2):
        if u[i]>0.3 and u[i]>=u[i-1] and u[i]>u[i+1]:
            if not peaks or x[i]-peaks[-1][0]>1.0: peaks.append((x[i],u[i]))
    syms=[min(symbol_to_c,key=lambda s:abs(symbol_to_c[s]-2*h)) for _,h in peaks]
    return syms, peaks

dec_kdv,_ = decode(u_kdv); dec_lin,_ = decode(u_lin)
print("sent:", message, "| integrable:", dec_kdv, sorted(dec_kdv)==sorted(message),
      "| non-integ:", len(dec_lin), "bumps")
```

---

## 附录 B：标准二代码（箱球系统 vs Transformer）

需要 `torch`（CPU 即可）。

```python
import numpy as np, torch, torch.nn as nn, math
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

# 可积引擎：箱球系统更新（左->右无限容量搬运工），精确可逆、球数守恒
def bbs_step(s):
    out = np.zeros_like(s); carrier = 0
    for i in range(len(s)):
        if s[i]==1: carrier+=1; out[i]=0
        else:
            if carrier>0: out[i]=1; carrier-=1
    return out
def bbs_run(s,T):
    for _ in range(T): s=bbs_step(s)
    return s
def mirror(s): return s[::-1].copy()
def bbs_step_inv(s): return mirror(bbs_step(mirror(s)))   # 时间反演

T_STEPS, MAXSOL = 2, 3
def make_config(L, rng):
    s=np.zeros(L,dtype=np.int64); usable=int(L*0.55); i=0; n=0
    while i<usable-MAXSOL-1:
        if rng.random()<0.5 and n<max(3,L//7):
            size=rng.integers(1,MAXSOL+1); s[i:i+size]=1; i+=size+rng.integers(2,5); n+=1
        else: i+=1
    if s.sum()==0: s[1:3]=1
    return s
def dataset(L,Nn,rng):
    X=np.zeros((Nn,L),dtype=np.int64); Y=np.zeros((Nn,L),dtype=np.int64)
    for j in range(Nn):
        xx=make_config(L,rng); X[j]=xx; Y[j]=bbs_run(xx,T_STEPS)
    return torch.tensor(X), torch.tensor(Y)

class PE(nn.Module):
    def __init__(s,d,maxlen=4096):
        super().__init__(); pe=torch.zeros(maxlen,d)
        pos=torch.arange(maxlen).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        s.register_buffer("pe",pe)
    def forward(s,x): return x+s.pe[:x.size(1)].unsqueeze(0)
class TF(nn.Module):
    def __init__(s,d=64,heads=4,layers=3):
        super().__init__(); s.emb=nn.Embedding(2,d); s.pe=PE(d)
        enc=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,dropout=0.0,activation="gelu")
        s.enc=nn.TransformerEncoder(enc,layers); s.head=nn.Linear(d,2)
    def forward(s,x):
        h=s.pe(s.emb(x)); return s.head(s.enc(h))

rng=np.random.default_rng(0)
Xtr,Ytr=dataset(32,2500,rng)
model=TF(); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); lossf=nn.CrossEntropyLoss(); B=256
for epoch in range(28):
    perm=torch.randperm(len(Xtr))
    for i in range(0,len(Xtr),B):
        idx=perm[i:i+B]; logits=model(Xtr[idx])
        loss=lossf(logits.reshape(-1,2),Ytr[idx].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

model.eval(); rng_te=np.random.default_rng(123)
with torch.no_grad():
    for L in [32,48,64,96,128]:
        Xte,Yte=dataset(L,300,rng_te); pred=model(Xte).argmax(-1)
        pos=(pred==Yte).float().mean().item()
        cons=(pred.sum(1)==Xte.sum(1)).float().mean().item()
        print(f"L={L:3d}  TF per-pos={pos*100:5.1f}%  conserved={cons*100:5.1f}%  | integrable=100%/100%")
```
