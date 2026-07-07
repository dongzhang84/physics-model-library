# Benchmark proposal: family-level integrable / reversible extrapolation

> Status: proposal draft. This file defines what the benchmark should become. The current
> `README.md` records an existing benchmark attempt and why it is not valid yet.

---

## ⛔ 前置探针 — 先跑这个,再决定要不要建下面整套（Claude Code 加，2026-07-07）

**这份 proposal 好,但它把整篇论文押在一个从没跑过的系统 `colored BBS`（§7 Tier 2、§11 第 3 主力）上。这是"多孤子 benchmark 信誓旦旦、跑完 leak 审计才发现是硬编"的原样翻版。在建 Phase A–E 之前，必须先用一个 ~1 天的单种子探针验证核心假设，否则可能搭完整套框架才发现主力是空的。**

**要探的不是裸 colored BBS，是 `colored × finite-carrier BBS` 的组合**——因为"深"和"真学"是**正交**的两件事，各由不同机制提供：

- **颜色 / 多 species** → 让守恒量丰富（带标签的孤子内容）→ 治"单一守恒可被 bolt-on 复制"；
- **finite-carrier（有界容量 K）** → 让 emit 门**非平凡**（carrier-blind ≠ 满分，如 K=6 上 ≈89%）→ 治"硬编 leak"。

裸 colored BBS 仍是一个**固定确定性 CA 规则**，一个写死正确规则的 carrier-blind 很可能照样满分（和 plain BBS 的 `emit=1−cell` 一样硬编）。所以"又深又真学"必须**两者组合**，而这个组合**一次都没验证过**（三重未证：颜色够不够深？有界容量上真学得出带标签内容？free-form 真的会崩？）。

**探针（单种子，~1 天）：** 实现 `colored finite-carrier BBS`（精确 rollout + 带标签孤子内容提取器 + 自检守恒/可逆），跑三方——**结构模型（学）/ carrier-blind（写死规则、不训练）/ free-form + bolt-on**——只问一个问题：

> **关掉学习（carrier-blind），带标签孤子内容的精确率掉不掉？**
> - **掉（blind ≪ 学习模型）** → 又深（颜色）、又真学（有界容量）、bolt-on 补不上 → **核心假设成立，值得按下面搭。**
> - **不掉（blind 也高）** → 又一个硬编死结 → **proposal 主力塌，别建，重想。**

通过这个探针之前，下面所有 tier / spec / Phase 都是**基于未验证假设的设计**。**一天探针，省两周框架。**

---

## 0. One-sentence goal

Build a benchmark that tests whether a **learned** model with integrable / reversible structure can
preserve exact long-horizon behavior under length, time, and collision extrapolation, in a way that
generic sequence models and bolt-on constraints cannot replicate.

The benchmark should not merely show that a hard-coded rule wins. It should show that the right
physical structure makes a model learn the rule **more exactly**, so that the learned transition can
be composed many times without drift.

## 1. Why the current demos are not enough

The existing experiments already establish useful pieces:

- Hard-coded BBS shows the ceiling: exact structure extrapolates in length, conserves, and reverses.
- Finite-carrier BBS removes the plain-BBS leak and shows that a structured carrier can learn a
  non-trivial residual.
- Margolus block CA shows the same recipe transfers to a different reversible backbone: free-form
  models learn one step well but drift under `T proportional to L`; the structural model stays exact.
- Toda is a scoped negative result: on smooth continuous flow, free-form models already solve the
  task well enough that the structural discriminator disappears.

Those are demonstrations, not yet a benchmark. A real benchmark must close four gaps:

1. **No single-rule leak.** A carrier-blind, identity, lookup, or other fixed audit must not solve
   the task.
2. **No hard-coded winner.** The structural entrant must contain a learned component whose removal
   measurably hurts.
3. **No single invariant as the metric.** Ball count alone is too weak because it can be bolted on.
4. **No one-system claim.** The benchmark must contain a family of related systems, not only one
   hand-picked cellular automaton.

## 2. Claim the benchmark should adjudicate

The benchmark is designed around this claim:

> For discrete reversible / integrable dynamics with many exact interactions, structure helps not
> because it adds a single conserved quantity, but because it makes the learned local transition
> exact enough that long compositions preserve the whole interaction content.

The phrase "interaction content" is deliberate. The target is not just total mass. It includes
objects, labels, amplitudes, phases, collision outcomes, reversibility, and any system-specific
conserved signature that distinguishes integrable behavior from ordinary conservation.

## 3. Benchmark name and scope

Working name:

**Integrable Extrapolation Benchmark**

Conservative name, if we want to avoid overclaiming before Toda / KdV succeeds:

**Reversible Interaction Extrapolation Benchmark**

Recommendation for the first public version: use the conservative subtitle:

**Integrable Extrapolation: discrete reversible systems benchmark**

This is honest: the first version lives on discrete systems, where the current evidence says the
discriminator is strongest. Continuous true-integrable systems remain a later tier, not the main
claim of v1.

## 4. Core protocol

Each task instance is a dynamical system family member:

```text
initial state x_0
system parameters theta
horizon T
target state x_T = F_theta^T(x_0)
```

The model receives `x_0`, `theta`, and `T` where appropriate, and predicts `x_T`. A second mode asks
the model to learn one step and compose it for `T` steps. The benchmark should report both modes
when possible:

- **Direct horizon prediction:** can the model map `(x_0, T) -> x_T`?
- **Composed transition:** can the learned one-step transition be iterated without drift?

The composed mode is the main discriminator. It is where small errors accumulate, where bolt-on
constraints fail to recover the true interaction content, and where exact structure should matter.

## 5. Train / test splits

The benchmark should define extrapolation along four axes:

| axis | train | test |
|---|---|---|
| length | `L = 32, 48` | `L = 64, 128, 256, 384` |
| horizon | `T = 1..4` or short `T` | `T = 8, 16, 32`, plus `T proportional to L` |
| interaction density | few solitons / few active blocks | more solitons, higher density, more collisions |
| system parameter | small set of capacities / rules | held-out capacities / rules / boundary settings |

The key is that testing should require many interactions, not only more empty space. Length
extrapolation without collision extrapolation is too easy and can reward trivial transport.

## 6. Metrics

Every leaderboard row should report at least:

1. **State accuracy**
   - per-cell accuracy for binary / categorical states
   - Hamming error or exact-state accuracy where appropriate
2. **Primary conservation**
   - ball count, species count, block charge, or the equivalent system mass
3. **Reversibility**
   - apply predicted forward transition and inverse protocol; measure exact return rate
   - for models without an inverse, evaluate whether predicted states lie on a reversible orbit
4. **Interaction-content fidelity**
   - soliton-content exact match / IoU for BBS-like systems
   - species-labeled soliton content for colored BBS
   - block-rule orbit class / phase consistency for Margolus-like systems
5. **Bolt-on gap**
   - same base model with the obvious invariant forced by projection
   - report whether projection fixes only conservation or also recovers state accuracy and content
6. **Leak-audit gap**
   - replace learned gates with fixed blind rules
   - the benchmark is invalid if the blind audit reaches the structural model

The money metric should be a paired statement:

```text
accuracy high + interaction content high + conservation high + reversibility high
```

A model that preserves ball count but destroys soliton content has not solved the benchmark.

## 7. Systems in v1

### Tier 1: finite-carrier BBS

Purpose: stable, cheap, no plain-BBS `emit = 1 - cell` leak.

Required variations:

- carrier capacity `K` varies by task instance
- train on some capacities, test on held-out capacities
- test on longer length, longer horizon, and higher collision density
- include carrier-blind and fixed-gate audits

What it tests:

- learned residual over a structured carrier
- length / horizon extrapolation
- primary conservation and reversibility

Risk:

- still close to the existing BBS structure, so it cannot carry the whole benchmark alone.

### Tier 2: colored / multi-species BBS

Purpose: make "soliton identity" and collision content harder than total ball count.

Why this is likely the most important new system:

- total mass is not enough;
- species counts, soliton amplitudes, and label ordering create a richer invariant signature;
- collisions must preserve more than a scalar;
- a bolt-on top-N projection cannot recover the right labeled soliton content.

Possible task shape:

```text
state cells: empty or species in {1, ..., C}
parameters: carrier capacity K, species priority / interaction rule
target: state after T steps
metrics: per-cell accuracy, per-species count, labeled soliton content, reversibility
```

This tier is the best bridge between the current toy BBS and a more convincing integrable benchmark.
It stays discrete and verifiable, but it makes the conserved signature richer.

### Tier 3: Margolus block CA family

Purpose: verify that the benchmark is not only carrier scanning.

Required variations:

- multiple reversible, conservative block permutations
- held-out block rules at test time if the rule is provided as a condition
- `T proportional to L` rollout
- bolt-on conservation projection baseline
- identity / fixed-permutation leak audit

What it tests:

- exact composition under a different reversible backbone
- whether structural parameterization, not only conservation, buys accuracy

### Tier 4: Toda / near-Toda challenge

Purpose: optional hard tier for true-integrable continuous or semi-continuous structure.

Status:

- not suitable as the v1 main benchmark, because the existing Toda experiment suggests free-form
  models can solve smooth continuous trajectories well enough that the structural gap is weak.

Use it as:

- a challenge tier;
- a negative-result anchor;
- a future route toward Lax / isospectral / generated-conservation structure.

It should not block v1.

## 8. Required baselines

Each system should include:

- Transformer
- GRU / LSTM
- SSM or Mamba-like scan model
- task-matched scan baseline
- the same baselines with bolt-on conservation projection
- structural model with learned component
- structural model with learned component removed or frozen
- hard-coded oracle, reported only as a ceiling

The hard-coded oracle must never be presented as the learned result. It is the ceiling line.

## 9. What counts as success

The benchmark succeeds if it produces a table with this qualitative pattern:

| model class | state accuracy | conservation | reversibility | interaction content |
|---|---:|---:|---:|---:|
| hard-coded oracle | high | high | high | high |
| structural learned model | high | high | high or verified | high |
| free-form models | maybe high short-term, lower long-term | drifts | weak | low |
| free-form + bolt-on | improves conservation | high | weak | still low |
| leak audit | below structural learned model | varies | varies | below structural learned model |

The benchmark fails if:

- a blind structural audit solves it;
- bolt-on constraints recover the full result;
- the only advantage is total mass conservation;
- a generic model solves all tiers under the same training budget;
- the task requires so much hard-coded system knowledge that learning becomes cosmetic.

## 10. Implementation plan

### Phase A: spec before models

Create a small benchmark API:

```python
sample(theta, L, density, split) -> x0
step(theta, x) -> x_next
rollout(theta, x0, T) -> x_T
invariants(theta, x) -> dict
content(theta, x) -> structured signature
inverse(theta, x) -> x_prev or verification protocol
```

Define fixed train / validation / test splits and store seeds.

### Phase B: finite-carrier BBS benchmark

Port the existing finite-carrier generator and structural carrier into the benchmark API.

Add:

- held-out `K`
- collision-density splits
- leak audits
- bolt-on baselines
- multi-seed runner

### Phase C: colored BBS

Implement the simplest colored BBS variant that has:

- exact rollout;
- content extractor;
- reversible or at least verified inverse protocol;
- non-trivial fixed-rule audit below the learned structural model.

This is the highest-value next system.

### Phase D: Margolus family

Generalize the existing Margolus experiment from one block permutation to a family of conservative
reversible permutations.

### Phase E: public leaderboard format

Produce:

- one `results.json` per run;
- one aggregate `leaderboard.csv`;
- plots for accuracy / conservation / content / reversibility versus length and horizon;
- a leak-audit section that must be filled before a result can be called valid.

## 11. Paper role

This benchmark should become the experimental spine of the paper:

1. BBS demo: why integrable structure is plausible.
2. Finite-carrier BBS: learned structure, no plain-rule leak.
3. Colored BBS: richer interaction content; not just ball count.
4. Margolus family: not carrier-specific.
5. Toda: boundary / negative result explaining why v1 focuses on discrete systems.

The paper's main sentence should be:

> On a family-level discrete reversible benchmark, learned structural models preserve accuracy,
> conservation, reversibility, and interaction content under long composition; generic models and
> bolt-on constraints fail at least one of these, usually interaction content.

That is the clean version of the project claim.

---

## 评价（Claude Code，2026-07-07；综合另一份 Claude 评价）

**结论：这是目前最规整、防坑意识最强的一版（§6/§9 把 leak 审计、bolt-on gap 写成硬性通过条件，把踩出来的照妖镜制度化了，这点该认）。但它有三个致命隐患，都是已踩过的坑的翻版；不先解决，很可能"搭完框架发现主力是空的"。**

1. **整篇论文押在一个从没跑过的系统（colored BBS）上。** 它自己把 colored BBS 定为"最重要的新系统"、写进论文第 3 主力（§11），但这个系统一次没跑、carrier-blind 一次没审计。这和多孤子 benchmark 当初"信誓旦旦、跑完 leak 才发现硬编"是同一个位置——未验证却已当主力。
2. **"深 + 真学"仍靠不同 tier 各证一块、拼起来充数。** finite-carrier = 真学但浅、多孤子 = 深但硬编，**两者兼得的单一系统一个都没有**。proposal 没解决这个核心缺口，只是赌 colored BBS 能填——填不填得上取决于那个没做的探针，所以整个结构悬空。
3. **防坑机制放错了阶段。** 审计被当成"设计门槛"，却没先用探针实测哪个系统能过门槛。按 Phase A–E 搭完，跑到 Tier 2 才发现 leak 审计没过 → 主力塌、框架大半白费。**顺序要反：先探针，再决定搭不搭。**

**补一刀（第一手经验）：** 裸 colored BBS 很可能就是 plain-BBS 那种硬编 leak——"颜色"让**不变量**丰富（治 bolt-on），但不让**学习**非平凡（那来自 finite-carrier 的有界容量，与颜色正交）。所以真正该探的是 **`colored × finite-carrier` 组合**，且它是三重未验证。见本文件顶部的「前置探针」。

**一句话：这份 proposal 值得认真对待，但在跑那个探针之前，它的核心仍是一张空头支票。**

