# Benchmark proposal: family-level integrable / reversible extrapolation

> Status: proposal draft. This file defines what the benchmark should become. The current
> `README.md` records an existing benchmark attempt and why it is not valid yet.

## 0. Phase 0 gate: validate the candidate flagship first

> **Outcome (2026-07-07): gate PASSED via Path B — single-color finite-carrier BBS + soliton-amplitude
> content.** Rather than the harder colored rule, the cheaper single-color combination already clears
> both conditions (5 seeds): structural 100 ± 0 amplitude-content, carrier-blind leak audit only
> 40 ± 3 (a 60-point learned margin — genuine, no leak), and a ball-count bolt-on recovers 0.7 ± 0.8 %
> amplitude content (depth: the multiset is not bolt-on-able). This is the **validated Tier 1**
> ([`RESULT_tier1_finite_carrier.md`](RESULT_tier1_finite_carrier.md)). The **colored** rule was
> tried first (`colored_bbs.py`) but a naive colored carrier is not exactly reversible (~86%), so a
> correct colored system needs the crystal / combinatorial-R construction; **colored is deferred to a
> future Tier 2 enhancement**, not needed for a valid first tier. The original gate reasoning is kept
> below for the record.

Before building the full benchmark framework, run a small probe for the most important unverified
assumption: whether a **colored finite-carrier BBS** task can be both richer than scalar
conservation and still genuinely learned.

This matters because "richer invariant content" and "non-trivial learning" come from different
mechanisms:

- **Colored / multi-species BBS** makes the conserved signature richer: labeled soliton content,
  species counts, and collision identity are harder to recover than total ball count.
- **Finite-carrier BBS** makes the learned emit rule non-trivial: a carrier-blind fixed rule should
  no longer reach the learned structural model, as in the plain-BBS leak.

Plain colored BBS may still be a fixed deterministic CA with a hard-coded carrier solution. The
first serious probe should therefore be the combination:

```text
colored BBS x finite carrier capacity K
```

Minimal probe:

1. Implement exact rollout, invariant checks, a labeled-content extractor, and a reversibility check
   or inverse verification protocol.
2. Run three entrants on one seed:
   - structural model with a learned gate;
   - carrier-blind / fixed-gate audit with learning disabled;
   - free-form model plus bolt-on conservation.
3. Ask one question: does the carrier-blind audit lose labeled-content fidelity relative to the
   learned structural model?

Interpretation:

- If `carrier-blind << learned structural model`, the candidate system has both richer content and
  genuine learning, so it is worth building the full benchmark around it.
- If `carrier-blind` is also high, the system has another hard-coding leak; do not promote it to the
  flagship tier.

Until this gate passes, colored BBS is a candidate flagship, not an established benchmark system.

## 1. One-sentence goal

Build a benchmark that tests whether a **learned** model with integrable / reversible structure can
preserve exact long-horizon behavior under length, time, and collision extrapolation, in a way that
generic sequence models and bolt-on constraints cannot replicate.

The benchmark should not merely show that a hard-coded rule wins. It should show that the right
physical structure makes a model learn the rule **more exactly**, so that the learned transition can
be composed many times without drift.

## 2. Why the current demos are not enough

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

## 3. Claim the benchmark should adjudicate

The benchmark is designed around this claim:

> For discrete reversible / integrable dynamics with many exact interactions, structure helps not
> because it adds a single conserved quantity, but because it makes the learned local transition
> exact enough that long compositions preserve the whole interaction content.

The phrase "interaction content" is deliberate. The target is not just total mass. It includes
objects, labels, amplitudes, phases, collision outcomes, reversibility, and any system-specific
conserved signature that distinguishes integrable behavior from ordinary conservation.

## 4. Benchmark name and scope

Working name:

**Integrable Extrapolation Benchmark**

Conservative name, if we want to avoid overclaiming before Toda / KdV succeeds:

**Reversible Interaction Extrapolation Benchmark**

Recommendation for the first public version: use the conservative subtitle:

**Integrable Extrapolation: discrete reversible systems benchmark**

This is honest: the first version lives on discrete systems, where the current evidence says the
discriminator is strongest. Continuous true-integrable systems remain a later tier, not the main
claim of v1.

## 5. Core protocol

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

## 6. Train / test splits

The benchmark should define extrapolation along four axes:

| axis | train | test |
|---|---|---|
| length | `L = 32, 48` | `L = 64, 128, 256, 384` |
| horizon | `T = 1..4` or short `T` | `T = 8, 16, 32`, plus `T proportional to L` |
| interaction density | few solitons / few active blocks | more solitons, higher density, more collisions |
| system parameter | small set of capacities / rules | held-out capacities / rules / boundary settings |

The key is that testing should require many interactions, not only more empty space. Length
extrapolation without collision extrapolation is too easy and can reward trivial transport.

## 7. Metrics

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

## 8. Systems in v1

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

### Tier 2: colored finite-carrier BBS candidate

Purpose: make "soliton identity" and collision content harder than total ball count while preserving
the no-leak property of finite-carrier BBS.

Why this is the candidate flagship:

- total mass is not enough;
- species counts, soliton amplitudes, and label ordering create a richer invariant signature;
- collisions must preserve more than a scalar;
- a bolt-on top-N projection cannot recover the right labeled soliton content.
- finite carrier capacity should make the learned gate non-trivial, preventing the plain-BBS
  carrier-blind leak.

Possible task shape:

```text
state cells: empty or species in {1, ..., C}
parameters: carrier capacity K, species priority / interaction rule
target: state after T steps
metrics: per-cell accuracy, per-species count, labeled soliton content, reversibility
```

This tier is the best bridge between the current toy BBS and a more convincing integrable benchmark.
It stays discrete and verifiable, but it makes the conserved signature richer. It must pass the
Phase 0 gate before being treated as a main paper result.

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

## 9. Required baselines

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

## 10. What counts as success

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

For the colored finite-carrier tier specifically, the benchmark fails if the carrier-blind audit
matches the learned structural model on labeled-content fidelity.

## 11. Implementation plan

### Phase 0: colored finite-carrier probe

Before building the shared framework, run the single-seed probe described at the top of this file.
This is a gate, not a full result.

Deliverables:

- exact colored finite-carrier rollout;
- labeled-content extractor;
- conservation and reversibility checks;
- structural learned model;
- carrier-blind / fixed-gate audit;
- free-form plus bolt-on conservation baseline;
- one short table for accuracy, conservation, reversibility, and labeled-content fidelity.

Decision:

- pass: proceed to Phase A and make colored finite-carrier BBS the candidate flagship tier;
- fail: do not build the v1 benchmark around colored BBS; return to system search.

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

### Phase C: colored finite-carrier BBS

If Phase 0 passes, implement the simplest colored finite-carrier BBS variant that has:

- exact rollout;
- content extractor;
- reversible or at least verified inverse protocol;
- non-trivial fixed-rule audit below the learned structural model;
- richer labeled-content metrics than total ball count.

This is the highest-value next system only after the Phase 0 gate passes.

### Phase D: Margolus family

Generalize the existing Margolus experiment from one block permutation to a family of conservative
reversible permutations.

### Phase E: public leaderboard format

Produce:

- one `results.json` per run;
- one aggregate `leaderboard.csv`;
- plots for accuracy / conservation / content / reversibility versus length and horizon;
- a leak-audit section that must be filled before a result can be called valid.

## 12. Paper role

This benchmark should become the experimental spine of the paper:

1. BBS demo: why integrable structure is plausible.
2. Finite-carrier BBS: learned structure, no plain-rule leak.
3. Colored finite-carrier BBS, if the gate passes: richer interaction content; not just ball count.
4. Margolus family: not carrier-specific.
5. Toda: boundary / negative result explaining why v1 focuses on discrete systems.

The paper's main sentence should be:

> On a family-level discrete reversible benchmark, learned structural models preserve accuracy,
> conservation, reversibility, and interaction content under long composition; generic models and
> bolt-on constraints fail at least one of these, usually interaction content.

That is the clean version of the project claim. If the colored finite-carrier gate fails, the paper
should keep BBS and Margolus as evidence for the current narrower claim, and treat benchmark design
as an open system-search problem rather than a completed result.
