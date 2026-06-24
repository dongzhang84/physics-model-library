# Physics Model Library

**Physics is a systematically mineable library of candidate models for AI.**

Most AI architecture search is a lottery: try random designs in parallel, lock in the few that happen to work, throw the rest away. This project makes a different bet — that the better move is to **mine the ~100 mature mathematical skeletons physics has spent centuries validating** (Ising, Langevin/diffusion, Hamiltonian, tensor networks, symmetry/equivariance, renormalization group, spin glasses, integrable systems, chaos, …). Fewer than one in ten of these families is seriously used in AI today. Each unmined one may hide the next breakthrough.

This isn't speculation. **Diffusion models are non-equilibrium thermodynamics ported wholesale into generation.** Hopfield networks and Boltzmann machines (spin glass → associative memory; Ising → learning and generation) won the 2024 Nobel Prize in Physics. Oscillator/Hamiltonian structure is turning into long-range sequence models (coRNN, LinOSS). The mine has gold, and that has been proven more than once.

## Evolution, not design

There has been one famous attempt to organize AI around physical structure: **Geometric Deep Learning** (Bronstein et al., 2021), which tried to unify all of deep learning under a single principle — symmetry — modeled on Klein's Erlangen Program. We deliberately do **not** take that road. Klein could unify geometry because geometry was already mature. AI is still in its **pre-paradigmatic era**: new paradigms keep erupting from unexpected directions, and a premature grand unification ends up both too abstract and quickly outdated.

We follow an **evolutionary** logic instead, on three pillars:

- **Breadth** — turn every physics root-family into a candidate AI model; cast a wide net.
- **Engineering** — push each candidate through the full pipeline: *physical principle → trainable network → benchmark → head-to-head against mainstream models*.
- **Match to task** — build a lookup from *task features → best-fit family*, so each task finds the physical structure that suits it.

The bet is not that any one paradigm wins. It is that **continuously mining physics** stays valuable for decades. That bet is very hard to lose.

## The mining map

The core artifact is a living map of physics root-families against the AI capabilities they are naturally good at, and how mature each one is. Maturity: 🟢 established · 🟡 in progress · 🔴 nearly untouched (high-potential).

| Physics family | Mechanism | AI capability | Maturity |
|---|---|---|---|
| Ising / spin glass | energy landscape, relaxation to ground state | associative memory, denoising, energy models, generation | 🟢 |
| Non-equilibrium thermo / diffusion | noise–denoise diffusion | generation (image, audio, …) | 🟢 |
| Symmetry / equivariance | invariance and equivariance | geometric data, molecules, graphs | 🟢 |
| Hamiltonian / oscillator | conserved state along time evolution | long-range sequence modeling | 🟡 |
| Renormalization group | scale-by-scale coarse-graining | multiscale hierarchy, long memory | 🟡 |
| Tensor networks | matrix-product compression of states | compression, classification, generation | 🟡 |
| Chaos / reservoir | edge-of-chaos dynamics | sequence prediction (near training-free) | 🟡 |
| **Integrable systems** | **infinitely many conserved quantities, solitons, exact reversibility** | **length extrapolation, conserved/reversible sequences, crosstalk-free channels** | 🔴 |
| Lindblad / open quantum | dissipation plus noise | unexplored (noisy inference?) | 🔴 |
| KPZ / surface growth | scaling-law stochastic growth | unexplored | 🔴 |
| Hubbard / Heisenberg | strongly-correlated many-body | unexplored | 🔴 |

Full version — with the AI-capability, physics-capability, and maturity cross-tables, plus a candid related-work section — lives in **[CATALOG.md](CATALOG.md)**. The 🔴 rows are the targets this project cares about most. This repo claims the first one: **integrable systems**.

## How a demo earns its place

Two acceptance bars:

- **Bar 1 — light up a blank cell.** A physics family does a legitimate AI task, lighting up one cell in the family × capability map. No need to beat anyone; just show the family can do it, and record what it is good at.
- **Bar 2 — structure beats scale.** A physics family does something today's AI (Transformer/LLM) does *badly* — proven on a falsifiable comparison (e.g. zero-shot, beyond-training-length, with hard guarantees).

One rule governs both, learned from diffusion: **physics is only the engine; the task must be a genuine AI task whose subject has nothing to do with physics.** Diffusion uses thermodynamics as its engine but generates images and music. The failure mode to avoid is making a physics model solve physics (the three-body problem, a PDE) and then bragging that a Transformer can't — that wins on physics' home turf, which proves nothing, and drowns in the AI-for-Science red ocean.

## Why integrable systems first

Integrable systems sit in a 🔴 near-empty zone, and they have a temperament unique in all of physics:

> **Infinitely many exact conserved quantities + full time-reversibility + never chaotic + exactly solvable by the inverse scattering transform.**

They can push a *nonlinear* system forward arbitrarily far and rewind it exactly, without losing a single bit. Their signature is the **soliton**: two waves collide, pass through each other, and come out with their shapes perfectly intact, off by only a phase shift.

By contrast: Hamiltonian networks give only approximate energy conservation and can go chaotic; linear state-space models are structurally too weak; a Transformer's learned conservation is approximate and drifts as sequences grow. **Only integrable systems hold nonlinear + exactly reversible + exactly conserved + non-chaotic all at once.** Bar 2 is staked on that combination.

## Demos

### Demo 1 — Soliton channel · Bar 1

Encode four abstract symbols `[3, 1, 2, 0]` as four KdV solitons of different heights, send them down a *single* channel, let them collide in transit, and read all four back at the far end. The data is abstract symbols; nothing about the task is physical.

![Soliton channel](demos_1/standard1_soliton_channel/soliton_channel_demo.png)

- **Integrable engine (top):** the solitons collide and separate with shape and height untouched → all four symbols decoded.
- **Non-integrable control (bottom, pure linear dispersion):** the same symbols smear into noise → symbols lost.

Same AI task, the integrable engine does it and the non-integrable one cannot → the integrable cell lights up. **Honest boundary:** this shows *structure can do it*, not yet an engineered, trainable, benchmarked model — and attention routes multiplexed information fairly well too, so this is Bar 1, not Bar 2. → [details](demos_1/standard1_soliton_channel/README.md)

### Demo 2 — Box-Ball System vs Transformer · Bar 2

The Box-Ball System (BBS) is the ultradiscrete limit of KdV — an integrable cellular automaton. Task: given a 0/1 state, predict its state several steps later. We train a small Transformer (3 layers, d=64, sinusoidal positions) **only on length L=32**, then test on lengths 48 / 64 / 96 / 128 — out of distribution by length alone (soliton size and density held fixed across lengths).

![Box-Ball System vs Transformer](demos_1/standard2_box_ball_system/bbs_standard2_demo.png)

| Lattice length | Transformer per-cell accuracy | Transformer conserved ball-count | Integrable engine |
|---|---|---|---|
| 32 (train) | 94.2% | 20.0% | 100% / 100% |
| 48 | 83.6% | 0.0% | 100% / 100% |
| 64 | 81.6% | 0.0% | 100% / 100% |
| 96 | 81.7% | 0.0% | 100% / 100% |
| 128 | 81.0% | 0.0% | 100% / 100% |

The integrable engine is exact, conserved, and **bit-perfectly reversible** at every length; the trained Transformer collapses past its training length and destroys the conserved quantity. **Structure beats scale, on the out-of-distribution length that is the Transformer's acknowledged blind spot.**

Honest boundaries, three of them:

1. The integrable rule is *built in* (it is the ground-truth generator), so 100% is "structure by construction," not "learned better." What the demo shows is that an integrable inductive bias extrapolates where pure learning does not.
2. "Recurrent / structured models extrapolate in length better than Transformers" overlaps existing literature (see CATALOG). Integrable's *exclusive* contribution is the "exact conservation + exact reversibility" line.
3. Toy task, a few minutes on CPU. Turning this into publishable evidence means scaling up, adding stronger baselines, and designing tasks that demand *both* nonlinearity and exact conservation.

→ [details](demos_1/standard2_box_ball_system/README.md)

## What is and isn't novel here

Honest, in three layers (full version in [CATALOG.md](CATALOG.md)):

- **Integrable systems + deep learning** — plenty of people do it, but in the *opposite* direction: using neural nets to solve or discover integrable systems (Lax-pair networks, Darboux-transform nets, conservation-law PINNs, SILO). That is "AI as the tool, physics as the goal" — exactly what we avoid.
- **The Box-Ball System** — well studied in mathematical physics (Takahashi–Satsuma 1990; Ferrari et al. 2018), but apparently no one has run it as a *sequence model trained on an AI task and benchmarked against a Transformer*.
- **Length extrapolation** — that structured/recurrent models extrapolate where Transformers don't is an established result (I-BERT; Liu et al. 2023; Merrill & Sabharwal 2023/2024). So the novelty here is the *carrier and the framing, not the capability*.

The collisions are all at the level of parts; nothing collides at the level of the skeleton. **No one has assembled these into "physics is a systematically mineable model library, integrable systems are one un-mined mine, and the proof happens on AI's own turf."** That organizing logic is the point.

## Repository

- **[MANIFESTO.md](MANIFESTO.md)** — why this project exists, the two acceptance bars, why integrable systems first.
- **[CATALOG.md](CATALOG.md)** — the mining map (full cross-tables) and a candid related-work map.
- **[docs/](docs/)** — the long-form proposal: physics methodology as the foundation for the next fifty years of AI.
- **[demos_1/](demos_1/)** — runnable demos (Python plus figures).

```bash
# Demo 1 — soliton channel
cd demos_1/standard1_soliton_channel && pip install numpy matplotlib && python soliton_channel.py

# Demo 2 — box-ball system vs Transformer (CPU is enough)
cd demos_1/standard2_box_ball_system && pip install torch numpy matplotlib && python box_ball_system.py
```

**Status:** early. The map is a draft and the demos are toy-scale. The catalog is meant to be a living document — corrections, additions, and claims on any 🔴 cell are welcome.
