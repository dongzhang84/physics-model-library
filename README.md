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

## The first mine: integrable systems

The first 🔴 cell this repo claims is **integrable systems** — the one physics family that holds *nonlinear + exactly reversible + exactly conserved + non-chaotic* all at once, so it can push a nonlinear system forward arbitrarily far and rewind it exactly, without losing a bit. Two finished demos already earn both bars: a **crosstalk-free soliton channel** (Bar 1 — four symbols collide down one channel and decode intact) and the **Box-Ball System vs a Transformer** (Bar 2 — exact, conserved, and bit-perfectly reversible at every length, where the trained Transformer collapses past its training length and destroys the conserved quantity).

The full rationale, both demos with figures and honest boundaries, the candid related-work map, and the research plan toward a paper live in the project itself:

**→ [integrable_reversible/](integrable_reversible/)** — the integrable & reversible systems research project (the Physics Library's first mined family).

## Repository

- **[MANIFESTO.md](MANIFESTO.md)** — why this project exists, the two acceptance bars, why integrable systems first.
- **[CATALOG.md](CATALOG.md)** — the mining map (full cross-tables) and a candid related-work map.
- **[docs/](docs/)** — the long-form proposal: physics methodology as the foundation for the next fifty years of AI.
- **[integrable_reversible/](integrable_reversible/)** — the first mined family: the integrable & reversible systems research project (proposal, finished demos, and the in-progress paper).

```bash
# Demo 1 — soliton channel
cd integrable_reversible/demos/soliton_channel && pip install numpy matplotlib && python soliton_channel.py

# Demo 2 — box-ball system vs Transformer (CPU is enough)
cd integrable_reversible/demos/box_ball_system && pip install torch numpy matplotlib && python box_ball_system.py
```

**Status:** early. The map is a draft and the demos are toy-scale. The catalog is meant to be a living document — corrections, additions, and claims on any 🔴 cell are welcome.
