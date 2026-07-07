# Making a *true*-integrable model learnable — the Toda lattice (attempt 1: a negative result)

**English** · (中文版可后补)

> **Status: honest negative result (F2) on the first structural approach.** Ground-truth
> integrability ✅ · action-angle learnable model ✅ built · **it does not beat the strongest
> bolt-on and does not extrapolate — F2.** Pre-registered before coding
> ([`PREREGISTRATION.md`](PREREGISTRATION.md)); every step logged
> ([`OBSTACLES.md`](OBSTACLES.md)). Next step is a decision (write up / try a different
> structure) — see the end.

## Why this experiment (the real target)

The Box-Ball and [Margolus](../margolus_block_ca/) experiments showed the recipe transfers,
but a **bolt-on control** proved their headline invariant (*conservation*) is **bolt-on-able**:
force it onto a free-form model and it holds — what the structure really buys is *exactness*.
Both systems are also **shallow**: they use only one conservation law, so a skeptic can still
say *"you never made the integrable system itself into a model."*

Toda is the first **genuinely integrable** system (a real **Lax pair**, **N conserved
quantities in involution**, action-angle structure). The goal: make its core structure a
**learnable** model that buys something **no finite set of bolted-on constraints can
replicate**. Per the pre-registration, the criterion is set where it *can't* be bolted on:

> Pin **all N conserved quantities** onto a free-form time-stepper and it should *still* fail
> to extrapolate long-time — because what integrability adds over its conserved values is the
> **linear evolution of the angle variables** (action-angle), not the values themselves.

## The system is genuinely integrable (Step 1 ✅)

[`01_toda_system.py`](01_toda_system.py) — open Toda chain, velocity-Verlet integrator,
Flaschka → Lax matrix. Self-check on generic / soliton / 2-soliton initial conditions:

```
energy drift    ~1e-5     (symplectic)
momentum drift  ~1e-14    (machine precision)
ISOSPECTRAL     ~1e-6     ← the N Lax eigenvalues are conserved  (the integrability signature)
1-soliton       discrete eigenvalue λ = cosh κ = 1.185 > 1       (a genuine bound soliton)
```

The structure the model must exploit — a conserved spectrum + linearly-evolving angles — is
**really there and numerically exact**. (Honest caveat, logged: the *analytic* soliton IC is
only approximate; irrelevant to Step 1 and to the probe, which uses bounded quasi-periodic
states and reads any phase shift straight off the exact integrator.)

## The task and the models

**Time extrapolation** (the analogue of BBS/Margolus length extrapolation): given `s(0)`,
predict `s(t)`; train on `t ≤ 5`, test out to `t = 20`. Periodic Toda, `N = 8`, moderate
amplitude (genuinely nonlinear, not just phonons). [`02_probe.py`](02_probe.py), single seed.

| model | what it is |
|---|---|
| **structural v1** | learnable action-angle: encode `s0→(I,φ0)`, evolve `φ(t)=φ0+ω(I)·t`, decode `(I,cosφ,sinφ)→s(t)`. Actions **learned**. |
| **structural v2** | same, but the actions `I` are the **true conserved quantities** `tr(Lᵏ)` (given, not learned — the principled iteration); only φ0, ω(I), decoder are learned. |
| **free-form** | direct map `(s0,t)→s(t)`, no structure. |
| **bolt-on (strongest)** | free-form **+ all N conserved quantities `tr(Lᵏ)` pinned** by a training penalty. The sharpest litmus: *pin every conserved quantity.* |

Integrability is an **inductive bias** (conserved actions + linear angles); the hard inverse
map (spectrum + angle → state) is **learned**, not a hard-coded IST — avoiding the Demo-2
tautology (pre-registration §4).

## Result — F2 (the structure does not win)

State MSE vs prediction time (predict-the-mean baseline ≈ **0.245**), 4 buckets:

| model | t ≤ 5 (train) | t 5–10 | t 10–15 | t 15–20 |
|---|---|---|---|---|
| structural v1 (learn actions) | 0.038 | 0.436 | 0.516 | 0.501 |
| structural v2 (true actions) | 0.224 | 0.466 | 0.502 | 0.532 |
| free-form | 0.022 | 0.860 | 1.703 | 3.078 |
| **bolt-on (all N cons pinned)** | 0.277 | **0.380** | **0.381** | **0.504** |

**Reading it, without dressing it up:**

- **The structural model does not extrapolate.** Both variants fit the training window but
  drift out of distribution, reaching MSE ~0.5 at `t 15–20` — **worse than predicting the
  mean (0.245)**.
- **Giving it the true conserved quantities did not help** (v2 ≈ v1; v2 is even worse
  in-distribution — the extra structure traded off fit without buying extrapolation).
- **The structure does not beat the strongest bolt-on.** In the informative OOD band
  (`t 5–15`) the bolt-on (0.38) is actually **better** than either structural model
  (0.44–0.52). Only the unconstrained free-form is clearly worst (and its conserved
  quantities explode to 10⁸ — it leaves the physical manifold entirely).
- The clean story we hoped for — *bolt-on keeps conservation but drifts in state; structure
  stays flat* — **did not materialise**. Everything drifts long-time.

**Verdict: F2** (a pre-registered outcome). The learnable **action-angle** approach — with
actions learned *or* given true — does **not** learn to exploit Toda's integrability for time
extrapolation, and does not beat a free-form model with every conserved quantity bolted on.

## The obstacle (what specifically blocked it)

**R2 — joint learnability of accurate frequencies + the inverse-spectral decoder.** Long-time
prediction needs `φ(t)=φ0+ω·t` with `ω` accurate (a small error amplifies 4× from `t=5` to
`t=20`) **and** a decoder that reconstructs the 16-D state from an 8-D torus + 8 actions — the
inverse-spectral reconstruction. An end-to-end MLP did not capture this, even handed the exact
actions. (Full diagnosis, ruled-out hypotheses, and the frequency-window argument in
[`OBSTACLES.md`](OBSTACLES.md) #2–#3.)

## What this does and does **not** claim

- ✅ **Does** claim: *the action-angle-MLP route, in this rough-demo setup, fails to make
  Toda's integrable structure learnably useful, and the blocker is R2.* A scoped, honest
  negative result — exactly the "obstacle analysis" the project values.
- ❌ **Does not** claim "Toda × integrability is unlearnable." Only one structural family was
  tried. A reviewer could fairly say we picked a hard architecture.

## Next (a decision)

1. **Write it up as the negative result** — clean and honest, and a planned outcome.
2. **A new pre-registration for a different structure: "Neural-Lax"** — evolve the Lax matrix
   directly under `dL/dt = [B,L]` (isospectrality by construction), **sidestepping the
   inverse-spectral decoder** that blocked this attempt. Either it works (the real target
   delivered) or it also hits F2 (a much stronger *two-independent-routes* negative). A fresh
   attempt with its own criterion — **not** hyperparameter chasing.

## Reproduce (CPU, ~2 min)

```
python3 01_toda_system.py   # integrability self-check (energy / momentum / isospectral)
python3 02_probe.py         # structural (v1/v2) vs free-form vs strongest bolt-on
```
