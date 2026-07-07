# Making a *true*-integrable model learnable — the Toda lattice (two attempts: a negative result + a finding)

**English** · (中文版可后补)

> **Status: honest negative result — and a more interesting one than expected.** Ground-truth
> integrability ✅. Two independent structural approaches tried (pre-registered before coding):
> **(1) action-angle** → F2 (doesn't extrapolate, doesn't beat bolt-on); **(2) Neural-Lax** →
> F2′ (technical failure) **and** it surfaced the real finding — **a plain free-form composing
> stepper already solves continuous-Toda time extrapolation, so there is no gap for integrable
> structure to fill.** This empirically confirms the roadmap's *discrete→continuous* concern:
> the discriminator that made structural conservation shine on discrete BBS/Margolus does **not**
> transfer to smooth continuous Toda. Pre-registrations: [`PREREGISTRATION.md`](PREREGISTRATION.md),
> [`PREREGISTRATION_v2_neural_lax.md`](PREREGISTRATION_v2_neural_lax.md); every step logged in
> [`OBSTACLES.md`](OBSTACLES.md).

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

## Attempt 1 result — F2 (action-angle does not win)

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

## Attempt 2 — Neural-Lax, and the deeper finding

[`03_neural_lax.py`](03_neural_lax.py). A different structure to sidestep attempt 1's R2: keep
the state as a Lax matrix `L` (Flaschka — trivial both ways), and take an **exactly isospectral
step** `L → Q L Qᵀ, Q = exp(Δ·B_θ(L))`, learning the Lax-flow generator `B_θ = Σ_k c_k(P₊(L^k)−P₋(L^k))`.
All three models are now **composing steppers** (learn a small step `Δ=0.25`, compose to `t=20`).

State MSE (predict-mean ≈ **0.251**):

| model | t ≤ 5 | t 5–10 | t 10–15 | t 15–20 |
|---|---|---|---|---|
| Neural-Lax (isospectral, learn c_k) | 1.60 | 9.59 | 28.4 | 49.6 |
| **free-form** | **0.0002** | **0.0008** | **0.0016** | **0.0026** |
| bolt-on (all N cons pinned) | 0.234 | 0.529 | 0.378 | 0.341 |

**This flipped the picture — and the surprise is the real result:**

- **A plain free-form composing stepper *solves* the task** — MSE ~0.003, essentially flat, and
  it conserves as a **by-product of accuracy** (conserved-qty error ~0.03, no penalty needed).
- **Neural-Lax failed technically** (F2′): the reprojection to the tridiagonal band **broke the
  exact isospectrality** the conjugation was supposed to give (conserved-qty error ~1.0, not 0),
  and it did not learn `c₁≈1` (`c_k = [0.39, 0.03]`) — so its flow was wrong and it blew up.
- **The bolt-on penalty back-fired** (worse than plain free-form on *both* axes) — pinning
  large-dynamic-range `tr(L^k)` destabilised training.

**Why free-form wins here — the finding.** With a small step and a **smooth** flow, the one-step
map is trivially learnable and composing it barely accumulates error. This is the opposite of the
**discrete** BBS/Margolus regime, where the exact rule is a discrete map a free-form model can't
quite nail (98.7% ≠ 100%) and `T ∝ L` composition amplifies the residual into a conservation
collapse. **On smooth continuous Toda there is no hard-to-nail exact step to compound, so
free-form composing already extrapolates — leaving no gap for integrable structure to fill.**

This empirically confirms the roadmap's (`../../proposal.md` §7) **discrete → continuous** concern:
the discriminator that made structural conservation shine on discrete systems does **not** transfer
to continuous Toda.

## What this does and does **not** claim

- ✅ **Does** claim: *two independent structural routes (action-angle, Neural-Lax) failed to make
  Toda's integrability learnably win; and — more importantly — a free-form composing stepper already
  solves continuous-Toda time extrapolation, so this task offers no gap for structure.* The
  discrete→continuous barrier is real and empirical.
- ❌ **Does not** claim "no integrable-structure model can ever help on any continuous task." It
  claims this *time-extrapolation task on smooth Toda* does not discriminate, and two natural
  structural designs did not win.

## Next (a decision)

Two disciplined attempts are spent; per the pre-registration we **stop iterating architectures**
(no third shot — that would be the bottomless pit we agreed to avoid). The honest deliverable is
this **negative result + finding**: *integrable structure did not beat free-form on continuous
Toda, because the discrete-system discriminator (tiny-residual-compounds-under-composition) has no
analogue in a smooth flow.* Options: write it up as-is; or step back and pick a **different kind of
continuous task** where free-form composing would *not* already win (e.g. one requiring the global
scattering structure — long-range, not a smooth local step) — a **new** experiment, not a re-tune.

## Reproduce (CPU, ~2 min each)

```
python3 01_toda_system.py   # integrability self-check (energy / momentum / isospectral)
python3 02_probe.py         # attempt 1: action-angle (v1/v2) vs free-form vs bolt-on
python3 03_neural_lax.py    # attempt 2: Neural-Lax vs free-form vs bolt-on (composing)
```
