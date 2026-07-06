# Structural conservation on a second reversible system — the Margolus block CA

**English** · (中文完整版将在实验收尾时补上)

> **Status: in progress.** Ground truth ✅ · single-seed probe ✅ (the discriminator appears) · multi-seed rigor + full write-up ⏳.

## Why this experiment

The [Box-Ball experiment](../box_ball_learned_vs_transformer/) showed that welding
**conservation** into a network's *structure* keeps the invariant exact where
free-emit scan models drift. But that was **one** system, and its natural prior is
a long-range left→right **carrier scan**. The open question the proposal raises is
**generality**: does the recipe *transfer* to a reversible system whose natural
structure is different — and does a different system indeed want a different
structure?

Margolus is the deliberate contrast: a **block / partitioning** CA with **no
long-range carrier**. Same recipe (structural conservation + a learned residual),
a genuinely different backbone.

## The system (1D Margolus block CA)

- Ring of length `L` (divisible by 3), periodic boundary.
- **Block size 3**; the partition **offset cycles 0 → 1 → 2** with the step index —
  the 1D analogue of Margolus' alternating neighbourhood.
- Each block is mapped by a fixed permutation **φ** that fixes `000`/`111` and
  **rotates the 1-ball class `{100,010,001}` and the 2-ball class `{110,101,011}`**
  by a 3-cycle. φ is a bijection (**reversibility structural**) and preserves
  ball-count per block (**conservation structural**); it is *not* the identity, so
  the rotation is a **non-trivial residual that must be learned**.

`01_margolus_system.py` establishes the ground truth (self-check + leak audit):

```
ball-count conserved : True
reversible (block-inverse, reversed phase order) : True
leak audit — identity (blind) per-cell accuracy: 52.8%   (≪100% ⇒ rotation is non-trivial)
```

## Why the horizon grows with length (T ∝ L)

Margolus' one-step rule is **local** (range ≈ block size), unlike BBS's O(L) carrier
sweep. At a fixed small horizon *any* model would nail it, so there would be nothing
to measure. To recreate a length-generalization challenge we let the horizon grow
with the ring: **at test length `L` we compose `T = L/2` steps.** A composing model
extrapolates in `T` for free; the question is whether its *invariant* survives the
long composition.

## Probe result (single seed — trend signal, not the final number)

`02_probe.py`: both models learn the **single (phase-aware) step**, then compose
`T = L/2` on growing rings. Free-form baseline is a **bidirectional** GRU — the fair
prior here, since a block map needs intra-block context (a causal GRU, natural for
BBS, is the *wrong* prior for Margolus and was rejected as unfair).

| | single-step acc / cons | L=48, T=24 | L=96, T=48 | L=192, T=96 | L=384, T=192 |
|---|---|---|---|---|---|
| **structural block-CA** | **100 / 100** | **100 / 100** | **100 / 100** | **100 / 100** | **100 / 100** |
| bidirectional GRU (free-form) | 98.6 / 66.7 | 63 / 7 | 63 / 6 | 62 / 4 | 60 / 1 |

**Reading it.** The fair free-form model learns the step *well* (98.6%) but not
**exactly**; that ~1.4% residual, compounded over `T ∝ L` steps, drives accuracy to
~60% and conservation to ~1%. The structural model is exact per step and stays
100/100 at any horizon. **The Box-Ball finding transfers to a structurally different
reversible system — and it sharpens the thesis: structural conservation matters most
precisely when you compose many steps, which is the reversible-systems regime.**

**Honest boundary (already visible).** The edge comes from the free-form model not
hitting *exactly* 100% single-step — not from it "failing to learn the rule" (98.6%
is high). A model that reached an exact single step would compose cleanly too; but
hitting *exactly* 100% is the hard part, and the structural model gets it for free.

## Next

- **Multi-seed rigor** (5 seeds, mean ± std), mirroring Box-Ball Test 5.
- One or two more **fair** free-form baselines (e.g. bidirectional LSTM; a small
  Transformer as the single-step learner) to show the drift is not GRU-specific.
- Figures (accuracy + conservation vs length) and the full bilingual write-up.
- Later escalation: the **2D classic Margolus** (2×2 blocks, e.g. critters / HPP
  lattice gas) — richer count classes, the canonical version.

## Reproduce (CPU, ~1 minute each)

```
python3 01_margolus_system.py   # ground truth: conservation + reversibility + leak audit
python3 02_probe.py             # single-seed probe: structural stays exact, free-form drifts
```
