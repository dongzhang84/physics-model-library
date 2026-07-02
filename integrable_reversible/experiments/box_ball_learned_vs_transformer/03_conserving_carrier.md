# Test 3 · Conserving carrier — learns it, and conserves + reverses exactly

**English** · [中文](03_conserving_carrier.zh-CN.md)

> Script: [`03_conserving_carrier.py`](03_conserving_carrier.py)　Figure: [`03_conserving_carrier.png`](03_conserving_carrier.png)
> One run (single seed, CPU). Status: **the result we were after** — a genuinely learned model that matches the cheat ceiling on accuracy *and* conserves + reverses exactly, and extrapolates to any length.

## One-line takeaway

**A learned left→right carrier whose per-site update is constrained to preserve `cell + carrier` learns the BBS rule to 100% accuracy at every length, conserves ball-count exactly (100%), and is exactly reversible (100%, mirror trick) — with no cheating (the rule is learned; loss → 0.0001).** The previous full-Route-C attempt (local gated-swap CA) had the guarantees but couldn't learn; this keeps the carrier's reach and adds the conservation constraint, and lands both.

## What changed from v1

v1 (a local gated-swap CA) guaranteed conservation+reversibility but the *local* structure couldn't express BBS (accuracy stuck ~82%). v2 keeps the **carrier** (left→right reach = expressive, learns BBS) and constrains the **per-site update to conserve**: with carrier count `k` and cell `c`, total `t = c + k`, the only count-preserving outcomes are *emit* (`out=1, k'=t−1`) or *hold* (`out=0, k'=t`); a learned gate picks between them. BBS itself is just the fixed rule "emit iff cell==0" — here that rule is *learned*.

## The four models

| # | Line | Reach | Rule | Conserves / reversible |
|---|---|---|---|---|
| 1 | Hard-coded integrable | — | written in (cheat) | exact |
| 2 | Plain carrier | left→right | learned, **free-emit** | drifts / no |
| 3 | **Conserving carrier (Route C v2)** | left→right | learned, **emit/hold** | **exact (see below)** |
| 4 | Transformer | global attention | learned | broken / no |

## Results (trained on L=32 only; tested OOD by length)

![four-way figure](03_conserving_carrier.png)

| Length | all-zeros | Transformer acc / cons | Plain carrier acc / cons | **Conserving carrier** acc / cons | Integrable |
|---|---|---|---|---|---|
| 32 (train) | 84.2% | 94.5 / 29.0 | 98.9 / 76.3 | **100.0 / 100.0** | 100 / 100 |
| 48 | 83.7% | 83.7 / 0.0 | 99.1 / 70.7 | **100.0 / 100.0** | 100 / 100 |
| 64 | 82.9% | 82.0 / 0.0 | 98.7 / 57.3 | **100.0 / 100.0** | 100 / 100 |
| 96 | 82.7% | 81.8 / 0.0 | 98.4 / 41.0 | **100.0 / 100.0** | 100 / 100 |
| 128 | 82.3% | 80.9 / 0.0 | 98.4 / 38.7 | **100.0 / 100.0** | 100 / 100 |

Reversibility (conserving carrier, whole-sequence, via the BBS mirror trick `inv = mirror·step·mirror`): **100.0%**.

## What is guaranteed vs learned (precise)

- **Per-site conservation — structural (any gate).** Each site preserves `cell + carrier` by construction; the gate can only choose emit vs hold, never create or destroy a ball.
- **Whole-output conservation = 100% — structural bias + learned.** The output ball-count equals the input's *minus whatever stays in the carrier at the end*; here the learned gate empties the carrier over the trailing zeros, so the measured conservation is exactly 100% at all lengths. (A deterministic end-of-scan flush would make this a hard guarantee for any gate; it wasn't needed.)
- **Reversibility = 100% — learned rule + structure, verified.** The model learned (essentially) BBS, which is exactly invertible by the mirror trick; verified whole-sequence at 100%. This is inherited from learning the reversible rule, not a hard guarantee for arbitrary gates.
- **Accuracy — learned, and it extrapolates.** Loss → 0.0001 on L=32; because the carrier update is weight-shared and length-independent, it holds 100% out to L=128.

## The clean isolation

**#2 plain carrier vs #3 conserving carrier** have the *same* left→right reach and both learn the task (~99% / 100% accuracy). The only difference is the per-site parameterization: free-emit vs count-preserving emit/hold. That single change moves conservation from **76 → 38% (drifting with length)** to a **flat 100%** — clean evidence that the conservation comes from the structural constraint, not from scale, data, or reach.

## Significance

This is the shape of the publishable result the project was aiming for: **a genuinely trained model with an integrable-style inductive bias that (i) beats the Transformer on OOD length, (ii) matches the hard-coded integrable ceiling on accuracy, and (iii) adds exact conservation and reversibility — reached by learning, not by hard-coding.** It turns the earlier "cheat vs learn" and "guarantee vs expressivity" tensions into a single line that tracks the ceiling on both panels.

## Honest boundaries

1. **Single seed, toy scale, one task (BBS).** The next step is multiple seeds (mean ± error bars) and more than one reversible system (Margolus CA, Toda) to show the recipe transfers — exactly the `data/reversible_systems/` plan in the proposal.
2. **The architecture is a close structural match to BBS** (carrier + emit/hold). That is the point (the right inductive bias makes the rule learnable and exactly extrapolable), but it also means the hard part now is *generality*: does the same conserving-carrier idea learn other integrable/reversible dynamics, not just the one it structurally fits?
3. **Global conservation relied on the carrier emptying** (it did, 100%); a flush would make it unconditional.
