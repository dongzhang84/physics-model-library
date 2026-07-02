# Test 1 · Plain learned carrier — removes the cheat, but doesn't conserve

**English** · [中文](01_plain_carrier.zh-CN.md)

> Script: [`01_plain_carrier.py`](01_plain_carrier.py)　Figure: [`01_plain_carrier.png`](01_plain_carrier.png)
> One run (single random seed, tens of seconds on CPU). Status: the first brick of Route C, not a final result.

## One-line takeaway

Once Demo 2's "cheat" is removed, **a genuinely trained model carrying a length-independent structure still holds ~92% per-position accuracy beyond its training length (the Transformer collapses to the trivial baseline) — yet its conserved ball-count still drifts toward zero.** The conclusion splits in two: **structure + learning buys "accuracy extrapolation", but not "exact conservation".** The latter is integrable systems' exclusive gift and must be welded in by structure — precisely the missing piece of full Route C.

---

## 1. Why this experiment (removing Demo 2's cheat)

In Demo 2 the "integrable engine" is the **hard-coded rule** — i.e. the very generator of the labels. Pitting that answer key against a Transformer that is trying to learn it, then declaring the answer key 100%, is close to tautological — Demo 2's biggest weakness.

This experiment removes the cheat: **keep the hard-coded integrable line only as a ceiling**, and add a **genuinely trained** model to test whether a *learned* model can also extrapolate.

## 2. The three models

| Model | Structure | Where the rule comes from | Role |
|---|---|---|---|
| Hard-coded integrable (BBS carrier) | — | Written in (it is the ground-truth generator) | Ceiling / cheat reference |
| **Learned carrier (the protagonist)** | Weight-shared left→right recurrent scan with a small "carrier" state; learn the single step, apply it T times | **Learned from L=32 data**, never sees the BBS rule | Route C prototype |
| Transformer | Generic attention (sinusoidal PE, so it can accept longer inputs) | Learned from L=32 data | Lower bound / no-structure control |

- **Task**: given a 0/1 state, predict its state after **2 steps**.
- **Training**: both learned models are trained on **L=32 only**.
- **Testing**: L = 32 / 48 / 64 / 96 / 128; soliton size and density fixed across lengths → **pure length extrapolation**.
- **Why the learned carrier can run at any length**: its weights are position-independent and its carrier state is fixed-size, so it runs at any length; but **what it computes is learned** — that is what removes the cheat.

## 3. Results

![three-line figure](01_plain_carrier.png)

| Length | All-zeros baseline (sparsity) | Transformer acc / conserved | **Learned carrier** acc / conserved | Hard-coded integrable |
|---|---|---|---|---|
| 32 (train) | 84.2% | 94.5% / 29.0% | 94.1% / 32.7% | 100% / 100% |
| 48 | 83.7% | 83.7% / **0.0%** | **93.3%** / 21.7% | 100% / 100% |
| 64 | 82.9% | 82.0% / 0.0% | **92.5%** / 12.0% | 100% / 100% |
| 96 | 82.7% | 81.8% / 0.0% | **92.5%** / 6.0% | 100% / 100% |
| 128 | 82.3% | 80.9% / 0.0% | **92.1%** / 3.3% | 100% / 100% |

## 4. Reading the result (two separate things)

**① Accuracy extrapolation: structure wins, and without cheating.**
The learned carrier stays at ~92% across all lengths, tracking just under the ceiling; the Transformer, once past its training length, **falls to the all-zeros baseline** — out of distribution it has learned almost nothing (confirming the earlier point that "81% per-position is inflated by sparsity": the all-zeros baseline is already ~82%). **Key: the protagonist is genuinely trained and never saw the rule, yet it extrapolates. The cheat is gone, and the conclusion — "learning can extrapolate, the Transformer cannot" — still holds.**

**② Exact conservation: a recurrent structure alone is not enough.**
The most valuable finding — **the protagonist's conserved ball-count also drifts toward zero** (32.7 → 21.7 → 12 → 6 → 3.3%), just more slowly than the Transformer (which drops straight to 0). Reason: this prototype's reversibility/conservation is **approximately learned, not structurally guaranteed**; small ~92%-per-position errors accumulate over length until exact conservation collapses.

**Together**: a generic recurrent structure **buys accuracy extrapolation but not exact conservation**. And "exact conservation + exact reversibility" is exactly what sets integrable systems apart from ordinary recurrent models. So this prototype pinpoints the missing piece of full Route C: **weld conservation and reversibility in by structure** (coupling layers + a structural invariant) so the protagonist's conservation line also becomes a flat 100%.

## 5. Honest boundaries

1. **Single run, single seed, toy scale** (tens of seconds on CPU). The trend is clear, but numbers wobble across seeds; a real version needs multiple seeds with mean ± error bars.
2. **What the protagonist welds in is the "recurrent/local" computational form.** The length extrapolation this buys overlaps existing literature (recurrent models extrapolate better than Transformers). **What this prototype exclusively contributes is quantifying the *absence* of conservation** — evidence that "exact conservation must be structural, not learned".
3. **Reversibility is not measured here** (the protagonist is a forward scan, not structurally reversible). That belongs to the next step.
4. Per-position accuracy is still a soft metric (inflated by sparsity), so the report also reports the all-zeros baseline and conservation as two harder quantities.

## 6. Next step

Upgrade the "learned carrier" into a **structurally reversible + structurally conserving** version (full Route C):
- reversibility welded in via coupling layers (invertible no matter what the inner function learns);
- ball-count conservation enforced as a structural constraint;
- goal: turn the protagonist's conservation column from "drifting to 0" into "a flat 100%", while accuracy still tracks the ceiling.

Reach that, and this comparison graduates from "prototype demo" to "publishable result".
