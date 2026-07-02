# Experiments — the Route C arc

**English** · [中文](README.zh-CN.md)

The research question behind Demo 2: once you remove its **cheat** (the "integrable engine" is the hard-coded rule that generated the labels), can a *genuinely trained* model carrying an integrable-style inductive bias beat a Transformer on the Box-Ball System length-extrapolation task — and add exact conservation and reversibility?

**We ran 3 tests** — three iterations of the *same* investigation (Route C on the Box-Ball System), not three separate experiments, so they live in this one folder as `01/02/03`. Read in order:

| # | Test | What it tried | Outcome |
|---|---|---|---|
| 1 | [`01_carrier_vs_transformer.md`](01_carrier_vs_transformer.md) | Remove the cheat: a learned left→right **carrier** (rule learned, not hard-coded) vs Transformer | Carrier **extrapolates in accuracy** (~99% at all lengths) where the Transformer collapses — but its **conservation drifts** (recurrence buys accuracy, not the invariant). |
| 2 | [`02_reversible_swap_ca.md`](02_reversible_swap_ca.md) | Weld the guarantees: a **reversible + conservative gated-swap CA** (conservation & reversibility exact by construction) | Guarantees **verified exact**, but the *local* swap structure **couldn't learn** BBS (accuracy stuck ~82%). Exposed a **guarantee-vs-expressivity tension**. |
| 3 | [`03_conserving_carrier.md`](03_conserving_carrier.md) | Unite both: keep the carrier's reach, constrain its per-site update to **conserve** (emit/hold) | **The result.** A learned model at **100% accuracy** (all lengths), **100% conservation**, **100% reversible** — tracks the cheat ceiling on both panels, no cheating (loss → 0.0001). |

## The punchline

| model | accuracy (OOD) | conservation | reversible | learned? |
|---|---|---|---|---|
| Transformer | collapses (~81%) | ~0% | no | yes |
| plain carrier | ~99% | drifts 76→39% | no | yes |
| reversible swap-CA | ~82% (trivial) | 100% | 100% | yes |
| **conserving carrier** | **100%** | **100%** | **100%** | **yes** |
| hard-coded integrable | 100% | 100% | 100% | no (cheat) |

The two failures were not detours — they located the answer. #1 showed recurrence buys accuracy but not the invariant; #2 showed a structural guarantee that can't express the rule is worthless; #3 kept the reach of #1 and the discipline of #2 (conserve at every site) and landed all three. The cheat-vs-learn and guarantee-vs-expressivity tensions collapse into a single learned line that tracks the ceiling.

## Where it stands / next

This is one seed, one task (BBS), and an architecture that closely matches BBS's structure — so the open question is now **generality**: does the same conserving-carrier recipe learn *other* reversible systems (Margolus CA, Toda), across seeds, with error bars? That is the `data/reversible_systems/` plan in [`../../proposal.md`](../../proposal.md), and the step that turns this from a clean demo into a publishable result.
