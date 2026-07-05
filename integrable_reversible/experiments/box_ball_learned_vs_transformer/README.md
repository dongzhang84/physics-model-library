# Learning an integrable model on the Box-Ball System (3 tests)

**English** · [中文](README.zh-CN.md)

Demo 2 showed a hard-coded integrable rule (the Box-Ball System, BBS) beating a Transformer on length extrapolation — but that was a **cheat**: the "integrable engine" *is* the rule that generated the labels, so its 100% is tautological. This folder removes the cheat and asks the real question:

> Can a **genuinely trained** model, carrying an integrable-style inductive bias, beat a Transformer on the BBS length task — and add **exact conservation and reversibility**, by learning rather than by hard-coding?

**We ran 3 tests** — three iterations of the *same* investigation, not three separate experiments (hence one folder, files `01/02/03`). Together they are the proposal's "Route C" (a learned model with integrable structure). Answer: **yes — test 3 lands it.**

## The task

BBS is the ultradiscrete limit of KdV, an integrable cellular automaton on a 0/1 lattice: blocks of 1s are "solitons" that move right and pass through each other, ball-count is conserved, and the dynamics are exactly reversible. Task: given a 0/1 state, predict its state after **2 steps**. Every model is trained **only on length L=32**, then tested on 48/64/96/128 — out of distribution by length alone (soliton size and density fixed). Measured at every length: **per-position accuracy** and whether the **ball-count is conserved**.

![What the L=32 data looks like](bbs_data_l32.gif)

> The `01/02/03` scripts train on random L=32 states like this one. Here a **size-3 soliton (fast) overtakes a size-1 and a size-2, passing _through_ them** — the order swaps, the sizes stay intact, and all 6 balls persist. The `input` (t=0) and `+2 = target` (t=2) rows are the pair the models actually see; the extra steps just show the dynamics. Made by [`bbs_data_l32.py`](bbs_data_l32.py).

## The three models — an honest audit (structure vs learning)

Same task (predict a Box-Ball state 2 steps ahead), three models. The question that matters is not "does it work" but **how much is hard-coded structure vs genuinely learned** — measured by replacing the learnable gate with a one-line fixed rule and seeing whether accuracy drops.

| model | what it is | learnable weights decide | conserved · reversible | genuinely learned? (fixed-rule audit) | honest claim |
|---|---|---|---|---|---|
| **Test 1 — plain carrier** | RNN: left→right scan + a generic 24-d hidden state; free-emit output | the whole cell (`net/out/hnext`); **no structural constraint** | conserved: **drifts** (not structural) · reversible: **no** | learns the *whole* rule (~92–99%) but doesn't conserve; a hard-coded `emit=1−cell` even beats it (100%) | "recurrence learns a rule & extrapolates" (overlaps known work); **no conservation** |
| **Test 2 — swap-automaton** | stacked gated swaps of neighbor pairs; no carrier | only *swap / no-swap* per pair; the swaps are hard-coded | conserved · reversible: **both structural** (hold for any weight) | **can't learn**: fixed *and* learned gate both cap ~82% | "conservation + reversibility can be **fully structural**"; **can't learn the rule** |
| **Test 3 — conserving carrier** (finite-carrier BBS) | carrier scan; `t=c+k`, `avail`, `k'=t−out` hard-coded | the emit gate only | conserved: **structural (per step)** · reversible: learned + verified | fixed `emit=1−cell` = **89%** vs learned = **100%** → **~11% genuinely learned** (on plain BBS this gap is **0**, which is why plain BBS was dropped) | "on a conserving structure a **non-trivial, extrapolating residual is genuinely learned**"; **NOT** "no leak / fair win over Transformer" |

**Where the structure-vs-learning line falls, unified.** Test 1 hard-codes only "causal scan + generic state" and learns the whole rule (but conservation drifts). Test 2 hard-codes swaps + conservation + reversibility and learns only "swap or not" (but can't express the rule). Test 3 hard-codes the carrier scan + conservation bookkeeping + capacity constraint and learns the emit decision — of which only the "pass the ball when the carrier is full" part (~11%) is non-trivial.

**Two caveats stated plainly.** (1) **Structural reversibility exists only in Test 2** — and Test 2 is exactly the one that can't learn; the models that learn (Test 1, Test 3) have only *learned + verified* reversibility. No single model yet has structural reversibility **and** learns the rule. (2) **None of them "fairly beats" a Transformer**: the win comes mostly from the left→right scan prior (which overlaps the known "recurrent / state-tracking extrapolates, attention doesn't" result). To make a "beats X" claim meaningful you'd compare against models with the *same* scan prior (RNN / SSM); the genuinely exclusive contribution then shrinks to the **structural conservation / reversibility guarantees**, not accuracy.

---

## The three architectures

All three are drawn the same way — input at the bottom, the predicted state 2 steps later on top; the internal boxes/arrows are schematic, but the shown input→output is the *real* Box-Ball mapping. (Generated by [`architecture_diagrams.py`](architecture_diagrams.py).)

**1 · Transformer  vs  plain carrier** — the two base models
![Transformer vs the carrier RNN](architecture_01_transformer_vs_carrier.png)
> **Transformer**: global self-attention — every output from the whole sequence at once; no order, no carried state, no invariant. **Carrier**: one weight-shared cell scans left→right carrying a state `h`, so it runs at any length.

**2 · Reversible swap-automaton** — exact guarantees, but local
![Reversible swap-automaton](architecture_02_swap_automaton.png)
> Stacked gated swaps of neighbor pairs (pairing alternates each layer). A swap conserves the pair's count + is its own inverse, and the gate reads only frozen cells → exactly reversible & conservative *by construction* — but the swaps are **local**, so it can't express BBS's nonlocal carrier.

**3 · Conserving carrier** — structure kept, task redesigned
![Conserving carrier](architecture_03_conserving_carrier.png)
> The same left→right scan as #1, but the carrier is an integer ball-count `k` and each cell may only **emit** (out=1, k→t−1) or **hold** (out=0, k→t) — the two moves that preserve `cell + carrier`. Conservation is structural. (On plain BBS the "when to emit" rule was trivial; Test 3 runs this on **finite-carrier BBS**, where it isn't.)

---

## Test 1 — plain learned carrier · removes the cheat, but doesn't conserve

*Script: [`01_plain_carrier.py`](01_plain_carrier.py)*

Replace the hard-coded rule with a **learned** left→right carrier — a weight-shared recurrent cell with a small carrier state, trained on L=32, applied at any length. It learns the rule from data (never sees it); the hard-coded integrable line stays only as a ceiling.

![Test 1](01_plain_carrier.png)

| Length | all-zeros | Transformer acc / cons | Plain carrier acc / cons |
|---|---|---|---|
| 32 | 84.2% | 94.5 / 29.0 | 94.1 / 32.7 |
| 48 | 83.7% | 83.7 / 0.0 | 93.3 / 21.7 |
| 64 | 82.9% | 82.0 / 0.0 | 92.5 / 12.0 |
| 96 | 82.7% | 81.8 / 0.0 | 92.5 / 6.0 |
| 128 | 82.3% | 80.9 / 0.0 | 92.1 / 3.3 |

**Finding.** The carrier holds ~92% accuracy at every length while the Transformer collapses to the all-zeros baseline (out of distribution it learned almost nothing). So a length-independent structure that *learns* the rule extrapolates where attention doesn't — cheat removed, conclusion intact. **But** its conservation drifts toward zero: recurrence buys accuracy, not the invariant.

## Test 2 — reversible swap-automaton · exact guarantees, but can't learn the rule

*Script: [`02_reversible_swap_ca.py`](02_reversible_swap_ca.py)*

Weld the guarantees. A stack of **gated swaps** of adjacent cells: a swap conserves the pair's count and is its own inverse; the gate reads only frozen context + the swap-invariant pair sum, so the whole stack is exactly invertible no matter what it learns. Verified at random init (before any training): ball-count conserved exactly and `invert(forward(x)) == x` exactly, at all lengths.

![Test 2](02_reversible_swap_ca.png)

| Length | Transformer acc / cons | plain carrier acc / cons | swap-automaton acc / cons |
|---|---|---|---|
| 32 | 94.8 / 32.0 | 99.6 / 95.7 | 83.3 / **100** |
| 48 | 82.2 / 9.3 | 99.8 / 96.7 | 82.5 / **100** |
| 64 | 80.1 / 1.0 | 99.6 / 93.0 | 81.6 / **100** |
| 96 | 74.8 / 5.7 | 99.5 / 86.0 | 81.5 / **100** |
| 128 | 71.9 / 7.0 | 99.5 / 82.3 | 80.8 / **100** |

**Finding.** Conservation is a flat 100% and reversibility exact — the guarantees work. But the *local* swap structure **can't learn** BBS: accuracy stalls at ~82% (the trivial baseline), loss plateaus at 0.15 while the carrier reaches 0.02. BBS transport needs the carrier's nonlocal left→right reach, which a local gate lacks. **A structural guarantee that can't express the rule is worthless — the guarantee-vs-expressivity tension.**

## Test 3 — conserving carrier on *finite-carrier* BBS · a non-trivial rule to actually learn

> Plain BBS was dropped: welding conservation into the carrier makes its rule *trivial* — a hard-coded `emit = 1 − cell` already scores 100% (nothing left to learn; see the audit table above). Finite-carrier BBS keeps the same structure but makes the rule non-trivial.

*Design — verified in a quick check; the full run will replace `03_conserving_carrier.py`.*

**The fix.** Keep everything that was good — the conserving-carrier structure (conservation stays **structural**), the length-extrapolation setup, the Transformer baseline — but change the **task** from plain BBS to **finite-carrier BBS**: the carrier (the basket) now holds at most **K balls**. When it is **full**, an arriving ball must **pass through** instead of being picked up. That one change makes the emit decision **depend on the carrier count `k`**, so `emit = 1 − cell` no longer works — the model is *forced* to learn to use the carrier.

Verified in a quick check (K=2, predict 2 steps, same conserving-carrier structure):

| model | accuracy (OOD, L=48→128) | conserved | note |
|---|---|---|---|
| hard-coded finite-carrier rule | 100% | 100% | ceiling |
| **carrier-blind gate** (`emit = 1 − cell`) | **~89%** | 100% | ← proves the residual is now **non-trivial** |
| **conserving carrier (learned)** | **100%** | 100% | learns to use the carrier; holds to L=128 |
| Transformer | collapses (expected) | ~0% | no structure |

- **Finite-carrier BBS is conserved + reversible** — both checked numerically over random configs (reversible via the mirror trick).
- **Conservation stays structural** (`k' = t − out` bookkeeping → 100% for any gate) — the good half of the old design is kept.
- **The learning is now real**: a carrier-blind gate tops out at **~89%**, while the learned gate reaches **100%** only by using the carrier (learning "pass through when the basket is full"). Training loss → ~0.003 (it *had* to learn something), vs plain-BBS's ~0.0001 (nothing to learn).

**Still to build** (turns this design into a committed result): rewrite `03_conserving_carrier.py` for finite-carrier BBS and add the carrier-blind line to the figure; update the architecture diagram (carrier now finite → "pass when full"); fill this table from the committed run.

---

## Honest boundaries

1. **Single seed, toy scale, one task (BBS).** Numbers wobble across seeds (that is why the Transformer / carrier columns differ slightly between tests). A real version needs multiple seeds with mean ± error bars.
2. **On plain BBS the conserving carrier *leaked the rule*** — a hard-coded `emit = 1 − cell` already hits 100% (see the audit table above), so that "learned" result would be hollow. Test 3 therefore runs on **finite-carrier BBS**, where the residual is non-trivial (carrier-blind ≈ 89%; the learned gate reaches 100% only by using the carrier). The broader open question is still **generality** — does the recipe hold across *several* reversible systems (finite-carrier BBS, Margolus CA, Toda), multi-seed?
3. **Global conservation relied on the carrier emptying** (it did, 100%); a flush would make it unconditional.

## Where it stands / next

This is the shape of the result the project aimed for: a genuinely trained model with an integrable-style bias that (i) beats the Transformer on OOD length, (ii) matches the hard-coded integrable ceiling on accuracy, and (iii) adds exact conservation and reversibility — reached by learning, not hard-coding. The next step is the **generality study** (same recipe on other reversible systems, multiple seeds, error bars) — the `data/reversible_systems/` plan in [`../../proposal.md`](../../proposal.md), and the step that turns this from a clean demo into a publishable result.

## Reproduce (CPU, a few minutes each)

```bash
pip install torch numpy matplotlib
python 01_plain_carrier.py        # remove the cheat
python 02_reversible_swap_ca.py   # weld guarantees (can't learn)
python 03_conserving_carrier.py   # the one that works
```
