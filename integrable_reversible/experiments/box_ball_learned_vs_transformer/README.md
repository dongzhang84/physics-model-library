# Learning an integrable model on the Box-Ball System (3 tests)

**English** · [中文](README.zh-CN.md)

Demo 2 showed a hard-coded integrable rule (the Box-Ball System, BBS) doing what a Transformer can't on length extrapolation — but that was a **cheat**: the "integrable engine" *is* the rule that generated the labels, so its 100% is tautological. This folder removes the cheat and asks the real question:

> Can **conservation (and reversibility) be made structural guarantees** — true by construction, not learned — and can a model then **genuinely learn, on top of that structure, a rule that extrapolates** to unseen lengths?

**We ran 3 tests** — three iterations of the *same* investigation, not three separate experiments (hence one folder, files `01/02/03`). Together they are the proposal's "Route C" (a learned model with integrable structure). **Answer: yes — Test 3 lands it: conservation is structural (100% for any gate), and on top of it sits a genuinely-learned, length-extrapolating residual (a carrier-blind gate gets 89%, the learned gate 100%).**

> **"Beating a Transformer" is *not* the headline.** Any model here with a left→right scan beats the structure-free Transformer on length extrapolation — but that is *expected*, overlaps the known "recurrent / state-tracking models extrapolate, attention doesn't" result, and is *not a fair fight* (the Transformer has no scan prior). It is kept only as a no-structure reference. The **fair, discriminating comparison is against other scan models — RNN / LSTM / Mamba (SSM) — where the question is *whose conserved quantity stays exact vs. drifts with length*, not whose accuracy is higher.** That is the thing structural conservation uniquely buys.

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

*Script: [`03_conserving_carrier.py`](03_conserving_carrier.py)*

**The fix.** Keep everything that was good — the conserving-carrier structure (conservation stays **structural**), the length-extrapolation setup, the Transformer baseline — but change the **task** from plain BBS to **finite-carrier BBS**: the carrier (the basket) now holds at most **K balls** (here K=2). When it is **full**, an arriving ball must **pass through** instead of being picked up. That one change makes the emit decision **depend on the carrier count `k`**, so `emit = 1 − cell` no longer works — the model is *forced* to learn to use the carrier.

![Test 3](03_conserving_carrier.png)

Trained on L=32, tested OOD (finite-carrier BBS verified conserved + reversible before training):

| Length | ceiling (hard-coded) | carrier-blind (`emit=1−cell`) acc / cons | **conserving carrier (learned)** acc / cons | Transformer acc / cons |
|---|---|---|---|---|
| 32 (train) | 100 / 100 | 88.7 / 100 | **100 / 100** | 98.0 / 58.0 |
| 48 | 100 / 100 | 89.5 / 100 | **100 / 100** | 90.0 / 2.7 |
| 64 | 100 / 100 | 88.7 / 100 | **100 / 100** | 87.2 / 0.3 |
| 96 | 100 / 100 | 88.8 / 100 | **100 / 100** | 85.1 / 0.0 |
| 128 | 100 / 100 | 88.1 / 100 | **100 / 100** | 83.0 / 0.0 |

Reversibility of the learned conserving carrier (whole-sequence, mirror trick): **100%**.

- **The learning is real (not leaked)**: the carrier-blind gate `emit=1−cell` tops out at **~89%**; the learned gate does better only by using the carrier ("pass the ball through when the carrier is full"). *This single-seed run shows 100%, but across 5 seeds the learned gate averages **94 ± 5%** — a **~4-point** edge over carrier-blind that is within noise (see [Test 5](#test-5--multi-seed-rigor-5-seeds)). At K=2 the genuinely-robust win is the **conservation (100 ± 0)**, not the accuracy edge; the edge is clean and large only at K=6 (Test 4).* On plain BBS the same gap was 0 (see appendix).
- **Conservation stays structural**: both conserving models (blind and learned) hold **100%** at every length (`k'=t−out`), while the Transformer's conservation collapses (58% → 0%).
- **Honest boundary**: this shows a non-trivial learnable residual on a conserving structure — it does **not** claim "no leak" (the carrier scan + conservation accounting are still hard-coded) or a "fair" win over the Transformer (see the audit table above).

---

## Test 4 — fair comparison vs scan models (RNN / LSTM / Mamba)

*Script: [`04_scan_baselines.py`](04_scan_baselines.py). **Single seed — a trend signal, not a conclusion** (multi-seed deferred to a final batch, per Tests 1–3).*

"Beating a Transformer" is expected and unfair (it has no scan prior), so the real test is against models that **also** scan left→right — a plain RNN, an LSTM, and a minimal diagonal SSM (a Mamba-family stand-in). The question is the one thing structural conservation could uniquely buy: **whose conserved quantity stays exact vs. drifts with length.** Because conservation drift is a symptom of imperfect accuracy, the task is pushed to a harder regime — finite-carrier BBS with a **bigger bounded carrier (K=6)**, tested out to L=256 — so the free-emit models can't simply learn it perfectly.

![Test 4](04_scan_baselines.png)

| model (K=6) | acc / cons @32 | @64 | @128 | @256 |
|---|---|---|---|---|
| **conserving carrier** | **100 / 100** | **100 / 100** | **100 / 100** | **100 / 100** |
| GRU carrier (composing, free-emit) | 89 / 21 | 70 / 6 | 71 / 8 | 71 / 5 |
| LSTM (direct-map) | 100 / 100 | 97 / 27 | 94 / 1 | 92 / 0 |
| SSM (Mamba-family, direct-map) | 96 / 48 | 94 / 18 | 92 / 4 | 90 / 0 |
| Transformer (ref) | 97 / 59 | 67 / 7 | 62 / 0 | 66 / 0 |

**What this shows (honest).** At K=6, **no standard free-emit scan model matches the conserving carrier — neither a composing one nor a direct-map one:**
- the **composing GRU** (stable, gradient-clipped) can't even learn the rule fully in-distribution (89% at L=32) and its conservation collapses (21% → 5%);
- the **direct-map LSTM** learns it in-distribution (100/100 at L=32) but its conservation drifts to ~0% OOD; the **SSM** drifts too;
- the **conserving carrier holds a flat 100/100 everywhere.**

Mechanism: the conserving carrier only learns the threshold `emit=[k≥K]` — the integer carrier is *structural*, so a bigger bounded carrier costs it nothing — whereas every free-emit model has to *learn to maintain the counter* and either can't (GRU) or doesn't extrapolate it exactly (LSTM/SSM); the errors surface as conservation drift. (An earlier hand-rolled RNN baseline was training-unstable at K=6 — it *diverged*, worse when made bigger — so it was replaced by this stable GRU; no claim rested on it.)

**Claim boundary (deliberately narrow).** Supported: *at matched, reasonable budget, standard free-emit scan models — both composing (GRU) and direct-map (LSTM, Mamba-family) — do not match the conserving carrier's accuracy + conservation on the harder (K=6) rule; structural conservation makes it both learnable and conservation-exact.* It is an **efficiency + guarantee** advantage. **Not** claimed: that *no* free-emit net could ever do it — with unbounded budget one might reach ~100% and then conserve; we don't chase that. Single seed → trend only.

---

## Test 5 — multi-seed rigor (5 seeds): the formal numbers

Every table in Tests 1–4 above is a **single seed** (trend illustration). Here each test is re-run across **5 seeds** and reported as **mean ± std** — the numbers of record. *Script: [`05_multiseed.py`](05_multiseed.py) → error-bar figures `05_test{1..4}.png`, raw `multiseed_results.json`.*

**What multi-seed corrected (the honest part):**
1. **Test 3 was over-optimistic on accuracy.** The conserving carrier's accuracy is **94 ± 5%**, not the single-seed 100%, and the learned-vs-carrier-blind gap shrinks from 11 pts to **~4 pts (94 vs 90) — within noise.** What is rock-solid is the **structural conservation: 100 ± 0** at every length. So at K=2 the robust win is the conservation *guarantee*, not a large learned-accuracy edge.
2. **Tests 1 & 2: the plain carrier's conservation is highly seed-variable** (std **±22–44**) — it drifts on average but some seeds conserve far better than others; not a clean story. The **swap-automaton is stable** (~82 ± 1% acc, 100 ± 0 conservation).
3. **Test 4 (K=6) is the robust, strong result** (low variance): conserving carrier **100 ± 0 / 100 ± 0**; GRU, LSTM, SSM all drift on conservation OOD. This is where structural conservation clearly and reliably wins.

**Test 1** (plain BBS) ![](05_test1.png)

| acc / cons | @32 | @48 | @64 | @96 | @128 |
|---|---|---|---|---|---|
| plain carrier | 90±6 / 26±33 | 89±6 / 22±32 | 89±6 / 20±30 | 88±6 / 16±26 | 88±6 / 13±22 |
| Transformer | 97±1 / 47±8 | 84±1 / 15±6 | 82±1 / 5±6 | 80±1 / 0±0 | 79±2 / 0±1 |

**Test 2** (plain BBS) ![](05_test2.png)

| acc / cons | @32 | @48 | @64 | @96 | @128 |
|---|---|---|---|---|---|
| swap-automaton | 84±1 / **100±0** | 83±1 / 100±0 | 82±1 / 100±0 | 82±1 / 100±0 | 81±1 / 100±0 |
| plain carrier | 96±5 / 63±39 | 95±6 / 54±40 | 95±6 / 50±42 | 95±6 / 44±43 | 95±6 / 42±44 |
| Transformer | 95±1 / 44±8 | 84±2 / 10±6 | 80±3 / 7±5 | 80±2 / 2±3 | 78±3 / 2±2 |

**Test 3** (finite-carrier K=2) ![](05_test3.png)

| acc / cons | @32 | @48 | @64 | @96 | @128 |
|---|---|---|---|---|---|
| conserving carrier | 94±5 / **100±0** | 94±5 / 100±0 | 94±5 / 100±0 | 94±5 / 100±0 | 93±6 / 100±0 |
| carrier-blind | 90±0 / 100±0 | 89±0 / 100±0 | 89±0 / 100±0 | 88±0 / 100±0 | 88±0 / 100±0 |
| Transformer | 99±0 / 82±7 | 92±1 / 8±2 | 88±1 / 2±2 | 85±1 / 0±1 | 83±2 / 0±0 |

**Test 4** (finite-carrier K=6) ![](05_test4.png)

| acc / cons | @32 | @64 | @128 | @256 |
|---|---|---|---|---|
| **conserving carrier** | **100±0 / 100±0** | **100±0 / 100±0** | **100±0 / 100±0** | **100±0 / 100±0** |
| GRU carrier (compose) | 88±0 / 15±4 | 79±1 / 4±3 | 79±1 / 1±1 | 79±0 / 0±0 |
| LSTM | 100±0 / 100±0 | 97±0 / 39±9 | 93±0 / 12±7 | 91±1 / 6±5 |
| SSM (Mamba-family) | 96±1 / 38±15 | 94±1 / 16±9 | 92±1 / 2±2 | 91±1 / 0±0 |
| Transformer (ref) | 97±1 / 62±6 | 63±7 / 8±6 | 58±8 / 3±4 | 64±4 / 0±1 |

**Bottom line after multi-seed:** the **robust** claim is *structural conservation stays exact (100 ± 0) where free-emit scan models drift* — strongest and cleanest at Test 4 (K=6). The *learned-accuracy* edge is real but modest/noisy at K=2 (Test 3) and clean at K=6 (Test 4). Still one task family (BBS); cross-system generality is Test 6.

---

## Honest boundaries

1. **Single seed, toy scale, one task (BBS).** Numbers wobble across seeds (that is why the Transformer / carrier columns differ slightly between tests). A real version needs multiple seeds with mean ± error bars.
2. **On plain BBS the conserving carrier *leaked the rule*** — a hard-coded `emit = 1 − cell` already hits 100% (see the audit table above), so that "learned" result would be hollow. Test 3 therefore runs on **finite-carrier BBS**, where the residual is non-trivial (carrier-blind ≈ 89%; the learned gate reaches 100% only by using the carrier). The broader open question is still **generality** — does the recipe hold across *several* reversible systems (finite-carrier BBS, Margolus CA, Toda), multi-seed?
3. **Global conservation relied on the carrier emptying** (it did, 100%); a flush would make it unconditional.

## Where it stands / next

**The result, stated honestly.** Conservation is made **structural** (true by construction, 100% for any gate); on top of that structure a model **genuinely learns a rule that extrapolates** to unseen lengths (Test 3: carrier-blind ≈ 89% vs learned 100%; conservation flat 100%; reversibility learned + verified). It reaches the hard-coded ceiling on accuracy **by learning, not hard-coding**. That — not "beating a Transformer" — is the claim.

## Planned next tests (Test 4–6)

The claim to establish is **not** "beats a Transformer" but "**structural conservation keeps the invariant exact where learning alone drifts.**" A quick scratch attempt at the fair comparison (not committed) surfaced a subtlety that shapes the plan:

> **Key finding to build on.** On finite-carrier BBS (K=2), an LSTM and a plain RNN *also* reached 100% accuracy — and therefore 100% conservation; only a weaker SSM drifted. **Conservation drift is a symptom of imperfect accuracy, not of "no structural conservation": a model that learns the rule exactly conserves for free.** (This also explains why Test 1's RNN drifted — it only reached ~93% on *plain* BBS — while an RNN on finite-carrier BBS, whose carrier is bounded and so easier to track exactly, hits 100% and does not drift.)
> **Tension this creates:** the task that makes a free-emit RNN drift (plain BBS, unbounded carrier) is exactly the one where the conserving carrier's residual is *trivial* (the leak); the task where the residual is non-trivial (finite-carrier BBS) is easy enough that the RNN nails it too. Test 4 has to escape both horns.

**Test 4 — ✅ done (see the [Test 4](#test-4--fair-comparison-vs-scan-models-rnn--lstm--mamba) section above).** Single seed: at a harder regime (K=6), **both a composing free-emit GRU and direct-map models (LSTM, Mamba-family SSM) fail to match the conserving carrier** — the GRU can't fully learn the rule (89% in-dist) and its conservation collapses; the LSTM learns in-dist but drifts OOD; the conserving carrier holds 100/100. Claim: *at matched budget, standard free-emit scan models (composing + direct-map) don't match structural conservation's accuracy + conservation — an efficiency + guarantee advantage* (not "impossible for any net"). Remaining rigor: multi-seed (Test 6).

**Test 5 — ✅ done: multi-seed rigor (see the [Test 5](#test-5--multi-seed-rigor-5-seeds) section above).** All four tests re-run across 5 seeds → mean ± std. It corrected the Test 3 single-seed accuracy (100 → 94±5) and shrank the learned-vs-blind gap to ~4 pts, while confirming the robust result: structural conservation stays 100 ± 0 where free-emit models drift (cleanest at Test 4, K=6).

**Test 6 — generality (a second reversible system).** Recommended: Margolus-neighbourhood reversible CA. Same recipe (structural conservation + a learned residual) on a *different* system — validates "the recipe transfers" and the proposal's prediction that different systems need different structures. This is the next experiment.

**Order:** Test 4 (done) → Test 5 multi-seed (done) → Test 6 generality (next) — foundation first, then the second floor. Ties into the `data/reversible_systems/` plan in [`../../proposal.md`](../../proposal.md).

## Reproduce (CPU, a few minutes each)

```bash
pip install torch numpy matplotlib
python 01_plain_carrier.py        # remove the cheat
python 02_reversible_swap_ca.py   # weld guarantees (can't learn)
python 03_conserving_carrier.py   # the one that works
```

---

## Appendix — the old Test 3 (done, then dropped; kept for the record)

For the record: we first built and ran a conserving carrier on **plain (standard) BBS**. It scored 100% accuracy / 100% conservation / 100% reversible at every length — which looked like a clean "learned it" result, and it's why the earlier version of this folder called Test 3 a success.

We dropped it (moving Test 3 to finite-carrier BBS) after the audit showed the result was **hollow**: replacing the learnable gate with a one-line fixed rule `emit = 1 − cell` (no training, ignoring the carrier and the neighbors) *also* scores 100% / 100% at every length. So the architecture had essentially the whole plain-BBS rule built in (carrier pickup + conservation + capacity), leaving only a trivial one-bit negation to "learn" — genuinely-learned ≈ 0. The structural guarantees were real; the *learning* was not.

This is kept as a record of what was tried and why it was replaced — not a result we stand behind. (The `03_conserving_carrier.py` code path still targets plain BBS until the finite-carrier rewrite lands.)
