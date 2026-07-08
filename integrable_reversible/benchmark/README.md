# Integrable Extrapolation benchmark — Tier 1 validated; full family in progress

**English** · (中文版待补)

> **Status.** Tier 1 (finite-carrier BBS + soliton-amplitude content) is a **validated** benchmark
> tier: 5-seed confirmation that a learned integrable-structured model preserves an invariant that
> generic sequence models and a conservation bolt-on cannot, and the advantage is genuine
> (leak-audited), on a **discrete** integrable system. Full write-up:
> **[`RESULT_tier1_finite_carrier.md`](RESULT_tier1_finite_carrier.md)**.
> The full multi-system benchmark (`BENCHMARK_PROPOSAL.md`) is **not** built. An earlier attempt on
> plain-BBS multi-soliton is kept below as history — it leaked.

## Tier 1 (validated) — finite-carrier BBS + soliton-amplitude content

Full write-up and boundaries: [`RESULT_tier1_finite_carrier.md`](RESULT_tier1_finite_carrier.md).
Headline (K=3, compose T=8, 5 seeds, mean ± std):

- structural conserving carrier: **100 ± 0** on accuracy / ball-count / amplitude-content;
- carrier-blind **leak audit**: only **40 ± 3** amplitude-content — a 60-point learned margin, so
  the advantage is genuine (not the plain-BBS leak);
- **bolt-on** (ball-count pinned): 100 % ball count but **0.7 ± 0.8 %** amplitude content —
  conservation is bolt-on-able, the amplitude multiset is not;
- free-form GRU / LSTM / Transformer: 0 % amplitude content.

Reproduce: `python3 fc_bbs.py`, `python3 fc_gate_multiseed.py`. Scope: one discrete integrable
system, K=3 — not the full family; discrete only (Toda showed the effect does not transfer to
continuous flow).

---

## Earlier attempt (superseded, kept as history) — plain-BBS multi-soliton, which leaked

> **This earlier attempt is NOT a valid benchmark**; it is retained only as the record of why the
> flagship moved to finite-carrier. On *plain* BBS the structural winner is hard-coded (its leak
> audit scores 100 too), so it does not show learned success. Tier 1 above supersedes it.

### What this attempt was for

The paper's thesis is that **integrability is a method that makes sense** — on the right task, a
model that builds in the integrable structure beats generic sequence models, and the advantage is
one that a *bolted-on constraint cannot replicate*. This benchmark is meant to make that concrete
and measurable.

Lesson banked from the [Toda experiment](../experiments/toda_lattice/) (a negative result): the
discriminator only exists on **discrete** integrable systems, where the exact rule is hard to nail
and errors compound under composition — on a smooth continuous flow a free-form model just solves
it. So the benchmark lives on discrete systems.

### Earlier attempted system: the multi-soliton Box-Ball System (`soliton_bbs.py`)

A block of `k` consecutive 1s is a **soliton of amplitude `k`**; it moves right at speed `k`, larger
solitons overtake and pass through smaller ones, each keeping its amplitude. The **multiset of
soliton amplitudes is an exact invariant** — BBS's conserved quantities (as many as there are
solitons: the integrable signature, not one conservation law). `soliton_bbs.py` provides exact
(vectorised) dynamics, `soliton_content` (extract the invariant), a generator, and a **passing
self-check** (content conserved incl. the textbook {3,1} overtake; reversible).

### Task and metric (this attempt)

Learn **one** BBS step, then **compose** it over a long horizon on multi-soliton states (many
collisions); train on few solitons / short, test on more solitons / long. The discriminating metric
is **soliton-content fidelity**: does the predicted state carry the right multiset of soliton
amplitudes, not just the right ball count?

### This attempt's result (`run_benchmark.py`, single seed, compose T=12) — leaked

| entrant | acc % | ball-cons % | soliton-exact % | soliton-IoU % |
|---|---|---|---|---|
| conserving carrier (structural) | 100 | 100 | **100** | **100** |
| &nbsp;&nbsp;└ carrier-blind (leak audit) | 100 | 100 | **100** | **100** |
| GRU (free-form) | 92 | 0 | 0 | 17 |
| LSTM (free-form) | 93 | 0 | 0 | 1 |
| Transformer (free-form) | 88 | 3 | 0 | 33 |
| bolt-on = GRU + ball-count pinned | 89 | **100** | 2.5 | 33 |

**What is real here:** generic sequence models (GRU / LSTM / Transformer) **fail to preserve the
soliton content** over many collisions, and — the money row — the **bolt-on holds ball count at 100%
but soliton content at ~2.5%**. So on this task **conservation (ball count) is bolt-on-able, but the
soliton content (integrability) is not**; only the carrier-structured model keeps it.

### Why it leaked — the structural winner is hardcoded, not learned

The `carrier-blind` row is a **leak audit**: it replaces the learnable emit gate with a fixed
`emit = 1 − cell` (no training). It **also scores 100/100/100/100.** So on *plain* BBS the structural
model's win comes entirely from the **carrier bookkeeping structure**, not from learning — it is
essentially the hardcoded BBS solver. (This is the same leak that pushed the
[box-ball experiment](../experiments/box_ball_learned_vs_transformer/) from plain BBS to
finite-carrier BBS.) A new metric (soliton content) does **not** fix a system-level leak.

So this leaderboard currently supports **"integrable structure as an architectural prior is
necessary and generic models can't replicate it"** — it does **not** yet support "a genuinely
*learned* integrable model wins." Those are different claims; only the weaker one is demonstrated here.

### What this attempt did / did not give

- ✅ The attempt's system + verified conserved invariant (`soliton_bbs.py`).
- ✅ Metrics + one single-seed leaderboard (`run_benchmark.py`), with a leak audit.
- ❌ A **genuine-learning** entrant with no leak — this attempt did not have one; that is what the
  finite-carrier **Tier 1** above provides (carrier-blind 40 % ≠ structural 100 %, no leak).

### Reproduce (this attempt)

```
python3 soliton_bbs.py     # system self-check: soliton content conserved + reversible
python3 run_benchmark.py   # single-seed leaderboard + leak audit (leaked — see above)
```
