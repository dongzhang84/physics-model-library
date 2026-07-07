# Integrable Extrapolation — a benchmark *attempt* (not yet valid — the current result leaks)

**English** · (中文版待补)

> **This is NOT a working benchmark yet — do not call it one.** As a demonstration of the paper's
> real goal ("a genuinely *learned* integrable model wins"), it **fails**: a leak audit shows the
> structural winner is **hardcoded**, not learned (carrier-blind also scores 100). What *is* real:
> generic sequence models and a bolt-on fail to preserve soliton content — necessity of the
> structure, not learned success. The name "benchmark" is earned only once a genuine-learning
> (finite-carrier) entrant discriminates without a leak. Single seed; no spec; no multi-seed.

## What this benchmark is for

The paper's thesis is that **integrability is a method that makes sense** — on the right task, a
model that builds in the integrable structure beats generic sequence models, and the advantage is
one that a *bolted-on constraint cannot replicate*. This benchmark is meant to make that concrete
and measurable.

Lesson banked from the [Toda experiment](../experiments/toda_lattice/) (a negative result): the
discriminator only exists on **discrete** integrable systems, where the exact rule is hard to nail
and errors compound under composition — on a smooth continuous flow a free-form model just solves
it. So the benchmark lives on discrete systems.

## Flagship system: the multi-soliton Box-Ball System (`soliton_bbs.py`)

A block of `k` consecutive 1s is a **soliton of amplitude `k`**; it moves right at speed `k`, larger
solitons overtake and pass through smaller ones, each keeping its amplitude. The **multiset of
soliton amplitudes is an exact invariant** — BBS's conserved quantities (as many as there are
solitons: the integrable signature, not one conservation law). `soliton_bbs.py` provides exact
(vectorised) dynamics, `soliton_content` (extract the invariant), a generator, and a **passing
self-check** (content conserved incl. the textbook {3,1} overtake; reversible).

## Task and metric

Learn **one** BBS step, then **compose** it over a long horizon on multi-soliton states (many
collisions); train on few solitons / short, test on more solitons / long. The discriminating metric
is **soliton-content fidelity**: does the predicted state carry the right multiset of soliton
amplitudes, not just the right ball count?

## Current result (`run_benchmark.py`, single seed, compose T=12)

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

## ⚠️ Honest caveat — the structural winner is hardcoded, not learned

The `carrier-blind` row is a **leak audit**: it replaces the learnable emit gate with a fixed
`emit = 1 − cell` (no training). It **also scores 100/100/100/100.** So on *plain* BBS the structural
model's win comes entirely from the **carrier bookkeeping structure**, not from learning — it is
essentially the hardcoded BBS solver. (This is the same leak that pushed the
[box-ball experiment](../experiments/box_ball_learned_vs_transformer/) from plain BBS to
finite-carrier BBS.) A new metric (soliton content) does **not** fix a system-level leak.

So this leaderboard currently supports **"integrable structure as an architectural prior is
necessary and generic models can't replicate it"** — it does **not** yet support "a genuinely
*learned* integrable model wins." Those are different claims; only the weaker one is demonstrated here.

## What is done / not done

- ✅ Flagship system + verified conserved invariant (`soliton_bbs.py`).
- ✅ Metrics + one single-seed leaderboard (`run_benchmark.py`), with a leak audit.
- ❌ Benchmark spec (formal splits/format), multi-seed, unified README across systems.
- ❌ A **genuine-learning** entrant with no leak — needs **finite-carrier BBS**, where the emit gate
  is non-trivial (carrier-blind ≈ 89%, learning matters), already shown to discriminate in the
  box-ball experiment (Test 4/5).

## Reproduce (CPU)

```
python3 soliton_bbs.py     # system self-check: soliton content conserved + reversible
python3 run_benchmark.py   # single-seed leaderboard + leak audit
```
