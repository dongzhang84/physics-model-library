# Validated result — Tier 1: finite-carrier Box-Ball System

## Scope (read first)

- This is **one validated benchmark tier**: a single discrete integrable system (finite-carrier
  BBS, capacity K=3), confirmed over 5 seeds. It is **not** the full benchmark family; the
  multi-system plan (Phase A–E in `BENCHMARK_PROPOSAL.md`) is not built. Do not describe this as a
  completed benchmark.
- The claim is scoped to **discrete integrable dynamics**. The Toda experiment
  ([`../experiments/toda_lattice/`](../experiments/toda_lattice/)) showed the effect does not
  transfer to smooth continuous flow, so this is **not** a claim about integrability in general.

## Task

Finite-carrier BBS: a carrier of capacity K sweeps left→right; a ball is picked up if the carrier
has room, otherwise it passes through; an empty cell receives a drop if the carrier holds a ball.
Models learn one step and **compose** it over a long multi-soliton horizon (train on few solitons,
test on more — many collisions). The metric is **soliton-amplitude content**: the multiset of
soliton amplitudes, a conserved invariant that is richer than the ball count (two states with the
same ball count can have different amplitude multisets, e.g. {5,3,1} and {4,4,1}).

## Claim this tier establishes

On a discrete integrable system, a model that builds in the integrable structure (a carrier with
structural ball-count conservation) learns to preserve the soliton-amplitude content under long
composition; generic sequence models and a conservation bolt-on do not — and the advantage is
**genuinely learned, not hard-coded** (the leak audit below).

## Leaderboard

Finite-carrier BBS, K=3, compose T=8, 5 seeds, mean ± std. Metrics: per-cell accuracy;
ball-count conservation; soliton-amplitude content exact-match and IoU.

| entrant | acc % | ball-count % | amp-content exact % | amp-content IoU % |
|---|---|---|---|---|
| **conserving carrier (structural)** | **100.0 ± 0.0** | **100.0 ± 0.0** | **100.0 ± 0.0** | **100.0 ± 0.0** |
| GRU (free-form) | 91.7 ± 1.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 5.9 ± 11.2 |
| LSTM (free-form) | 92.7 ± 1.3 | 1.7 ± 3.3 | 0.0 ± 0.0 | 13.5 ± 11.6 |
| Transformer (free-form) | 71.6 ± 10.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 20.3 ± 3.6 |
| bolt-on (GRU + ball-count pinned) | 89.2 ± 2.5 | 100.0 ± 0.0 | 0.7 ± 0.8 | 21.8 ± 5.3 |
| carrier-blind (leak audit) | 88.9 ± 0.5 | 100.0 ± 0.0 | 40.0 ± 3.2 | 57.9 ± 2.3 |

## Two audits (the honesty guardrails)

- **Leak audit — carrier-blind.** Replace the learned emit gate with a fixed `emit = 1 − cell` rule
  (no training). It reaches only 40.0 ± 3.2 % amp-content exact, vs the learned model's 100 % — a
  **60-point gap**. So the advantage is genuinely learned; this is not the plain-BBS leak, where a
  carrier-blind rule reaches 100 %.
- **Bolt-on — ball-count pinned.** Force the free-form model's output to carry the input's ball
  count (top-N projection). Ball count goes to 100 %, but amp-content stays at 0.7 ± 0.8 %.
  Conservation is bolt-on-able; the amplitude content is not.

## Reading

- The structural model preserves accuracy, ball count, and amplitude content exactly (100 ± 0 on
  all four), across 5 seeds.
- Free-form models drift: they keep no ball count and 0 % amplitude content under composition.
- The bolt-on holds ball count at 100 % but recovers ~0 % amplitude content — pinning the scalar
  invariant does not recover the richer one.
- The leak audit isolates the learned contribution: the gate is worth ~60 points of amplitude
  content over the fixed rule.

## Boundaries (explicit)

1. **Validated Tier 1 only** — one system (finite-carrier BBS), one capacity (K=3), 5 seeds. This
   is a validated tier, not the full benchmark family; the benchmark is not "built."
2. **Discrete integrable only** — the Toda experiment showed this discriminator does not transfer
   to smooth continuous dynamics. No claim about integrability in general.
3. **Structure given, gate learned** — the carrier scan and structural conservation are an
   inductive bias supplied to the model; the learned part is the emit gate, whose contribution
   (≈60 points here) is what the leak audit measures. This is "integrable structure makes the rule
   learnable-to-exact," not "the integrable structure was learned from scratch."

## Reproduce

```
python3 fc_bbs.py             # ground truth: amplitude content conserved + reversible (K=2/4/6)
python3 fc_gate_multiseed.py  # 5-seed leaderboard (resumable; writes fc_gate_multiseed.json)
```
