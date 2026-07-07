# Changelog

All notable changes to **physics-model-library** are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/). This is a
research project in its pre-paradigmatic phase — there are **no versioned
releases yet**, so entries are grouped by dated milestones (newest first).
Routine CI bookkeeping commits (`chore: update SPRINT.md`) are omitted.

Scope so far: the repo's thesis and mining map (root docs), plus the first mined
root-family — **integrable / reversible systems** — carried through the full
pipeline on the Box-Ball System and now extended to a second system (Margolus).

---

## [Unreleased]

### In progress — Experiment 2: Margolus block CA (a second reversible system) · 2026-07-06

Testing whether the "structural conservation + learned residual" recipe
**transfers** off the Box-Ball System to a structurally different reversible
system (a block/partitioning CA with no long-range carrier).

- **Added** `integrable_reversible/experiments/margolus_block_ca/`.
  - `01_margolus_system.py` — the 1D Margolus block CA (block size 3, partition
    offset cycles 0→1→2, count-preserving rotation φ). Ground-truth self-check:
    conservation ✅ and reversibility ✅ are structural; leak audit shows the
    rotation is a non-trivial residual (blind identity = 52.8%).
  - `02_probe.py` — single-seed probe. Because Margolus' one-step rule is local,
    the horizon grows with length (`T = L/2`) to recreate a length-generalization
    challenge. A fair **bidirectional** free-form GRU learns the single step well
    (98.6%) but not *exactly*; compounded over `T ∝ L` steps its accuracy falls to
    ~60% and conservation to ~1%, while the structural block-CA is exact per step
    and stays **100/100** at every horizon.
  - `README.md` — design, ground truth, probe result, honest boundary, next steps.
- **Finding (single seed):** the Box-Ball result transfers, and sharpens the
  thesis — structural conservation matters most precisely when composing many
  steps (the reversible-systems regime).
- **Next:** multi-seed rigor (5 seeds), 1–2 more fair baselines (bi-LSTM, small
  Transformer), figures, bilingual write-up; later a 2D classic Margolus.

---

## Experiment 1: Box-Ball System — learning an integrable model · 2026-07-02 → 2026-07-05

`integrable_reversible/experiments/box_ball_learned_vs_transformer/` — five
multi-seed tests asking whether an integrable-style structure can be *learned*
(not hard-coded) and buy something a plain sequence model can't. **Finalized.**

### Added
- **Test 1** — plain learned carrier: removes Demo 2's "cheat", but a free-emit
  carrier does not conserve.
- **Test 2** — reversible swap-automaton (Margolus-style gated swap): exact
  guarantees by construction, but hits an expressivity wall learning the rule
  (~82%, reported honestly as a negative result).
- **Test 3** — conserving carrier on **finite-carrier BBS**: conservation is
  structural (`k' = t − out`) and a genuinely non-trivial residual sits on top.
- **Test 4** — fair comparison vs scan models (composing GRU / LSTM / Mamba-SSM),
  with the Transformer kept only as a no-structure reference.
- **Test 5** — multi-seed rigor (5 seeds, mean ± std) over Tests 1–4, with
  error-bar figures and `multiseed_results.json`.
- Architecture diagrams (Transformer vs carrier; swap-automaton; conserving
  carrier) and an animated view of the L=32 data.
- Bilingual write-up (`README.md` + `README.zh-CN.md`).

### Changed
- Reframed the core claim away from "beats a Transformer" toward the honest,
  discriminating one: *structural conservation stays exact where free-emit scan
  models drift* — an efficiency + guarantee edge among scan models.
- Consolidated three Route-C iterations into one folder; renamed the cryptic
  `route_c_bbs` → `box_ball_learned_vs_transformer`; de-jargoned test titles;
  folded 8 per-test markdown files into 2 comprehensive READMEs.
- Test 4: replaced an unstable hand-rolled RNN with a stable gradient-clipped
  composing GRU, and narrowed the claim accordingly.
- README title corrected 3 → 5 tests; added an explicit **Conclusion** (best
  model, result table, mechanism, honest boundaries).
- Proposal synced to actual Route-C progress.

### Fixed
- **Rule leak in the original Test 3:** a hard-coded `emit = 1 − cell` already
  scored 100%, making the "learned" result vacuous. Redesigned onto finite-carrier
  BBS where the residual is non-trivial; the scrapped version is kept as an
  appendix for the record.
- Multi-seed run corrected single-seed over-claims (Test 3 accuracy 100 → 94±5;
  learned-vs-blind gap 11 → ~4 points).

---

## Demo 1: Soliton channel (KdV / Box-Ball) · 2026-07-01

`integrable_reversible/demos/` — the motivating demo.

- **Added** the real motivation write-up (attention is lossy; solitons aren't),
  a crisp native-resolution animated GIF of the soliton channel, and a
  non-expert README.
- **Changed** to keep the heavy `.mp4` local and git-ignored (GIF is the
  in-repo artifact).

---

## Restructure & project scaffold · 2026-06-30 → 2026-07-01

- **Changed** `demos_1` → the `integrable_reversible` research project, with its
  own README, visual directory tree, `proposal.md`, and sub-structure
  (`data/ demos/ eval/ experiments/ models/ paper/`).
- **Changed** to move integrable-specific content out of the root README into the
  project, keeping the root as the library-wide front door.
- **Fixed / reverted** notify-playbook CI so it skips cleanly when
  `PLAYBOOK_TOKEN` is unset; dropped placeholder READMEs in empty scaffold dirs.

---

## Initial commit · 2026-06-24

- **Added** the repository and its thesis ("physics is a systematically mineable
  library of candidate models for AI"):
  - English README front door; `MANIFESTO.md` and `CATALOG.md` (the mining map of
    physics root-families vs. AI capabilities and maturity).
  - `docs/物理模型作为下一代AI的方法论基础.md` — the 322-line methodology
    proposal (goals, research programme, 50-year framing).
  - `scripts/extract-sprint-summary.py` and sprint-sync CI wired to the
    indie-product-playbook.
