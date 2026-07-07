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

### Benchmark: Integrable Extrapolation — flagship system + first leaderboard (single seed) · 2026-07-07

The paper's intended centrepiece: a benchmark showing integrable structure is the
method that makes sense. **Partial and with an important honest caveat.**

- **Added** `integrable_reversible/benchmark/soliton_bbs.py` — flagship system:
  multi-soliton Box-Ball System. Exact vectorised dynamics (Skorokhod-reflection
  closed form), `soliton_content` (the conserved multiset of soliton amplitudes —
  the integrable signature), generator, and a passing self-check (soliton content
  is an exact invariant; reversible).
- **Added** `run_benchmark.py` — one single-seed leaderboard. Result *does*
  discriminate: over 12 composed steps (many collisions), a carrier-structured model
  preserves soliton content (100/100/100/100) while GRU/LSTM/Transformer collapse
  (soliton-IoU 1–33%, exact 0%), and the bolt-on (GRU + ball-count pinned) holds
  ball count at 100% but soliton content at ~2.5% — **conservation is bolt-on-able,
  soliton content is not.**
- **⚠️ Honest caveat (leak audit):** on *plain* BBS the structural model = the
  hardcoded BBS solver (carrier + emit=1−cell), so a carrier-blind entrant also
  scores 100 — the win is from the **structure, not learning** (the same leak that
  moved the box-ball experiment to finite-carrier). So this currently shows
  "integrable structure as an architectural prior is necessary; generic models
  can't replicate it," **not** "a genuinely *learned* integrable model." Not yet
  done: benchmark spec / README, multi-seed, and a genuine-learning (finite-carrier)
  entrant.

### Exp 3: Toda lattice — a scoped negative result (two attempts, both F2) · 2026-07-07

The real target (make a *genuinely* integrable system a learnable model that beats
every bolted-on constraint). Pre-registered before coding; **the honest outcome is
negative, and it is recorded, not used for the paper.**

- **Added** `PREREGISTRATION.md` + `PREREGISTRATION_v2_neural_lax.md` (criteria set
  where they can't be bolted on), `OBSTACLES.md` (every step logged), and a
  bilingual-pending `README.md` writing up the result.
- **Step 1** (`01_toda_system.py`): symplectic integrator + Lax matrix; integrability
  confirmed (energy ~1e-5, momentum ~1e-14, **isospectral ~1e-6**).
- **Attempt 1 — action-angle** (`02_probe.py`): F2. Learnable action-angle model
  (actions learned *or* given true) does not extrapolate and does not beat the
  strongest bolt-on. Blocker = R2 (frequencies + inverse-spectral decoder).
- **Attempt 2 — Neural-Lax** (`03_neural_lax.py`): F2′ (technical failure) **and** the
  real finding — a plain free-form composing stepper already *solves* continuous-Toda
  time extrapolation (MSE ~0.003), so there is no gap for structure. Confirms the
  roadmap's **discrete→continuous** concern: the discriminator that works on discrete
  BBS/Margolus does not transfer to a smooth continuous flow.
- **Stopped** iterating architectures per the pre-registration (no bottomless pit).

---

### Reframe: exactness is the crux, conservation is a bolt-on-able by-product · 2026-07-07

A **bolt-on control** — force conservation onto a free-form model (top-N
projection) and ask whether accuracy recovers — settled the old "anyone can add a
conservation constraint to an LSTM" objection: yes they can, and it does **not**
rescue accuracy.

- **Added** the control to **Margolus** (`03_multiseed.py`, `03_bolt_on.png`):
  conservation forced to 100% for GRU / LSTM / Transformer, yet accuracy still
  collapses to ~51–56% at L=384 vs the structural model's 100%.
- **Added** the same control to **Box-Ball** (`05_multiseed.py`, `05_bolt_on.png`):
  conservation forced to 100%, accuracy essentially unchanged (79–92%) and still
  below the conserving carrier's 100% — a **modest 9–21 pt gap** here (Test 4
  composes only T=2, far milder than Margolus's T∝L collapse). Direct confirmation
  on BBS, not transferred.
- **Changed** the whole repo's headline accordingly (both experiments' READMEs, EN +
  zh, and proposal §3/§7): from "structural conservation stays exact where free-form
  drifts" to "**structural *exactness* composes without drift — even bolting
  conservation onto the opponent can't catch up**." Conservation is a by-product;
  exactness is the part that can't be bolted on.
- **Established** the bolt-on control as a **standard** for every deep-water
  experiment going forward (proposal §7).

---

### Experiment 2: Margolus block CA (a second reversible system) — 1D done · 2026-07-06

Tested whether the "structural conservation + learned residual" recipe
**transfers** off the Box-Ball System to a structurally different reversible
system (a block/partitioning CA with no long-range carrier). It does.

- **Added** `integrable_reversible/experiments/margolus_block_ca/`.
  - `01_margolus_system.py` — the 1D Margolus block CA (block size 3, partition
    offset cycles 0→1→2, count-preserving rotation φ). Ground-truth self-check:
    conservation ✅ and reversibility ✅ are structural; leak audit shows the
    rotation is a non-trivial residual (blind identity = 52.8%).
  - `02_probe.py` — single-seed probe. Because Margolus' one-step rule is local,
    the horizon grows with length (`T = L/2`) to recreate a length-generalization
    challenge (learn one step, compose it many times on a bigger ring).
  - `03_multiseed.py` — the formal numbers: 5 seeds, three fair free-form
    baselines (bidirectional GRU / LSTM, small Transformer), each given ample
    equal budget (120 epochs) so all reach **~99% single-step**. Composed under
    `T ∝ L` all three collapse (conservation → ~0–1%, accuracy → ~chance by
    L=384) while the structural block-CA stays **100/100**. Plus error-bar figures
    and a single-step diagnostic.
  - `README.md` + `README.zh-CN.md` — bilingual write-up.
- **Finding (5 seeds):** the Box-Ball result transfers and sharpens — the drift is
  not GRU-specific (three architectures, ~99% single-step, all drift), and a tiny
  single-step residual is fatal under long composition; structural conservation
  matters most precisely when composing many steps. Recorded honestly as **same
  recipe, different backbone** (block-partition vs carrier scan) — not a different
  structure, which answers the proposal's per-system-structure question with "no"
  for Margolus.
- **Optional next:** 2D classic Margolus (breadth already banked by 1D).

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
