# Phase 5 Plan — Completing the JAX and Bayesian Research Surface

This document is the **design plan for PLAN.md Priority 5**: the remaining
JAX gaps (5.1), safe advanced geometry inference (5.2), and robust sampling
with model comparison (5.3). Revise this plan first if the approach changes.

> **Status: drafted 2026-07, not yet started.** Soft dependency on the
> Phase 3 `bayes` package split (which decides file placement for the new
> sampling/diagnostics code) and on Phase 4.1's noise-scale vocabulary
> (which 5.2's hierarchical scales must match). Neither is a hard blocker:
> every item can be written against today's layout.
>
> **Lifecycle.** Transient: when Priority 5 ships, settled decisions
> migrate to `PLAN.md` and `docs/bayes.md`/`docs/backend.md`/
> `docs/gradients.md`, and this file is retired to git history.

---

## 1. Hard Constraints

1. **Additive and optional.** Everything here lives behind the `jax` /
   `bayes` extras. `import geodef` on a base install must not change
   behavior, import time, or requirements (the Phase 3.1 base-install test
   enforces this). No existing public signature changes; new capability
   arrives as new functions, new keyword-only arguments, or new optional
   result fields.
2. **NumPy parity is the contract.** Every JAX-accelerated path that has a
   NumPy equivalent (L-curve, CV, strain kernels) gets an exact-API,
   numerical-parity test against the NumPy path, following the pattern the
   existing ABIC sweep tests set.
3. **Tutorials unaffected; CI budget respected.** Chapters 10 and 14 are
   already optional/gated; nothing here touches Parts 0–V. New gated tests
   follow the standing budget rule (tutorials/OUTLINE.md §11 decision 9):
   tiny problems, short chains, shared compiled kernels — shrink the gated
   path where possible rather than growing it.
4. **Samplers may not leak through the interface.** BlackJAX API details
   stay inside `bayes._sampling`; results are sampler-independent records
   (§4 below) so a BlackJAX upgrade or an additional sampler is a
   contained change.

---

## 2. Item 5.1 — Close the Remaining JAX Gaps

### Batched hyperparameter sweeps

`LinearSystem._abic_sweep_jax` (invert.py:1244) already demonstrates the
pattern: batch the λ axis with broadcasting/`vmap`, dispatch on
`backend.get_backend() == "jax"` (invert.py:1654), return NumPy arrays.

- [ ] **`_lcurve_sweep_jax`** — batched solves + misfit/model-norm norms
  across the λ grid; dispatched inside `lcurve` exactly as the ABIC sweep
  is; corner detection stays NumPy (it is O(grid), not O(n³)).
- [ ] **`_cv_sweep_jax`** — batch over folds × λ: precompute per-fold
  index masks, batch the held-out normal-equation solves. CV fold logic is
  reused, not duplicated — one fold-definition function feeds both
  backends so parity is structural.
- [ ] **Parity tests** (gated): identical fold seeds and λ grids, assert
  curves and selected λ agree with NumPy within documented float64
  tolerance; wall-clock is *not* asserted (benchmark harness territory).
- [ ] Note: the Phase 4.2 generalized-eigendecomposition sweep gives the
  NumPy path a large speedup independently; land whichever comes first,
  keep the parity tests shared.

### Differentiable strain/stress kernels

`gradients` currently covers displacement only (`rect_displacement`,
`tri_displacement`, their Jacobians, `rect_greens`/`tri_greens`,
`los_project`). Strain paths in `greens` use the kernels through
`backend.namespace()` and are close to traceable already.

- [ ] **Audit the strain kernels for traceability**: no in-place writes,
  no data-dependent Python branching, singularity handling via
  `backend.masked_eval` (exists) rather than boolean indexing. Fix inside
  the adapter layer — the reference ports themselves stay visibly
  traceable to their sources (Phase 3 constraint).
- [ ] **`gradients.rect_strain` / `gradients.tri_strain`** — jitted,
  vmapped strain (and derived stress via `ElasticMedium`) forward models
  mirroring the displacement API, plus geometry Jacobians.
- [ ] **Differentiable stress kernel** — `fault.stress_kernel`'s JAX
  equivalent in `gradients`, unlocking gradient-based use of
  `regularization='stresskernel'` and, later, Phase 6.1's cycle modeling.
- [ ] **Finite-difference validation** away from documented singular
  boundaries (patch edges, the free surface for internal-strain
  evaluation); the *documentation* of those boundaries is part of the
  deliverable (`docs/gradients.md` gains a "where derivatives are
  trustworthy" section).

### Backend state hygiene

- [ ] **Prepared objects capture their backend.** `LinearSystem` and the
  posterior classes record `(backend, precision)` at construction; a
  mid-lifecycle global switch raises an informative error on next use
  ("this system was prepared under numpy/float64; rebuild it or restore
  the backend") instead of silently mixing dtypes or retracing. Compiled
  kernels close over concrete dtypes at trace time rather than reading
  the global config mid-trace. `set_backend(...)` remains the one simple
  entry point — this item removes *hidden* dependence, not the switch.
- [ ] **Compilation-cost visibility.** `docs/backend.md` gains a
  compilation-cache section (what triggers retracing, shape buckets,
  first-call vs steady-state cost); add a lightweight
  `backend.compile_report()` that summarizes trace counts per kernel
  (via a wrapper counter, not JAX internals) so users can diagnose
  shape-churn; the geometry-search and sampling docs state their one-time
  compile cost explicitly.

---

## 3. Item 5.2 — Advanced Geometry Inference, Set Up Safely

### Priors, frames, and pre-sampling checks

- [ ] **Named prior mappings with units.** `RectPosterior` (and
  `geometry_search` bounds) accept priors keyed by the 1.3 geometry names
  (`e0`, `n0`, `depth`, `strike`, `dip`, `length`, `width`) with plain
  tuple specs — `('normal', mean, sd)`, `('uniform', lo, hi)` — in the
  public units (meters, degrees). No distribution classes; the specs are
  parsed once into the internal arrays the samplers already use. The
  `theta` array path remains for experts.
- [ ] **Prior-predictive geometry checks.**
  `bayes.prior_predictive(post, *, n, seed)` draws geometries from the
  prior and returns them as `Fault` objects plus a validation summary
  (reusing `Fault.validate()`: above-surface breaches, absurd aspect
  ratios); `plot.prior_geometry(...)` renders the drawn fault outlines
  over the data footprint. Documented as the mandatory cheap step before
  burning sampler time; the half-space check failing produces the
  concrete remedy text (tighten `depth`/`dip` priors).
- [ ] **Explicit `LocalFrame`** threaded through the posterior
  constructors (accepting `frame=`, defaulting to the fault/data frame
  with a mismatch error) so geometry priors are unambiguous in space.

### Initialization and multi-start search

- [ ] **`geometry_search(..., n_starts=, seed=)`** — multi-start local
  optimization with starts drawn from bounds/priors; result gains the
  per-start table (converged θ, objective, flag) and the basin count —
  the Chapter 09 local-minimum lesson, now API-supported.
- [ ] **`bayes.init_from_search(result)`** — seed NUTS/SMC starting
  points from a deterministic fit (jittered around the optimum), clearly
  documented as initialization, not inference; keeps the optimization →
  sampling hand-off one line.

### Diagnostics with remedies

`PosteriorResult` already carries `split_rhat`, `effective_sample_size`,
divergence counts, and acceptance rates. This item is the interpretive
layer the roadmap asks for:

- [ ] **Posterior-predictive checks.** `bayes.posterior_predictive(post,
  result, *, n, seed)` — draw → forward-predict → return per-dataset
  predictive envelopes; works against held-out datasets from
  `data.split` (Phase 4.3) for genuine out-of-sample checks.
  `plot.predictive_check(...)` shows data vs envelope per dataset.
- [ ] **Sampling plots**: `plot.trace` (per-parameter trace + density),
  `plot.rank` (rank histograms across chains), `plot.divergences`
  (divergence locations projected onto parameter pairs),
  `plot.prior_posterior` (overlaid marginals). All small wrappers over
  the existing result arrays — matplotlib only, no ArviZ dependency.
- [ ] **Warnings with concrete remedies.** `PosteriorResult.summary()`
  gains a diagnostics section that names the failing statistic, the
  affected parameter, and the remedy list (more warmup, tighter priors,
  reparameterize, more chains) — the text the tutorials' Chapter 14
  teaches, produced by the library.

### Hierarchical scales without parameter-vector surgery

- [ ] **Multiple regularization scales**: accept per-component or
  per-region λ as a mapping (`{'strike': λ1, 'dip': λ2}` or a patch-mask
  keyed mapping) in `solve` and the slip posteriors, assembled internally
  into the block-diagonal penalty — users never build the stacked
  parameter vector.
- [ ] **Sampled dataset noise scales**: the Bayesian counterpart of
  Phase 4.1's `noise_scales`, sharing its naming (`noise_scales='sample'`
  or per-name prior specs); implemented in the collapsed posteriors as
  additional hyperparameters with the marginalization the collapsed
  construction already performs.

---

## 4. Item 5.3 — Robust Sampling and Model Comparison

- [ ] **Sampler-independent result contract first.** Define, in
  `bayes._sampling`, the field contract every sampler fills: `draws`,
  `log_prob`, per-parameter `rhat`/`ess` where meaningful, `seed`,
  `sampler` name, and a sampler-specific `extras` mapping (NUTS:
  divergences, acceptance; SMC: weights, temperatures, log-evidence).
  `PosteriorResult` is extended compatibly (new optional fields, nothing
  removed); `sample()`'s signature and behavior are unchanged.
- [ ] **`bayes.sample_smc(post, *, n_particles, seed, target_ess=0.5,
  mutation='nuts', ...)`** — BlackJAX adaptive tempered SMC: prior draws
  → adaptive temperature schedule (ESS-controlled) → NUTS mutation moves
  → weighted posterior + log-evidence estimate. Works on the same
  posterior objects `sample` accepts (`RectPosterior`, `SlipPosterior`,
  `TriPosterior`) using only their shared `x0`/`logpdf`/`n_params`/bounds
  surface — plus a `log_prior`/`log_likelihood` split, which the
  collapsed posteriors must expose for tempering (small internal
  refactor: the terms already exist separately inside `logpdf`).
- [ ] **Validation ladder** (in order, each a test):
  1. Analytic low-dimensional targets: Gaussian (evidence known in closed
     form), a banana/curved target (shape), a two-component Gaussian
     mixture (multimodality — the case NUTS structurally fails).
  2. NUTS parity on a unimodal `RectPosterior`: posterior moments agree
     within Monte-Carlo error on the teaching-scale problem.
  3. Linear-Gaussian evidence check: SMC log-evidence vs the analytic
     marginal likelihood ABIC is built on — ties Chapter 04/14 math to
     the sampler output.
  4. A deliberately multimodal geometry problem (e.g. dip ambiguity /
     conjugate-plane construction) demonstrating mode coverage NUTS
     misses; this becomes the `examples/` companion, not a tutorial.
- [ ] **Model comparison surface.** Log-evidence differences exposed with
  honest error bars (SMC evidence variance across seeds/replicates);
  documentation warns explicitly about prior sensitivity of evidence —
  the Chapter 14 prior-sensitivity practice applies doubly here.

---

## 5. Suggested Delivery Order

1. Backend-capture hygiene + compile-report (5.1) — small, de-risks the
   rest.
2. Batched L-curve/CV sweeps + parity tests (5.1).
3. Strain/stress kernel traceability audit, then the differentiable
   kernels + FD validation (5.1).
4. Prior mappings + prior-predictive checks + `LocalFrame` threading
   (5.2).
5. Multi-start search + init helper (5.2).
6. Diagnostics plots, PPC, summary remedies (5.2).
7. Result-contract refactor, then `sample_smc` + validation ladder (5.3).
8. Hierarchical scales (5.2, after Phase 4.1's `noise_scales` naming
   ships, so the vocabularies match).

---

## 6. Settled Decisions (proposed here, confirm at review)

1. **No ArviZ dependency.** Rank/trace/PPC plots are small matplotlib
   functions over `PosteriorResult` arrays; adopting ArviZ would drag a
   heavy dependency into the `bayes` extra for plots the course needs to
   explain anyway.
2. **Prior specs are plain tuples/mappings, not distribution objects.**
   Two forms (`'normal'`, `'uniform'`) cover the geometry use cases; a
   distribution-class hierarchy fails the object-budget test.
3. **SMC mutation kernel is NUTS** (reusing the tuned machinery), with
   the random-walk fallback only if NUTS-within-SMC proves fragile in
   validation — record the outcome here.

## 7. Open Questions

1. **Where does `log_prior`/`log_likelihood` splitting stop?**
   `SlipPosterior`'s positivity truncation makes the prior
   normalization intractable; tempering `p(d|m)` only (likelihood
   tempering) avoids needing it — confirm BlackJAX's tempered SMC can
   run likelihood-tempering with the truncated prior as the base
   measure, on a toy problem, before committing the API.
2. **`compile_report` mechanics**: wrapper-based trace counting is easy
   but only covers geodef-owned kernels; is that enough, or do we
   document `jax.log_compiles` instead and skip the helper? Prototype
   both on the geometry-search example.
3. **Result-schema coordination with Phase 4** (shared open question):
   if `PosteriorResult` gains persisted form later, align the version
   numbering with the `InversionResult` schema bump planned in 4.x.
4. **CV-sweep memory on JAX**: batching folds × λ multiplies the
   normal-equation batch; confirm the teaching-scale and realistic-scale
   memory footprint before choosing full batching vs a λ-only batch with
   a Python loop over folds.
