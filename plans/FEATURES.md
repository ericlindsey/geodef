# Feature Improvements — Menu and Discussion

This is a menu of improvements to things GeoDef already does: better noise
handling for inversions it already runs, supported APIs for workflows the
tutorials already teach, faster paths for computations that already exist.
The companion menus are `plans/ARCHITECTURE.md` (internal structure, with
fuller discussion) and `plans/CAPABILITIES.md` (things GeoDef cannot do at
all today).

Each entry states the value, a priority judgment, effort, and dependencies,
with a design sketch. Detail is deliberately lighter than in the
architecture menu; designs are elaborated when an item is picked.

---

## How to choose from this menu

| # | Item | Value in one line | Effort | Depends on |
|---|---|---|---|---|
| F1 | Synthetic-test helpers | The course's resolution-test workflow becomes real API | Small | — |
| F2 | First-class coupling | A declared target application stops being a manual exercise | Medium | — |
| F3 | Noise modeling tools | Honest covariances for real data, estimated and inspected | Medium | A3 |
| F4 | Nuisance parameters | Ramps and frame shifts stop contaminating slip | Medium | — (A3 helps) |
| F5 | Data preprocessing algebra | Downsampling and splits with provenance, not manual slicing | Medium | — |
| F6 | Large-problem paths | Realistic InSAR scenes become tractable | Medium–large | A3, A4 |
| F7 | Sweep acceleration | L-curve/CV selection goes from O(n³)-per-point to cheap | Small–medium | A4 (NumPy), parity tests (JAX) |
| F8 | Guided Bayesian setup | Geometry inference becomes safe to set up without expertise | Medium | — |
| F9 | Differentiable strain/stress | Completes the JAX surface; unlocks cycle modeling later | Medium | A2 |
| F10 | Examples, docs site, golden workflows | The proof that the above works end to end | Ongoing | F2–F4 for the examples |

**Recommended near-term picks: F1 and F2.** Both are small-to-medium, have
no architecture prerequisites, and are strategic: F1 closes the last gap
between what the course teaches and what the API supports, and F2 serves
one of the package's two declared target applications. After those, the
F3 → F4 → F5 cluster (with A3 underneath) is what unblocks the two
real-data examples in F10. F6 matters when your actual datasets demand it;
F8 matters most for research users of the Bayesian surface; F7 and F9 are
opportunistic.

---

## F1. Synthetic-test helpers and the scenario builder

**What.** A new `geodef.synthetic` module — `scenario(...)` (the
documented builder for the course's recurring thrust-fault setup),
`checkerboard(fault, ...)`, `spike(fault, ...)`, and
`noisy_data(fault, slip, ..., seed=...)` — plus
`geodef.invert.compare(result, true_slip)` for recovered-versus-input
comparison. All pure functions; the design is settled in
`tutorials/OUTLINE.md` §11 (decision 7), including the naming decision
that `compare` lives in `invert`, unaliased.

**Value and priority.** High priority, small effort, zero dependencies.
The tutorials teach checkerboard/spike resolution testing, but every
student currently re-implements it from notebook cells — the last
"taught but not supported" gap from the structural audit. Also the
prerequisite for replacing repeated tutorial setup cells with one
documented builder (the remaining 2.2 notebook item), and for the golden
workflow tests in F10. `tutorials/OUTLINE.md` is retired once this ships.

## F2. First-class interseismic coupling

**What.** Coupling is one of GeoDef's two declared applications, and today
it is done by manually re-signing slip vectors (tutorial Chapter 12 shows
the `components='plate'` + bounds recipe). Promote it to vocabulary:

- `slip.plate_rate_from_euler(fault, pole)` (magnitude, m/yr) beside the
  existing `plate_rake_from_euler`, plus an overload accepting per-patch
  `(rate, rake)` for non-Euler convergence models.
- `components='coupling'` in `solve`/`LinearSystem`, requiring
  `plate_rate=` and `plate_rake=`: solves for the coupling fraction with
  columns `G_plate · diag(rate)`, default bounds `(0, 1)`, regularization
  acting on the *fraction* (settled: dimensionless smoothing makes λ
  comparable across faults; the example shows the rate-smoothing
  alternative once, in prose). Result gains named `coupling` and
  `backslip_rate` views.
- `invert.moment_deficit_rate(result, fault)` → `(rate, sigma)` in
  N·m/yr, propagating `C_m`, with an error if the datasets are
  displacements rather than velocities (the 1.5 velocity metadata doing
  real work). Coupling plots render fractions in [0, 1], not signed slip.
- Documented, tested Euler glue: one pole feeds both the block correction
  (`euler.remove_pole`) and the plate rate/rake, so rigid-block
  prediction and coupling inversion share one vocabulary. Co-estimating
  block motion stays out of scope but must not be precluded — the F4
  rotation basis is the intended future hook.

**Value and priority.** High priority, medium effort. The backslip/
coupling convention is already settled and documented
(`docs/conventions.md`); the bounds and fixed-direction machinery already
exist, so this needs **no** noise or nuisance work first — it can land
now. Chapter 12 gains the one-sidebar swap it was written to receive, and
the interseismic example (F10) is delivered through this API.

## F3. Noise modeling tools

**What.** The user-facing half of the whitening work (architecture A3):

- The `geodef.noise` constructors themselves (diagonal, dense, block,
  sparse, low-rank) as documented user API.
- Covariance *estimation*, kept separate from inversion so assumptions
  stay visible: `noise.empirical_variogram(dataset, ...)` (seeded pair
  subsampling for large scenes), `noise.fit_covariance(variogram,
  model='exponential'|'gaussian')` feeding the parameters
  `data.spatial_covariance` already accepts, and `plot.variogram(...)`
  showing empirical points, fit, and honest scatter. The workflow is
  estimate → inspect → construct → invert, each step visible.
- Per-dataset variance scales: `solve(..., noise_scales={name: s})`,
  recorded in result provenance (one schema bump per A9). The *sampled*
  Bayesian version shares this name (F8).

**Value and priority.** High for anyone touching real InSAR — correlated
noise is the difference between honest and decorative uncertainties, and
Chapter 06 already teaches the theory. Medium effort once A3 exists.

## F4. Nuisance parameters

**What.** Real data contain signals that are not fault slip: InSAR
orbital ramps and offsets, GNSS reference-frame translations/rotations,
inter-dataset biases. Add explicit linear nuisance bases —
`data.nuisance_ramp(insar, order=0|1|2)`, `data.nuisance_offset(...)`,
`data.nuisance_translation(gnss)`, `data.nuisance_rotation(gnss)` — each
a named record `(dataset_name, parameter_names, basis)`. Then
`solve(..., nuisance=[...])` appends unregularized columns after the slip
block: `L` never touches them, slip views are unchanged, results gain
named `nuisance_parameters`, and predictions split with
`with_nuisance=True|False` so "fault signal" versus "fitted ramp" is one
keyword. Settled: nuisance identifiability is the user's responsibility,
but `summary` reports each parameter's correlation with total moment as a
health check. A small analytic-marginalization helper is included for the
Bayesian path to consume later. Consumers to audit: `prediction`,
`residual`, `diagnostics`, the model-assessment functions (slip-block
partitioning), `save`/`load`, `plot.fit`.

**Value and priority.** High — without this, ramps leak into slip and
users pre-process outside the package where provenance is lost. Medium
effort; independent of A3, though the earthquake example (F10) wants both.

## F5. Data preprocessing algebra

**What.** Three related conveniences that replace manual array slicing:

- **Quadtree downsampling.** `data.quadtree(insar, *, variance_limit,
  min_cells, max_cells)`: split while cell variance exceeds the limit;
  cell means with propagated uncertainties; returns a new `InSAR` with
  provenance (source name, cell corners, counts); `plot.insar` gains a
  cell-outline overlay. Open question to prototype: when the input
  carries a covariance *model*, re-evaluate cell-to-cell covariance at
  cell centers (cheap) or propagate exactly (expensive) — compare both on
  the Chapter 06 scenario before deciding.
- **Train/validation splits.** `data.split(dataset, fraction, *, seed,
  block_size=None)` — random or spatially blocked (blocked mode addresses
  the correlated-neighbor leakage the Chapter 04 CV caveat teaches).
- **Masking/subsetting/concatenation.** `DataSet.mask(keep)`,
  `DataSet.subset(indices)`, `data.concat([a, b])` — immutable returns,
  names/uncertainties/covariance handled consistently, provenance kept.
- **Multi-epoch bridge (metadata only).** Optional `epoch`/`time_span`
  dataset metadata plus a design note on temporal bases; actual
  time-dependent slip stays a capability (C10).

**Value and priority.** Medium-high; quadtree in particular is a standard
step every InSAR inversion needs. Medium effort, no hard dependencies.

## F6. Large-problem paths

**What.** The user-facing half of A3/A4 for problems that outgrow dense
arrays:

- Iterative solvers (`method='lsqr'|'lsmr'`) on the augmented whitened
  system, returning a convergence report (iterations, stop reason,
  residual norms, conditioning estimate) stored on the result. Bounded
  large problems: evaluate `scipy.optimize.lsq_linear(method='trf')` with
  operator input *before* promising it; if it can't, document the limit.
- A practical preconditioner hook with one shipped default (Jacobi from
  column norms), documented as expert-tier.
- Chunked Green's assembly (`greens.matrix(..., chunk_stations=)`) with
  optional memmap output and cache support. First slice: vectorize the
  per-column Python loop in `greens.project` — an exact-equivalence pure
  win that can land alone, anytime.
- `invert.estimate_memory(fault, datasets)` reporting dense footprints
  before allocation; `LinearSystem` warns with the estimate and the
  operator/chunking alternatives instead of dying opaquely.
- The documented scale boundary: use the benchmark harness to measure
  teaching → realistic → large problems and write the guidance table
  (when to downsample, when operators, when JAX/GPU) into
  `docs/invert.md` — never a single speedup number without problem size,
  backend, precision, and hardware.

**Value and priority.** Entirely driven by your actual data sizes: if
studies stay at thousands of observations after quadtree (F5), this can
wait; the moment a scene needs 10⁴–10⁵ points retained, it is the
difference between possible and not. Medium-large effort on top of A3/A4.

## F7. Hyperparameter sweep acceleration

**What.** Two independent speedups for L-curve/ABIC/CV selection, sharing
parity tests: the NumPy generalized-eigendecomposition sweep (A4) that
makes each λ point O(n²); and batched JAX sweeps — `_lcurve_sweep_jax`
and `_cv_sweep_jax` following the existing `_abic_sweep_jax` pattern
(batch λ, and folds × λ, via vmap; one shared fold-definition function
feeds both backends so parity is structural; corner detection stays
NumPy). Gated parity tests assert curves and selected λ agree with NumPy
within documented float64 tolerance; wall-clock claims live in the
benchmark harness. Open question: CV batching multiplies memory by folds
— measure before choosing full batching versus λ-only batching with a
Python loop over folds.

**Value and priority.** Nice-to-have until sweeps become the bottleneck
of someone's workflow; land whichever half arrives first with its parity
tests. Small-medium effort.

## F8. Guided Bayesian geometry inference

**What.** The Bayesian surface is powerful but demands package expertise
to set up safely. The interpretive layer:

- Named prior mappings with units: priors keyed by the geometry names
  (`e0`, `n0`, `depth`, `strike`, `dip`, `length`, `width`) with plain
  tuple specs — `('normal', mean, sd)`, `('uniform', lo, hi)` — in
  meters/degrees. No distribution classes (settled); the `theta` array
  path remains for experts. An explicit `LocalFrame` threads through the
  posterior constructors so priors are unambiguous in space.
- Prior-predictive checks: `bayes.prior_predictive(post, *, n, seed)`
  draws geometries as `Fault` objects with a validation summary (reusing
  `Fault.validate()`); `plot.prior_geometry(...)` renders them over the
  data footprint. Documented as the mandatory cheap step before burning
  sampler time.
- `geometry_search(..., n_starts=, seed=)` multi-start optimization with
  a per-start table and basin count (the Chapter 09 local-minimum lesson,
  API-supported), and `bayes.init_from_search(result)` to seed samplers —
  clearly documented as initialization, not inference.
- Diagnostics with remedies: `bayes.posterior_predictive(...)` with
  per-dataset envelopes (pairs with `data.split` from F5 for genuine
  out-of-sample checks); `plot.trace`/`plot.rank`/`plot.divergences`/
  `plot.prior_posterior` as small matplotlib wrappers (settled: no ArviZ
  dependency); `PosteriorResult.summary()` names the failing statistic,
  the affected parameter, and concrete remedies.
- Hierarchical scales without parameter-vector surgery: per-component or
  per-region λ as a mapping, and sampled dataset noise scales sharing
  F3's `noise_scales` vocabulary.

**Value and priority.** High for research users — this is the difference
between the Bayesian tools being usable by their author versus by
colleagues and students. Medium effort, few hard dependencies (the
noise-scale naming should follow F3; the rest can proceed anytime).

## F9. Differentiable strain and stress kernels

**What.** `gradients` covers displacement only. Add
`gradients.rect_strain`/`tri_strain` (jitted, vmapped strain and derived
stress via `ElasticMedium`, plus geometry Jacobians) and the JAX
equivalent of `fault.stress_kernel`, validated against finite differences
away from documented singular boundaries — with the documentation of
those boundaries ("where derivatives are trustworthy" in
`docs/gradients.md`) part of the deliverable.

**Value and priority.** Completes the JAX research surface: unlocks
gradient-based use of `regularization='stresskernel'` and is a hard
prerequisite for the cycle-modeling capability (C6). Medium effort;
depends on the adapter-layer traceability audit (A2).

## F10. Examples, documentation site, and golden workflows

**What.** The remaining Priority 2 delivery items, kept together because
they are proof-of-work for everything above:

- One end-to-end interseismic coupling example (rides F2, ideally F3) and
  one earthquake example with nuisance parameters and correlated noise
  (rides F3 + F4), both in the uniform example structure with a reduced
  machine-executed path.
- Three golden workflows (first forward model, first inversion, joint
  GNSS+InSAR study) with their complete public call sequences under test.
- A searchable documentation site built from the existing Markdown and
  docstrings — only after navigation and content hierarchy are settled;
  the content already exists.
- Periodic observation sessions with novice geophysicists; each repeated
  confusion becomes an API, error-message, or documentation test. (A
  human-schedule task, listed so it isn't lost.)

**Value and priority.** The examples are the acceptance tests for the
F2–F4 cluster and should land with it. The docs site is presentation-only
polish — worthwhile before advertising the package, idle work before
that. Ongoing effort rather than one block.
