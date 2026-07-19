# Phase 4 Plan — Scale to Real Geodetic Datasets

This document is the **design plan for PLAN.md Priority 4**: noise and
whitening operators (4.1), linear operators and large inversions (4.2),
observation preprocessing and nuisance parameters (4.3), and first-class
interseismic coupling (4.4). Revise this plan first if the approach changes.

> **Status: drafted 2026-07, not yet started.** Depends on nothing from
> Phase 3 except (soft) the `invert` package split, which decides which
> file new code lands in; every item here can be written against today's
> layout and moved.
>
> **Lifecycle.** Transient: when Priority 4 ships, settled decisions
> migrate to `PLAN.md`, `docs/conventions.md`, and the module docs, and
> this file is retired to git history.

---

## 1. Hard Constraints

1. **Additive API only — no breaking changes.** Every item is a new
   module, new function, or new keyword-only argument with a default that
   reproduces today's behavior. The tutorials (including the in-flight
   Priority 2 rewrite) must run unmodified before and after this phase.
2. **Dense, transparent paths remain the defaults** at teaching scale.
   `stack_weights`, dense `spatial_covariance`, and the direct dense solve
   are what Chapters 03–06 teach; they are kept, tested, and documented as
   the small-problem/educational path — never deprecated by this phase.
3. **Numerical parity tests for every new path.** Operator/whitened/
   iterative routes must reproduce the dense route on teaching-sized
   problems to a stated tolerance (documented per test; not bit-identical
   — factorization order differs — but tight, e.g. rtol ≲ 1e-10 for
   float64 well-conditioned cases).
4. **Result-file schema changes are versioned with migration**, following
   the `_migrate_v2_regularization_keys` precedent: bump the schema
   version, keep loading old files, test the migration.
5. **Memory honesty.** Any path that would allocate a large dense array
   reports the estimate first (4.2) rather than dying opaquely.

Cross-phase note: the Priority 2.3 real-data examples (interseismic
coupling; earthquake with nuisance + correlated noise) are *blocked on*
4.1/4.3/4.4 and return here as delivery vehicles — each major item below
ends with the example or golden workflow that proves it.

---

## 2. Item 4.1 — Noise and Whitening Operators

### Current state

`DataSet.covariance` is a dense `(n, n)` array; `greens.stack_weights`
(greens.py:818) inverts each block with `np.linalg.inv` and block-diagonals
the result; `LinearSystem` materializes `W` and forms `GᵀWG`/`Gᵀwd`. For a
50 000-point InSAR scene, `C_d` alone is 20 GB — the dense route is a
teaching tool, not a production one, and PLAN.md audit item 8 calls this
out.

### Design

- [ ] **New module `geodef.noise`** (functions only). A *whitening* is the
  one small record this phase adds (justified under product principle 9:
  it carries a cross-call invariant — "apply and logdet describe the same
  `C_d`" — that a bare callable cannot):

  ```python
  @dataclass(frozen=True)
  class Whitening:
      apply: Callable[[np.ndarray], np.ndarray]  # x -> C^{-1/2} x (rows)
      logdet: float                              # log det C_d
      n: int
  ```

  Constructor functions (no class hierarchy, per the roadmap):
  - `noise.diagonal(sigma)` — per-datum standard deviations.
  - `noise.dense(cov)` — Cholesky-backed; validates SPD via the existing
    0.2 covariance validation.
  - `noise.block_diagonal(parts)` — one block per dataset; also the
    internal join used for multi-dataset solves.
  - `noise.sparse(cov)` — `scipy.sparse` input, factorized with `splu`
    (optional `scikit-sparse`/CHOLMOD used if importable, never required).
  - `noise.low_rank(U, diag)` — `C = D + U Uᵀ` via Woodbury; the natural
    form for InSAR atmosphere screens.
- [ ] **Dataset integration.** Datasets keep accepting `covariance=` as
  today (dense array or per-datum sigmas); additionally accept a
  `Whitening` there. `DataSet.covariance` stays for the dense case;
  a new `DataSet.whitening` property returns the operator form (built
  from the dense matrix when that is what was given).
- [ ] **Solver integration.** `LinearSystem` solves via whitened rows —
  `Gw = C^{-1/2}G`, `dw = C^{-1/2}d`, `GᵀWG = GwᵀGw` — instead of forming
  `W = C^{-1}` when any dataset carries an operator whitening; the dense-`W`
  code path is kept verbatim for dense input (bit-stable teaching path).
  ABIC gets `log det C_d` from `Whitening.logdet`. `stack_weights` is
  untouched (educational/small-problem helper, so documented).
- [ ] **Parity tests**: diagonal/dense/block/sparse/low-rank whitenings vs
  the dense `W` route on the teaching scenario, for `solve`, `lcurve`,
  `compute_abic`, and `model_covariance`.
- [ ] **Covariance estimation, separate from inversion.**
  `noise.empirical_variogram(dataset, *, n_bins, max_lag, seed)` (pair
  subsampling for large scenes, seeded), `noise.fit_covariance(variogram,
  model='exponential'|'gaussian')` returning the parameters that
  `data.spatial_covariance` already accepts, and `plot.variogram(...)`
  showing empirical points, the fit, and honest scatter. The workflow is
  estimate → inspect → construct → invert, each step visible.
- [ ] **Per-dataset variance scales.** `solve(..., noise_scales={name:
  s})` multiplies dataset `name`'s covariance by `s²` (equivalently scales
  its whitening); recorded in `InversionResult` provenance (schema bump).
  The hierarchical (sampled) version of the same vocabulary lands in
  Phase 5.2 so deterministic and Bayesian workflows share the name.

---

## 3. Item 4.2 — Linear Operators and Large Inversions

- [ ] **Operator-valued `G` and `L`.** `LinearSystem` accepts
  `scipy.sparse.linalg.LinearOperator`-compatible objects (anything with
  `shape`, `matvec`, `rmatvec`) for the Green's matrix and regularizer in
  addition to dense arrays. Dense stays the default everything-else path;
  operator inputs simply make the direct dense solve unavailable and
  require an iterative `method`.
- [ ] **Iterative solvers.** `method='lsqr'|'lsmr'` solving the augmented
  whitened system `[C^{-1/2}G; √λ L] m ≃ [C^{-1/2}d; √λ L m_ref]` with
  `scipy.sparse.linalg`; bounded/nonneg large problems route through
  projected/possibly `trf` least squares (evaluate `scipy.optimize.
  lsq_linear(method='trf')` with operator input). Each returns a
  **convergence report** (iterations, stop reason, residual norms,
  conditioning estimate from LSQR) stored on `InversionResult` (new
  optional field + schema bump shared with 4.1's provenance additions —
  batch the schema changes into one version bump).
- [ ] **Preconditioning hooks.** Keyword `preconditioner=` accepting a
  callable; ship one practical default helper (diagonal/Jacobi from
  column norms). Documented as expert-tier.
- [ ] **Chunked assembly.** `greens.matrix(..., chunk_stations=)` assembles
  in station blocks (bounding peak memory) and optionally writes to a
  `np.memmap`; the cache layer stores/loads memmapped matrices without
  materializing them. Also: vectorize the per-column Python loop in
  `greens.project` (greens.py:880) — an exact-equivalence, pure-win
  speedup that can land first.
- [ ] **Reusable factorizations for sweeps.** For dense systems, one
  generalized eigendecomposition of `(GᵀWG, LᵀL)` makes every point of an
  L-curve/ABIC/CV sweep O(n²) instead of O(n³); `LinearSystem` caches it
  beside the existing `_eig_LtL`. This is the NumPy analogue of the
  existing `_abic_sweep_jax` batching (invert.py:1244) and shares its
  tests. Batched *JAX* sweeps stay in Phase 5.1.
- [ ] **Memory estimation.** `invert.estimate_memory(fault, datasets)`
  reporting the dense `G`, `C_d`, and factorization footprints before
  allocation; `LinearSystem` warns (with the estimate and the operator/
  chunking alternatives) past a threshold, rather than OOMing silently.
- [ ] **Documented scale boundary.** Using the Phase 3.4 benchmark
  harness: measure teaching → realistic → large problems and write the
  guidance table (when to downsample, when operators, when JAX/GPU) into
  `docs/invert.md`. Never a single speedup number without problem size,
  backend, precision, and hardware.

---

## 4. Item 4.3 — Observation Preprocessing and Nuisance Parameters

### Nuisance bases

- [ ] **Basis constructors** in `geodef.data` (they are data-side objects;
  no new module): `data.nuisance_ramp(insar, order=0|1|2)` (offset /
  planar ramp / quadratic in local coordinates), `data.nuisance_offset
  (dataset)`, `data.nuisance_translation(gnss)` (per-component frame
  translation), `data.nuisance_rotation(gnss)` (rigid rotation about the
  frame origin). Each returns a named record `(dataset_name,
  parameter_names, basis)` with `basis` of shape `(n_obs, k)`.
- [ ] **Solver integration.** `solve(..., nuisance=[...])` augments the
  system with unregularized columns appended after the slip block:
  regularization matrices are zero-padded so `L` never touches nuisance
  parameters; bounds/constraints machinery sees them as unbounded unless
  explicitly bounded. Slip views (`strike_slip`, `dip_slip`,
  `slip_vector`) are unchanged (slip block only); results gain
  `nuisance_parameters` (per dataset, named) and predictions can be split
  `with_nuisance=True|False` so "fault signal" vs "fitted ramp" is one
  keyword. Touched consumers to audit: `prediction`, `residual`,
  `diagnostics`, `model_covariance`/`resolution`/`uncertainty` (slip-block
  partitioning), `save`/`load` (schema), `plot.fit`.
- [ ] **Marginalization option** for the Bayesian path: analytic
  Rao-Blackwellization of linear nuisance under Gaussian priors inside the
  collapsed posteriors (Phase 5 consumes this; only the linear-algebra
  helper lands here).

### Downsampling, partitioning, dataset algebra

- [ ] **Quadtree downsampling.** `data.quadtree(insar, *, variance_limit,
  min_cells, max_cells)`: split while the cell variance exceeds the limit;
  cell values are means with propagated uncertainties (and reduced-sample
  covariance if a full `C_d`/covariance model is present); returns a new
  `InSAR` with provenance metadata (source name, cell corners, counts).
  `plot.insar` gains an optional cell-outline overlay.
- [ ] **Train/validation splits.** `data.split(dataset, fraction, *,
  seed, block_size=None)` — random or spatially blocked (blocked mode
  addresses the correlated-neighbor leakage the Chapter 04 CV caveat
  teaches); returns two datasets with provenance.
- [ ] **Masking/subsetting/concatenation.** `DataSet.mask(keep)`,
  `DataSet.subset(indices)`, `data.concat([a, b])` — immutable returns,
  names/uncertainties/covariance rows handled consistently, provenance
  recorded; replaces the parallel-manual-array-slicing anti-pattern.

### Multi-epoch bridge

- [ ] **Design note first, minimal API second.** Datasets already carry
  velocity-vs-displacement semantics (1.5). Add only: an optional
  `epoch`/`time_span` on dataset metadata and a design note
  (`plans/`-level, folded into this file) on temporal basis functions —
  implementation is deliberately deferred until nuisance and noise are
  proven, per the roadmap's "bridge, not time-dependent slip" wording.
  Full kinematic slip histories remain Phase 6.3.

---

## 5. Item 4.4 — First-Class Interseismic Coupling

The backslip **convention is already settled** (tutorials/OUTLINE.md §11
decision 8) and enters `docs/conventions.md` with tutorial Chapter 12:
velocities are block-corrected into a declared reference frame using the
same Euler pole(s) that supply per-patch plate rates; backslip is
anti-parallel to local plate motion, non-negative, bounded by the plate
rate; coupling = backslip rate / plate rate ∈ [0, 1]. This phase builds the
API on that convention — it does not reopen it.

- [ ] **Per-patch plate-rate vector.** `slip.plate_rake_from_euler(fault,
  pole)` exists; add `slip.plate_rate_from_euler(fault, pole)` (magnitude,
  m/yr) so direction and rate come from one pole in one vocabulary, plus
  an overload accepting a per-patch `(rate, rake)` pair for non-Euler
  sources (e.g. a published convergence model).
- [ ] **Coupling parameterization.** `components='coupling'` in
  `solve`/`LinearSystem`, requiring `plate_rate=` (scalar or `(N,)`) and
  `plate_rake=`: solves for the coupling fraction `c` with columns
  `G_plate · diag(rate)`, default bounds `(0, 1)` applied automatically
  (overridable), regularization acting on `c` (dimensionless — settled
  below). Result gains named views `coupling` (the fraction) and
  `backslip_rate` (fraction × rate, m/yr); `slip_vector` remains the
  expert blocked view of the solved parameters.
- [ ] **Moment-deficit rate.** `invert.moment_deficit_rate(result, fault)`
  → `(rate, sigma)` in N·m/yr: `Σ μ · c_k · rate_k · A_k`, with `σ` by
  propagating `C_m` through this linear functional; `μ` from
  `fault.medium`; units and epoch semantics tied to the velocity metadata
  (error if the datasets are displacements, not velocities — the 1.5
  semantics doing real work). Companion `plot` support: coupling maps
  render as fractions in [0, 1], not signed slip.
- [ ] **Euler workflow glue.** Document (and test) the two-step teaching
  path end to end: `euler.best_fit_pole`/`remove_pole` for the block
  correction, the same pole into `plate_rate/rake_from_euler`, then the
  coupling solve. Co-estimating block motion with coupling is explicitly
  out of scope (roadmap keeps it later) but the API must not preclude it —
  the nuisance machinery from 4.3 (rotation basis ≈ local Euler
  correction) is the intended future hook.
- [ ] **Deliverables through the new API.** The Priority 2.3 interseismic
  example (real data, reduced executed path) is built on this; tutorial
  Chapter 12's "revised when 4.4 lands" hook is exercised (swap the
  basis-and-bounds construction for `components='coupling'` in one
  sidebar); promote the example to a golden workflow (2.4) once stable.

---

## 6. Suggested Delivery Order

1. `greens.project` vectorization + memory estimator (small, standalone).
2. `geodef.noise` constructors + whitened `LinearSystem` path + parity
   tests (4.1) — the enabling layer for everything after.
3. Variogram estimation + plots (4.1).
4. Nuisance bases + solver integration + one schema bump shared with the
   convergence-report fields (4.3 + 4.2 result additions).
5. Iterative solvers + operator `G`/`L` + chunked assembly (4.2).
6. Quadtree, splits, dataset algebra (4.3).
7. Coupling parameterization + moment deficit + Euler glue (4.4).
8. The two Priority 2.3 examples, then the scale-boundary documentation.

---

## 7. Settled Decisions (proposed here, confirm at review)

1. **`Whitening` is the single new public record** of this phase; noise
   models are constructor functions returning it, not a class hierarchy.
2. **Coupling regularization acts on the coupling fraction**, not the
   backslip rate: smoothing a dimensionless field makes λ comparable
   across faults with varying convergence rate, and "smooth coupling" is
   the geophysical prior actually intended. The example must demonstrate
   the alternative (smoothing rate) for comparison once, in prose.
3. **Nuisance parameters are appended, unregularized, and excluded from
   slip views**; their identifiability is the user's responsibility, but
   `summary` reports their values and the correlation of each with total
   moment as a health check.
4. **One result-schema version bump for the whole phase** (noise scales,
   convergence report, nuisance parameters, coupling views), landed with
   the first item that needs it and covering the rest — not four bumps.

## 8. Open Questions

1. **Sparse Cholesky dependency:** is optional `scikit-sparse` worth the
   packaging friction for `noise.sparse`, or is `splu` enough until users
   ask? Default: `splu` only; revisit on demand.
2. **`lsq_linear(method='trf')` with LinearOperator input** — verify it
   actually supports operators and bounds together at useful scale before
   promising the bounded-iterative path; if not, large bounded problems
   are documented as unsupported this phase (fall back to projected
   gradient or defer).
3. **Quadtree covariance:** when the input carries a covariance *model*
   (function) rather than a matrix, should cell-to-cell covariance be
   re-evaluated from the model at cell centers (cheap, approximate) or
   propagated exactly (expensive)? Prototype both on the Chapter 06
   scenario and compare inversion results before deciding.
4. **Where the schema bump lands** if Phase 5 also wants result changes —
   coordinate the version numbering with 5.2/5.3 result records before
   the first bump ships.
