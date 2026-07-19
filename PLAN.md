# PLAN.md — GeoDef Roadmap

GeoDef's next phase is not a choice between a small teaching library and a
powerful research package. It should be both: a novice should be able to express
a geophysical problem with a few memorable functions and durable domain values,
while an expert can still reach the Green's matrices, covariance operators,
autodiff kernels, constrained solvers, and Bayesian posteriors underneath.

This roadmap replaces the implementation diary that previously occupied this
file. Git history, tests, and the module documentation preserve the details of
completed work; this document describes what remains and why it matters.

**Read `PYTHON.md` before editing any code.** Use red/green TDD, update relevant
documentation with every public change, and commit each independently useful
unit with its relevant tests passing. Run the full routine suite when a pull
request or major change is wrapping up.

---

## Current foundation (v0.1)

The following capabilities are complete and are foundations, not open roadmap
items:

- Rectangular Okada 1985/1992 and triangular Nikkhoo & Walter dislocation
  engines, with surface and internal displacement/strain paths.
- `Fault`, `GNSS`, `InSAR`, and `Vertical` domain objects; rectangular and
  triangular mesh construction; Green's assembly and disk caching.
- Weighted, bounded, constrained, fixed-direction, and regularized linear slip
  inversion; L-curve, ABIC, cross-validation, uncertainty, resolution, and
  per-dataset diagnostics.
- NumPy as the portable default and optional JAX acceleration, float32/float64
  controls, differentiable rectangular and triangular forward models, and
  gradient-based planar geometry search.
- Collapsed and positivity-aware Bayesian inference, including
  `RectPosterior`, `SlipPosterior`, `TriWarp`, `TriPosterior`, NUTS sampling,
  convergence diagnostics, conditional slip draws, and prediction.
- Eleven executed teaching notebooks, real/worked examples, per-module Markdown
  documentation, and a large reference and integration test suite.

The old roadmap's still-open research items are retained below: batched JAX
L-curve/CV sweeps, differentiable stress kernels, tempered SMC,
earthquake-cycle modeling, and additional Green's engines.

---

## Product principles

These are acceptance criteria for every roadmap item.

1. **One obvious beginner path.** The first successful forward model and
   inversion should use a small, stable vocabulary. Alternative solvers and raw
   matrices remain available, but should not be prerequisites.
2. **Geophysical names before array conventions.** Public results should expose
   `strike_slip`, `dip_slip`, station predictions, and coordinate frames by
   name. Blocked vectors and stacked rows remain available as explicit linear-
   algebra views.
3. **Units and frames are never implicit.** SI units remain the default. Every
   non-SI parameter carries a suffix, and every local coordinate array is tied
   to a declared origin and axis convention.
4. **Reveal complexity progressively.** A novice can use sensible defaults; a
   researcher can inspect or replace the Green's matrix, noise model,
   regularizer, parameterization, backend, and solver.
5. **Teach the model, including its limits.** Documentation explains the
   assumptions behind a result and makes covariance, resolution, geometry
   uncertainty, and prior sensitivity visible.
6. **Optional power stays optional.** JAX, BlackJAX, Cartopy, mesh generation,
   and future cycle/layered-earth dependencies must not burden the base install.
7. **Stable public surface, replaceable internals.** Refactors preserve public
   imports and numerical behavior. Deprecations include warnings, migration
   examples, and at least one minor-release transition period.
8. **Array transparency without array traps.** NumPy arrays remain accepted and
   returned where natural, but common shape/order mistakes should be prevented
   by named accessors and precise validation.
9. **Functions are the default abstraction.** Add a public class only when it
   owns durable state, preserves an invariant that functions cannot preserve,
   or amortizes expensive preparation across calls. Parameter bundles and
   transient array views stay as keyword arguments, arrays, and module
   functions.

---

## Structural audit (2026-07)

### What already serves learners well

- The core relation `d = G m` is visible rather than hidden behind a framework.
- `Fault.planar(...)`, `fault.displacement(...)`, and `geodef.solve(...)` form
  a compact path from geometry to a solution.
- NumPy is the default; advanced compilation and sampling are opt-in.
- Durable domain objects are immutable, synthetic tutorials are reproducible, and the
  teaching sequence follows the concepts of an inverse problem rather than the
  package's module layout.
- Low-level engines are independently accessible and extensively cross-checked,
  which is valuable for both learning and research verification.

### Friction and design debt

1. **Internal storage conventions leak into ordinary use.** Slip is often a
   blocked `2N` vector, observations from multiple datasets are one stacked
   vector, and users manually slice predictions before plotting. The same
   result also exposes an `(N, 2)` slip array, creating two equally prominent
   representations.
2. **There are multiple workflow levels but no explicit map between them.** A
   user can call `geodef.invert`, construct `LinearSystem`, call
   `greens.greens`, or use `Fault.greens_matrix`; the distinctions are sound,
   but discovery currently depends on reading several module references.
3. **Coordinates are correct but cognitively expensive.** Public calls mix
   geographic `lat/lon/depth`, local ENU vertices, a seven-element local
   geometry vector, depth-positive-down arrays, and kernel-native coordinates.
   Ordering is inconsistent: `Fault.planar` starts with `lat, lon`, dataset
   constructors start with `lon, lat`, `Fault.centers` stores `[lat, lon,
   depth]`, and `Mesh.centers_geo` stores `[lon, lat, depth]`. The reference
   origin is sometimes carried separately from the values.
4. **Units are not fully uniform.** Most geometry is in meters, but some mesh
   APIs use kilometers, while all values are plain floats. This is easy to miss
   in a notebook and hard to detect after the fact.
5. **Data construction is array-heavy.** A 3-component `GNSS` constructor has
   eight positional arrays, InSAR look vectors must already be components, and
   there is no named table/column ingestion layer. These interfaces are precise
   but unfriendly at the point where many novices first meet the package.
6. **Results do not retain enough problem context.** `InversionResult` cannot
   split predictions by dataset, plot itself, reproduce the solve, or report
   the assumptions that produced it without the caller re-supplying the fault
   and datasets. `chi2` is a reduced chi-squared value, while diagnostics also
   use `chi2` for the unreduced statistic.
7. **Regularization terminology has drifted.** Runtime code consistently uses
   `lambda * ||Lm||^2` (and `sqrt(lambda) * L` in the augmented system), while
   parts of the tutorial outline still write `lambda^2 * ||Lm||^2`. The public
   name `smoothing_strength` also covers damping, stress, and custom priors.
8. **Noise handling is mathematically general but operationally dense.** Full
   covariance matrices are materialized, inverted, and block-stacked. This is
   simple for teaching-sized problems but becomes a memory and numerical
   bottleneck for realistic InSAR, and it offers no natural home for diagonal,
   sparse, low-rank, or parametric noise models.
9. **Large modules obscure responsibilities.** `bayes.py`, `invert.py`,
   `plot.py`, and `fault.py` each contain several separable subsystems. Their
   public APIs can remain stable while implementation is divided into smaller
   units with explicit dependency direction.
10. **Packaging and documentation metadata can drift.** The mesh documentation
    requires `netCDF4`, but the `mesh` extra currently installs only `meshpy`;
    tutorial status text still refers to ten notebooks in places; recorded test
    counts become obsolete quickly; and examples are not all held to the same
    execution contract.
11. **Validation focuses more on shapes than physical plausibility.** Public
    constructors should consistently detect non-finite values, invalid dip or
    depth, non-unit look vectors, non-symmetric/non-positive covariance, empty
    datasets, degenerate triangles, and incompatible coordinate frames, with
    messages that name the bad field and expected units.
12. **The research API has grown faster than the beginner vocabulary.** The JAX
    and Bayesian work is powerful, but geometry priors, parameter-vector layout,
    backend state, initialization, and posterior diagnostics demand substantial
    package knowledge. A guided setup layer can reduce this without weakening
    the expert API.
13. **Cache keys are incomplete relative to the computation's inputs.** The
    Green's-matrix cache key omits Poisson's ratio, the backend precision, and
    any kernel or format version. Today `nu` is unreachable from the `Fault`
    path so keys cannot collide yet, but a float32 run and a float64 run share
    a key, and any future kernel fix would silently serve stale matrices. The
    cache is a trust feature; its keys must be provably complete.
14. **Elastic material parameters are implicit.** `nu=0.25` is hard-coded
    below `Fault.greens_matrix`, and `mu=30e9` appears as an independent
    default in `moment`, `magnitude`, and `stress_kernel`. There is no single
    place where a user declares the elastic medium, which also blocks the
    planned layered-earth engines from having a clean parameter home.
15. **Release and legal scaffolding is missing.** The repository has no
    LICENSE, no citation metadata, no CI configuration at all, no `py.typed`
    marker despite a fully typed and mypy-clean API, and the version string is
    duplicated between `pyproject.toml` and `__init__.py`. The ported engines
    derive from the author's own MIT/BSD sources (`geometry/`), so licensing
    is tractable — but it must be declared before any public release.
16. **Top-level names collide.** After `import geodef`, `geodef.invert` is the
    function and shadows the `invert` module, while `geodef.data` and
    `geodef.fault` remain modules — so `geodef.invert.LinearSystem` fails and
    docs that qualify names through the module path silently break. Three
    different callables are named `resolution` (`greens.resolution`,
    `plot.resolution`, `model_resolution`), and `plot.map` shadows a builtin.
17. **Taught workflows lack API support.** The tutorials teach checkerboard
    and spike resolution tests, but the package offers no synthetic-test
    helpers; every student re-implements the pattern from notebook cells.
18. **Interseismic coupling has no dedicated vocabulary.** Coupling is one of
    the package's two declared target applications, yet there is no backslip
    convention, no coupling-fraction (0–1) parameterization, no plate-rate
    bound helper, and no moment-deficit-rate output; `euler` block velocities
    are not connected to the inversion layer.

---

## Priority 0 — Consistency and trust (small, high-impact work)

Complete these before adding a new high-level abstraction. They establish the
semantics used by the function-oriented public API.

### 0.1 Audit and freeze mathematical conventions

- [x] Declare one regularization convention everywhere:
  `Phi = r.T @ W @ r + lambda * ||L(m - m_ref)||^2`, with augmented rows
  `sqrt(lambda) * L`. Correct the tutorial equations and define how published
  sources using `alpha` or `lambda^2` map to GeoDef.
- [x] Add a convention test spanning direct inversion, `LinearSystem`, ABIC,
  geometry search, and Bayesian fixed-lambda modes.
- [x] Define unambiguous vocabulary: `reduced_chi2` is the reduced statistic and
  `chi2` is the unreduced statistic. Rename the pre-release
  `InversionResult.chi2` field directly to `reduced_chi2`; reserve `chi2` for
  unreduced values such as those exposed by dataset diagnostics.
- [x] Write a single reference page for coordinate axes, depth sign, strike,
  dip, rake, slip azimuth, row order, column order, and units; link every public
  geometry/data API to it.
- [x] Settle one geographic ordering policy. Make ambiguous multi-array calls
  keyword-only over a deprecation cycle, add explicitly named coordinate
  accessors, and never silently reinterpret existing positional arguments.

### 0.2 Make invalid physical inputs fail early

- [x] Centralize validation helpers for one-dimensional numeric arrays, finite
  values, broadcast rules, angle ranges, positive dimensions/uncertainties, and
  matching lengths. Error messages must identify the argument, received shape
  or range, and required unit.
- [x] Validate covariance symmetry and positive definiteness with a useful
  remediation message; provide an explicit escape hatch only for advanced
  semidefinite/operator cases.
- [x] Validate or explicitly normalize InSAR look vectors; provide a diagnostic
  for likely satellite-to-ground versus ground-to-satellite sign reversal.
- [x] Add `Fault.validate()`, `DataSet.validate()`, and `Mesh.validate()` reports
  for interactive workflows, including patch/triangle degeneracy, above-surface
  sources, extreme aspect ratios, duplicate stations, and coordinate bounds.
- [x] Replace user-triggerable `assert` statements in public paths with typed,
  informative exceptions; keep trace-only kernel assertions private.

### 0.3 Repair packaging, licensing, and documentation drift

- [x] Choose and add a LICENSE, confirm attribution for the ported engines
  (the `geometry/` sources are the author's own MIT/BSD code), add a
  `CITATION.cff`, and complete `pyproject.toml` metadata (license, authors,
  URLs, classifiers, readme) before any public release.
- [x] Stand up CI from scratch (none exists): ruff, mypy, the routine test
  suite, and executed tutorials, on every push.
- [x] Ship a `py.typed` marker so downstream type checkers see the existing
  annotations, and single-source the package version instead of duplicating
  it between `pyproject.toml` and `__init__.py`.
- [x] Make optional extras match actual imports (`meshpy`, `netCDF4`, `pyproj`,
  Cartopy, JAX, and BlackJAX), and test each documented install tier in CI.
- [x] Test the declared Python/NumPy/SciPy version range, including the oldest
  supported and newest released combinations; calibrate reference tolerances
  from physical/numerical requirements rather than one platform's roundoff.
- [x] Remove hard-coded test counts and stale ten-versus-eleven tutorial status
  text. Add one authoritative capability/version table.
- [x] Add an API documentation check for broken examples, missing public
  members, and signature drift. Prefer executable short examples over copied
  signatures.
- [x] Add a changelog and a written compatibility/deprecation policy before the
  next public release.

### 0.4 Make caching and material parameters provably safe

- [x] Include every input that affects a cached result in its key: elastic
  parameters, backend precision, and a kernel/format version stamp that is
  bumped whenever an engine's numerics change. Add tests asserting that each
  varied input produces a cache miss.
- [x] Define the cache invalidation story: what a version bump invalidates,
  how users clear stale entries, and what `cache.info()` reports about them.
- [x] Thread Poisson's ratio through `Fault.greens_matrix` and the assembly
  layer instead of freezing the kernel default, and give `mu` one declared
  home shared by `moment`, `magnitude`, and `stress_kernel`. Design this as a
  small elastic-medium parameter object so layered engines (6.2) extend it
  rather than adding parallel keyword arguments.

---

## Priority 1 — A small, function-oriented everyday API

The public vocabulary should remain smaller than the set of concepts in the
implementation. GeoDef keeps objects for durable domain state and prepared
computations; ordinary transformations and one-shot workflows are functions.

### 1.1 Set and enforce the public object budget

- [x] Keep `Fault`, `GNSS`/`InSAR`/`Vertical`, `Mesh`, `LocalFrame`, and
  `ElasticMedium`: each carries durable identity or invariants shared by many
  computations. Keep immutable result records where named fields prevent tuple
  or ordering mistakes.
- [x] Remove the draft `PlanarGeometry` and `TriGeometry` public wrappers before
  release. `Fault` already represents rectangular or triangular geometry;
  retain the useful validation and conversions as functions.
- [x] Replace the draft `SlipModel` and `Displacement` wrappers with `slip`
  conversion functions, ordinary arrays, direct named `InversionResult` views,
  and the existing three-array displacement return.
- [x] Keep `LinearSystem` as an expert prepared/cache object for repeated
  sweeps and assessment, not as the beginner workflow and not behind a second
  `SlipProblem` facade.
- [x] Require an explicit justification in API review for every new public
  class: durable state, enforced cross-call invariant, or measured reuse of an
  expensive preparation. Do not add classes solely to bundle keyword arguments.

### 1.2 Give the functional namespaces memorable names

- [x] Make the module path the primary discovery surface. Prefer specific names
  over umbrella verbs: `geodef.invert.solve`, `lcurve`, `abic_curve`,
  `diagnostics`, and `model_covariance`; `geodef.greens.matrix`,
  `project`, and `laplacian`; `geodef.slip.pack`, `unpack`, `from_rake`,
  `from_azimuth`, `from_plate`, `to_plate`, `magnitude`, and `rake`.
- [x] Resolve the top-level `geodef.invert` function/module collision directly
  before release: `geodef.invert` is the module, `geodef.invert.solve(...)` is
  the primary call, and `geodef.solve(...)` is the short alias. No deprecation
  shim or callable-module proxy is needed because the draft API has no users.
- [x] Represent a slip basis with explicit function keywords (`components`,
  `rake`, `slip_azimuth`, `plate_rake`) and conversion functions. Do not create
  `SlipBasis`, `Regularization`, or `Bounds` configuration classes.
- [x] Standardize patch ordering utilities and provide `fault.reshape_patches`
  / `fault.flatten_patches` rather than requiring learners to know which grid
  axis varies fastest.

### 1.3 Keep coordinates named without duplicating geometry

- [x] Attach one immutable `LocalFrame` to every `Fault` and `Mesh` that owns
  local coordinates; reject incompatible frames unless explicitly transformed.
- [x] Put conversions in `transforms`/`geometry` functions accepting
  `frame=...`. Keep `Fault.planar(...)` keyword geometry and
  `Fault.from_triangles(..., frame=...)` as the named construction paths.
- [x] Replace unexplained seven-element geometry arrays in beginner docs with
  keyword calls and named result fields. Advanced JAX/Bayesian functions may
  retain `theta`, but must accept a mapping keyed by `e0`, `n0`, `depth`,
  `strike`, `dip`, `length`, and `width` in addition to the array view.
- [x] Return optimized geometry as a `Fault` plus the expert `theta` array and
  frame on the existing result record, rather than introducing a parallel
  geometry value hierarchy.

### 1.4 Result records plus assessment functions

- [x] Keep `InversionResult` a compact, serializable data record. Add direct
  named slip views (`strike_slip`, `dip_slip`, `slip_magnitude`, `slip_rake`)
  where they are unambiguous; keep `slip_vector` as the blocked expert view.
- [x] Record dataset names and row slices, solver status, regularization
  selection, backend, warnings, and minimal provenance needed to interpret and
  reproduce a solve. Do not retain live `Fault` or dataset objects in results.
- [x] Add module functions `invert.prediction`, `invert.residual`,
  `invert.diagnostics`, and `invert.summary`, plus corresponding `plot`
  functions. Do not turn the result record into a workflow facade.
- [x] Define a versioned, safe result file schema with metadata and migration;
  retain `.npz` portability and add a human-readable manifest.

### 1.5 Friendlier data functions

- [x] Add `data.gnss`, `data.horizontal_gnss`, `data.insar`, and
  `data.vertical` functions with keyword-only component names and sensible
  defaults. They return the existing validated dataset classes; class
  constructors remain available for compatibility.
- [x] Add `data.from_table` with explicit column mappings, units,
  missing-value handling, and station names. Keep dataframe libraries optional
  and accept the Python dataframe interchange protocol rather than coupling the
  core to one implementation.
- [x] Separate displacement from velocity semantics in metadata (units and
  epoch/time span) without duplicating all dataset classes.
- [x] Introduce dataset names as first-class identifiers so joint results and
  plots are stable and readable.

### 1.6 Pre-documentation breaking-change batch

The draft API still has no users, so these remaining renames and unit fixes
land directly, without deprecation shims — the precedent set in 1.2. They must
be complete before Priority 2 bakes today's names into the quickstart, course,
and examples; every one becomes a migration burden after v0.2.

- [x] Resolve the remaining name collisions from the structural audit: rename
  `greens.resolution` to `greens.resolution_matrix`; keep
  `invert.model_resolution` as the primary assessment call and
  `plot.resolution` as the plotting name (no longer ambiguous once the
  `greens` name is specific). Rename `plot.map`, which shadows the builtin
  and reads like the `geomap` module, to `plot.map_view`.
- [x] Standardize regularization vocabulary on the math the tutorials teach
  (`Phi = r.T @ W @ r + lambda * ||L(m - m_ref)||^2`): rename the
  `smoothing=` keyword to `regularization=`, `smoothing_strength=` to
  `regularization_strength=`, and `smoothing_target=` to
  `regularization_target=` across `invert`, `bayes`, and `plot`, including
  the `InversionResult` field and the result-file schema (bump the schema
  version and migrate the old key on load). The values `'laplacian'`,
  `'damping'`, and `'stresskernel'` are unchanged; `'damping'` is no longer
  described as a kind of smoothing.
- [x] Remove the legacy latitude-first `Fault.centers`; `centers_geo` and
  `centers_local` remain as the two documented accessors. Update the notebook
  cells that still use it.
- [x] Unify mesh units on meters: `mesh.from_slab2` takes `target_length` and
  `max_depth` in kilometers while `from_polygon`/`from_trace` take meters, and
  `from_trace` mixes a kilometer `max_depth` with a meters-argument `dip`
  callable in one signature. Convert every public mesh length/depth argument
  to meters, per the enduring unit convention.
- [x] Make `invert.solve` arguments keyword-only after `fault` and `datasets`,
  matching the `Fault.planar` policy, so `solve(fault, data, 0.5)` cannot
  silently pass a strength where the regularization type belongs.
- [x] Decide the API stability tiers now (beginner-public, expert-public,
  private) and trim the top-level `__all__` to the documented set, so
  Priority 2 documents the final vocabulary. The expert names stay importable
  as `geodef.<name>` for now; removing those top-level re-exports is folded
  into the 2.2 notebook rewrite so the affected cells are edited once.
  Publishing the map and the dependency-direction and import-cycle test work
  remain in 3.1.

Settled naming decision: `greens.matrix` keeps its name. `design_matrix` was
considered and rejected — the module path already says Green's, the package
prefers geophysical vocabulary over statistics vocabulary, and
`Fault.greens_matrix` provides the spelled-out form where no module qualifier
is present.

---

## Priority 2 — Learning experience and documentation architecture

`tutorials/OUTLINE.md` is the design document for this phase: it plans the
textbook-depth course revision (fifteen chapters 00–14 after two merges and a
one-time renumbering, chapter anatomy, solution notebooks), the homes for the
2.1 "start here" artifacts (`docs/quickstart.md`, `docs/glossary.md`, and a
combined workflow-map/decision-guides `docs/workflow.md`), and the
`geodef.synthetic` helper module (with recovered-versus-input comparison as
`invert.compare`, unaliased). Its §11 records the settled design decisions,
including the backslip/coupling convention (the 4.4 convention sub-item,
pulled forward so it is defined once) and the tutorial-suite CI budget. The
outline is transient: when this phase ships, durable content migrates to the
glossary, `docs/conventions.md`, and release notes.

### 2.1 Build a true “start here” path

- [x] Create a five-minute, copy-paste quickstart that performs forward
  modeling, adds synthetic noise, solves slip, and plots observations versus
  predictions without manual vector packing or slicing.
- [x] Add a visual workflow page linking the three levels of API:
  domain functions → matrices/operators → physics kernels.
- [x] Add a glossary of geophysical and inverse-theory terms, with package names
  beside the mathematical symbols.
- [x] Provide “which function do I use?” and “which assumption am I making?”
  decision guides for geometry, slip basis, regularization, covariance,
  constraints, geometry uncertainty, and Bayesian inference.

### 2.2 Revise the course around the improved API

- [x] Preserve the equation-first pedagogy and manual `G @ m` demonstrations,
  but use named results for routine operations so students only manipulate
  ordering when ordering is the lesson.
- [x] Add explicit learning objectives, prerequisites, estimated time, recap,
  and tested exercises to every tutorial; publish solution notebooks separately.
- [x] Add a preflight notebook covering arrays, shapes, broadcasting, plotting,
  units, and coordinate conventions for geophysicists new to scientific Python.
- [x] Add short conceptual notebooks or examples for triangular faults,
  interseismic coupling, model misspecification, prior sensitivity, and
  posterior diagnostics. Keep advanced JAX/Bayesian material outside the core
  novice sequence unless it teaches a general concept.
- [ ] Replace repeated synthetic setup cells with a documented scenario builder
  only after students have seen the explicit construction once.
- [ ] Add synthetic-test helpers — checkerboard and spike slip patterns, noisy
  synthetic data generation with declared seeds, and recovered-versus-input
  comparison — so the resolution-testing workflow the tutorials teach is a
  supported API rather than notebook-only code.
- [ ] Finish the 1.6 top-level export reduction here, while the same cells are
  being rewritten, so each is edited once. 1.6 trimmed `geodef.__all__` to the
  beginner vocabulary but left the expert names (`lcurve`, `abic_curve`,
  `model_covariance`/`model_resolution`/`model_uncertainty`, `geometry_search`,
  `compute_abic`, `LinearSystem`, the `*Result` types, `stack_obs`/
  `stack_weights`/`select_slip_columns`, `spatial_covariance`, and the moment
  conversions) importable as `geodef.<name>` via redundant aliases. Migrate the
  notebooks, examples, and `docs/*.md` to the module path
  (`geodef.invert.lcurve`, `geodef.greens.stack_obs`,
  `geodef.data.spatial_covariance`, ...), then remove those top-level re-exports
  from `geodef/__init__.py`. About two dozen executed code lines across ~8
  notebooks change in lockstep, so land the removal and the migration in one
  commit to keep the tutorial suite green; the public-API contract test in
  `tests/test_public_api.py` moves those names from the "still importable"
  assertion to a "no longer top-level" one. (Do this before any public `0.2`
  tag, since a release makes the top level user-visible.)

### 2.3 Make real workflows reproducible

- [x] Convert examples to a uniform structure: question, data provenance,
  assumptions, preprocessing, model setup, validation, interpretation, and a
  machine-executed reduced-size path.
- [ ] Add one end-to-end interseismic coupling example and one earthquake
  example with nuisance parameters and correlated noise. Sequencing: these
  two examples depend on the 4.1 noise operators, 4.3 nuisance bases, and
  4.4 coupling vocabulary. Deliver the rest of Priority 2 against the
  post-1.6 API first and return for these once those pieces exist, rather
  than pulling all of Priority 4 forward.
- [x] Add reproducible environment metadata and deterministic seeds to every
  executable example; distinguish downloaded data from bundled test fixtures.
- [ ] Build a searchable documentation site from the existing Markdown and
  docstrings only after navigation and content hierarchy are settled.

### 2.4 Test usability, not only correctness

- [ ] Define three golden workflows (first forward model, first inversion,
  joint GNSS+InSAR study) and test their complete public call sequences.
- [x] Track beginner-facing metrics: number of required concepts/imports,
  manual reshapes/slices, ambiguous unit-bearing arguments, warning quality,
  and time to a labeled diagnostic plot.
- [ ] Run periodic observation sessions with novice geophysicists; convert each
  repeated confusion into an API, error-message, or documentation test.

---

## Priority 3 — Legible and extensible internals

This work must be behavior-preserving and land in small extraction commits. Do
not reorganize numerical reference ports merely to make their style conventional.

### 3.1 Establish package layers and public boundaries

- [ ] Publish an API stability map: beginner-public, expert-public, and
  private. The tier decision and top-level `__all__` trim happen in 1.6, and
  the actual removal of the expert top-level re-exports is done in 2.2 with the
  notebook rewrite, so documentation here targets the final set; this item
  publishes the map and keeps advanced modules importable. (The name collisions
  this item originally covered — the `invert` function/module shadow, three
  `resolution` callables, `plot.map` — were resolved directly in 1.2 and 1.6
  while the API had no users.)
- [ ] Define dependency direction: domain types → operators/problem assembly →
  solvers/results, with plotting and I/O at the edges and kernels below all of
  them. Remove imports through `geodef.__init__` from internal modules.
- [ ] Add import-cycle, base-install, optional-import, and public-API snapshot
  tests. Importing `geodef` must not initialize JAX or require optional stacks.

### 3.2 Split large modules behind stable re-exports

- [ ] Split `invert.py` into result types, system assembly, regularization,
  solvers, hyperparameter selection, diagnostics, and nonlinear geometry.
- [ ] Split `bayes.py` into posterior models, slip transforms, geometry
  parameterizations, samplers, diagnostics, and result types.
- [ ] Split `fault.py` into core geometry, factories, I/O adapters, and forward
  conveniences; split `plot.py` by geometry, data, fit, and assessment plots.
- [ ] Deduplicate data save/load logic and validation without introducing a deep
  inheritance hierarchy.
- [ ] Keep `okada85.py`, `okada92.py`, and `tri.py` visibly traceable to their
  published sources; wrap them with clearer adapters rather than cosmetically
  rewriting formulas.

### 3.3 Replace string dispatch with callable contracts

- [ ] Define typed function signatures for Green's engines, data projection,
  noise whitening, regularization operators, and solvers. Accept callables and
  SciPy-style operators directly; introduce a public protocol type only where
  static typing materially improves extension safety.
- [ ] Register engines explicitly instead of expanding `if engine == ...`
  branches across `Fault`, `greens`, gradients, Bayesian code, and plotting.
- [ ] Require engine capability declarations (surface/internal displacement,
  strain, autodiff, supported source geometry) and produce actionable errors
  when a workflow requests an unsupported capability.
- [ ] Avoid a plugin framework until at least two external engines demonstrate
  the callable contract; start with ordinary functions and registration.

### 3.4 Strengthen numerical contracts

- [ ] Add array shape/dtype/backend contracts at module boundaries and property-
  based tests for packing, projection, coordinate round trips, and linearity.
- [ ] Add conditioning diagnostics and stable solve fallbacks. Do not use normal
  equations solely for speed when their squared condition number can change the
  answer; benchmark QR/SVD/Cholesky choices on representative problems.
- [ ] Separate exact numerical equivalence tests from tolerance-based physical
  validation and from performance benchmarks.
- [ ] Record benchmark problem definitions, compilation cost, steady-state cost,
  memory, backend, precision, and hardware; never report a single speedup number
  without those qualifiers.

---

## Priority 4 — Scale to real geodetic datasets

### 4.1 Noise and whitening operators

- [ ] Accept a whitening callable or `LinearOperator` and provide constructor
  functions for diagonal, dense, block-diagonal, sparse, and low-rank-plus-
  diagonal cases. Do not require users to adopt a `NoiseModel` class hierarchy.
- [ ] Solve via whitening or factorizations rather than explicitly forming
  `W = C^-1`. Preserve `stack_weights()` as an educational/small-problem helper.
- [ ] Add parametric spatial covariance fitting, variograms, and honest
  diagnostics; keep covariance estimation distinct from slip inversion so the
  assumptions are visible.
- [ ] Support dataset-specific variance scale factors and hierarchical noise
  scales in deterministic and Bayesian workflows.

### 4.2 Linear operators and large inversions

- [ ] Allow Green's matrices and regularizers to be dense arrays, sparse arrays,
  or `LinearOperator`-like objects; add chunked/memory-mapped assembly for large
  InSAR scenes.
- [ ] Add iterative least-squares and constrained solver paths with convergence
  reports and preconditioning. Dense direct solves remain the transparent
  default for teaching-sized systems.
- [ ] Add reusable factorizations and batched solves for hyperparameter sweeps;
  report estimated memory before allocating dense covariance or Green's arrays.
- [ ] Benchmark and document the scale boundary where users should downsample,
  use operators, or move to JAX/GPU.

### 4.3 Observation preprocessing and nuisance parameters

- [ ] Add explicit linear nuisance bases for InSAR offsets/ramps, GNSS frame
  translations/rotations, and dataset biases; solve or marginalize them without
  mixing them into fault slip.
- [ ] Provide transparent quadtree/spatial downsampling and train/validation
  partition helpers that preserve coordinates, uncertainties, and provenance.
- [ ] Add masking/subsetting/concatenation methods returning immutable datasets
  with provenance rather than encouraging parallel manual array slicing.
- [ ] Support multiple epochs and temporal basis functions as a bridge from
  static inversion to time-dependent slip.

### 4.4 First-class interseismic coupling

Coupling is a declared target application and should not remain an exercise in
manually re-signing slip vectors.

- [ ] Define and document the backslip convention once: sign, rake/azimuth
  relationship to plate motion, and how a backslip inversion maps onto the
  existing slip basis and bounds machinery.
- [ ] Add a coupling-fraction parameterization: solve for coupling in [0, 1]
  against a per-patch plate-rate vector, with the plate rate derivable from
  `euler` block models so rigid-block prediction and coupling inversion share
  one vocabulary.
- [ ] Report moment-deficit rate (and its uncertainty) alongside moment, with
  explicit units and epoch semantics tied to the velocity metadata from 1.5.
- [ ] Deliver the Priority 2 interseismic example through this API and promote
  it to a golden workflow once stable.

### 5.1 Close the remaining JAX gaps

- [ ] Batch L-curve and cross-validation sweeps on JAX using the established
  ABIC pattern; verify exact API/numerical parity with NumPy.
- [ ] Make rectangular and triangular strain/stress kernels traceable,
  gradient-safe, jitted, and vmapped; validate derivatives against finite
  differences away from documented singular boundaries.
- [ ] Remove hidden dependence on global backend state from compiled kernels
  and prepared systems while retaining `set_backend(...)` as the simple entry
  point.
- [ ] Add compilation-cache guidance and shape-change diagnostics so users can
  distinguish compilation time from solve time.

### 5.2 Make advanced geometry inference easier to set up safely

- [ ] Accept geometry keyword mappings, an explicit `LocalFrame`, and prior
  mappings with units; generate prior-predictive geometry plots and half-space
  checks before sampling.
- [ ] Add multi-start geometry search and an initialization helper that can seed
  NUTS from deterministic fits while clearly separating optimization from
  posterior inference.
- [ ] Add posterior predictive checks, rank/trace plots, divergence summaries,
  prior-versus-posterior comparisons, and warnings with concrete remedies.
- [ ] Support multiple smoothing scales (by component, region, or operator) and
  dataset noise scales without forcing users to construct parameter vectors.

### 5.3 Robust sampling and model comparison

- [ ] Implement `bayes.sample_smc` around BlackJAX adaptive tempered SMC with
  prior draws, NUTS mutation, an adaptive temperature schedule, weighted
  posterior output, and log-evidence estimates.
- [ ] Validate SMC against analytic low-dimensional targets and NUTS on unimodal
  cases, then use it for deliberately multimodal geometry examples.
- [ ] Define sampler-independent result records and sampling functions so
  BlackJAX API changes or future samplers do not leak through the GeoDef user
  interface.

---

## Priority 6 — New physics and research capabilities

These efforts should use the engine/operator interfaces above so new physics
does not multiply special cases in beginner-facing code.

### 6.1 Quasi-dynamic earthquake-cycle modeling

- [ ] Write a separate design note defining scope, state variables, sign/unit
  conventions, validation targets, and the boundary between static GeoDef
  values and a new optional `geodef.cycle` module.
- [ ] Port and independently validate stress-kernel-driven quasi-dynamic
  rate-and-state evolution from `related/stress-shadows/unicycle/`.
- [ ] Add friction-law functions/callables, adaptive ODE integration, event
  detection, restart/checkpoint files, and energy/moment diagnostics. Use a
  state record only if checkpointing invariants require one.
- [ ] Support rectangular and triangular faults, CPU first and differentiable
  JAX integration only after the reference CPU implementation is trusted.
- [ ] Deliver a small pedagogical spring-slider example before a large fault-
  system example.

### 6.2 Additional Green's engines

- [ ] Add Meade (2007) triangular dislocations as an independent cross-check.
- [ ] Add compound dislocation / point-source models for volcanic deformation.
- [ ] Add layered half-space displacement Green's functions behind an optional
  dependency, beginning with a well-bounded elastic layering use case.
- [ ] Evaluate viscoelastic and poroelastic engines only after source/engine
  callable contracts can represent time and material parameters cleanly.
- [ ] For every engine: cite equations, preserve a reference implementation,
  cross-validate published cases, declare capabilities/coordinate conventions,
  and show one end-to-end example through the same high-level workflow.

### 6.3 Richer fault and slip models

- [ ] Add tensile/opening components without disturbing the two-component
  default; make basis and moment semantics explicit.
- [ ] Add multiple faults/segments with continuity or boundary constraints and
  named result partitions.
- [ ] Add mesh quality metrics, adaptive refinement driven by geometry and data
  sensitivity, and transfer operators between meshes.
- [ ] Add elastic-parameter sensitivity and uncertainty before exposing joint
  inversion for elastic structure.
- [ ] Explore kinematic time-dependent slip histories after multi-epoch data and
  nuisance bases are established.

---

## Delivery sequence

The priorities are ordered deliberately, but each phase should deliver useful
increments rather than becoming a long-lived rewrite.

1. **v0.1 foundation:** Priorities 0 and 1 — consistency, validation,
   packaging/licensing/CI, safe caching, explicit material parameters, and the
   small function-oriented everyday API.
2. **v0.2 learning release:** Priority 2's quickstart, revised course,
   reproducible examples, and usability tests.
3. **Later 0.x releases:** implementation boundaries, callable operators,
   nuisance parameters, large-problem diagnostics, and the remaining JAX and
   Bayesian work. `LinearSystem` remains the optional prepared-system API for
   repeated analyses.
4. **v1.0:** only after this roadmap is complete and the resulting public
   workflows have gone through human testing. New engines or a large feature
   count do not by themselves justify 1.0.

For each user-facing phase, require:

- a before/after golden workflow showing fewer manual transformations;
- unit, integration, documentation, and backwards-compatibility tests;
- an executed tutorial or example using the public path;
- numerical equivalence against the current implementation where semantics are
  unchanged;
- release notes with migration examples and explicit non-goals.

---

## Enduring design conventions

- Public Cartesian coordinates are East, North, Up; public depth is positive
  down. Kernel-native conventions are converted at adapters.
- Base geometry lengths, displacements, and slip use meters; stress uses Pa;
  angles use degrees unless an API name explicitly says otherwise.
- The expert linear-algebra slip view remains blocked as
  `[:N]` strike-slip and `[N:]` dip-slip. Named domain views are preferred in
  everyday code.
- NumPy remains the default runtime and JAX remains optional.
- Hash-based caching may accelerate pure computations but must never change
  results or hide problem provenance. Cache keys include every input that
  affects the result — geometry, observation locations, material parameters,
  precision, and a kernel version — or the computation is not cached.
- `PLAN.md` is forward-looking. When work ships, replace detailed checklists
  with a concise baseline entry or release note rather than accumulating an
  implementation diary here.
- Update this plan in the same logical commit when scope or a settled design
  decision changes.
- Do not add `Co-Authored-By` trailers. AI model attribution is maintained in
  `README.md`, `AGENTS.md`, and `CLAUDE.md` according to repository policy.
