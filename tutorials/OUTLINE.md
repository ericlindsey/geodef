# GeoDef Tutorial Course — Master Outline

This document is the **design plan for the tutorial course**. It is the single
source of truth for what each chapter teaches, the order concepts are
introduced, the math that must appear, and the code each chapter uses. Write or
revise notebooks against this outline; revise this outline first if the plan
changes.

> **Status: course and documentation delivered 2026-07 for the v0.2 learning
> release** (`PLAN.md` Priority 2). The textbook-style chapters 00–14, frozen
> numbering, merged regularization chapter, module-path documentation calls,
> separate worked solutions, start-here layer, and reproducible example
> structure are implemented and manually executed. Runtime/test work is
> deliberately outside this docs-only branch: `geodef.synthetic`,
> `invert.compare`, removal of redundant top-level aliases, and atomic pytest
> enumeration remain open in `PLAN.md`. Compatibility notebook links keep the
> existing execution harness green until that test migration lands.
>
> **Lifecycle.** This outline is a working document, not a permanent one.
> Once the v0.2 course ships, its durable content migrates to its real
> homes — notation to `docs/glossary.md`, conventions to
> `docs/conventions.md`, settled decisions to `PLAN.md` release notes —
> and this file is retired to git history.

---

## 1. Purpose and Audience

These notebooks are a **course in geodetic inverse methods**, taught through
the GeoDef library — not a tour of its modules. A reader who finishes the
course should understand how surface geodetic data (GNSS, InSAR) are turned
into images of fault slip, *why* each step is done, and what the results can
and cannot be trusted to say. GeoDef is the vehicle; the destination is the
methods.

The course is written for two overlapping audiences:

- **Geophysics students and researchers** who know the physics vocabulary but
  may be new to inverse theory, linear algebra in practice, or scientific
  Python. Chapter 00 exists for them.
- **Quantitative scientists from other fields** who know least squares but not
  faults. The geophysical context (dislocations, moment, coupling) is
  developed rather than assumed.

### A course, not a demo gallery

The target register is a **textbook chapter, not a README**: each chapter
should read as a self-contained lesson that develops its subject with enough
prose, derivation, and worked interpretation that the markdown alone teaches
the method. This is a deliberate deepening relative to the first generation of
the notebooks. Concretely:

- **Theory is primary and developed, not stated.** Key results are derived
  (e.g. the weighted-least-squares estimator from the objective's gradient,
  the posterior covariance, the resolution matrix), with the intuition for
  *why* the result looks the way it does, not just the formula. Where a full
  derivation would derail the chapter, a boxed sketch plus a reference is
  acceptable — but "here is the equation, here is the call" is not.
- **Interpretation is taught.** Every worked demonstration ends with prose
  interpreting the figure: what the reader should see, why it looks that way,
  and what would change it. Plots are never left to speak for themselves.
- **Code is minimal and illustrative.** Snippets stay short — usually a few
  lines — and exist to *demonstrate* the concept just explained, not to build
  a production workflow. Depth goes into markdown, math, and figures, **not**
  into longer scripts or heavier computation.
- **Each chapter targets roughly 45–90 minutes of engaged study** (reading,
  running, exercises). Topics chapters (Part VI) may be shorter (30–60
  minutes); the two merged chapters (04 and 14) are the course's heaviest and
  may run toward two hours — if either overruns that in practice, §11
  decision 1 records the fallback split. If any other chapter outgrows ~90
  minutes it should be split or material moved to `examples/`.
- **Data is synthetic and reproducible.** Chapters generate their own small
  geometries and noise with fixed seeds so they run fast, execute under
  pytest, and isolate the concept being taught. Depth must never come from
  compute: the executed suite stays inside the CI budget (§11 decision 9).
- **Each chapter ends with exercises**, now backed by separately published
  solution notebooks (§9).

### Tutorials vs. Examples

| | `tutorials/` | `examples/` |
|---|---|---|
| Goal | Teach one method per chapter | Show a complete real workflow |
| Data | Synthetic, tiny, seeded | Real, bundled or downloaded |
| Code | Short snippets | Full end-to-end scripts |
| Math | Developed in detail | Assumed; referenced |
| Tested | Executed by `tests/test_tutorials.py` | Reduced-size executed path |

Longer worked examples (e.g. the Gorkha earthquake, Bayesian geometry) stay
under `examples/`. When a tutorial would balloon into a full case study, the
case study moves to `examples/` and the tutorial keeps only the minimal
teaching version.

---

## 2. The "Start Here" Documentation Layer

Priority 2.1 adds a small set of orientation documents that live **outside**
the notebooks but are designed together with them. The course links into this
layer constantly; the layer links back to specific chapters.

| Artifact | Home | Relationship to the course |
|---|---|---|
| **Five-minute quickstart** — copy-paste forward model → noise → solve → observed-vs-predicted plot, no manual packing/slicing | `README.md` (short form) and `docs/quickstart.md` (annotated form) | The *same code* appears as the closing preview of Chapter 00 and is exercised by a golden-workflow test (2.4), so the README can never silently rot. |
| **Workflow map & decision guides** — one page: the visual map of the three API levels (domain functions → matrices/operators → physics kernels), followed by the "which function do I use?" / "which assumption am I making?" guides for geometry, slip basis, regularization, covariance, constraints, geometry uncertainty, and Bayesian inference | `docs/workflow.md` (combined; a separate decision-guides file was considered and rejected — the guides *are* the map read question-first) | Introduced in Chapter 00; Chapters 01–02 walk down the levels explicitly (the double-demo *is* the map in action). Each guide ends by naming the chapter that teaches the underlying method; chapters end their recap with a pointer back. |
| **Glossary** — geophysical and inverse-theory terms with the math symbol and the package name side by side (e.g. *regularization strength*, `λ`, `regularization_strength=`) | `docs/glossary.md` | Every chapter links glossary terms on first use instead of re-defining them inconsistently. **Once created, the glossary is the source of truth for notation**; §5's table seeds it and thereafter defers to it. |

Rules of engagement between the layers:

- The quickstart is **not** a tutorial and teaches nothing; it exists to give
  a complete, working, satisfying result in five minutes and to name the
  chapters where each line is explained.
- The glossary is the only place a term is *defined*; chapters may restate a
  definition when pedagogically necessary but must link it.
- The decision guides never contain method exposition — they route to
  chapters.

---

## 3. Course Map

The course is fifteen notebooks, 00–14. Two first-generation pairs are merged
(regularization + choosing λ; Bayesian inversion + diagnostics were planned
separately and are now designed as one), and the sequence is renumbered once,
during the 2.2 rewrite (§11 decision 1). Old→new file mapping is given below
the table.

| # | Chapter | Part | Status | Time | Requires |
|---|---|---|---|---|---|
| 00 | Preflight: scientific Python for geodesy | 0 | **[new]** | 60–90 min (skippable) | — |
| 01 | The elastic dislocation forward model | I — The forward problem | revise & deepen | 60–90 min | — |
| 02 | Discretization and the Green's matrix | I | revise & deepen | 60–90 min | 01 |
| 03 | Least squares and the failure of naive inversion | II — The inverse problem | revise & deepen | 60–90 min | 02 |
| 04 | Regularization and how to choose it | II | **merge** of old 04+05, deepen | 90–120 min | 03 |
| 05 | Multiple datasets: GNSS + InSAR | III — Real data | revise & deepen (was 06) | 60–90 min | 04 |
| 06 | Correlated noise | III | revise & deepen (was 07) | 45–75 min | 05 |
| 07 | Bounds, constraints, and slip bases | III | revise & deepen (was 08) | 60–90 min | 04 |
| 08 | Uncertainty, resolution, and synthetic tests | IV — Assessment | revise & deepen (was 09) | 60–90 min | 04 |
| 09 | Nonlinear geometry search | V — Geometry | revise & deepen (was 10) | 60 min | 08 |
| 10 | Gradient-based geometry inversion (JAX) | V | revise & deepen (was 11) | 60 min | 09; `geodef[jax]` |
| 11 | Triangular faults | VI — Topics | **[new]** | 30–60 min | 02 |
| 12 | Interseismic coupling | VI | **[new]** | 45–60 min | 07 |
| 13 | Model misspecification | VI | **[new]** | 45–60 min | 08, 09 |
| 14 | Bayesian inversion: priors, sampling, and diagnostics | VI | **[new]** | 90–120 min | 08; `geodef[bayes]` |

File migration (executed with `git mv` in lockstep with
`tests/test_tutorials.py` and `tutorials/README.md`, per §11 decision 1):

| Old | New |
|---|---|
| `04_regularization.ipynb` + `05_choosing_regularization.ipynb` | `04_regularization.ipynb` (merged; old 05 deleted) |
| `06_multiple_datasets.ipynb` | `05_multiple_datasets.ipynb` |
| `07_correlated_noise.ipynb` | `06_correlated_noise.ipynb` |
| `08_bounds_and_constraints.ipynb` | `07_bounds_and_constraints.ipynb` |
| `09_uncertainty_and_resolution.ipynb` | `08_uncertainty_and_resolution.ipynb` |
| `10_nonlinear_geometry.ipynb` | `09_nonlinear_geometry.ipynb` |
| `11_gradient_geometry.ipynb` | `10_gradient_geometry.ipynb` |
| — | `11_triangular_faults.ipynb`, `12_interseismic_coupling.ipynb`, `13_model_misspecification.ipynb`, `14_bayesian_inversion.ipynb` |

Chapters 10 and 14 depend on optional extras and are gated in CI the way the
JAX notebook already is; they are clearly marked optional in
`tutorials/README.md` so the core course (00–09, 11–13) never requires more
than the base install.

Part VI chapters are **conceptual topics**, not a continuation of the linear
narrative: each is entered from the chapter listed in "Requires" and they may
be read in any order. Advanced JAX/Bayesian machinery stays out of Parts 0–V;
14 earns its place because priors, sensitivity, convergence, and predictive
checking are general inverse-theory concepts, not library features.

---

## 4. Chapter Anatomy

Every chapter follows this template, in order. The first-generation notebooks
already follow an informal version; the revision makes it explicit and
complete.

1. **Header block:** title; one-sentence promise; **learning objectives**
   (3–6, observable: "derive…", "predict…", "diagnose…"); **prerequisites**
   (chapters + concepts); **estimated time** (a band, not false precision).
2. **Motivation** — the geophysical question this method answers, and what
   goes wrong without it (1–3 paragraphs; where possible, a figure of the
   failure the chapter will fix).
3. **Theory** — the mathematical development, in sections. Derivations
   follow the notation table (§5); every symbol is introduced before use;
   assumptions are stated when made, not discovered later.
4. **Worked demonstration(s)** — short code interleaved with the theory it
   illustrates, each followed by written interpretation of the output.
   Double-demos (§5) appear here where prescribed.
5. **Checkpoint questions** — 2–4 quick self-test questions with answers in
   collapsed `<details>` blocks (§11 decision 10), testing the theory before
   the exercises apply it.
6. **Common mistakes** — the 2–4 errors novices actually make with this
   method (wrong units, wrong ordering, misread diagnostics), each with the
   symptom the reader would observe.
7. **Recap** — bullet summary of the objectives, now restated as facts the
   reader can verify they know; a pointer into `docs/workflow.md`'s relevant
   decision guide (§2); explicit hand-off to the next chapter.
8. **Exercises** — 3–6, ordered from parameter-variation ("change dip,
   predict the pattern before running") to a final challenge exercise that
   requires combining the chapter with an earlier one. Every exercise is
   answerable with the material taught so far and has a worked solution in
   the solutions notebook (§9).
9. **Further reading** — 2–5 references: the primary sources (Okada 1985/92,
   Nikkhoo & Walter 2015, Savage 1983, …) and standard texts (Menke; Aster,
   Borchers & Thurber; Tarantola; Segall 2010), each with one line on what
   it adds.

---

## 5. Shared Conventions

So chapters stay consistent and composable:

- **Coordinates and units.** Local Cartesian `x = East`, `y = North`,
  `z = Up`; geographic `lon`/`lat` (in that order for named APIs) where
  geometry is built; depth positive down; SI units with degrees for angles.
  `docs/conventions.md` is the single authority — chapters link it rather
  than re-deriving it, and Chapter 00 teaches it.
- **Slip vector blocking.** `m = [s_strike(0..N-1), s_dip(0..N-1)]`: the
  first `N` entries are strike-slip, the next `N` dip-slip. Taught explicitly
  in Chapters 01–02 where ordering *is* the lesson.
- **Named views by default, blocked vectors when ordering is the lesson.**
  Routine operations use the named API: `result.strike_slip`,
  `result.dip_slip`, `result.slip_magnitude`, `result.slip_rake`,
  `result.reduced_chi2`, `geodef.invert.prediction(result)` /
  `residual` / `diagnostics` / `summary`, and `slip.pack`/`slip.unpack` at
  the boundary. The blocked `result.slip_vector` and hand-stacked systems
  appear only where the linear algebra is itself the subject (01–04, 08) —
  students manipulate ordering only when ordering is being taught.
- **Module-path API policy.** Beginner-public names are used bare
  (`geodef.Fault`, `geodef.solve`, `geodef.GNSS`); everything else is spelled
  through its module (`geodef.invert.lcurve`, `geodef.greens.matrix`,
  `geodef.data.spatial_covariance`, `geodef.slip.pack`). The notebooks'
  migration off the legacy top-level aliases (`geodef.lcurve`,
  `geodef.model_covariance`, …) lands **in one commit with** the removal of
  those aliases from `geodef/__init__.py` and the corresponding
  `tests/test_public_api.py` flip, per PLAN.md 2.2 — do not mix old and new
  spellings across chapters in between.
- **The elastic medium is declared, not implicit.** Chapters that use `μ` or
  `ν` (01 moment, 04 stress kernel, 13 misspecification) construct or name an
  `ElasticMedium` so students see where material parameters live.
- **Forward operator.** Always written `d = G m` (data = Green's matrix ×
  slip); the regularized objective is always
  `Φ(m) = (Gm − d)ᵀ W (Gm − d) + λ ‖L (m − m_ref)‖²`, with augmented rows
  `√λ L`. No `λ²` or `α` variants; published sources that use them are mapped
  in `docs/conventions.md`.
- **Notation.** The table below seeds `docs/glossary.md`; once the glossary
  exists it is the source of truth and this table defers to it (§11
  decision 4).
  - `d` — data vector, `m` — model (slip) vector, `G` — Green's matrix.
  - `ε` — noise; `C_d` — data covariance; `W = C_d^{-1}` — weight matrix.
  - `λ` — regularization strength; `L` — regularization operator;
    `m_ref` — reference model.
  - `C_m` — posterior model covariance; `R` — model resolution matrix.
  - `N` — number of patches; `2N` — slip parameters (both components).
  - `χ²` — unreduced weighted misfit; `χ²_ν` — reduced (`reduced_chi2`).
  - `μ`, `ν` — shear modulus, Poisson's ratio; `M_0`, `M_w` — moment,
    moment magnitude; `θ` — nonlinear geometry parameters.
- **No special characters in code.** Greek letters (`λ`, `χ²`, `θ`) are fine
  in markdown; code uses the plain ASCII names GeoDef exposes
  (`regularization_strength`, `reduced_chi2`, `theta`).
- **Recurring synthetic scenario.** One small planar thrust fault
  (`Fault.planar(...)`, coarse grid, e.g. `n_length ≈ 8`, `n_width ≈ 5`) with
  a smooth "true" slip bump and seeded Gaussian noise is tracked from Chapter
  03 through 08 (and reused by 11–14), so readers follow a single problem end
  to end. Chapters 01–02 use per-lesson illustrative geometries instead.
- **Scenario delivery — explicit first, then the builder.** Chapters currently
  construct their compact scenario explicitly because this course branch is
  restricted to notebooks and documentation. The explicit cells keep every
  chapter standalone and make seeds visible. Once the §6 runtime API lands,
  Chapters 05+ replace only those copied cells with the documented scenario
  builder; Chapters 03–04 remain explicit so students see the construction.
- **Random seeds.** Every chapter sets `rng = np.random.default_rng(0)` (or a
  stated seed) and passes seeds explicitly to the synthetic helpers.
- **Standard imports.** `import numpy as np`,
  `import matplotlib.pyplot as plt`, `import geodef`.
- **Double-demo (used sparingly).** When it genuinely clarifies *what GeoDef
  is doing*, show the underlying calculation by hand first (a few lines of
  NumPy), then the one-line GeoDef equivalent, asserting agreement with
  `np.allclose(...)`. Prescribed instances: `G @ m` vs.
  `fault.displacement()` (01); building `G` column by column vs.
  `geodef.greens.matrix()` (02); normal equations by hand vs.
  `geodef.solve()` (03); the augmented system vs.
  `regularization=` (04). Everywhere else the default remains one clear
  `geodef.*` call plus a labeled, interpreted plot.

---

## 6. Scenario Builder and Synthetic-Test Helpers **[new API; runtime-deferred]**

PLAN.md 2.2 requires that the synthetic workflows the course teaches become
supported API rather than notebook-only code. Home: a new **`geodef.synthetic`**
module (functions only, per the object-budget policy; approved — §11
decision 7):

- `synthetic.scenario(...)` — the documented builder for the recurring
  teaching scenario: returns a named record of `fault`, `true_slip`,
  `datasets` (with declared seed, noise level, station layout, and geometry
  keywords all overridable). Used by Chapters 05+ and by the golden-workflow
  tests.
- `synthetic.checkerboard(fault, ...)` / `synthetic.spike(fault, ...)` —
  patterned slip vectors for resolution tests (checker size / spike location
  as keywords, both components supported).
- `synthetic.noisy_data(fault, slip, datasets_or_layout, *, seed, ...)` —
  forward-model plus seeded noise consistent with each dataset's declared
  uncertainties.
- `geodef.invert.compare(result, true_slip)` — recovered-versus-input
  comparison metrics (per-patch difference, recovery fraction within a mask)
  backing the checkerboard/spike workflow in Chapter 08. This lives in
  `invert` beside the other assessment functions — it consumes an
  `InversionResult`, not a synthetic scenario — and gets **no alias** in
  `synthetic`: one name, one home, per the same principle under which 2.2 is
  deleting the legacy top-level aliases.

Design constraints: pure functions, NumPy default, seeds explicit and
required for anything random, returns are ordinary arrays/records that the
rest of the API accepts, and each helper is what the corresponding chapter
would otherwise write inline — the tutorials are the acceptance test for
their ergonomics. TDD as usual; document in `docs/` with the same executable-
example policy as other modules.

---

## 7. Visualization Strategy

There is no standalone plotting tutorial. Each `geodef.plot` function is
introduced at the moment its underlying concept is taught:

| Plot | Introduced in | Because |
|---|---|---|
| `plot.slip`, `plot.patches` | 01 | first time slip-on-fault is shown |
| `plot.map_view`, `plot.vectors` | 01 | surface displacement field |
| `plot.fault3d` | 01 | fault geometry at depth |
| `plot.fit`, `plot.prediction`, `plot.residual` | 03 | observed vs. predicted diagnostics |
| L-curve / ABIC / CV curve plots | 04 | hyperparameter selection |
| `plot.insar` | 05 | first InSAR dataset |
| `plot.diagnostics`, `plot.summary` | 05 | per-dataset fit reporting |
| `plot.resolution`, `plot.uncertainty` | 08 | model assessment |
| `plot.slip_interpolated` | 11 | smooth slip on triangular meshes |

The exhaustive gallery survives as its own unnumbered
`tutorials/reference_plots.ipynb` (not part of the numbered course path).
Every figure in every chapter carries titles, axis labels, units, and
colorbars — and, per §4, a written interpretation.

---

## 8. The Chapters

Each entry specifies: **Goal**, **Concepts & math** (the theory that must
appear, at textbook depth), **Key calls**, **Plots**, and **Exercises**.
Chapters that already exist additionally carry **Revision deltas** — the
concrete changes this revision makes to the shipped notebook. The template of
§4 (objectives, motivation, checkpoints, common mistakes, recap, further
reading) applies to every chapter and is not repeated below.

> **Reuse existing material.** The shipped notebooks are the starting point
> for their own revisions — deepen, merge, and migrate them; do not rewrite
> from scratch. For new chapters, check `related/` (e.g.
> `related/shakeout_v2/`, `related/stress-shadows/`) for adaptable material
> before starting fresh.

---

### Chapter 00 — Preflight: Scientific Python for Geodesy **[new]**
*The array, plotting, and convention skills the course assumes.*

**Goal.** Equip geophysicists new to scientific Python with exactly the NumPy,
matplotlib, and convention knowledge the course uses — and let everyone else
verify they can skip ahead.

**Concepts & math.**
- A **self-test up front**: five questions (shape of a broadcast result, what
  a slice views vs. copies, reading a colorbar, converting km→m, depth sign);
  readers who pass are told to jump to Chapter 01.
- Arrays: `shape`, `dtype`, 1-D vs 2-D vs `(N, 3)` layouts; why GeoDef
  returns plain arrays and what "vector of length 2N" means concretely.
- Indexing and slicing: views, fancy indexing, boolean masks — enough to read
  `m[:N]` and `d[2::3]` fluently.
- **Broadcasting** and vectorization: elementwise rules, `np.newaxis`, why
  loops over patches/stations don't appear in this course.
- Linear algebra: `@`, `np.linalg.solve` vs. explicit inverses, `lstsq` — a
  first look at the machinery Chapters 02–03 build on.
- Matplotlib anatomy: figure/axes, labels, colorbars, subplots — the elements
  every course figure carries.
- **Units and conventions:** SI defaults, degree-bearing names, ENU axes,
  depth positive down, `lon, lat` order, `LocalFrame` in one paragraph — a
  guided reading of `docs/conventions.md`.
- Floating point in two paragraphs: roundoff, why `np.allclose` not `==`.
- Reproducibility: `np.random.default_rng(seed)`; reading GeoDef's validation
  errors and `.validate()` reports as a feature, not a failure.
- **Closing preview:** the five-minute quickstart (§2) run top to bottom,
  with one sentence per line naming the chapter that explains it.

**Key calls.** NumPy/matplotlib primitives; `geodef.Fault.planar(...).validate()`
as the error-message showcase; the quickstart sequence verbatim.

**Plots.** A deliberately unlabeled figure fixed into a fully labeled one;
broadcasting diagram; the quickstart's observed-vs-predicted map.

**Exercises.** Reshape and re-block a fake `2N` vector; vectorize a small
loop; convert a `lat, lon, depth-in-km` table to convention; fix three
intentionally broken calls using only the error messages.

---

### Chapter 01 — The Elastic Dislocation Forward Model
*How slip on a buried fault produces surface displacement.*

**Goal.** Understand the dislocation forward problem and compute surface
displacement for a rectangular source.

**Concepts & math.**
- Elastic rebound and why geodesy sees faults at all; the **half-space
  idealization** (homogeneous, isotropic, linear-elastic) and a first honest
  statement of when it fails (layering, topography, inelasticity) —
  foreshadowing Chapter 13.
- A **dislocation** as a displacement discontinuity across a surface;
  Volterra's construction in words and pictures; from Steketee to the Okada
  (1985) closed form. No full derivation — a boxed lineage with references —
  but the *structure* of the solution (source geometry in, three displacement
  components out, linear in slip) is developed carefully.
- Fault geometry vocabulary with a labeled 3-D diagram: `strike`, `dip`,
  `rake`, `length`, `width`, `depth`; hanging wall/footwall; mechanism types
  as regions of rake.
- Forward vs. inverse problems, tied by `d = G m + ε`. For *fixed* geometry
  displacement is **linear in slip** — the hinge for Chapters 02–08,
  demonstrated numerically (double a slip patch, watch the field double).
- Slip's two in-plane components and the **blocked** vector
  `m = [strike-slip | dip-slip]` of length `2N`; `slip.pack`/`unpack` as the
  named boundary.
- Reading predictions: interleaved `[e, n, u]` rows and the three-array
  return of `fault.displacement(...)`.
- The **elastic medium**: `ElasticMedium(mu, nu)` as the declared home of
  material parameters; how `μ` enters moment and `ν` enters the kernels.
- Seismic moment `M_0 = μ Σ s_k A_k`, moment magnitude `M_w` (Hanks &
  Kanamori), with a worked "how big is a metre of slip?" calculation.
- **Double-demo:** `G @ m` unpacked by hand vs. `fault.displacement(...)`.

**Revision deltas.** Add the elasticity/motivation development and geometry
diagram; introduce `ElasticMedium` explicitly (was implicit); replace
`fault.centers` with `centers_geo`/`centers_local`; adopt named views and
`slip.pack`; add checkpoints/common-mistakes/further-reading per §4; expand
moment section with the worked magnitude calculation.

**Key calls.** `Fault.planar(...)` (keyword geometry), `fault.greens_matrix(...)`,
`fault.displacement(...)`, `slip.pack(...)`, `fault.moment(slip)` /
`fault.magnitude(slip)`, `ElasticMedium`, attributes `n_patches`,
`grid_shape`, `centers_geo`, `centers_local`, `areas`.

**Plots.** `plot.fault3d` geometry colored by depth; `plot.slip`;
`plot.map_view` + `plot.vectors` (horizontal arrows, vertical dots) over the
fault footprint.

**Exercises.** Vary dip and depth, predicting pattern and peak amplitude
before running; switch to pure strike-slip and interpret the quadrant
pattern; compute `M_w` for a published earthquake's dimensions; challenge —
superpose two sources by hand and verify linearity.

---

### Chapter 02 — Discretization and the Green's Matrix
*From a continuous fault to a linear system.*

**Goal.** Build the Green's matrix for a multi-patch fault and understand its
structure as the discrete forward operator.

**Concepts & math.**
- `G` as a **design matrix**: warm up with the two-column line fit
  `y = a x + b` before generalizing to fault slip.
- Discretization as **function approximation**: slip as a piecewise-constant
  expansion over `N` patch basis functions; what refinement does and does not
  buy (convergence of the field vs. explosion of unknowns).
- Superposition from linearity ⇒ `d = G m`; column `j` of `G` is the surface
  response to **unit slip on patch `j`**; the blocked column layout and
  interleaved row layout; **units of `G` entries** (dimensionless m/m) and
  why that matters when mixing datasets later.
- How dataset **projection** turns the raw 3-component response into observed
  rows (E/N/U for GNSS; LOS foreshadowed for Chapter 05).
- Patch ordering made concrete with `fault.reshape_patches` /
  `fault.flatten_patches` — students never memorize which grid axis varies
  fastest.
- **Double-demo:** `G` built column by column (unit slip per patch) vs.
  `fault.greens_matrix(...)` / `geodef.greens.matrix(...)`.
- **Sidebar — caching as a trust feature:** assembling `G` is expensive and
  repeated, so GeoDef hashes *every* input that affects the result (geometry,
  stations, medium, precision, kernel version) and caches to disk; a short
  timing demo, plus one paragraph on why incomplete cache keys would be
  dangerous.
- Why finer discretization trades resolution for stability — the
  ill-posedness teaser, now stated via the growth of the condition number
  with refinement (computed, plotted, deferred to Chapter 03 for theory).

**Revision deltas.** Add the function-approximation framing and the
condition-number-vs-refinement figure; add `reshape_patches`/
`flatten_patches`; expand the caching sidebar with the cache-key trust
paragraph; state units of `G`; §4 template sections.

**Key calls.** `Fault.planar(...)` with multiple patches,
`fault.greens_matrix`, `geodef.greens.matrix(fault, dataset)`,
`np.linalg.lstsq` (warm-up), `fault.patch_index`, `fault.reshape_patches`,
`geodef.cache.info()` / `set_dir()`.

**Plots.** Line-fit design matrix; `G` via `imshow` with the block structure
annotated; one column of `G` as a surface response map; condition number vs.
refinement; a two-asperity model and its predicted field.

**Exercises.** Refine the grid and track `G.shape` and conditioning; compare
a strike-slip vs. dip-slip column for one patch; compare deep vs. shallow
patch columns and relate similarity to resolvability; challenge — build the
projection rows for a made-up instrument that measures only East.

---

### Chapter 03 — Least Squares and the Failure of Naive Inversion
*Estimation theory meets an ill-posed problem.*

**Goal.** Estimate slip by (weighted) least squares, understand the estimator
as statistics rather than recipe, and watch unregularized inversion overfit.

**Concepts & math.**
- The linear inverse problem `d = G m + ε`; the noise model `ε ~ N(0, C_d)`
  stated as an *assumption* with consequences.
- Ordinary least squares **derived**: objective, gradient, normal equations.
- **Weighted** least squares with `W = C_d^{-1}`:
  `m̂ = (Gᵀ W G)^{-1} Gᵀ W d`, derived; why weighting is not optional when
  uncertainties differ; Gauss–Markov (BLUE) stated in one box.
- The **SVD picture of ill-conditioning**: singular values of the (whitened)
  `G`, noise amplification by `1/σ_i`, condition number; the spectrum of the
  teaching scenario plotted and interpreted. This is the chapter's new
  theoretical core and the foundation Chapters 04 and 08 build on.
- Over- vs. under-determined systems; rank deficiency.
- Goodness of fit: residuals, `χ²`, **reduced `χ²_ν`** and its expected value
  under a correct model — the vocabulary (`chi2` unreduced, `reduced_chi2`
  reduced) fixed once here.
- **The overfitting catastrophe:** invert noisy data unregularized; wild
  oscillation, excellent fit; why "fits better" is not "is better".
- **Double-demo:** normal equations by hand vs. `geodef.solve(...)`.

**Revision deltas.** Add the WLS derivation, Gauss–Markov box, and the SVD
section with spectrum figure; adopt named result views (`strike_slip`,
`reduced_chi2`) and `geodef.invert.prediction`/`residual`/`summary` — the
blocked `slip_vector` appears only inside the double-demo; explicit scenario
construction stays here (§5); §4 template sections.

**Key calls.** `geodef.solve(fault, datasets)` (beginner spelling of
`geodef.invert.solve`), `InversionResult` named views,
`geodef.invert.prediction` / `residual` / `summary`, `np.linalg.svd`,
`plot.fit`, `plot.slip`.

**Plots.** True vs. recovered slip; observed-vs-predicted (`plot.fit`);
singular-value spectrum; the oscillatory unregularized solution.

**Exercises.** Increase noise and re-invert, tracking `reduced_chi2`; drop
stations below `N` and observe instability; whiten by hand and verify against
`solve`; challenge — relate the largest oscillation pattern to the smallest
singular vectors.

---

### Chapter 04 — Regularization and How to Choose It
*Making the inverse problem well-posed — and defensible.*

Merged from the first-generation regularization and choosing-λ notebooks:
operators and selection criteria are one lesson taught in two movements. The
course's heaviest core chapter (90–120 min); the seam between the movements
is the fallback split point if it overruns (§11 decision 1).

**Goal.** Stabilize the inversion with prior information, understand what
each regularization operator assumes, and replace eyeballing `λ` with
quantitative criteria whose own assumptions are understood.

**Concepts & math — movement 1: regularization.**
- Ill-posedness ⇒ additional information is required; Tikhonov's idea.
- The regularized objective
  `Φ(m) = (Gm − d)ᵀ W (Gm − d) + λ ‖L (m − m_ref)‖²`, minimized in closed
  form; the augmented/stacked system view (`√λ L` rows) **derived**, not
  asserted.
- **The filter-factor view**: on the SVD of Chapter 03, Tikhonov damping
  multiplies each mode by `σ_i²/(σ_i² + λ)` (damping case worked; smoothing
  case described) — under- and over-smoothing become visible in mode space.
- Choices of `L` and the prior each encodes:
  - **Smoothing** — discrete Laplacian; the 5-point stencil written out,
    boundary handling stated; penalizes roughness.
  - **Damping** — `L = I`; penalizes magnitude/moment; when that is and is
    not a sensible prior.
  - **Stress-kernel** — inter-patch elastic interactions; penalizes stress
    heterogeneity; one paragraph of physics and the `ElasticMedium`
    dependency.
  - `m_ref` — regularizing toward a nonzero reference and when to use it.
- **The Bayesian reading** (one section, foreshadowing Chapter 14):
  regularization ⇔ Gaussian prior, `λ` ⇔ prior precision, the regularized
  estimate ⇔ MAP. Planted here so Chapter 14 is a continuation, not a leap.
- Bias–variance: what `λ` buys and what it costs.
- **Double-demo:** the augmented system solved by `lstsq` vs.
  `regularization='laplacian'`.

**Concepts & math — movement 2: choosing λ.**
- The misfit–roughness trade-off curve as `λ` varies; why there is no
  assumption-free "right" answer.
- **L-curve:** model norm vs. misfit (log–log); corner as curvature maximum;
  what the corner balances and known failure modes.
- **ABIC**: the hierarchical model (prior with hyperparameter `λ`), marginal
  likelihood integrating slip out — a boxed derivation sketch kept compact
  for the merged chapter — Occam's-razor intuition and the
  effective-degrees-of-freedom reading; minimize ABIC. (The course's first
  marginalization — flagged as such, and reused in Chapter 14.)
- **Cross-validation:** hold-out prediction error, `k`-fold mechanics,
  spatial caveats (correlated neighbors leak — foreshadows Chapter 06).
- When the criteria agree/disagree; reporting sensitivity to the choice
  rather than hiding it.

**Revision deltas.** Merge the two notebooks at the trade-off seam (movement
1's closing under/over-smoothed panel becomes movement 2's motivation —
currently duplicated setup between the two notebooks disappears); migrate
`geodef.lcurve`/`geodef.abic_curve`/`geodef.compute_abic` to `geodef.invert.*`
module paths (with the 2.2 export removal); add the filter-factor
development, Laplacian stencil, stress-kernel physics paragraph, Bayesian
reading, ABIC sketch, effective-DOF discussion, and CV spatial caveat;
explicit scenario construction retained here (last time, per §5); §4 template
sections.

**Key calls.** `geodef.solve(..., regularization='laplacian'|'damping'|'stresskernel',
regularization_strength=..., regularization_target=...)`;
`geodef.invert.lcurve(...)`, `geodef.invert.abic_curve(...)`,
`geodef.invert.compute_abic(...)`,
`geodef.solve(..., regularization_strength='abic'|'cv', cv_folds=...)`;
`geodef.greens.laplacian` referenced conceptually.

**Plots.** Panel grid of recovered slip across `λ` from under- to
over-smoothed; filter factors vs. `σ_i` for several `λ`; smoothing vs.
damping at matched misfit; L-curve with marked corner; ABIC vs. `λ`; CV
error vs. `λ`; the three chosen solutions side by side.

**Exercises.** Swap Laplacian for damping and explain the differences from
the priors; regularize toward a nonzero `m_ref` and interpret; compare `λ`
from L-curve, ABIC, and CV on one problem, then change the noise level and
find which criterion is most stable; challenge — implement a custom `L`
(e.g. gradient operator), pass it as a matrix, and select its `λ` by ABIC.

---

### Chapter 05 — Multiple Datasets: GNSS + InSAR
*Joint inversion and relative weighting.*

**Goal.** Combine complementary datasets in one inversion, with the data
built through the friendly constructors and the results read per dataset.

**Concepts & math.**
- Why joint: GNSS (sparse, 3-component, absolute) vs. InSAR (dense, 1-D
  line-of-sight, relative); complementary sampling in space and component.
- **InSAR viewing geometry developed**: from satellite heading and incidence
  angle to the unit look vector; `LOS = u · l̂`; ascending vs. descending;
  the sign convention stated against `docs/conventions.md` and the
  ground-to-satellite reversal diagnostic.
- Building datasets the named way: `data.gnss(...)`, `data.insar(...)` with
  keyword components and per-dataset **names** as first-class identifiers.
- Stacking the forward problem: concatenated `G` and `d`, block `C_d`; shown
  once by hand (`greens.stack_obs`) and thereafter left to `solve`.
- **Relative weighting**: weights as covariance scaling; what over-weighting
  a dataset does; per-dataset `reduced_chi2` via
  `geodef.invert.diagnostics(result)` as the balance check; honest
  reporting when one dataset is systematically misfit.
- Velocity vs. displacement semantics (interseismic foreshadow for
  Chapter 12): the metadata distinction in one paragraph.
- **Scenario builder introduced** (§5, §6): the one-cell
  `synthetic.scenario(...)` sidebar showing equivalence to Chapters 03–04's
  explicit construction.

**Revision deltas.** Renumber from 06; migrate dataset construction to
`data.gnss`/`data.insar` with names; use `dataset_slices`/named per-dataset
predictions instead of manual slicing; add the viewing-geometry derivation
and weighting-balance section; add `plot.diagnostics`/`plot.summary`;
introduce the scenario builder; §4 template sections.

**Key calls.** `geodef.data.gnss(...)`, `geodef.data.insar(...)`,
`geodef.solve(fault, [gnss, insar], ...)`, `geodef.invert.diagnostics`,
`geodef.synthetic.scenario(...)`, `plot.insar`, `plot.vectors`,
`plot.diagnostics`.

**Plots.** GNSS vectors and InSAR LOS for one scenario; joint vs.
single-dataset slip; per-dataset fit panels (`plot.diagnostics`).

**Exercises.** Down-weight one dataset and watch slip migrate; add a
descending track and quantify the vertical/east separation it buys; misname a
look-vector sign and diagnose it from residuals; challenge — reproduce the
joint solution by hand-stacking with `greens.stack_obs`/`stack_weights`.

---

### Chapter 06 — Correlated Noise
*Beyond diagonal data covariance.*

**Goal.** Represent spatially correlated noise — especially InSAR
atmosphere — and see what ignoring it costs.

**Concepts & math.**
- Why InSAR noise is **spatially correlated** (atmosphere, orbits); the
  diagonal-`C_d` assumption of earlier chapters examined and rejected here.
- Covariance **functions** (exponential, Gaussian) with variance and
  correlation length; building a full `C_d` with
  `geodef.data.spatial_covariance(...)`; positive-definiteness as a real
  constraint (validation catches violations).
- **Whitening**: the Cholesky view `C_d = F Fᵀ`, transforming to independent
  data; the effective number of independent observations, computed and
  plotted against correlation length.
- Consequences: how off-diagonal terms change the estimate *and* its
  uncertainty; why ignoring correlation gives overconfident error bars
  (quantified on the scenario).
- Practical note: dense-`C_d` cost scaling, downsampling, and a pointer to
  the operator-based noise roadmap (PLAN.md 4.1) — taught honestly as a
  current limitation.
- Estimating covariance from data (variograms) as an outlook paragraph.

**Revision deltas.** Renumber from 07; remove the historical "blocked" status
(the `spatial_covariance` support landed); migrate to the
`geodef.data.spatial_covariance` module path; add the whitening development,
effective-N figure, and overconfidence quantification; §4 template sections.

**Key calls.** `geodef.data.spatial_covariance(...)`, `geodef.data.insar(...)`
with full covariance, `geodef.solve(...)`.

**Plots.** Covariance function and matrix; diagonal vs. full-`C_d` inversions
with uncertainties; effective-N vs. correlation length.

**Exercises.** Sweep correlation length from ~0 to fault-scale and track the
solution and error bars; deliberately invert with the wrong correlation
length both ways; challenge — build a two-length-scale covariance and predict
which scale dominates the estimate.

---

### Chapter 07 — Bounds, Constraints, and Slip Bases
*Enforcing physically admissible slip.*

**Goal.** Add inequality and sign constraints, reduce parameters with fixed
slip directions, and understand the cost of each.

**Concepts & math.**
- Why constraints: rupture does not reverse sense mid-fault; magnitude
  bounds; geologic priors. Why quadratic regularization cannot express them.
- **Non-negative least squares** `m ≥ 0`: active-set intuition in a short
  boxed sketch (which patches end up pinned at zero and why).
- **Bounded least squares** `lb ≤ m ≤ ub`; scalar, per-component, and
  per-parameter bounds.
- **Linear inequality constraints** `C m ≤ d` as a quadratic program; one
  worked example (e.g. total-moment cap).
- **Fixed-direction bases derived**: the reduction matrix that maps one
  amplitude per patch at fixed `rake` (or geographic `slip_azimuth`, or
  `plate_rake` for plate-motion-aligned problems) into the blocked `2N`
  space; how a basis encodes a sense prior *exactly* rather than softly.
  The plate basis is planted here as the vocabulary Chapter 12 builds on.
- Constraints vs. regularization: admissibility, bias, and the danger of
  constraint-induced artifacts (positivity piling slip at edges) —
  demonstrated.
- A one-paragraph bridge: positivity as a *prior* has a principled Bayesian
  treatment (Chapter 14).

**Revision deltas.** Renumber from 08; add the fixed-basis derivation and the
`components='plate'`/`plate_rake` option; add the constraint-artifact
demonstration and Bayesian bridge; §4 template sections.

**Key calls.** `geodef.solve(..., bounds=(0, None))` (auto-NNLS),
`bounds=(lb, ub)`, `method='constrained', constraints=(C, d)`,
`components='rake', rake=...`, `components='azimuth', slip_azimuth=...`,
`components='plate', plate_rake=...`; `geodef.slip.from_rake` /
`from_azimuth` / `from_plate` for the by-hand basis.

**Plots.** Unconstrained vs. non-negative slip (spurious back-slip removed);
fixed-rake vectors; a constraint-artifact example.

**Exercises.** Compare WLS, NNLS, and fixed-rake on one dataset; tighten an
upper bound until it visibly biases the fit and find that threshold;
challenge — build the rake-basis reduction matrix by hand and verify against
`components='rake'`.

---

### Chapter 08 — Uncertainty, Resolution, and Synthetic Tests
*How well is the slip actually resolved?*

**Goal.** Quantify and visualize uncertainty and resolution, run principled
synthetic tests, and report derived quantities with error bars.

**Concepts & math.**
- **Posterior model covariance** derived from the linear-Gaussian model:
  `C_m = (Gᵀ W G + λ LᵀL)^{-1}`; the diagonal as per-patch uncertainty; what
  the off-diagonals mean (and why neighboring patches anti-correlate under
  smoothing).
- **Model resolution matrix** `R = G^{-g} G` derived; recovered model =
  `R m_true` + noise term; rows as averaging kernels, diagonal as a
  resolution map; connection to the SVD/filter factors of Chapters 03–04.
- The **resolution–uncertainty trade-off** as `λ` varies — the same
  trade-off from Chapter 04 seen from the model side, plotted as paired maps.
- **Synthetic tests as supported API** (§6): checkerboard and spike tests via
  `synthetic.checkerboard`/`spike` + `noisy_data` + `invert.compare`; what
  checkerboards do and do not establish (linearity caveat, pattern-scale
  dependence) — an honest-assessment discussion, not just the ritual.
- Derived quantities with uncertainty: moment and `M_w` error propagation
  from `C_m` through the linear moment functional.

**Revision deltas.** Renumber from 09; migrate `geodef.model_*` calls to the
`geodef.invert.*` module paths; replace notebook-local checkerboard code with
the new `geodef.synthetic` helpers and `geodef.invert.compare`; add the `C_m`
and `R` derivations, off-diagonal discussion, and the checkerboard-caveat
section; add moment error propagation; §4 template sections.

**Key calls.** `geodef.invert.model_covariance(...)`,
`geodef.invert.model_resolution(...)`, `geodef.invert.model_uncertainty(...)`,
`geodef.invert.diagnostics(...)`, `geodef.invert.compare(...)`,
`geodef.synthetic.checkerboard` / `spike` / `noisy_data`,
`fault.moment` / `fault.magnitude`, `plot.resolution`, `plot.uncertainty`.

**Plots.** Per-patch uncertainty map; resolution-diagonal map; one full
resolution row as an averaging kernel; checkerboard input/recovered pair;
paired resolution/uncertainty maps at two `λ`.

**Exercises.** Run checkerboards at two scales and two `λ` and tabulate
recovery; find the depth below which spikes are unrecoverable; report
`M_w ± σ`; challenge — show numerically that `R → I` as noise → 0 and
regularization → 0.

---

### Chapter 09 — Nonlinear Geometry Search
*When the geometry itself is unknown.*

**Goal.** Estimate nonlinear fault parameters (location, strike, dip, depth)
on top of the linear slip inversion.

**Concepts & math.**
- Why geometry is **nonlinear**: `G(θ)` depends nonlinearly on position and
  orientation; `d = G(θ) m` is bilinear at best.
- **Variable projection** developed: for any trial `θ`, the inner slip
  problem is the linear inversion of Chapters 03–07, so the outer search is
  over few parameters; the projected objective `Φ(θ) = min_m Φ(θ, m)`;
  reference to Golub & Pereyra.
- **Identifiability and trade-offs**: depth–size–slip covariance shown via
  2-D misfit surfaces; why single-dataset geometries can be poorly
  constrained.
- The objective surface over `θ`: local minima, the value of a coarse grid
  search before local refinement; `scipy.optimize.minimize` mechanics
  (starting points, bounds, convergence flags read critically).
- **Outlook:** gradient-based search (Chapter 10) and full Bayesian geometry
  posteriors (`geodef.bayes`, worked in `examples/bayesian_geometry.ipynb`).

**Revision deltas.** Renumber from 10; replace the stale `emcee`/"future
study" outlook with pointers to Chapter 10 and the existing `geodef.bayes`
example; add the variable-projection development and
misfit-surface/trade-off section; §4 template sections.

**Key calls.** A small objective wrapping `geodef.solve(...)` inside
`scipy.optimize.minimize` / a grid loop; `geodef.synthetic.scenario` with a
perturbed starting geometry.

**Plots.** Misfit vs. one scanned parameter (dip); a 2-D misfit surface
(depth vs. width) with the trade-off valley; recovered vs. true geometry and
slip.

**Exercises.** Grid-search dip then refine with `minimize`; start from a
wrong basin and document the local minimum; scan misfit vs. depth at two
noise levels; challenge — map the depth–width trade-off and explain it
physically.

---

### Chapter 10 — Gradient-Based Geometry Inversion (JAX)
*Differentiable forward models for geometry search.*

Optional: requires `geodef[jax]`; CI-gated like the JAX test modules.

**Goal.** Use automatic differentiation to make geometry search fast and to
attach curvature-based uncertainties.

**Concepts & math.**
- What **automatic differentiation** is (forward/reverse in two paragraphs,
  no implementation detail) and why an analytic-kernel forward model is
  differentiable end to end.
- The differentiable variable-projection objective; gradients of `Φ(θ)`;
  L-BFGS-B refinement vs. Chapter 09's derivative-free search (iteration
  counts compared on the same problem).
- Multi-parameter recovery: scaling of search difficulty with `dim(θ)`;
  parameter scaling/preconditioning in practice.
- **Gauss–Newton curvature ⇒ approximate geometry covariance**: error bars
  on `θ` from the Jacobian at the optimum; when the Gaussian approximation
  is trustworthy (connects to Chapter 14).
- Backend mechanics kept to one section: `set_backend('jax')`, float64,
  compilation vs. execution time (measured once), and NumPy parity asserted.
- Sequencing note: this chapter stays deliberately independent of Parts
  0–IV's core path; nothing later requires it except as an alternative to
  Chapter 09's search.

**Revision deltas.** Renumber from 11. The shipped notebook predates this
outline's inclusion of a gradient chapter: bring it under the §4 template
(it currently lacks the formal spec), add the AD-concepts and
trustworthiness sections, and align its API spellings with the module-path
policy.

**Key calls.** `geodef.backend.set_backend('jax')`,
`geodef.invert.geometry_search(...)`, its result record (optimized `Fault`,
`theta`, covariance), `geodef.gradients` referenced conceptually.

**Plots.** Convergence paths (gradient vs. grid+simplex); recovered geometry
with error ellipses on `θ` pairs; timing bar (compile vs. solve).

**Exercises.** Recover 3, then 5, then 7 geometry parameters and track
difficulty; compare wall-clock and iteration count against Chapter 09;
challenge — verify one gradient component by finite differences.

---

### Chapter 11 — Triangular Faults **[new]**
*The same linear system on an unstructured mesh.*

**Goal.** Show that everything in Chapters 02–08 carries over unchanged to
triangular dislocations, and when triangles are worth the trouble.

**Concepts & math.**
- Why triangles: curved fault surfaces, slab interfaces, trace-following
  geometry; what rectangles cannot represent without gaps/overlaps.
- The Nikkhoo & Walter (2015) triangular dislocation element in one boxed
  paragraph (artefact-free construction; same half-space assumptions).
- A small mesh built **on the fly** (a handful of triangles via
  `Fault.from_triangles` or a one-call `geodef.mesh` helper) — real meshing
  workflows (traces, polygons, slab2.0) stay in `examples/mesh_generation.ipynb`.
- The invariance lesson: `G` assembly, solve, regularization, and assessment
  calls are *identical*; the Laplacian on an unstructured mesh
  (adjacency-based) contrasted with the structured stencil of Chapter 04.
- Rake vs. azimuth bases on curved surfaces: why `slip_azimuth` (or
  `plate_rake`) is the meaningful fixed direction when strike varies —
  reusing Chapter 07's basis machinery.
- Rect vs. tri on the same scenario: agreement where geometry is planar;
  what changes on a curved geometry.

**Key calls.** `Fault.from_triangles(..., frame=...)`, a minimal
`geodef.mesh` construction, `geodef.solve(...)` unchanged,
`plot.slip_interpolated`, `plot.fault3d`.

**Plots.** The triangular mesh in 3-D; interpolated slip; rect-vs-tri
solution comparison on matched geometry.

**Exercises.** Refine the mesh and re-run the Chapter 08 assessment; fix slip
azimuth on a curved mesh and compare against per-patch rake; challenge —
build a two-segment fault with a bend and discuss the Laplacian across the
join.

---

### Chapter 12 — Interseismic Coupling **[new]**
*Imaging locked faults with the backslip idea.*

> **Convention settled (2026-07; see §11 decision 8).** The backslip
> sign/frame convention below is decided and must be written into
> `docs/conventions.md` before or with this chapter, so it is defined once.
> The chapter teaches the concept with today's basis-and-bounds vocabulary
> and is revised when the PLAN.md 4.4 coupling API (coupling-fraction
> parameterization, moment-deficit outputs) lands. The full real-data
> coupling example (PLAN.md 2.3) remains deferred until 4.1/4.3/4.4 exist.

**Goal.** Understand the interseismic velocity field of a locked fault and
pose coupling estimation as the same linear inversion with a different
parameterization.

**Concepts & math.**
- Elastic rebound across the earthquake cycle; what GNSS velocities look
  like near locked strike-slip and subduction faults (the arctangent profile
  derived for the infinite strike-slip screw dislocation — the course's one
  closed-form inversion-free result).
- **Reference frames first**: velocities are corrected per station for
  rigid-block motion — normally into the overriding-plate-fixed frame —
  using the same Euler pole(s) that later supply the per-patch plate rates
  (one vocabulary for correction and coupling). Stated plainly: stations on
  the downgoing plate carry the full convergence rate after this correction,
  and pole uncertainty propagates into the "data" — an honest paragraph
  tying into Chapter 13.
- **Backslip** (Savage 1983): steady plate motion + backslip on the locked
  patches = interseismic velocity field; backslip is slip **anti-parallel to
  the local plate-motion vector** on the fault surface — normal sense
  (inverse of thrust) on a megathrust — with **non-negative amplitude
  bounded by the local plate rate**. The kinematic-device nature of the
  construction and its limits stated honestly (not mechanics).
- Sign bookkeeping made explicit once: positive backslip has a *negative*
  dip-slip component in GeoDef's raw convention; the plate basis
  (`components='plate'` with `plate_rake` in the backslip direction) hides
  this, and the conventions page records the mapping for raw-component
  users.
- Velocity data vs. displacement data: the units/epoch semantics from
  Chapter 05 now doing real work.
- **Coupling** = backslip rate / plate rate ∈ [0, 1]; coupling maps plotted
  as fractions, not signed slip; moment-deficit rate computed by hand from
  the result (`μ` × backslip rate × area, summed), pending the 4.4 API.
- Rigid-block context: where the plate rate comes from (`geodef.euler`), and
  why block motion and coupling must share one vocabulary (roadmap 4.4);
  co-estimating block motion with coupling noted as the research-grade path
  (roadmap 4.3/4.4), which this two-step teaching treatment must not
  preclude.

**Key calls.** `geodef.data.gnss(...)` velocities,
`geodef.slip.plate_rake_from_euler(...)`,
`geodef.solve(..., components='plate', plate_rake=..., bounds=...)`,
`geodef.euler` for the plate rate and block correction.

**Plots.** The arctangent profile vs. locking depth; synthetic interseismic
velocity field before/after block correction; recovered coupling map;
moment-deficit accumulation vs. time.

**Exercises.** Vary locking depth and fit the profile; invert a synthetic
coupling pattern and run the Chapter 08 checkerboard on it (coupling
resolution is usually poor at depth — see it); challenge — express the same
problem in raw strike/dip components and show why the plate basis is the
honest parameterization.

---

### Chapter 13 — Model Misspecification **[new]**
*What happens when the assumed model is wrong.*

**Goal.** Develop the habit of asking "wrong how?": see how errors in fixed
assumptions bias slip estimates in ways formal uncertainties do not capture.

**Concepts & math.**
- The uncomfortable theorem of Chapter 08: `C_m` and `R` are computed *under
  the model*; they say nothing about being wrong about `G` itself.
- Systematic experiments on the scenario, each: truth generated with one
  model, inverted with another, bias mapped and compared against the formal
  `σ`:
  - wrong **dip** / **depth** (geometry misspecification; connects to
    Chapter 09 — this is why geometry search matters);
  - wrong **elastic medium** (`ν`, layering-vs-half-space stand-in;
    `ElasticMedium` finally varied);
  - wrong **discretization** (too-coarse patches aliasing slip);
  - wrong **noise model** (Chapter 06's correlated noise treated as white).
- **Residual forensics**: spatially coherent residuals as the misspecification
  signal; per-dataset diagnostics re-read with suspicion.
- What to report: sensitivity analyses alongside formal errors; the
  difference between precision and accuracy in published slip models.

**Key calls.** Nothing new — `geodef.synthetic.*`, `geodef.solve`,
`geodef.invert.diagnostics`, `plot.residual`; the chapter's novelty is
methodological.

**Plots.** Bias maps per experiment with formal-`σ` contours overlaid;
coherent-residual maps; a summary table of bias/`σ` ratios.

**Exercises.** Find the dip error that biases `M_w` by 0.1; determine which
misspecification the residuals detect most easily and which is nearly
invisible; challenge — design the station geometry that best *exposes* a
depth error.

---

### Chapter 14 — Bayesian Inversion: Priors, Sampling, and Diagnostics **[new]**
*From regularization to posterior distributions — and knowing when to trust them.*

Optional: requires `geodef[bayes]` (JAX + BlackJAX); CI-gated. Designed as
one chapter (formulation, prior sensitivity, and diagnostics together) to
keep the course compact; like Chapter 04 it is a two-movement chapter with a
recorded fallback split (§11 decision 1). Runs stay tiny (coarse fault,
short chains, fixed seeds) — the point is the workflow, not converged
science.

**Goal.** Recast the regularized inversion as Bayesian inference, obtain
posterior distributions instead of point estimates, test how conclusions
depend on the prior, and audit the sampler before believing any of it.

**Concepts & math — movement 1: inference and priors.**
- Bayes' theorem for the linear-Gaussian model; the Chapter 04 bridge paid
  off: Tikhonov solution = posterior mean, `C_m` = posterior covariance,
  ABIC = marginal likelihood — nothing new, *reorganized*.
- Where sampling earns its keep: **positivity** (Chapter 07's bounds as a
  truncated prior via `SlipPosterior`), hyperparameter uncertainty, and
  nonlinear geometry (`RectPosterior`, marginalizing slip — the collapsed
  construction in one boxed sketch).
- NUTS in two paragraphs (what a sampler produces; internals deferred to
  movement 2); credible intervals vs. confidence intervals, said carefully
  once.
- **Prior sensitivity as a first-class practice**: re-run with defensible
  alternative priors (smoothness scale, positivity on/off, geometry prior
  widths); a sensitivity table of the quantities of interest (`M_w`, peak
  slip, depth); which conclusions are robust and which are prior-driven.

**Concepts & math — movement 2: diagnostics.**
- What can go wrong silently: unconverged chains that look smooth; the
  danger of single-chain, single-seed results.
- **Convergence diagnostics** at working depth: trace plots read critically,
  split-`R̂`, effective sample size (why correlated draws count less),
  divergences and what they signal about geometry (with the concrete remedy
  list: reparameterize, tighten priors, more warmup).
- **Prior predictive checks** as the cheap *pre*-sampling sanity test
  (geometry draws that break the half-space are found before burning
  compute); **posterior predictive checks**: draw → predict → compare
  against held-out and in-sample data — the Bayesian cousin of Chapter 13's
  residual forensics.
- One compact worked pathology: a deliberately hard posterior (e.g.
  near-surface patch with weak data) sampled badly, diagnosed, fixed,
  re-diagnosed.
- Roadmap honesty: some diagnostic conveniences (rank plots, automated
  warnings with remedies) arrive with PLAN.md 5.2; the chapter teaches the
  concepts with the diagnostics `geodef.bayes` exposes today and is upgraded
  when 5.2 lands.

**Key calls.** `geodef.bayes.SlipPosterior` / `RectPosterior`, the NUTS
sampling entry point, convergence-diagnostic outputs, credible-interval and
conditional-draw utilities, posterior predictive draws via the prediction
utilities, `geodef.synthetic` for held-out data,
`geodef.backend.set_backend('jax')`.

**Plots.** Posterior slip mean vs. Tikhonov solution; per-patch credible
intervals vs. Chapter 08's `σ`; prior-vs-posterior overlays for two geometry
parameters; the prior-sensitivity table as a figure; good vs. bad trace
plots; `R̂`/ESS summary table; divergence locations; PPC panels (data vs.
predictive envelope).

**Exercises.** Verify the analytic correspondence (posterior mean vs.
regularized solution) numerically; switch positivity on/off and compare
shallow-slip conclusions; halve and double the prior smoothness scale and
report which findings move; run four chains from dispersed starts and
compare to one long chain; construct a PPC that *passes* while the model is
wrong (tie back to Chapter 13); challenge — compute ABIC's `λ` and the
hierarchical posterior's `λ` distribution and reconcile them.

---

## 9. Solution Notebooks **[new]**

Every chapter's exercises get worked solutions, published separately so the
course pages stay spoiler-free:

- **Location:** `tutorials/solutions/NN_<same_stem>_solutions.ipynb`, one per
  chapter, listed at the bottom of `tutorials/README.md` under their own
  heading (not interleaved with the course table).
- **Content:** each exercise restated, then a worked solution with the same
  interpretation standard as the chapters — the *reasoning*, not just the
  output. Challenge exercises may include discussion of alternative
  approaches.
- **Tested:** executed by `tests/test_tutorials.py` alongside their chapters
  (same optional-dependency gating and the same CI budget, §11 decision 9),
  so solutions cannot drift from the API. Solutions reuse the scenario
  builder and chapter setup verbatim.
- **Authoring rule:** a chapter and its solutions notebook are one unit of
  work — an exercise may not land without its solution.

---

## 10. Authoring Checklist (per chapter)

Before a chapter is considered done:

- [ ] Header block: title, promise, learning objectives, prerequisites,
      estimated time band (§4).
- [ ] Motivation section before any math; theory before the code that uses
      it; assumptions stated when introduced.
- [ ] Notation matches the glossary (seeded from §5); glossary terms linked
      on first use; conventions cited to `docs/conventions.md`, not restated.
- [ ] API spellings follow the module-path policy (§5); no legacy top-level
      expert aliases.
- [ ] Named result views for routine operations; blocked vectors only where
      ordering is the lesson.
- [ ] Uses the recurring scenario (03–08) or the builder (05+) per §5; seeds
      fixed and visible.
- [ ] Every figure labeled (titles, axes, units, colorbars) **and**
      interpreted in prose.
- [ ] Checkpoint questions with `<details>` answers; common-mistakes section;
      recap with decision-guide pointer; further reading with annotations.
- [ ] 3–6 exercises ending in a challenge; solutions notebook complete and
      executing (§9).
- [ ] Runs top-to-bottom under `tests/test_tutorials.py` within the CI
      budget; optional-dependency chapters correctly gated.
- [ ] `tutorials/README.md` updated (course table, time, extras flags).

---

## 11. Design Decisions (settled)

Recorded so they are not re-litigated; historical decisions from the first
generation are retained where they still bind. Decisions 3–4 and 7–10 record
the 2026-07 resolutions of this outline's former open-questions list.

1. **Course shape: two merges, one renumbering, then frozen.** Regularization
   and choosing-λ are one chapter (04); Bayesian formulation, prior
   sensitivity, and diagnostics are one chapter (14) — fifteen notebooks
   total (00–14) rather than seventeen. The renumbering (§3 table) happens
   once, via `git mv` in lockstep with `tests/test_tutorials.py` and
   `tutorials/README.md`, as part of the 2.2 rewrite; after that, numbers
   are frozen. Fallback: if 04 or 14 proves unteachable at 90–120 minutes,
   each has a recorded seam (operators/selection; inference/diagnostics) at
   which to split — renumber at most once more, only then.
2. **Quickstart lives in the docs layer, previewed in Chapter 00.** The
   five-minute quickstart's home is `README.md` + `docs/quickstart.md`; the
   identical code closes Chapter 00 and is pinned by a golden-workflow test
   (2.4). There is no separate "quickstart notebook."
3. **The docs layer is three files.** `docs/quickstart.md`,
   `docs/glossary.md`, and `docs/workflow.md` — the last combining the
   API-levels map with the decision guides, which are the map read
   question-first. Flat files now; the 2.3 documentation site reorganizes
   navigation later without changing these homes.
4. **The glossary is the source of truth for notation.** §5's table seeds
   `docs/glossary.md`; once that file exists, chapters and docs defer to it
   and §5 is reduced to a pointer. Consistency is enforced by the API-doc
   drift check (PLAN.md 0.3) rather than generation machinery. This outline
   itself is transient (see the Lifecycle note at the top).
5. **Scenario delivery — explicit first, builder after.** Supersedes the
   first-generation "always copy/paste" rule; see §5. The docs-only delivery
   keeps explicit setup in all chapters. Chapters 05+ migrate to
   `synthetic.scenario(...)` atomically when that runtime API lands.
6. **Depth in prose, not compute.** The textbook deepening must not slow the
   executed suite materially; heavy demonstrations are shrunk or moved to
   `examples/`.
7. **`geodef.synthetic` is approved; `compare` lives in `invert`, unaliased.**
   The module ships `scenario`, `checkerboard`, `spike`, and `noisy_data` as
   pure functions. Recovered-versus-input comparison is
   `geodef.invert.compare` — it consumes an `InversionResult` and sits with
   the other assessment functions — with no `synthetic` alias: one name, one
   home, the same principle under which 2.2 deletes the legacy top-level
   aliases.
8. **Backslip convention.** Interseismic velocities are first corrected per
   station for rigid-block motion into a declared reference frame — normally
   the overriding plate fixed — using the same Euler pole(s) that supply the
   per-patch plate-motion vectors, so block correction and coupling share
   one vocabulary. Backslip is slip anti-parallel to the local plate-motion
   vector on the fault surface (normal sense — the inverse of thrust — on a
   megathrust; opposite-sense strike-slip on a transform), with non-negative
   amplitude bounded by the local plate rate. Coupling = backslip rate /
   plate rate ∈ [0, 1]; moment-deficit rate integrates `μ` × backslip rate ×
   area. In existing machinery: `components='plate'` with `plate_rake`
   pointing in the backslip direction and `bounds=(0, rate)`. Note the raw
   convention consequence — positive backslip has negative dip-slip — which
   `docs/conventions.md` must record for raw-component users. This decision
   is the 4.4 convention sub-item pulled forward; the rest of 4.4
   (coupling-fraction API, moment-deficit outputs, block co-estimation)
   stays on its roadmap schedule and must not be precluded by the two-step
   teaching treatment.
9. **CI budget.** `tests/test_tutorials.py` (chapters + solutions, same job)
   targets ≤ ~5 minutes for the base-install path and ≤ ~10 minutes
   including the JAX/Bayes-gated chapters. Solutions notebooks get a pytest
   marker so they can be deselected locally. Standing goal alongside:
   shorten the existing JAX/Bayesian test modules where possible (shared
   compiled kernels, smaller draw counts) rather than letting the gated
   path grow to fit the budget.
10. **Checkpoint answers use collapsed `<details>` blocks.** They render
    collapsed on GitHub and in Jupyter; environments that auto-expand them
    degrade gracefully (answers visible below the question, still clearly
    marked).
11. **Old-generation dispositions stand.** Caching is a Chapter 02 sidebar;
    the plotting gallery is unnumbered `reference_plots.ipynb`; real mesh
    generation is an `examples/` study (Chapter 11 may build a small mesh
    inline); the old Tutorial-07 covariance blocker is resolved and that
    notebook exists (now Chapter 06).
12. **Module-path migration is atomic.** Notebook/example/docs migration off
    the legacy top-level expert aliases lands in one commit with the
    `geodef/__init__.py` removal and the `tests/test_public_api.py` flip
    (PLAN.md 2.2).
13. **Docs-only branch compatibility.** The canonical 00–14 files use the
    frozen numbering. Legacy numbered paths are relative notebook symlinks to
    those canonical files so the unchanged execution harness runs the same
    content. Remove the links in the later atomic test-enumeration commit;
    they are not a second course and are omitted from the course table.

---

## 12. Open Questions

None currently open — the 2026-07 review resolved the initial seven (see §11
decisions 1, 3, 4, 7, 8, 9, 10). Record new open questions here as they
arise, and move each to §11 when settled.
