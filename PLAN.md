# PLAN.md - Development Plan for GeoDef

## Goal

Build **GeoDef**: a flexible, student-friendly Python library for forward and
inverse modeling of fault slip in elastic half-spaces. Consolidate existing
Matlab and Python code into a single well-tested package that helps students
get started quickly while remaining capable for research.

**Important: Read `PYTHON.md` before editing any code.**

---

## Current State - 2026-05-20

GeoDef's core library is functional and well covered by runtime tests. The
remaining work is mostly stabilization of local API work, documentation and
notebook refresh, tooling cleanup, and then targeted extensions.

### Verification Snapshot

| Command | Result | Notes |
|---------|--------|-------|
| `uv run pytest` | 799 passed, 1 skipped, 800 collected | Runtime suite is green; 222 warnings |
| `uv run ruff check` | 515 errors | Mostly import order, line length, unused imports, notebooks/tests |
| `uv run mypy src/geodef` | 140 errors in 9 files | Missing stubs plus optional-array and matplotlib typing issues |

### Local Working Tree

At this review, the branch is `main` tracking `origin/main` with uncommitted
work already present. Treat these as in-progress user changes, not cleanup
targets:

- Modified package files: `fault.py`, `greens.py`, `invert.py`, `plot.py`
- Modified tests: `test_fault.py`, `test_invert.py`, `test_mesh.py`,
  `test_plot.py`
- Modified example: `examples/gorkha_earthquake/model_gorkha.ipynb`
- Untracked: `AGENTS.md`

The local diff appears to add public `Fault.strike`/`Fault.dip`, fixed-rake and
fixed-geographic-azimuth inversion modes, one-component slip plotting, and
Gorkha example updates. The runtime tests pass with these changes, but docs and
tooling are not yet caught up.

---

## Completed Core Work

| Area | Status |
|------|--------|
| Green's functions | `okada85`, `okada92`, `tri`, and unified `okada` dispatcher are implemented and cross-validated |
| Library structure | `src/` package layout, public `geodef` API, hatchling build config |
| Fault and data model | `Fault`, `GNSS`, `InSAR`, `Vertical`, Green's matrix assembly, caching, I/O |
| Inversion framework | WLS, NNLS, bounded LS, constrained QP, smoothing/damping/stress-kernel regularization, ABIC/CV/L-curve tuning |
| Assessment | Covariance, resolution, uncertainty, per-dataset diagnostics, moment and magnitude |
| Visualization | Slip, vectors, InSAR, fit, fault3d, map, resolution, uncertainty plots |
| Mesh generation | `Mesh`, trace/polygon/points/slab2.0 generation, `Fault.from_mesh()` |
| Docs | Per-module API reference exists in `docs/` |
| Examples | Four general notebooks plus a real-data Gorkha earthquake example |

Current package modules:

`cache`, `data`, `fault`, `greens`, `invert`, `mesh`, `okada`, `okada85`,
`okada92`, `plot`, `transforms`, `tri`, and package `__init__`.

---

## Immediate Next Steps

### 1. Stabilize Current Local WIP

Priority: high. This is the most important next logical unit because the code
already exists locally and passes tests.

- Decide whether to keep the new inversion API names:
  `components="rake"` with `rake=...`, and `components="azimuth"` with
  `slip_azimuth=...`.
- Finish documentation for those modes in `docs/invert.md`, `README.md`, and
  the Gorkha notebook.
- Decide whether `plot.slip()` should keep the old `component=` keyword as a
  backward-compatible alias after the local change to `components=`.
- Add or update tests for any chosen backward compatibility behavior.
- Run `uv run pytest` and commit only this coherent unit once docs and tests
  match the API.

### 2. Refresh User-Facing Docs and Examples

Priority: high. The docs are usable but now stale in several places.

- Update `README.md` test count from 669 to the current 800 collected tests.
- Update `docs/plot.md` for one-component slip vectors and the
  `components=`/`component=` decision.
- Update `docs/fault.md` for public `Fault.strike` and `Fault.dip` if the WIP
  lands.
- Update `docs/greens.md` once Green's matrix component selection is either
  implemented or explicitly deferred.
- Re-run or at least smoke-test notebooks that appear in the main examples
  table.
- Decide whether the progressive tutorial series should live in `tutorials/`
  or whether the existing `examples/01-04` notebooks should be treated as the
  tutorial path.

### 3. Restore Tooling Health

Priority: medium-high. Tests are green, but the project standards say Ruff and
mypy should pass before release-quality handoff.

- Decide Ruff scope: package only, package plus tests, or package plus notebooks.
- Add a narrow `pyproject.toml` lint config if notebooks are intentionally
  excluded from normal linting.
- Fix Ruff issues in small commits, starting with package files and tests before
  notebook formatting.
- Add missing type support or config for third-party packages (`scipy`,
  `pyproj`, `meshpy`, `matplotlib`) and then fix true mypy errors.
- Track runtime warnings separately:
  `np.row_stack` deprecation in `tri.py`, transform divide warnings, and the
  optional slab/mesh binary-compatibility warning.

### 4. Finish Small API Improvements

Priority: medium. These are good short development tasks after the WIP is
stabilized.

- **`greens.greens()` component selection**: add `components=` for custom
  workflows. If fixed-rake/azimuth lands, share the same column-selection logic
  with `invert()` so behavior is consistent.
- **Bounds semantics**: formalize scalar, per-component, and per-parameter
  bounds. Ensure `bounded_ls` and `constrained` handle arrays consistently and
  update docs/tests.
- **`Fault.from_triangles()` explicit connectivity**: add an optional
  `triangles` index array so imported meshes can preserve patch ordering.
- **GNSS E-N correlation**: add optional `rho` support and build the correct
  per-station covariance blocks.
- **Site names**: add optional `name` arrays to `GNSS` and `Vertical`, including
  save/load behavior.

### 5. Build Teaching Material

Priority: medium. This is the main remaining student-facing deliverable.

Recommended progressive sequence:

| # | Title | Key concepts |
|---|-------|-------------|
| 01 | Forward Model Basics | `Fault.planar()`, `displacement()`, map-view plot |
| 02 | Fault Discretization and the G Matrix | Multi-patch fault, `greens()`, column layout |
| 03 | Unregularized Inversion | `invert()`, `InversionResult`, overfitting demo |
| 04 | Regularization | Laplacian, damping, stress-kernel |
| 05 | Choosing Regularization Strength | L-curve, ABIC, cross-validation |
| 06 | Multiple Datasets | GNSS plus InSAR joint inversion |
| 07 | Correlated Noise and InSAR | Full covariance for InSAR |
| 08 | Bounds and Constraints | NNLS, bounded LS, inequality constraints |
| 09 | Uncertainty and Assessment | covariance, resolution, uncertainty |
| 10 | Nonlinear Geometry Search | `scipy.optimize`; optional `emcee` later |

Each tutorial should use synthetic data, include equations or conceptual
markdown where useful, show labeled plots inline, and end with student
exercises.

---

## Longer-Term Extensions

- Geographic plotting mode with optional Cartopy coastlines, borders, and
  topography.
- Interpolated slip visualization using `pcolormesh` for rectangular meshes and
  `tricontourf` for triangular meshes.
- Euler pole fitting from `related/shakeout_v2/euler_calc.py`.
- MCMC/Bayesian nonlinear inversion interface using `emcee`.
- Quasi-dynamic earthquake cycle modeling from
  `related/stress-shadows/unicycle/`.
- Additional Green's function engines: Meade (2007), compound dislocation
  model, and layered half-space support.

---

## Design Notes

- Public coordinates use local Cartesian `x=East`, `y=North`, `z=Up` plus
  geographic lat/lon where appropriate. Internal Green's function conventions
  are converted at module boundaries.
- Slip columns are blocked as `[:N]` strike-slip and `[N:]` dip-slip.
- Hash-based caching handles reuse of expensive Green's and regularization
  matrices.
- Optional dependencies should stay optional: slab2.0 mesh generation,
  Cartopy/geographic plotting, and Bayesian extensions should not burden the
  base install.
