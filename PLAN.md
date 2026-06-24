# PLAN.md - Development Plan for GeoDef

## Goal

Build **GeoDef**: a flexible, student-friendly Python library for forward and
inverse modeling of fault slip in elastic half-spaces. Consolidate existing
Matlab and Python code into a single well-tested package that helps students
get started quickly while remaining capable for research.

**Important: Read `PYTHON.md` before editing any code.**

---

## Current State - 2026-06-17

GeoDef's core library is functional and well covered by runtime tests. The
fixed-direction inversion API work has been stabilized, documented, tested, and
pushed to `origin/main`. A current-state `.md` documentation refresh, agent
policy update, tutorial/example split, and tutorial notebook execution checks
have also been completed. The progressive teaching sequence is now being
rebuilt against the `related/shakeout_v2/notebooks/` reference material:
notebooks 01 (forward model) and 02 (discretization / G matrix) are drafted and
executed under pytest, and the earlier feature notebooks are archived as
`tutorials/old_*`. The remaining work is the rest of the teaching sequence
(03-10), tooling cleanup, and targeted extensions.

### Verification Snapshot

| Command | Result | Notes |
|---------|--------|-------|
| `uv run pytest -q` | 814 passed, 1 skipped, 815 collected | Runtime suite is green; tutorial notebooks 01-02 execute; 232 warnings |
| `uv run ruff check` | 512 errors | Mostly import order, line length, unused imports, notebooks/tests |
| `uv run mypy src/geodef` | 130 errors in 9 files | Missing stubs plus optional-array and matplotlib typing issues |

### Local Working Tree

No planned local API WIP remains from the fixed-direction inversion work.
`AGENTS.md` is now treated as a first-class onboarding file alongside
`CLAUDE.md`.

The recently completed stabilization commit added public `Fault.strike` and
`Fault.dip`, fixed-rake and fixed-geographic-azimuth inversion modes,
one-component slip plotting, stress-kernel projection into the active slip
basis, and Gorkha example updates.

### Maintenance Policies

- When executing a step listed in this plan, update `PLAN.md` in the same
  logical unit so the roadmap remains current.
- Keep docs `.md` files up to date alongside minor code changes. If a large
  docs rewrite is needed, add it as its own plan step.
- Do not add `Co-Authored-By` trailers to commit messages. AI co-authorship is
  tracked in `README.md`, and the model list there plus `CLAUDE.md`/`AGENTS.md`
  should be updated when a new AI model materially contributes.

---

## Completed Core Work

| Area | Status |
|------|--------|
| Green's functions | `okada85`, `okada92`, `tri`, and unified `okada` dispatcher are implemented and cross-validated |
| Library structure | `src/` package layout, public `geodef` API, hatchling build config |
| Fault and data model | `Fault`, `GNSS`, `InSAR`, `Vertical`, Green's matrix assembly, caching, I/O |
| Inversion framework | WLS, NNLS, bounded LS, constrained QP, smoothing/damping/stress-kernel regularization, ABIC/CV/L-curve tuning |
| Fixed-direction inversion | `components="rake"` and `components="azimuth"` project Green's matrices and stress kernels into the selected one-parameter slip basis |
| Assessment | Covariance, resolution, uncertainty, per-dataset diagnostics, moment and magnitude |
| Visualization | Slip, vectors, InSAR, fit, fault3d, map, resolution, uncertainty plots, including one-component slip vectors |
| Mesh generation | `Mesh`, trace/polygon/points/slab2.0 generation, `Fault.from_mesh()` |
| Docs | Per-module API reference exists in `docs/`; current-state `.md` refresh completed 2026-05-20 |
| Tutorials | Progressive teaching sequence under construction in `tutorials/`; 01 (forward model) and 02 (discretization / G matrix) drafted and executed under pytest; earlier feature notebooks retained as `old_*` |
| Examples | Real-data/project examples live in `examples/`, currently the Gorkha earthquake workflow |

Current package modules:

`cache`, `data`, `fault`, `greens`, `invert`, `mesh`, `okada`, `okada85`,
`okada92`, `plot`, `transforms`, `tri`, and package `__init__`.

---

## Immediate Next Steps

### 1. Refresh Broader User-Facing Docs and Examples

Priority: high. The current `.md` docs have been refreshed for obvious API
drift, stale counts, onboarding rules, and AI-attribution policy. The
introductory notebooks now live in `tutorials/`, while real-data workflows stay
under `examples/`. Tutorial notebooks are executed by `tests/test_tutorials.py`.
The progressive sequence is being rewritten against the
`related/shakeout_v2/notebooks/` reference (matching its markdown depth, but with
much shorter GeoDef code and the double-demo convention; see
`tutorials/OUTLINE.md` for the master plan); 01-02 are done, 03-10 remain.

- Decide whether Gorkha should get a lightweight smoke test or remain a manual
  real-data example because it is heavier and data-dependent.
- Draft tutorials 03-10 from the teaching-material sequence below.
- Keep `README.md`, `tutorials/README.md`, and `examples/README.md` aligned as
  notebooks are added or moved.

### 2. Restore Tooling Health

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

### 3. Finish Small API Improvements

Priority: medium. These are good short development tasks after the WIP is
stabilized.

- **`greens.greens()` component selection**: add public `components=` support
  for custom workflows outside `invert()`, reusing the same column-selection
  semantics as the stabilized inversion path.
- **Bounds semantics**: formalize scalar, per-component, and per-parameter
  bounds. Ensure `bounded_ls` and `constrained` handle arrays consistently and
  update docs/tests.
- **`Fault.from_triangles()` explicit connectivity**: add an optional
  `triangles` index array so imported meshes can preserve patch ordering.
- **GNSS E-N correlation**: add optional `rho` support and build the correct
  per-station covariance blocks.
- **InSAR full covariance `C_d`**: the `InSAR` dataset currently has no good way
  to specify a full (non-diagonal) data covariance matrix. Add a way to build
  `C_d` from a covariance function (e.g. exponential/Gaussian with a correlation
  length) or pass an explicit matrix, and thread it through `invert()`. **This
  is a prerequisite for tutorial 07 (Correlated Noise and InSAR)** — that
  notebook is blocked until this lands.
- **Site names**: add optional `name` arrays to `GNSS` and `Vertical`, including
  save/load behavior.

### 4. Build Teaching Material

Priority: medium. This is the main remaining student-facing deliverable. The
sequence is being (re)built against the `related/shakeout_v2/notebooks/`
reference material: matching or exceeding its markdown depth, but replacing the
hand-rolled Okada / least-squares code with the much shorter GeoDef API.

The full design for this sequence — per-notebook goals, the math each must
develop, the minimal code surface, plots, and exercises — lives in
`tutorials/OUTLINE.md`, the master plan for notebook design. The tutorials are a
course in geodetic inverse methods (theory first, short illustrative code), not
an API tour; the earlier introduction-style notebooks 01–04
(forward/caching/plotting/mesh) are retired (kept as `tutorials/old_*` during the
transition, to be deleted once their content has migrated into the new tutorials
or `examples/`).

Progressive sequence (✅ = drafted and executed under pytest):

| # | Title | Key concepts | Status |
|---|-------|-------------|--------|
| 01 | Forward Model Basics | `Fault.planar()`, `displacement()`, map-view plot | ✅ |
| 02 | Fault Discretization and the G Matrix | Multi-patch fault, `greens()`, column layout | ✅ |
| 03 | Unregularized Inversion | `invert()`, `InversionResult`, overfitting demo | |
| 04 | Regularization | Laplacian, damping, stress-kernel | |
| 05 | Choosing Regularization Strength | L-curve, ABIC, cross-validation | |
| 06 | Multiple Datasets | GNSS plus InSAR joint inversion | |
| 07 | Correlated Noise and InSAR | Full covariance for InSAR (blocked; see InSAR C_d TODO in step 3) | |
| 08 | Bounds and Constraints | NNLS, bounded LS, inequality constraints | |
| 09 | Uncertainty and Assessment | covariance, resolution, uncertainty | |
| 10 | Nonlinear Geometry Search | `scipy.optimize`; optional `emcee` later | |

Each tutorial develops its theory in markdown, keeps code short and
illustrative, shows labeled plots inline, and ends with exercises. See
`tutorials/OUTLINE.md` for the full per-notebook breakdown, shared conventions
(including the **double-demo** convention — a hand calculation followed by the
one-line GeoDef equivalent, used sparingly where it clarifies the API), the
visualization strategy, and the disposition of the retired 01–04 notebooks.

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
