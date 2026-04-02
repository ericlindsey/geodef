# PLAN.md - Development Plan for GeoDef

## Goal

Build **GeoDef**: a flexible, student-friendly Python library for forward and inverse modeling of fault slip in elastic half-spaces. Consolidate existing Matlab and Python code into a single well-tested package that helps students get started quickly while remaining capable for research.

**Important: Read `PYTHON.md` before editing any code.**

---

## Completed Phases (1–8)

| Phase | What was built | Tests |
|-------|---------------|-------|
| 1. Green's Functions | `okada85`, `okada92`, `tri` — verified against Matlab references, cross-validated | 113 |
| 2. Library Structure | Package layout, `okada` dispatcher, tooling, code migration | 25 |
| 3. Fault & Data | `Fault` class, `DataSet`/`GNSS`/`InSAR`/`Vertical`, `greens` assembly, `cache` | 167 |
| 4. Inverse Framework | WLS/NNLS/bounded LS/constrained QP, Laplacian/damping/stress-kernel regularization, ABIC/CV/L-curve tuning | 126 |
| 5. Uncertainty | Model covariance/resolution/uncertainty, per-dataset diagnostics, moment/magnitude | 47 |
| 6. Visualization | `plot` module: slip, vectors, InSAR, fit, fault3d, map, resolution, uncertainty | 120 |
| 7. Mesh Generation | `mesh` module: `Mesh` dataclass, `from_trace`, `from_polygon`, `from_points`, `from_slab2`; `Fault.from_mesh()` | 70 |
| 8. I/O | `result.save_table()`, `GNSS`/`InSAR`/`Vertical` `.save()`/`.to_gmt()`/`.load()`, `InversionResult.save()`/`.load()` | — |

**Total: 669 tests passing.**

> **Note:** The docs reorganization (new `docs/` per-module reference, shortened README/CLAUDE.md) was done alongside Phase 9 planning and is not tracked as a numbered phase.

Key design decisions:
- **Flat module structure** — one file, one concept
- **Transparent Okada85/92 dispatch** — auto-selects based on observation depth
- **Progressive complexity** — quick-start → modular → low-level
- **Polymorphic data dispatch** — new data types work everywhere with zero changes
- **Blocked column layout** — `[ss_0..ss_N, ds_0..ds_N]` for Green's matrices and slip vectors

---

## Phase 9: Tutorial Notebooks

Progressive series in `tutorials/`, ported from `related/shakeout_v2/notebooks/` and rewritten to use `geodef`.

| # | Title | Key concepts |
|---|-------|-------------|
| 01 | Forward Model Basics | `Fault.planar()`, `displacement()`, map-view plot |
| 02 | Fault Discretization & the G Matrix | Multi-patch fault, `greens()`, column layout |
| 03 | Unregularized Inversion | `invert()` with WLS, `InversionResult`, overfitting demo |
| 04 | Regularization: Smoothing & Damping | Laplacian, damping, stress-kernel |
| 05 | Choosing Regularization Strength | L-curve, ABIC, cross-validation |
| 06 | Weighted Least Squares & Multiple Datasets | GNSS + InSAR joint inversion |
| 07 | Correlated Noise & InSAR | Full covariance for InSAR |
| 08 | Bounds & Constraints | NNLS, bounded LS, inequality constraints |
| 09 | Uncertainty & Model Assessment | `model_covariance()`, `model_resolution()`, `model_uncertainty()` |
| 10 | Nonlinear Inversion (MCMC) | `scipy.optimize`, `emcee` for fault geometry |

Each tutorial should:
- Build on the previous one (progressive complexity)
- Include clear markdown explanations of the math/concepts
- Use synthetic data so no external files needed
- Show plots inline with well-labeled axes
- End with exercises for students

Worked research-level examples with real data go in `examples/` (see existing notebooks 01–04 as models).

---

## Phase 10: Future Extensions

### 10.1 Small API Improvements

- **`greens.greens()` component selection** — `greens()` always returns the full `(M, 2*N)` matrix; component selection currently only happens inside `invert()` via a private helper. Add a `components='both'|'strike'|'dip'` argument to `greens.greens()` so users building custom workflows (e.g. manual G assembly, external solvers) don't have to know the blocked column layout and slice manually.
- **`invert()` per-component bounds** — `bounds` currently applies the same limits to both slip components. Add support for per-component specification, e.g. `bounds=[(0, None), (-1, 1)]` for non-negative strike-slip and bounded dip-slip.
- **`Fault.from_triangles()` with explicit connectivity** — currently derives triangles from ENU vertex coordinates only. Add an optional `triangles` parameter (index array, shape `(M, 3)`) so users can preserve a specific patch ordering when importing an existing slip model.
- **`GNSS` E-N correlation** — add optional `rho` parameter (scalar or per-station array) for the E-N correlation coefficient. Most datasets have `rho=0` but some processed solutions do not, and it affects the full covariance matrix used in inversion.
- **`GNSS` and `Vertical` site names** — add optional `name` parameter (string) to GNSS and Vertical data classes.

### 10.2 Geographic Projection & Cartopy Plotting
- `projection='geographic'` option on all plot functions
- Cartopy-based coastlines, country borders, topography
- Keep cartopy as an optional dependency

### 10.3 Quasi-Dynamic Earthquake Cycle Modeling
Port from `related/stress-shadows/unicycle/`: rate-and-state friction, stress interaction, ODE integration, tectonic loading.

### 10.4 Additional Green's Function Engines
- **Meade (2007)** — backslip triangular dislocations
- **Compound dislocation model (CDM)** — volcanic sources
- **Layered elastic half-space** — EDGRN/EDCMP

### 10.5 Euler Pole Fitting
Port `related/shakeout_v2/euler_calc.py` — fit rigid plate rotation to GNSS velocities.

### 10.6 MCMC / Bayesian Inversion Module
Structured interface using `emcee` for nonlinear inversion of fault geometry parameters.

### 10.7 Interpolated Slip Visualization
Smooth/contour-style rendering: `pcolormesh` for rectangular meshes, `tricontourf` for triangular.

---

## Notes

- **Coordinate conventions**: Public API uses East/North/Up and lat/lon. Green's functions handle internal conventions behind the interface.
- **Performance**: Hash-based caching handles reuse of expensive G and L matrices.
- **Mesh generation**: Depends on `meshpy`/`triangle` and `netCDF4` (optional).
- **Projections**: `.seg` format uses flat-earth; for large faults a proper projection should be added to `transforms.py`.
