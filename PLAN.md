# PLAN.md - Development Plan for GeoDef

## Goal

Build **GeoDef**: a flexible, student-friendly Python library for forward and inverse modeling of fault slip in elastic half-spaces. Consolidate existing Matlab and Python code into a single well-tested package that helps students get started quickly while remaining capable for research.

**Important: Read `PYTHON.md` before editing any code.**

---

## Phase 1: Foundation — Green's Functions & Testing [COMPLETED]

All three Green's function implementations are verified and cross-validated. **113 tests pass** across 4 test files (949 lines total).

### 1.1 Finalize Okada92 Python Implementation [DONE]
- Surface results verified to match `okada85.py` exactly (Okada92 at z=0 reproduces Okada85).
- Tested across multiple dip angles (15, 45, 70, 90 degrees) and all slip components.
- Depth variation, decay, linearity, and zero-slip tests all pass.

### 1.2 Expand Okada85 Test Coverage [DONE]
- 9 reference cases from Okada (1985) Table 2 for displacement, tilt, and strain (27 parametrized tests).
- Geometry tests across 4 dip angles x 3 slip types (12 tests).
- Symmetry: antisymmetry for vertical strike-slip, zero-slip, linearity.
- Far-field decay test, vectorized input tests.

### 1.3 Expand tdcalc Test Coverage [DONE]
- 4 Matlab reference configurations (FS_simple, FS_complex, HS_simple, HS_complex) for displacement and strain (8 tests).
- Property tests: zero-slip, linearity, far-field decay, full-space vs half-space deep source.
- Cross-validated against Okada85 at surface (triangle pair = rectangle, 9 geometry x slip combos).
- Cross-validated against DC3D (Okada92) at depth for displacement and strain.

### 1.4 Cross-Validation Suite [DONE]
- `test_cross_validation.py` (428 lines): Okada85 vs DC3D at surface, Okada85 vs Okada92 wrapper at surface, tdcalc vs Okada85 at surface, tdcalc vs DC3D at depth (displacement + strain).

---

## Phase 2: Library Structure & Packaging [COMPLETED]

### 2.1 Package Layout [DONE]

Design principle: **flat is better than nested**. Students should be able to do useful work with 1-2 imports. Advanced features are available but never required.

```
geodef/
├── pyproject.toml
├── src/geodef/
│   ├── __init__.py            # Top-level convenience API
│   ├── okada.py               # Unified dispatcher: auto-selects okada85 or okada92
│   ├── okada85.py             # Okada (1985) — surface displacements, tilts, strains
│   ├── okada92.py             # Okada (1992) — internal deformation at depth
│   ├── tri.py                 # Triangular dislocation interface (tdcalc)
│   ├── fault.py               # Fault geometry: create, load, discretize, plot
│   ├── data.py                # DataSet base + GNSS, InSAR, Vertical, etc.
│   ├── greens.py              # Green's matrix assembly (G for any source+data combo)
│   ├── invert.py              # Inversion: solve, regularize, tune hyperparameters
│   ├── cache.py               # Hash-based caching for expensive matrix computations
│   ├── transforms.py          # Coordinate transforms (geographic <-> local Cartesian)
│   ├── mesh.py                # Triangular mesh generation (from slabMesh)
│   └── plot.py                # Visualization utilities
├── tests/
├── tutorials/                 # Progressive notebook series (ported from shakeout_v2)
└── examples/                  # Research-level worked examples
```

### 2.2 Key Design Decisions [DONE]

**Flat module structure**: No deeply nested subpackages. `geodef.okada`, `geodef.fault`, `geodef.invert` — each is one file, one concept. This avoids `from geodef.greens.rectangular.okada85 import displacement` in favor of `geodef.okada.displacement(...)`.

**Transparent Okada85/92 selection**: `okada.py` is a thin dispatcher — it calls okada85 when all observation points are at the surface (z=0) for speed, and okada92 otherwise. The user never needs to know which is called. The underlying `okada85.py` and `okada92.py` remain as separate files (this is convention in the geodetic community), but users typically only interact with `okada.py`.

**Capitalization convention**: Classes use PascalCase (`Fault`, `Vertical`). Acronym class names stay fully capitalized (`GNSS`, `InSAR`). Standard scientific notation is preserved (`result.Mw`). Modules and functions are always snake_case. Data types are top-level classes (not nested under a `Data` namespace) with a shared `DataSet` base class — this is more Pythonic and plays better with type hints, pickle, and introspection. Examples:
- `geodef.Fault.planar(...)` — class with factory classmethods
- `geodef.GNSS(...)`, `geodef.InSAR(...)`, `geodef.Vertical(...)` — data type classes, importable directly
- `isinstance(gnss, geodef.DataSet)` — base class for type checking
- `result.Mw` — standard seismological notation
- `geodef.greens(...)`, `geodef.invert(...)` — module-level functions, lowercase

**Convenience top-level API**: Common 3-line workflows should work from `import geodef`:
```python
import geodef

# Create a discretized fault
fault = geodef.Fault.planar(lat=0, lon=100, depth=10e3, strike=320, dip=15,
                            length=100e3, width=50e3, n_length=10, n_width=5)

# Forward model with per-patch slip (50 patches = 10 x 5)
slip_strike = np.random.uniform(0, 2, fault.n_patches)  # (50,) array
slip_dip = np.zeros(fault.n_patches)
ue, un, uz = fault.displacement(obs_lat, obs_lon, slip_strike, slip_dip)

# Scalar slip values broadcast to all patches
ue, un, uz = fault.displacement(obs_lat, obs_lon, slip_strike=1.0, slip_dip=0.5)
```

**Progressive complexity**: The library exposes three levels:
1. **Quick start** — `geodef.Fault.planar(...)` + `.displacement()` / `.invert()` methods (notebook-friendly)
2. **Modular** — Compose `geodef.okada`, `geodef.greens`, `geodef.invert` for custom workflows
3. **Low-level** — Direct access to `geodef.okada.displacement(e, n, depth, ...)` for maximum control

### 2.3 Set Up Packaging & Tooling [DONE]
- `pyproject.toml` with hatchling build system, `uv`, `ruff`, `mypy`, `pytest`
- Installable via `uv pip install -e .`
- Minimal required dependencies: `numpy`, `scipy`, `matplotlib`
- Optional: `meshpy` (for mesh generation), `netCDF4` (for slab2.0), `emcee` (for MCMC)

### 2.4 Migrate Existing Code [DONE]
- `geometry/okada/okada85.py` → `src/geodef/okada85.py`
- `geometry/okada/okada92.py` → `src/geodef/okada92.py`
- New `src/geodef/okada.py` — dispatcher that auto-selects 85 vs 92
- `geometry/tdcalc/tdcalc.py` → `src/geodef/tri.py`
- `geometry/okada/okada_greens.py` + `okada_utils.py` → `src/geodef/greens.py`
- `related/shakeout_v2/fault_model.py` + `slip_model.py` → `src/geodef/fault.py`
- `related/shakeout_v2/geod_transform.py` → `src/geodef/transforms.py`
- `geometry/slabMesh/slabMesh.py` → `src/geodef/mesh.py`
- `related/shakeout_v2/test_geod_transform.py` → `tests/test_transforms.py`
- `related/shakeout_v2/test_laplacian.py` → `tests/test_greens.py`
- `related/shakeout_v2/notebooks/` → `tutorials/` (deferred to Phase 7)

---

## Phase 3: Fault & Data Abstractions [COMPLETED]

### 3.1 `geodef.Fault` — Fault Geometry [DONE]

Immutable `Fault` class with factory classmethods, engine abstraction, and vectorized forward modeling. **59 tests** in `tests/test_fault.py`.

**Factory classmethods:**
- `Fault.planar(lat, lon, depth, strike, dip, length, width, n_length, n_width)` — centroid-based
- `Fault.planar_from_corner(...)` — top-left corner-based
- `Fault.load(fname, format=..., ref_lat=..., ref_lon=...)` — center, topleft, or seg format
- `Fault.from_slab2(...)` — triangular mesh from slab2.0 (deferred)

**Properties:** `n_patches`, `centers`, `centers_local`, `areas`, `engine`, `grid_shape`, `laplacian`, `vertices_2d`, `vertices_3d`

**Methods:** `displacement()`, `greens_matrix()`, `stress_kernel()`, `moment()`, `magnitude()`, `patch_index()`, `save()`

**Module utilities:** `moment_to_magnitude()`, `magnitude_to_moment()`

**Seg format support:** Ported unicycle `flt2flt.m` algorithm for geometric depth-variable patch sizing. Loads/saves `.seg` files with `ref_lat`/`ref_lon` for geographic placement.

**Open issues:**
- Coordinate transforms: `.seg` format uses `translate_flat` (flat-earth); stress-shadows uses polyconic projection. For large faults, a proper projection system should be added to `transforms.py`.
- Patch ordering: `Fault.planar()` orders deepest-first; `flt2flt` orders shallowest-first.

```python
# Simple planar fault (rectangular patches → uses okada engine)
fault = geodef.Fault.planar(lat, lon, depth, strike, dip,
                            length, width, n_length=10, n_width=5)

# Load from unicycle seg file
fault = geodef.Fault.load("mentawai.seg", format="seg",
                          ref_lat=-2.0, ref_lon=100.0)

# Triangular mesh from slab2.0 (triangular patches → uses tri engine)
fault = geodef.Fault.from_slab2("cas_slab2_dep_02.24.18.grd", bounds=[...])

# Properties available on all faults
fault.centers        # (N, 3) patch centroids [lat, lon, depth]
fault.areas          # (N,) patch areas in m²
fault.n_patches      # number of patches
fault.engine         # 'okada' or 'tri' (set at creation, not changeable)
fault.laplacian      # (N, N) smoothing operator
fault.moment(slip)   # scalar seismic moment
```

### 3.2 `geodef.DataSet` — Geodetic Data [DONE]

All data types inherit from a common `DataSet` base class, which defines the polymorphic interface used by `greens()` and `invert()`. Each subclass specifies two things:
1. **`greens_type`** — what kind of raw Green's function it needs from the engine (`'displacement'` or `'strain'`)
2. **`project()`** — how to map those raw components to the data's observation space

Acronym names stay capitalized; non-acronym names use PascalCase per Python class convention. All are importable directly from `geodef`:

```python
gnss  = geodef.GNSS(lat, lon, ve, vn, vu, se, sn, su)
insar = geodef.InSAR(lat, lon, los, sigma, look_e, look_n, look_u)
vert  = geodef.Vertical(lat, lon, displacement, sigma)
```

Each data type also exposes a common interface for `invert()`:
- `data.obs` — observation vector (what was measured)
- `data.sigma` — uncertainties (1-sigma)
- `data.covariance` — optional full covariance matrix

Full covariance matrices are optional; if not provided, `sigma` is used to build a diagonal weight matrix.

**Base class and dispatch mechanism:**

```python
class DataSet:
    """Base class for all geodetic data types."""
    greens_type: str = 'displacement'  # what to request from the engine

    def project(self, *components):
        """Map raw Green's function output to this data type's observation space."""
        raise NotImplementedError

class GNSS(DataSet):
    greens_type = 'displacement'

    def project(self, ue, un, uz):
        # 3 components per station, interleaved by station: [e1,n1,u1, e2,n2,u2, ...]
        return np.column_stack([ue, un, uz]).ravel()

class InSAR(DataSet):
    greens_type = 'displacement'

    def project(self, ue, un, uz):
        # Project onto look vector → 1 scalar per pixel
        return self.look_e * ue + self.look_n * un + self.look_u * uz

class Vertical(DataSet):
    greens_type = 'displacement'

    def project(self, ue, un, uz):
        return uz

# Future: strain data types request strain Green's functions
class Strainmeter(DataSet):
    greens_type = 'strain'

    def project(self, exx, exy, exz, eyy, eyz, ezz):
        # Project strain tensor to observed components
        ...
```

Adding a new data type = write a new class with `greens_type` and `project()`. Nothing else in the library changes.

### 3.3 `geodef.greens` — Green's Matrix Assembly [DONE]

Polymorphic Green's matrix assembly using `DataSet.greens_type` dispatch and `DataSet.project()`. **32 tests** in `tests/test_greens_integration.py`.

**Core function:** `greens(fault, datasets)` — assembles the full Green's matrix for any combination of fault engine and data type. Slip columns are blocked: `[ss_0, ..., ss_N, ds_0, ..., ds_N]`.

```python
G = geodef.greens.greens(fault, gnss)              # single dataset
G = geodef.greens.greens(fault, [gnss, insar])     # joint: auto-stacks vertically
```

**Engine dispatch:** Four internal Green's functions handle all engine/data combinations:
- `displacement_greens()` — Okada engine, displacement data types
- `strain_greens()` — Okada engine, strain data types
- `tri_displacement_greens()` — Triangular engine, displacement data types
- `tri_strain_greens()` — Triangular engine, strain data types

**Stacking helpers:** `stack_obs()` and `stack_weights()` concatenate observation vectors and weight matrices across multiple datasets, matching the row layout of the joint G matrix.

**Also in `greens.py`:** `resolution()` matrix, `build_laplacian_2d()` / `build_laplacian_2d_simple()` regularization operators for structured grids, and `build_laplacian_knn()` for unstructured meshes (distance-weighted K-nearest-neighbors, sparse output).

**Changes from plan:** `build_patch_grid()` and `rect_fault_outline()` were moved from `greens.py` to `fault.py` as `Fault` methods/properties (`Fault.vertices_2d`, `Fault.vertices_3d`, `Fault.patch_outlines`), since they are fundamentally fault geometry operations.

### 3.4 Computation Caching (`geodef.cache`) [DONE]

Hash-based disk caching for expensive matrix computations, ported from the stress-shadows Matlab `DataHash` + HDF5 pattern. **29 tests** in `tests/test_cache.py`.

**Module API:**
```python
geodef.cache.set_dir("path/to/cache")  # default: .geodef_cache/
geodef.cache.get_dir()
geodef.cache.clear()
geodef.cache.enable() / geodef.cache.disable()
geodef.cache.is_enabled()
geodef.cache.info()  # {"n_files": int, "total_bytes": int}
```

**Transparent caching** — `greens()` and `stress_kernel()` automatically check for cached results:
```python
G = geodef.greens.greens(fault, data)  # first call: computes and saves
G = geodef.greens.greens(fault, data)  # second call: loads from cache
```

**Implementation:**
- `compute_hash(key_data)` — SHA-256 of serialized numpy arrays + metadata (dtype, shape, bytes), strings, floats, None
- `cached_compute(key_data, compute_fn)` — check cache, compute if missing, save compressed `.npz`
- Storage: `.geodef_cache/<hash[:2]>/<hash>.npz` (2-char subdirectories)
- Cache key includes: fault geometry arrays, engine, observation coordinates, data type, and type-specific fields (InSAR look vectors, GNSS component mode)
- Per-dataset caching in `greens()`: each dataset block is cached independently, so changing one dataset in a joint inversion doesn't invalidate the others

**Bug fix included:** `stress_kernel()` now correctly evaluates strain at patch centroid depths (using `okada92`/DC3D for internal deformation) instead of incorrectly computing surface strain. Added `obs_depth` parameter to `greens_matrix()`, `strain_greens()`, and `tri_strain_greens()`.

---

## Phase 4: Inverse Framework (`geodef.invert`)

### 4.1 One-Call Inversion

The simple call uses sensible defaults. Regularization is controlled via two main kwargs: `smoothing` (what type of regularization matrix) and `smoothing_strength` (how much weight it gets):

```python
# Simplest call: unregularized weighted least-squares
result = geodef.invert(fault, [gnss, insar])

# Laplacian smoothing with fixed weight
result = geodef.invert(fault, [gnss, insar],
                       smoothing='laplacian',
                       smoothing_strength=1e3,
                       bounds=(0, None),
                       method='bounded_ls')

result.slip          # (N, 2) strike-slip and dip-slip
result.residuals     # per-dataset residuals
result.chi2          # reduced chi-squared
result.moment        # scalar seismic moment
result.Mw            # moment magnitude
```

### 4.2 Regularization

Regularization is split into two concerns: **what matrix** to penalize with (`smoothing`) and **how strongly** to weight it (`smoothing_strength`).

**`smoothing`** — controls the regularization matrix L in the penalty term ||Lm||^2:

| `smoothing` value | What it builds |
|-------------------|---------------|
| `'laplacian'` (default) | Graph Laplacian for triangular meshes, finite-difference for rectangular grids. Area-weighted. |
| `'stresskernel'` | Stress interaction kernel (strain Green's functions evaluated at patch centers). For stress-shadow constraints (Lindsey et al., 2021). |
| `'damping'` | Identity matrix (Tikhonov / L2 damping). |
| `numpy.ndarray` | Custom (N, N) regularization matrix. Pass any matrix directly. |

Stress kernels are computed using the strain Green's functions from okada/tri, evaluated at fault patch centers — no additional Matlab porting needed beyond the existing strain implementations. These are also cached via `geodef.cache`.

**`smoothing_strength`** — controls the scalar weight on the regularization term:

| `smoothing_strength` value | Behavior |
|---------------------------|----------|
| `float` (e.g. `1e3`) | Fixed weight (lambda). |
| `'abic'` | Automatic via ABIC optimization. |
| `'cv'` | Automatic via cross-validation (use `cv_folds` kwarg, default 5). |

**Additional inversion kwargs:**

| Kwarg | Type | What it does |
|-------|------|-------------|
| `bounds` | `tuple` or `None` | Per-component slip bounds, e.g. `(0, None)` for non-negative |
| `cv_folds` | `int` | Number of folds when `smoothing_strength='cv'` (default 5) |
| `smoothing_range` | `tuple` | Search range for ABIC/CV, e.g. `(1e-2, 1e6)` |
| `method` | `str` | Solver choice (see 4.4) |

**Examples showing the full range:**

```python
# Laplacian smoothing, auto-tuned via ABIC
result = geodef.invert(fault, data, smoothing='laplacian', smoothing_strength='abic')

# Stress-shadow constraints with fixed weight
result = geodef.invert(fault, data, smoothing='stresskernel', smoothing_strength=1e4,
                       bounds=(0, None), method='constrained')

# Simple damping
result = geodef.invert(fault, data, smoothing='damping', smoothing_strength=1.0)

# Custom regularization matrix
result = geodef.invert(fault, data, smoothing=my_matrix, smoothing_strength=1e3)

# No regularization (default)
result = geodef.invert(fault, data)
```

### 4.3 Hyperparameter Tuning

```python
# Automatic via ABIC (works with any smoothing type)
result = geodef.invert(fault, data, smoothing='laplacian', smoothing_strength='abic')

# Automatic via cross-validation
result = geodef.invert(fault, data, smoothing='laplacian',
                       smoothing_strength='cv', cv_folds=5)

# Manual exploration: L-curve
lc = geodef.lcurve(fault, data, smoothing='laplacian',
                   smoothing_range=(1e-2, 1e6), n=50)
lc.plot()           # trade-off curve with optimal point marked
lc.optimal          # recommended smoothing_strength value
```

### 4.4 Solvers

| `method` value | Description | When to use |
|----------------|-------------|------------|
| `'wls'` (default) | Weighted least-squares | Fast, no constraints |
| `'nnls'` | `scipy.optimize.nnls` | Non-negative slip only |
| `'bounded_ls'` | `scipy.optimize.lsq_linear` | Bounded slip components |
| `'constrained'` | Quadratic programming | Stress-shadow inequality constraints |

---

## Phase 5: Uncertainty & Model Assessment

### 5.1 Model Covariance & Resolution
- `result.covariance` — model covariance matrix Cm
- `result.resolution` — resolution matrix R
- `result.uncertainty` — per-patch 1-sigma from diagonal of Cm

### 5.2 Fit Statistics
- `result.chi2`, `result.rms`, per-dataset residuals
- F-test for model comparison

### 5.3 Moment & Magnitude
- `result.moment`, `result.Mw` — computed automatically
- `fault.moment(slip)` — standalone calculation

---

## Phase 6: Visualization (`geodef.plot`)

Built-in plotting with sensible defaults:
```python
geodef.plot.slip(fault, result.slip)                        # colored patches
geodef.plot.fit(result, dataset_index=0)                    # observed vs predicted
geodef.plot.vectors(gnss, predicted=result.predict(gnss))   # gnss arrows
geodef.plot.lcurve(smoothing_values, misfits, norms)        # trade-off curve
```

### I/O
- `geodef.Fault.load()` / `.save()` — auto-detect format (seg, tri/ned, unicycle)
- `geodef.GNSS.load()` — common GNSS formats
- `geodef.InSAR.load()` — LOS + look vectors
- `result.to_gmt()` — export for GMT plotting

---

## Phase 7: Tutorials & Documentation

### 7.1 Tutorial Notebooks (port from `related/shakeout_v2/notebooks/`)
Progressive series rewritten to use `geodef`:
1. Forward model basics (single fault)
2. Fault discretization and G matrix
3. Regularized inversion
4. Choosing regularization strength
5. Multiple datasets and weighting
6. Real earthquake example
7. Interseismic coupling example
8. Nonlinear inversion (MCMC)

### 7.2 Documentation
- API reference (auto-generated from docstrings)
- Quick-start guide (5-minute first model)
- User guide with worked examples

---

## Execution Order

```
Phase 1 (Tests & Green's functions)    COMPLETE
    │
Phase 2 (Package scaffolding)          COMPLETE
    │
    ├── Phase 3 (Fault + Data + Greens + Cache)  COMPLETE (337 tests)
    │       │
    │       └── Phase 4 (Inversion)      ← NEXT
    │               │
    │               └── Phase 5 (Uncertainty)
    │
    ├── Phase 6 (Plotting + I/O)         ← can start during Phase 3
    │
    └── Phase 7 (Tutorials + Docs)       ← after core library is stable
```

---

## Phase 8: Future Additions

- **Angular-weighted KNN Laplacian**: The current `build_laplacian_knn` uses simple inverse-distance weighting. The Huiskamp (1991) formulation adds angular correction terms (`Theta_j / Theta_tot`) that account for uneven angular distribution of neighbors. This could be added as an option (e.g. `method='huiskamp'`) if cases arise where the simple version produces visibly different regularization behavior. See `related/stress-shadows/functions/compute_laplacian.m` for the Matlab reference implementation.

---

## Notes & Open Questions

- **Coordinate conventions**: Public API uses East/North/Up and lat/lon consistently. Green's functions handle internal conventions (Okada x=strike) behind the interface.
- **Performance**: For large inversions (thousands of patches, millions of InSAR pixels), the hash-based caching system (Phase 3.4) handles reuse of expensive G and L matrices. For truly massive problems, consider lazy G matrix evaluation or HDF5/memmap storage as a future extension.
- **Backslip model**: Careful sign conventions needed. Matlab logic in `Backslip_Translation_Source.m`.
- **Stress kernels**: Implemented in `fault.stress_kernel()`. Uses okada92/DC3D to evaluate strain at patch centroid depths (not the surface). Cached via `geodef.cache`.
- **Mesh generation**: `slabMesh` depends on `meshpy`/`triangle`. Keep as optional dependency.
- **Name**: The package is called `geodef` (geodetic deformation). Short, unique, pip-installable.
