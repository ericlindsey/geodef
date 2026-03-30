# PLAN.md - Development Plan for GeoDef

## Goal

Build **GeoDef**: a flexible, student-friendly Python library for forward and inverse modeling of fault slip in elastic half-spaces. Consolidate existing Matlab and Python code into a single well-tested package that helps students get started quickly while remaining capable for research.

**Important: Read `PYTHON.md` before editing any code.**

---

## Completed Phases (1-7)

Phases 1-7 are complete. See `CLAUDE.md` for the full module inventory, test counts, and architectural details.

| Phase | What was built | Tests |
|-------|---------------|-------|
| 1. Green's Functions | `okada85`, `okada92`, `tri` — verified against Matlab references, cross-validated | 113 |
| 2. Library Structure | Package layout, `okada` dispatcher, tooling, code migration | 25 |
| 3. Fault & Data | `Fault` class, `DataSet`/`GNSS`/`InSAR`/`Vertical`, `greens` assembly, `cache` | 167 |
| 4. Inverse Framework | WLS/NNLS/bounded LS/constrained QP, Laplacian/damping/stress-kernel regularization, ABIC/CV/L-curve tuning | 126 |
| 5. Uncertainty | Model covariance/resolution/uncertainty, per-dataset diagnostics, moment/magnitude | 47 |
| 6. Visualization | `plot` module: slip, vectors, InSAR, fit, fault3d, map, resolution, uncertainty; L-curve/ABIC refactor | 120 |
| 7. Mesh Generation | `mesh` module: `Mesh` dataclass, `from_trace`, `from_polygon`, `from_points`, `from_slab2`; `Fault.from_mesh()`, `Fault.from_triangles()` | 70 |

**Total: 669 tests passing.**

Key design decisions preserved from earlier phases:
- **Flat module structure** — `geodef.okada`, `geodef.fault`, `geodef.invert` (one file, one concept)
- **Transparent Okada85/92 dispatch** — auto-selects based on observation depth
- **Progressive complexity** — quick-start (Fault methods) → modular (compose modules) → low-level (raw Green's functions)
- **Polymorphic data dispatch** — `DataSet.greens_type` + `DataSet.project()` enables new data types with zero changes elsewhere
- **Blocked column layout** — `[ss_0..ss_N, ds_0..ds_N]` for Green's matrices and slip vectors

---

## Phase 6: Visualization (`geodef.plot`) — COMPLETE

A plotting module with sensible defaults that remain fully customizable. All plotting functions pass `**kwargs` through to underlying matplotlib calls, so any artist property can be overridden without the library getting in the way.

### 6.1 Core Design Principles

- **Defaults that look good out of the box** — colorbar, labels, title, appropriate colormap
- **Full customizability via kwargs** — any matplotlib artist property can be overridden
- **Axes-first pattern** — every function accepts an optional `ax` parameter; if `None`, creates a new figure/axes. Returns `ax` so users can continue customizing
- **No hidden state** — functions never call `plt.show()`; users control display
- **All functions return `ax`** — simple, consistent, composable. Users who need specific artists (colorbar, quiver, etc.) can grab them from the axes or capture them before calling the plot function
- **Local Cartesian coordinates** — all plots use x (East, km) / y (North, km) by default. Geographic projection (lon/lat with cartopy) is deferred to Phase 9

### 6.2 Slip Distribution Plot (`plot.slip`)

Plot fault slip as colored patches. Works for both rectangular (`PatchCollection`) and triangular (`PolyCollection`) meshes with a unified interface.

```python
ax = geodef.plot.slip(fault, result.slip)

ax = geodef.plot.slip(fault, result.slip,
    ax=ax,                          # plot on existing axes
    component='dip',                # 'strike', 'dip', 'magnitude' (default)
    cmap='RdBu_r',                  # any matplotlib colormap
    vmin=-2, vmax=2,                # color limits
    edgecolor='gray', linewidth=0.5,# patch outline style (**kwargs → PatchCollection)
    colorbar=True,                  # default True
    colorbar_label='Slip (m)',      # auto-generated if None
    colorbar_kwargs={...},          # passed to fig.colorbar()
    title='Coseismic slip',
)
```

**Implementation tasks:**
1. Internal helper `_get_patch_vertices(fault)` → list of (4,2) or (3,2) arrays in km
2. Internal helper `_get_slip_component(slip, n_patches, component)` → 1D array
3. Build `PatchCollection` / `PolyCollection`, set array, apply kwargs
4. Colorbar creation with label and kwargs passthrough
5. Axis labels ("East (km)", "North (km)"), equal aspect ratio, optional title

### 6.3 Data Observation Plots

#### 6.3.1 GNSS Vectors (`plot.vectors`)

Plot GNSS displacement or velocity vectors as quiver arrows, with optional uncertainty ellipses and predicted vectors overlaid.

```python
ax = geodef.plot.vectors(gnss,
    predicted=result.predicted,     # optional overlay of model predictions
    scale=1e3,                      # quiver scale factor
    obs_color='black',
    pred_color='red',
    ellipses=True,                  # uncertainty ellipses from gnss.sigma
    ellipse_kwargs={'alpha': 0.3},  # → Ellipse artist kwargs
    components='horizontal',        # 'horizontal', 'vertical', 'both'
    legend=True,
    quiver_kwargs={},               # → ax.quiver() kwargs
)
```

**Implementation tasks:**
1. Extract x, y, components from GNSS dataset
2. Horizontal: `ax.quiver(x, y, ux, uy, **quiver_kwargs)` for obs and pred
3. Vertical: plot as colored circles (up=one color, down=another) or scaled symbols
4. Uncertainty ellipses from sigma values using `matplotlib.patches.Ellipse`
5. Legend with obs/pred distinction
6. Support `components='both'` with side-by-side or combined layout

#### 6.3.2 InSAR LOS (`plot.insar`)

Plot InSAR line-of-sight data as colored scatter points, with optional predicted and residual panels.

```python
ax = geodef.plot.insar(insar,
    predicted=result.predicted,
    layout='obs_pred_res',          # 'obs', 'pred', 'residual', 'obs_pred_res' (3 panels)
    cmap='RdBu_r',
    vmin=-0.1, vmax=0.1,
    scatter_kwargs={'s': 2},        # → ax.scatter() kwargs
    colorbar=True,
)
```

**Implementation tasks:**
1. Single-panel: `ax.scatter(x, y, c=data, **scatter_kwargs)`
2. Multi-panel layout with shared colorbar and clim
3. Residual computation when predicted is provided
4. Shared vmin/vmax across panels for consistency

#### 6.3.3 Observed vs. Predicted Fit (`plot.fit`)

Simple diagnostic scatter plot of observed vs. predicted values (1:1 line).

```python
ax = geodef.plot.fit(result, datasets,
    dataset_index=0,                # which dataset (or 'all')
    style='scatter',                # 'scatter' or 'residual_histogram'
)
```

### 6.4 Fault Geometry Visualization

#### 6.4.1 3D View (`plot.fault3d`)

3D visualization of fault mesh geometry, colored by depth, area, or a user-supplied array.

```python
ax = geodef.plot.fault3d(fault,
    color_by='depth',               # 'depth', 'area', 1D array, or None
    cmap='viridis',
    show_edges=True,
    station_locations=gnss,         # optional: overlay station positions
)
```

**Implementation tasks:**
1. Build `Poly3DCollection` from fault vertices (3D coords in km)
2. Color by depth/area/custom array using `set_array()` + `set_clim()`
3. Invert z-axis for depth convention (positive down)
4. Optional station overlay via `ax.scatter3D()`

#### 6.4.2 Map View (`plot.map`)

2D map view of fault patches and optional station locations.

```python
ax = geodef.plot.map(fault,
    datasets=None,                  # overlay dataset locations
    show_patches=True,
    show_trace=True,                # surface projection of top edge
    patch_kwargs={},                # → PatchCollection kwargs
    trace_kwargs={'color': 'red', 'linewidth': 2},
)
```

### 6.5 Diagnostic & Tuning Plots

#### 6.5.1 Resolution and Uncertainty on Fault (`plot.resolution`, `plot.uncertainty`)

These reuse the same `plot.slip` machinery but with different default colormaps and labels.

```python
ax = geodef.plot.resolution(fault, R_diag,
    cmap='viridis', vmin=0, vmax=1,
    colorbar_label='Resolution',
)

ax = geodef.plot.uncertainty(fault, sigma,
    cmap='magma_r',
    colorbar_label='1-sigma uncertainty (m)',
)
```

**Implementation:** thin wrappers around the core `_plot_patch_scalar()` helper (shared with `plot.slip`).

#### 6.5.2 Update Existing L-curve / ABIC Plots

Refactor `LCurveResult.plot()` and `ABICCurveResult.plot()` in `invert.py` for consistency:
- Accept `ax=None` parameter (create figure if None, plot on existing axes if provided)
- Accept `**kwargs` passed through to the line/marker artists
- Return `ax` instead of `fig`
- Keep them as methods on their result objects (not moved to `plot` module)

### 6.6 Module Structure

```
src/geodef/plot.py          # single module (flat, consistent with project structure)
tests/test_plot.py          # tests for all plot functions
```

Public API:
- `geodef.plot.slip(fault, slip, ...)`
- `geodef.plot.vectors(dataset, ...)`
- `geodef.plot.insar(dataset, ...)`
- `geodef.plot.fit(result, datasets, ...)`
- `geodef.plot.fault3d(fault, ...)`
- `geodef.plot.map(fault, ...)`
- `geodef.plot.resolution(fault, values, ...)`
- `geodef.plot.uncertainty(fault, values, ...)`

### 6.7 Implementation Order

1. ~~**6.7a** — Internal helpers: `_get_patch_vertices()`, `_get_slip_component()`, `_plot_patch_scalar()`, axes creation helper~~ **DONE**
2. ~~**6.7b** — `plot.slip()` + `plot.resolution()` + `plot.uncertainty()` (all use `_plot_patch_scalar`)~~ **DONE**
3. ~~**6.7c** — `plot.vectors()` (GNSS quiver + ellipses)~~ **DONE**
4. ~~**6.7d** — `plot.insar()` (scatter + multi-panel layout)~~ **DONE**
5. ~~**6.7e** — `plot.fault3d()` (3D Poly3DCollection)~~ **DONE**
6. ~~**6.7f** — `plot.map()` (2D overview with trace + stations)~~ **DONE**
7. ~~**6.7g** — `plot.fit()` (obs vs pred scatter / residual histogram)~~ **DONE**
8. ~~**6.7h** — Refactor `LCurveResult.plot()` and `ABICCurveResult.plot()` for consistency (add `ax`, `**kwargs`, return `ax`)~~ **DONE**

### 6.8 Testing Strategy

- Use `matplotlib.testing` or `pytest-mpl` for image comparison where valuable
- At minimum: every function returns an `Axes`, no exceptions raised for valid input
- Test with both rectangular and triangular faults
- Test with and without optional arguments (colorbar, predicted, etc.)
- Test multi-panel layouts produce correct number of axes
- Test kwargs passthrough (e.g., edgecolor actually applied)

---

## Phase 7: Mesh Generation (`geodef.mesh`)

Migrate and modernize the existing `mesh.py` (currently a raw port from slabMesh) into a proper module with clean API, tests, and integration with `Fault`.

### 7.1 Clean Up Existing Code

- Add type hints, docstrings, and proper error handling
- Replace `print()` statements with `logging` or exceptions
- Remove `scipy.io` (Matlab `.mat` export) — keep only Python-native formats
- Make `netCDF4` and `meshpy` optional imports with clear error messages

### 7.2 API Design

```python
mesh = geodef.mesh.from_slab2("cas_slab2_dep.grd",
    bounds=[lon_min, lon_max, lat_min, lat_max],
    max_area=0.03,                  # uniform refinement threshold
    depth_variable=True,            # finer near trench, coarser at depth
    depth_factor=0.00007,           # scaling factor for depth-based refinement
)

fault = geodef.Fault.from_mesh(mesh)
# or one-step:
fault = geodef.Fault.from_slab2("cas_slab2_dep.grd", bounds=[...])
```

### 7.3 Mesh Object

Return a lightweight `Mesh` dataclass or named tuple:
- `points` — (N, 3) vertices in geographic coordinates (lon, lat, depth)
- `triangles` — (M, 3) connectivity (indices into points)
- `save(fname, format='ned')` — write `.ned` + `.tri` files (unicycle format)

### 7.4 Tests

- Round-trip: generate mesh → save → load → compare
- Basic properties: all triangles have positive area, no degenerate elements
- Bounds cropping produces mesh within specified region
- Depth-variable refinement produces smaller elements near trench

---

## Phase 8: I/O Additions

Extend import/export capabilities after plotting and mesh are solid.

- `result.to_gmt()` — export slip patches as GMT-plottable polygons with values
- `geodef.GNSS.load()` / `.save()` — common GNSS velocity/displacement formats
- `geodef.InSAR.load()` / `.save()` — LOS + look vectors (e.g. GMTSAR/ISCE format)
- `geodef.Fault.load()` — extend with triangular mesh (`.ned` + `.tri`) format support

---

## Phase 9: Tutorial Notebooks

Progressive series rewritten to use `geodef`, ported from `related/shakeout_v2/notebooks/`. Placed in `tutorials/`.

### 9.1 Scaffolded Tutorials (based on shakeout notebooks)

| # | Title | Source | Key concepts |
|---|-------|--------|-------------|
| 01 | Forward Model Basics | nb 03 | `Fault.planar()`, `displacement()`, single-fault map-view plot |
| 02 | Fault Discretization & the G Matrix | nb 04 | Multi-patch fault, `greens()`, column layout, matrix visualization |
| 03 | Unregularized Inversion | nb 04 | `invert()` with WLS, interpreting `InversionResult`, overfitting demo |
| 04 | Regularization: Smoothing & Damping | nb 05 | Laplacian, damping, stress-kernel; effect on slip distribution |
| 05 | Choosing Regularization Strength | nb 06 | L-curve, ABIC, cross-validation; `lcurve()`, `abic_curve()` |
| 06 | Weighted Least Squares & Multiple Datasets | nb 07 | GNSS + InSAR joint inversion, covariance matrices, data weighting |
| 07 | Correlated Noise & InSAR | nb 07b | Full covariance for InSAR, effect on slip uncertainty |
| 08 | Bounds & Constraints | — | Non-negative slip (NNLS), bounded LS, inequality constraints |
| 09 | Uncertainty & Model Assessment | — | `model_covariance()`, `model_resolution()`, `model_uncertainty()`, diagnostics |
| 10 | Nonlinear Inversion (MCMC) | nb 08 | `scipy.optimize`, `emcee` for fault geometry parameters |

Each tutorial should:
- Build on the previous one (progressive complexity)
- Include clear markdown explanations of the math/concepts
- Use synthetic data so no external files needed
- Show plots inline with well-labeled axes
- End with exercises or extension questions for students

### 9.2 Worked Examples with Real Data

Full research-level examples in `examples/`, demonstrating real-world workflows. Each uses actual fault geometries and geodetic data.

| # | Title | Data source | Key features |
|---|-------|------------|-------------|
| 01 | Forward Model Demo | — | Already exists (`01_forward_model.ipynb`) |
| 02 | Caching Demo | — | Already exists (`02_caching.ipynb`) |
| 03 | Cascadia Interseismic Coupling | `stress-shadows/cascadia/` | Slab2.0 mesh, GNSS velocities, backslip inversion, smoothing_target |
| 04 | Japan Megathrust Coupling | `stress-shadows/japan/` | Slab2.0 mesh, GNSS data, stress-kernel regularization |
| 05 | Nepal Earthquake Coseismic Slip | `stress-shadows/example_nepal_earthquake/` | Coseismic GNSS + InSAR, triangular mesh, joint inversion |
| 06 | 2D Earthquake Example | `stress-shadows/example_2d_earthquake/` | Simple planar fault, GNSS-only, good for validation |
| 07 | 3D Earthquake Example | `stress-shadows/example_3d_earthquake/` | Multi-segment fault, 3D geometry |
| TBD | Additional earthquake examples | User to provide | New coseismic events with data |

---

## Phase 10: Future Extensions

### 10.1 Quasi-Dynamic Earthquake Cycle Modeling

Port the earthquake cycle simulation framework from `related/stress-shadows/unicycle/` (Barbot et al.). This would extend geodef from static slip inversions to time-dependent earthquake cycle modeling.

**Key components:**
- **Rate-and-state friction** — aging law and slip law formulations on fault patches
- **Stress interaction** — use existing `stress_kernel()` infrastructure to compute inter-patch stress transfer
- **ODE integration** — adaptive time-stepping (e.g. Runge-Kutta) for coupled evolution of slip velocity and state variable on each patch
- **Driving load** — tectonic backslip loading between earthquakes
- **Output** — time series of slip, slip rate, and stress on each patch; synthetic surface displacement time series

**Design sketch:**
```python
cycle = geodef.CycleModel(fault,
    friction='aging',           # 'aging' or 'slip' law
    a=0.01, b=0.015,           # rate-state parameters (arrays or scalars)
    dc=0.01,                   # critical slip distance
    sigma_n=50e6,              # effective normal stress
    v_plate=40e-3 / YEAR,      # plate convergence rate
)
history = cycle.run(t_max=1000 * YEAR, dt_max=YEAR/12)
history.slip       # (n_patches, n_timesteps)
history.velocity   # (n_patches, n_timesteps)
history.time       # (n_timesteps,)
```

### 10.2 Finite-Volume Strain Green's Functions

Extend the Green's function library beyond point-source (Okada/triangular dislocation) solutions to include volume-based strain sources. These are needed for modeling distributed deformation (e.g., volcanic inflation, post-seismic viscoelastic relaxation).

**Key components:**
- **Volume strain source** — Mogi point source, finite spherical/ellipsoidal cavity, arbitrary volume elements
- **Finite-volume Green's functions** — displacement and strain at the surface or at depth due to volumetric strain in a finite element
- **Integration with existing framework** — new `DataSet.greens_type = 'volume_strain'` for seamless use with `greens()` and `invert()`

### 10.3 Geographic Projection & Cartopy Plotting

Add optional geographic coordinate support to the plotting module:
- `projection='geographic'` option on all plot functions
- Cartopy-based coastlines, country borders, topography
- Automatic coordinate transform from local Cartesian to lon/lat
- Keep cartopy as an optional dependency with clear error messages

### 10.4 Interpolated Slip Visualization

Add smooth/contour-style slip rendering as an alternative to flat-colored patches:
- For rectangular meshes: `pcolormesh` or `contourf` on a regular grid
- For triangular meshes: `matplotlib.tri.Triangulation` with `tricontourf`
- Useful for publication-quality figures where patch boundaries are distracting

### 10.5 Euler Pole Fitting

Port `related/shakeout_v2/euler_calc.py` for fitting rigid plate rotation (Euler pole) to GNSS velocities. Useful as a preprocessing step to remove plate motion before interseismic analysis.

```python
pole = geodef.euler_pole(gnss, plate='fixed')  # fit Euler pole to GNSS velocities
gnss_residual = gnss.remove_euler(pole)         # remove rigid rotation
```

### 10.6 Moment Tensor Decomposition

Port `related/shakeout_v2/moment_tensor.py` for computing and decomposing moment tensors from slip on discretized faults.

### 10.7 MCMC / Bayesian Inversion Module

Structured interface for nonlinear Bayesian inversion of fault geometry parameters (location, strike, dip, rake) using MCMC sampling (e.g., `emcee`). Currently covered in tutorial 10 as a manual workflow; could become a first-class module.

### 10.8 Additional Green's Function Engines

- **Meade (2007)** — backslip triangular dislocations (already in unicycle)
- **Compound dislocation model (CDM)** — for volcanic sources
- **Layered elastic half-space** — for more realistic Earth structure (e.g., EDGRN/EDCMP)

---

## Notes & Open Questions

- **Coordinate conventions**: Public API uses East/North/Up and lat/lon consistently. Green's functions handle internal conventions (Okada x=strike) behind the interface.
- **Performance**: Hash-based caching handles reuse of expensive G and L matrices. For truly massive problems, consider lazy G matrix evaluation or HDF5/memmap storage.
- **Backslip model**: Careful sign conventions needed. Matlab logic in `Backslip_Translation_Source.m`.
- **Mesh generation**: Depends on `meshpy`/`triangle` and `netCDF4`. Keep as optional dependencies.
- **Patch ordering**: `Fault.planar()` orders deepest-first; `flt2flt` orders shallowest-first.
- **Projections**: `.seg` format uses flat-earth; stress-shadows uses polyconic. For large faults, a proper projection system should be added to `transforms.py`.
