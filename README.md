# GeoDef

A flexible, student-friendly Python library for forward and inverse modeling of fault slip in elastic half-spaces. Targets both coseismic (earthquake) and interseismic (locked fault / coupling) applications.

## Current Status

**598 tests passing** across 14 test files.

- **Phase 1 (Green's functions)** -- complete
- **Phase 2 (Package structure)** -- complete
- **Phase 3 (Fault + Data + Greens + Cache)** -- complete
- **Phase 4 (Inversion)** -- complete
- **Phase 5 (Uncertainty & model assessment)** -- complete
- **Phase 6 (Visualization)** -- complete

See `PLAN.md` for the full development roadmap.

## Package Modules

| Module | What it provides |
|--------|-----------------|
| `okada85` | Surface displacements, tilts, strains (Okada 1985) |
| `okada92` | Internal deformation at depth (Okada 1992 / DC3D) |
| `tri` | Triangular dislocation displacements and strains (Nikkhoo & Walter 2015) |
| `okada` | Unified dispatcher: auto-selects okada85 (z=0) or okada92 (z<0) |
| `greens` | Green's matrix assembly, projection, stacking, Laplacian operators (structured + KNN) |
| `fault` | `Fault` class: fault creation, forward modeling, vertices, moment, file I/O |
| `data` | `DataSet` base class + `GNSS`, `InSAR`, `Vertical` data types |
| `invert` | Inversion: solvers, regularization, hyperparameter tuning, model assessment |
| `plot` | Visualization: slip, vectors, InSAR, fit, fault geometry, map, resolution, uncertainty |
| `cache` | Hash-based disk caching for Green's matrices and stress kernels |
| `transforms` | Geodetic transforms: ECEF, ENU, geodetic, Vincenty, haversine |
| `mesh` | Triangular mesh generation from slab2.0 NetCDF grids (optional deps) |

Green's function engines are cross-validated against each other (okada85 vs okada92 at the surface, triangular pairs vs rectangles, etc.).

## Installation

```bash
uv pip install -e .
```

## Quick Start

```python
import numpy as np
from geodef import Fault
```

### Creating a Fault

The `Fault` class uses factory classmethods -- you don't call `__init__` directly.

**From center parameters:**

```python
# 100 km x 50 km fault, 30 km deep, dipping 15 degrees, discretized 10x5
fault = Fault.planar(
    lat=0.0, lon=100.0, depth=30000.0,
    strike=90.0, dip=15.0,
    length=100_000.0, width=50_000.0,
    n_length=10, n_width=5,
)
# Fault(n_patches=50, engine='okada', grid=(10, 5))
```

**From top-left corner:**

```python
fault = Fault.planar_from_corner(
    lat=0.0, lon=100.0, depth=0.0,
    strike=90.0, dip=15.0,
    length=100_000.0, width=50_000.0,
    n_length=10, n_width=5,
)
```

**From a file:**

```python
# Center-format text file (default)
fault = Fault.load("fault_model.txt")

# Top-left corner format
fault = Fault.load("fault_model.txt", format="topleft")

# Unicycle .seg format (local Cartesian -- needs a geographic reference point)
fault = Fault.load("ramp.seg", format="seg", ref_lat=0.0, ref_lon=100.0)
```

### Fault Properties

```python
fault.n_patches      # int: number of patches
fault.centers        # (N, 3): [lat, lon, depth] per patch
fault.centers_local  # (N, 3): [east, north, up] in meters (lazy, cached)
fault.areas          # (N,): patch areas in m^2
fault.engine         # "okada" or "tri"
fault.grid_shape     # (n_length, n_width) or None
fault.laplacian      # (N, N): finite-difference smoothing operator (lazy, cached)
fault.vertices_2d    # (N, 4, 2): patch corners as [lon, lat]
fault.vertices_3d    # (N, 4, 3): patch corners as [lon, lat, depth_km]
```

All geometry arrays are immutable (read-only) after construction.

### Forward Modeling

Compute surface displacements from a slip distribution:

```python
# Observation points
obs_lat = np.array([0.1, 0.2, 0.3])
obs_lon = np.array([100.0, 100.1, 100.2])

# Uniform 1-meter dip slip on all patches
ue, un, uz = fault.displacement(obs_lat, obs_lon, slip_strike=0.0, slip_dip=1.0)
```

Slip is passed as an argument, not stored as state. The `displacement()` method builds the Green's matrix `G` and computes `G @ m` in one call. For more control, build `G` directly:

```python
# Green's matrix: shape (3*M, 2*N) for displacement, (4*M, 2*N) for strain
G = fault.greens_matrix(obs_lat, obs_lon, kind="displacement")
```

The slip vector `m` is interleaved as `[ss0, ds0, ss1, ds1, ...]` where `ss` and `ds` are the strike-slip and dip-slip components for each patch.

### Moment and Magnitude

```python
from geodef import moment_to_magnitude, magnitude_to_moment

slip = np.ones(fault.n_patches)  # 1 meter on every patch
M0 = fault.moment(slip, mu=30e9)       # seismic moment in N-m
Mw = fault.magnitude(slip, mu=30e9)    # moment magnitude

# Module-level utilities
Mw = moment_to_magnitude(1e20)    # 6.60
M0 = magnitude_to_moment(7.0)     # 1.41e19
```

### Stress Kernel

Compute the self-stress interaction kernel (strain Green's functions evaluated at the fault's own patch centers, scaled by shear modulus):

```python
K = fault.stress_kernel(mu=30e9)  # shape (4*N, 2*N)
```

### Laplacian (Smoothing Operator)

A discrete Laplacian is available for regularized inversions, and works for both structured and unstructured faults:

```python
L = fault.laplacian  # shape (N, N), cached after first access
```

For structured rectangular grids (created via `Fault.planar()`), this uses finite-difference stencils. For unstructured meshes (loaded from `.seg` files with geometric sizing, or triangular meshes), it uses a distance-weighted K-nearest-neighbors graph Laplacian (`k=6`). The KNN Laplacian assigns inverse-distance weights to the nearest neighbors and symmetrizes the graph, ensuring constants are in the nullspace.

The underlying functions are also available directly:

```python
from geodef.greens import build_laplacian_2d, build_laplacian_knn

# Structured grid
L = build_laplacian_2d(n_length=10, n_width=5)

# Unstructured mesh (returns a sparse matrix)
L = build_laplacian_knn(fault.centers_local, k=6)
```

### Grid Index Lookup

For structured grids, convert (strike, dip) indices to flat patch index:

```python
idx = fault.patch_index(strike_idx=3, dip_idx=1)
```

### Saving Faults

```python
# Center-format text file
fault.save("output.txt", format="center")

# Unicycle .seg format
fault.save("output.seg", format="seg", ref_lat=0.0, ref_lon=100.0)
```

## The .seg Format

GeoDef reads and writes the unicycle `.seg` format, which defines fault segments that are automatically subdivided into patches. Each segment line specifies:

| Field | Description |
|-------|-------------|
| `n` | Segment number |
| `Vpl` | Plate velocity |
| `x1, x2, x3` | Origin (North, East, Depth) in meters |
| `Length, Width` | Total segment dimensions in meters |
| `Strike, Dip, Rake` | Orientation in degrees |
| `L0, W0` | Initial patch size |
| `qL, qW` | Geometric growth factors (1.0 = uniform) |

With `qW > 1`, patch widths grow geometrically with depth, allowing coarser discretization at depth where resolution decreases. This is a port of unicycle's `flt2flt.m` algorithm.

## Coordinate Conventions

- Geographic coordinates: latitude, longitude, depth (meters, positive down)
- Local Cartesian: East, North, Up (meters) -- used internally for Green's functions
- Green's functions use Okada conventions internally (x=strike, y=updip) but convert at the interface
- The `.seg` format uses local Cartesian (North, East, Depth) with a user-supplied `ref_lat`/`ref_lon` for geographic placement

## Data Types

GeoDef provides three geodetic data classes, all inheriting from a common `DataSet` base:

```python
from geodef import GNSS, InSAR, Vertical

# Three-component GNSS (or horizontal-only with vu=None, su=None)
gnss = GNSS(lat, lon, ve, vn, vu, se, sn, su)
gnss = GNSS.load("stations.dat")                      # full 3-component
gnss = GNSS.load("stations.dat", components="en")     # horizontal only

# InSAR line-of-sight with look vectors
insar = InSAR(lat, lon, los, sigma, look_e, look_n, look_u)
insar = InSAR.load("ascending.dat")

# Single-component vertical (e.g. coral uplift, tide gauge)
vertical = Vertical(lat, lon, displacement, sigma)
vertical = Vertical.load("coral.dat")
```

Each data type provides:
- `data.obs` -- observation vector
- `data.sigma` -- 1-sigma uncertainties
- `data.covariance` -- full covariance matrix (diagonal from sigma by default)
- `data.project(ue, un, uz)` -- maps displacement components to observation space

## Green's Matrix Assembly

The `greens` module assembles the full Green's matrix for any combination of fault engine and data type:

```python
import geodef

# Single dataset
G = geodef.greens.greens(fault, gnss)

# Joint inversion: vertically stacks projected G blocks
G = geodef.greens.greens(fault, [gnss, insar])

# Matching observation and weight vectors
d = geodef.stack_obs([gnss, insar])
W = geodef.stack_weights([gnss, insar])
```

Slip columns are blocked: `[:N]` are strike-slip, `[N:]` are dip-slip. Each `DataSet` subclass defines how raw displacement/strain components are projected into its observation space (e.g., LOS projection for InSAR, interleaved E/N/U for GNSS).

## Inversion

`geodef.invert()` solves `d = Gm` for fault slip with one call:

```python
import geodef

# Simplest call: unregularized weighted least-squares
result = geodef.invert(fault, [gnss, insar])

# Laplacian smoothing with non-negative bounds
result = geodef.invert(fault, [gnss, insar],
                       smoothing='laplacian',
                       smoothing_strength=1e3,
                       bounds=(0, None))

result.slip          # (N, 2): [strike-slip, dip-slip] per patch
result.slip_vector   # (2N,): blocked solution vector
result.residuals     # (M,): observation minus prediction
result.predicted     # (M,): forward-modeled observations
result.chi2          # reduced chi-squared
result.rms           # RMS misfit
result.moment        # scalar seismic moment (N-m)
result.Mw            # moment magnitude
```

### Solvers

| `method` | Description | When to use |
|----------|-------------|------------|
| `'wls'` (default) | Weighted least-squares | Fast, no constraints |
| `'nnls'` | Non-negative least-squares | Non-negative slip only |
| `'bounded_ls'` | Bounded least-squares | Box constraints on slip |
| `'constrained'` | Quadratic programming (SLSQP) | Inequality constraints |

Auto-selection: `bounds=None` -> WLS, `bounds=(0, None)` -> NNLS, general bounds -> bounded_ls.

The `'constrained'` solver also accepts linear inequality constraints via `constraints=(C, d_ineq)`, enforcing `C @ m <= d_ineq` (e.g., stress positivity constraints).

### Regularization

Regularization is controlled by two parameters: **what matrix** (`smoothing`) and **how strongly** (`smoothing_strength`):

```python
# Laplacian smoothing (finite-difference or KNN, depending on fault type)
result = geodef.invert(fault, data, smoothing='laplacian', smoothing_strength=1e3)

# Stress-kernel regularization, non-negative
result = geodef.invert(fault, data, smoothing='stresskernel', smoothing_strength=1e4,
                       bounds=(0, None))

# Tikhonov / L2 damping
result = geodef.invert(fault, data, smoothing='damping', smoothing_strength=1.0)

# Custom regularization matrix
result = geodef.invert(fault, data, smoothing=my_matrix, smoothing_strength=1e3)

# Regularize toward a reference model (e.g., plate rate for coupling inversions)
result = geodef.invert(fault, data, smoothing='damping', smoothing_strength=1e3,
                       smoothing_target=m_ref)
```

### Component Selection

Solve for both slip components, or just one:

```python
result = geodef.invert(fault, data, components='both')     # strike + dip (default)
result = geodef.invert(fault, data, components='strike')   # strike-slip only
result = geodef.invert(fault, data, components='dip')      # dip-slip only
```

### Automatic Hyperparameter Tuning

The regularization weight can be selected automatically:

```python
# ABIC criterion (Fukuda & Johnson 2008)
result = geodef.invert(fault, data, smoothing='laplacian', smoothing_strength='abic')

# K-fold cross-validation
result = geodef.invert(fault, data, smoothing='laplacian',
                       smoothing_strength='cv', cv_folds=5)
```

### Exploring the Smoothing Parameter

Two exploration tools let you visualize how the smoothing parameter affects the inversion:

**L-curve** -- trade-off between data misfit and model roughness:

```python
lc = geodef.lcurve(fault, data, smoothing='laplacian',
                   smoothing_range=(1e-2, 1e6), n=50)
lc.plot()               # log-log misfit vs model norm, optimal marked
lc.optimal              # lambda at maximum curvature (the "corner")
lc.smoothing_values     # (50,) lambda array
lc.misfits              # (50,) data misfit norms
lc.model_norms          # (50,) regularized model norms
```

**ABIC curve** -- information criterion as a function of lambda:

```python
ac = geodef.abic_curve(fault, data, smoothing='laplacian',
                       smoothing_range=(1e-2, 1e8), n=50)
ac.plot()               # semilog ABIC vs lambda, optimal marked
ac.optimal              # lambda at minimum ABIC
ac.abic_values          # (50,) ABIC at each lambda
ac.misfits              # (50,) data misfit norms
ac.model_norms          # (50,) regularized model norms
```

Both return result objects with a `.plot()` method (accepts `ax=None`, returns `ax`) and an `.optimal` attribute for the recommended lambda. The optimal value is annotated on the plot by default.

### Model Assessment

These functions are computed on demand (not during `invert()`) since they require forming and inverting dense matrices.

**Per-dataset diagnostics** -- when inverting multiple datasets jointly, it is useful to evaluate how well each dataset is fit individually. This requires the hat matrix `H`, which describes how much influence each observation has on the estimated model. The hat matrix is defined as `H = G_w (G_w^T G_w + lambda L^T L)^{-1} G_w^T`, where `G_w` is the data-weighted Green's matrix. The diagonal entries of `H` (called *leverage*) measure how many effective model parameters are "used" by each observation. Summing the leverage over a dataset's observations gives the effective number of parameters consumed by that dataset, which determines its effective degrees of freedom (`dof = n_obs - leverage`) and hence its reduced chi-squared.

```python
diags = geodef.dataset_diagnostics(result, fault, [gnss, insar])
for i, d in enumerate(diags):
    print(f"Dataset {i}: chi2={d.chi2:.1f}, reduced_chi2={d.reduced_chi2:.2f}, "
          f"wrms={d.wrms:.4f}, n_obs={d.n_obs}, dof={d.dof:.1f}")
```

**Model covariance, resolution, and uncertainty:**

```python
# Model covariance: Cm = H_inv @ G^T W G @ H_inv  (regularized)
Cm = geodef.model_covariance(result, fault, data)

# Resolution matrix: R = (G^T W G + lambda L^T L)^{-1} G^T W G
# R = I for perfect resolution; diag(R) < 1 where regularization dominates
R = geodef.model_resolution(result, fault, data)

# Per-parameter 1-sigma uncertainty: sqrt(diag(Cm))
unc = geodef.model_uncertainty(result, fault, data)
```

## Visualization

`geodef.plot` provides publication-ready plots with sensible defaults and full customizability. Every function accepts `ax=None` (creates a figure) or an existing axes, returns `ax`, and never calls `plt.show()`.

### Slip Distribution

```python
# One line for a nice plot
geodef.plot.slip(fault, result.slip_vector)

# Full control: component, colormap, limits, patch styling
geodef.plot.slip(fault, result.slip_vector,
                 component='dip', cmap='RdBu_r', vmin=-2, vmax=2,
                 edgecolor='gray', colorbar_label='Dip slip (m)')
```

### GNSS Vectors

```python
# Observed vs predicted with scale arrow legend
geodef.plot.vectors(gnss, fault,
                    predicted=result.predicted[:gnss.n_obs],
                    scale=10, legend=True,
                    scale_arrow=0.5, scale_arrow_label="50 cm observed")

# Vertical component as color-coded circles
geodef.plot.vectors(gnss, fault, components='vertical', scale=5)
```

### InSAR

```python
# Three-panel layout: observed, predicted, residual
geodef.plot.insar(insar, fault, predicted=pred_insar, layout='obs_pred_res')
```

### Map View

Patches can be colored by a scalar array or slip vector:

```python
geodef.plot.map(fault, datasets=[gnss, insar],
                slip_vector=result.slip_vector, cmap='YlOrRd',
                colorbar_label='Slip (m)',
                show_trace=True, trace_kwargs={'color': 'red'})
```

### Composing Plots

Since every function accepts `ax`, you can layer plots freely:

```python
fig, ax = plt.subplots()
geodef.plot.slip(fault, result.slip_vector, ax=ax, cmap='YlOrRd')
geodef.plot.vectors(gnss, fault, ax=ax, scale=10, legend=True)
```

### Other Plot Types

```python
geodef.plot.patches(fault, values, ...)       # generic per-patch scalar
geodef.plot.fit(obs, pred, ...)               # obs vs pred scatter or residual histogram
geodef.plot.fault3d(fault, color_by='depth')  # 3D fault geometry
geodef.plot.resolution(fault, R_diag, ...)    # model resolution on patches
geodef.plot.uncertainty(fault, sigma, ...)    # model uncertainty on patches
```

L-curve and ABIC curve results also have `.plot()` methods that follow the same pattern:

```python
lc = geodef.lcurve(fault, data, smoothing='laplacian', smoothing_range=(1e-2, 1e6))
lc.plot()  # optimal lambda annotated automatically
```

See `examples/03_plotting.ipynb` for a full demo of every plot type.

## Caching

Green's matrices and stress kernels are automatically cached to disk for fast reuse:

```python
geodef.cache.set_dir("my_cache/")   # default: .geodef_cache/
geodef.cache.enable()                # on by default
geodef.cache.info()                  # {"n_files": 3, "total_bytes": 12345}
geodef.cache.clear()                 # remove all cached files
```

## Examples

| Notebook | What it covers |
|----------|---------------|
| `examples/01_forward_model.ipynb` | Fault creation, GNSS stations, Green's matrix, forward prediction, joint GNSS + InSAR |
| `examples/02_caching.ipynb` | Green's matrix and stress kernel caching for fast reuse |
| `examples/03_plotting.ipynb` | All plot types: slip, vectors, InSAR, fit, fault3d, map, resolution, uncertainty, L-curve/ABIC, composing plots |

## Planned Features

- **Mesh generation** -- slab2.0 triangular mesh creation (`geodef.mesh`)
- **I/O additions** -- GMT export, extended data/fault formats
- **Tutorial notebooks** -- progressive series from basic statistics to full geodetic inversion
- **Geographic projection** -- cartopy-based plotting with coastlines and topography
- **Earthquake cycle modeling** -- rate-and-state friction, quasi-dynamic simulations

## Testing

```bash
uv run pytest
```

## References

- Okada, Y., 1985. Surface deformation due to shear and tensile faults in a half-space. *Bull. Seismol. Soc. Am.*, 75(4), 1135--1154.
- Okada, Y., 1992. Internal deformation due to shear and tensile faults in a half-space. *Bull. Seismol. Soc. Am.*, 82(2), 1018--1040.
- Nikkhoo, M. & Walter, T.R., 2015. Triangular dislocation: an analytical, artefact-free solution. *Geophys. J. Int.*, 201(2), 1119--1141.
- Fukuda, J. & Johnson, K.M., 2008. A fully Bayesian inversion for spatial distribution of fault slip with objective smoothing. *Bull. Seismol. Soc. Am.*, 98(3), 1128--1146.
- Lindsey, E.O. et al., 2021. Slip rate deficit and earthquake potential on shallow megathrusts. *Nature Geoscience*, 14, 801--807.
