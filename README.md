# GeoDef

A flexible, student-friendly Python library for forward and inverse modeling of fault slip in elastic half-spaces. Targets both coseismic (earthquake) and interseismic (locked fault / coupling) applications.

## Current Status

**352 tests passing** across 11 test files.

- **Phase 1 (Green's functions)** -- complete
- **Phase 2 (Package structure)** -- complete
- **Phase 3 (Fault + Data + Greens)** -- complete
- **Phase 4 (Inversion)** -- next

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

Slip columns are interleaved as `[ss_0, ds_0, ss_1, ds_1, ...]`. Each `DataSet` subclass defines how raw displacement/strain components are projected into its observation space (e.g., LOS projection for InSAR, interleaved E/N/U for GNSS).

## Examples

See `examples/01_forward_model.ipynb` for a worked demo covering:
- Creating a discretized fault
- Defining synthetic GNSS stations
- Building the Green's matrix
- Predicting displacements from input slip
- Joint Green's matrix for GNSS + InSAR
- Visualizing slip distribution and predicted displacements

## Planned Features

- **Inversion** -- regularized least-squares with automatic hyperparameter tuning (ABIC, cross-validation)
- **Uncertainty** -- model covariance, resolution matrices, fit statistics
- **Triangular faults** -- slab2.0 mesh generation
- **Tutorial notebooks** -- progressive series from basic statistics to full geodetic inversion

## Testing

```bash
uv run pytest
```

## References

- Okada, Y., 1985. Surface deformation due to shear and tensile faults in a half-space. *Bull. Seismol. Soc. Am.*, 75(4), 1135--1154.
- Okada, Y., 1992. Internal deformation due to shear and tensile faults in a half-space. *Bull. Seismol. Soc. Am.*, 82(2), 1018--1040.
- Nikkhoo, M. & Walter, T.R., 2015. Triangular dislocation: an analytical, artefact-free solution. *Geophys. J. Int.*, 201(2), 1119--1141.
- Lindsey, E.O. et al., 2021. Slip rate deficit and earthquake potential on shallow megathrusts. *Nature Geoscience*, 14, 801--807.
