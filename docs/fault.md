# `geodef.fault` — Fault class

The `Fault` class holds an immutable collection of fault patches (rectangular or triangular) and provides methods for forward modeling and I/O. Always create via factory classmethods, not `__init__` directly.

---

## Factory classmethods

### `Fault.planar(lat, lon, depth, strike, dip, length, width, n_length=1, n_width=1)`

Create a discretized planar fault from its centroid.

```python
fault = Fault.planar(
    lat=0.0, lon=100.0, depth=30_000.0,
    strike=90.0, dip=15.0,
    length=100_000.0, width=50_000.0,
    n_length=10, n_width=5,
)
# → Fault with 50 rectangular patches, engine='okada'
```

### `Fault.planar_from_corner(lat, lon, depth, strike, dip, length, width, n_length=1, n_width=1)`

Same as `planar()` but the reference point is the top-left (shallowest, along-strike start) corner instead of the centroid.

### `Fault.from_mesh(mesh)`

Create a triangular fault from a `Mesh` object.

```python
from geodef.mesh import from_slab2
mesh = from_slab2("sum_slab2_dep.grd", bounds=(95, 106, -6, 6))
fault = Fault.from_mesh(mesh)
```

### `Fault.from_triangles(vertices, ref_lat=0.0, ref_lon=0.0)`

Create a triangular fault directly from ENU vertex coordinates. `vertices` has shape `(N, 3, 3)`.

### `Fault.load(fname, format=None, ref_lat=0.0, ref_lon=0.0)`

Load from a text file. Supported formats:

| `format` | Description |
|----------|-------------|
| `"center"` (default) | Whitespace-delimited: `id dipid strikeid lon lat depth L W strike dip` |
| `"topleft"` | Same columns but position is the top-left corner |
| `"seg"` | Unicycle segment format (local Cartesian; requires `ref_lat`/`ref_lon`) |
| `"ned"` | Unicycle `.ned`+`.tri` triangular mesh pair |

```python
fault = Fault.load("fault_model.txt")
fault = Fault.load("ramp.seg", format="seg", ref_lat=0.0, ref_lon=100.0)
fault = Fault.load("cascadia", format="ned")  # reads cascadia.ned + cascadia.tri
```

---

## Properties

| Property | Shape | Description |
|----------|-------|-------------|
| `n_patches` | scalar | Number of patches |
| `engine` | `str` | `"okada"` or `"tri"` |
| `grid_shape` | `(nL, nW)` or `None` | Structured grid dimensions |
| `centers` | `(N, 3)` | Patch centers as `[lat, lon, depth_m]` |
| `centers_local` | `(N, 3)` | Patch centers as `[east_m, north_m, up_m]` (lazy, cached) |
| `areas` | `(N,)` | Patch areas in m² |
| `laplacian` | `(N, N)` | Finite-difference (structured) or KNN Laplacian (unstructured); lazy, cached |
| `vertices_2d` | list of `(nc, 2)` | Per-patch corners in local km `[east, north]` |
| `vertices_3d` | list of `(nc, 3)` | Per-patch corners in local km `[east, north, depth]` |

All geometry arrays are read-only after construction.

---

## Forward modeling

### `fault.displacement(obs_lat, obs_lon, slip_strike, slip_dip, nu=0.25)`

Compute surface displacements for a uniform slip distribution.

```python
ue, un, uz = fault.displacement(obs_lat, obs_lon, slip_strike=0.0, slip_dip=1.0)
# ue, un, uz each have shape (n_obs,)
```

### `fault.greens_matrix(obs_lat, obs_lon, kind="displacement", nu=0.25)`

Build the raw Green's matrix.

```python
G = fault.greens_matrix(obs_lat, obs_lon)
# shape (3*n_obs, 2*N) for displacement
# Columns [:N] = strike-slip, [N:] = dip-slip
```

`kind='strain'` returns shape `(4*n_obs, 2*N)` for rectangular, `(6*n_obs, 2*N)` for triangular.

---

## Moment and magnitude

```python
M0 = fault.moment(slip, mu=30e9)      # slip shape (N,) or (N,2); returns N·m
Mw = fault.magnitude(slip, mu=30e9)  # moment magnitude

# Module-level utilities
from geodef import moment_to_magnitude, magnitude_to_moment
Mw = moment_to_magnitude(1e20)  # → 6.60
M0 = magnitude_to_moment(7.0)   # → 1.41e19
```

---

## Stress kernel

```python
K = fault.stress_kernel(mu=30e9)  # shape (4*N, 2*N)
```

Strain Green's functions evaluated at the fault's own patch centers, scaled by shear modulus.

---

## Grid lookup

```python
idx = fault.patch_index(strike_idx=3, dip_idx=1)
# Only valid for structured grids (Fault.planar or Fault.load with grid)
```

---

## Saving

```python
fault.save("output.txt", format="center")
fault.save("output.seg", format="seg", ref_lat=0.0, ref_lon=100.0)
```

---

## Coordinate conventions

- Geographic: latitude, longitude, depth in meters (positive down).
- Local Cartesian: East, North, Up in meters.
- The `.seg` format uses local Cartesian (North, East, Depth) with a user-supplied `ref_lat`/`ref_lon`.
