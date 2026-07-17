# `geodef.fault` — Fault class

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

The `Fault` class holds an immutable collection of fault patches (rectangular or triangular) and provides methods for forward modeling and I/O. Always create via factory classmethods, not `__init__` directly.

## What a discretized fault represents

Each patch carries a spatially uniform strike-slip and/or dip-slip value. The
continuous fault is therefore approximated by a finite basis: smaller patches
can represent finer structure but introduce more unknowns and generally demand
stronger data coverage or regularization. Patch size is a modeling choice, not
the resolution of the resulting inversion; use model-resolution and synthetic
recovery tests to learn what the data actually resolve.

Strike is clockwise from north, dip is downward from horizontal, and rake is
measured within the fault plane. GeoDef follows the local frame East, North,
Up and uses depth positive downward.

---

## Factory classmethods

### `Fault.planar(*, lat, lon, depth, strike, dip, length, width, n_length=1, n_width=1, medium=None, frame=None)`

Create a discretized planar fault directly from named geographic parameters:

```python
fault = Fault.planar(
    lat=0.0, lon=100.0, depth=30_000.0,
    strike=90.0, dip=15.0,
    length=100_000.0, width=50_000.0,
    n_length=10, n_width=5,
)
# → Fault with 50 rectangular patches, engine='okada'
```

Geographic arguments are keyword-only so latitude/longitude order cannot be
confused. Pass an explicit frame when the fault must share local coordinates
with another object:

```python
frame = geodef.LocalFrame(-2.0, 100.0)
fault = Fault.planar(..., frame=frame)
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

### `Fault.from_triangles(vertices, *, frame=None, ref_lat=None, ref_lon=None, triangles=None)`

Create a triangular fault from ENU arrays in either of two forms:

- Explicit corners: `vertices` has shape `(N, 3, 3)` (leave `triangles=None`).
- Node array + connectivity: pass a shared `(M, 3)` node array as `vertices`
  plus an `(N, 3)` index array as `triangles`. This preserves the exact patch
  order and node sharing of an imported mesh.

```python
fault = Fault.from_triangles(
    nodes, frame=frame, triangles=tris
)
```

### `Fault.load(fname, *, format=None, ref_lat=0.0, ref_lon=0.0)`

Load from a text file. Supported formats:

| `format` | Description |
|----------|-------------|
| `"center"` (default) | Whitespace-delimited: `id dipid strikeid lon lat depth L W strike dip` |
| `"topleft"` | Same columns but position is the top-left corner |
| `"seg"` | Unicycle segment format (local Cartesian; requires `ref_lat`/`ref_lon`) |
| `"ned"` | Unicycle `.ned` + `.tri` triangular mesh pair |

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
| `frame` | `LocalFrame` | Frame defining every local-coordinate view and triangular vertex |
| `grid_shape` | `(nL, nW)` or `None` | Structured grid dimensions |
| `centers_geo` | `(N, 3)` | Patch centers as `[lon, lat, depth_m]` (documented geographic order, matches `Mesh.centers_geo`) |
| `centers_local` | `(N, 3)` | Patch centers as `[east_m, north_m, up_m]` (lazy, cached) |
| `strike` | `(N,)` | Patch strike angles in degrees clockwise from north |
| `dip` | `(N,)` | Patch dip angles in degrees from horizontal |
| `areas` | `(N,)` | Patch areas in m² |
| `laplacian` | `(N, N)` | Finite-difference (structured) or KNN Laplacian (unstructured); lazy, cached |
| `vertices_2d` | `(N, 4, 2)` | Rectangular patch corners as `[lon, lat]` |
| `vertices_3d` | `(N, 4, 3)` | Rectangular patch corners as `[lon, lat, depth_km]` |
| `patch_outlines` | `(N, 5, 2)` | Closed rectangular patch outlines as `[lon, lat]` |

All geometry arrays are read-only after construction.

Use `fault.to_frame(target_frame)` to explicitly re-express local views. For
triangular faults it transforms every vertex while preserving its geographic
position; incompatible frames are never silently substituted.

---

## Forward modeling

### `fault.displacement(obs_lat, obs_lon, slip_strike, slip_dip=0.0)`

Compute surface displacements from strike-slip and dip-slip scalars or arrays.

```python
strike_slip = np.zeros(fault.n_patches)
dip_slip = np.ones(fault.n_patches)
east, north, up = fault.displacement(
    obs_lat,
    obs_lon,
    slip_strike=strike_slip,
    slip_dip=dip_slip,
)
```

### `fault.greens_matrix(obs_lat, obs_lon, kind="displacement", obs_depth=None)`

Build the raw Green's matrix.

```python
G = fault.greens_matrix(obs_lat, obs_lon)
# shape (3*n_obs, 2*N) for displacement
# Columns [:N] = strike-slip, [N:] = dip-slip
```

`kind='strain'` returns shape `(4*n_obs, 2*N)` for rectangular faults and
`(6*n_obs, 2*N)` for triangular faults. Pass `obs_depth` in meters, positive
down, for internal strain points; otherwise observations are at the surface.

---

## Moment and magnitude

Scalar seismic moment is `M0 = mu * sum(area_i * slip_i)`. It measures source
size, whereas moment magnitude is a logarithmic rescaling; neither says how
well the spatial slip distribution is resolved. `slip` here is slip magnitude,
not a signed strike- or dip-slip component.

```python
slip_magnitude = np.hypot(strike_slip, dip_slip)
M0 = fault.moment(slip_magnitude, mu=30e9)     # returns N·m
Mw = fault.magnitude(slip_magnitude, mu=30e9) # moment magnitude

# Module-level utilities
from geodef import moment_to_magnitude, magnitude_to_moment
Mw = moment_to_magnitude(1e20)  # → 6.60
M0 = magnitude_to_moment(7.0)   # → 1.41e19
```

---

## Stress kernel

```python
K = fault.stress_kernel(mu=30e9)
```

Strain Green's functions evaluated at the fault's own patch centers, scaled by
shear modulus. Rectangular faults return shape `(4*N, 2*N)` and triangular
faults return shape `(6*N, 2*N)`.

This is a discretized elastic interaction kernel, not a complete earthquake
failure model. Stress near patch edges and self-interaction depend on
discretization; Coulomb stress additionally requires a receiver orientation,
friction convention, and normal-stress sign convention.

---

## Grid lookup

```python
idx = fault.patch_index(strike_idx=3, dip_idx=1)
# Only valid for structured grids (Fault.planar or Fault.load with grid)

grid = fault.reshape_patches(values)  # (N, ...) -> (n_width, n_length, ...)
values = fault.flatten_patches(grid)  # inverse conversion
```

---

## Saving

```python
fault.save("output.txt", format="center")
fault.save("output.seg", format="seg", ref_lat=0.0, ref_lon=100.0)
fault.save("triangular_fault", format="ned")  # writes .ned + .tri
```

---

## Coordinate conventions

- Geographic: latitude, longitude, depth in meters (positive down).
- Local Cartesian: East, North, Up in meters.
- The `.seg` format uses local Cartesian (North, East, Depth) with a user-supplied `ref_lat`/`ref_lon`.
