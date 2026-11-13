# `geodef.fault` — Fault class

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

The `Fault` class holds an immutable collection of fault patches (rectangular or triangular) and provides methods for forward modeling and I/O. Create one with a factory classmethod; the keyword-only constructor `Fault(lon=, lat=, depth=, strike=, dip=, length=, width=)` (or `vertices=` for triangles) is for file readers and tests.

## What a discretized fault represents

Each patch carries a spatially uniform strike-slip and/or dip-slip value. The
continuous fault is therefore approximated by a finite basis: smaller patches
can represent finer structure but introduce more unknowns and generally demand
stronger data coverage or regularization. Patch size is a modeling choice, not
the resolution of the resulting inversion; use model-resolution and synthetic
recovery tests to learn what the data actually resolve.

Strike is clockwise from north, dip is downward from horizontal, and the fault
dips to the **right** of the strike direction. GeoDef follows the local frame
East, North, Up and uses depth positive downward.

---

## Factory classmethods

### `Fault.planar(*, lon, lat, depth, strike, dip, length, width, n_length=1, n_width=1, medium=None, frame=None)`

Create a discretized planar fault directly from named geographic parameters:

```python
fault = geodef.Fault.planar(
    lon=100.0, lat=0.0, depth=30_000.0,
    strike=90.0, dip=15.0,
    length=100_000.0, width=50_000.0,
    n_length=10, n_width=5,
)
# → Fault(n_patches=50, engine='okada', grid_shape=(10, 5))
```

`lon`, `lat`, and `depth` locate the **center** of the plane, so `depth` must be
at least `0.5 * width * sin(dip)` for the fault to stay below the surface.
`n_length` and `n_width` divide it into a grid whose first row is the
shallowest. Subdividing does not change the plane: uniform slip on a 10 x 5
grid produces the same displacement as on the single 100 km x 50 km patch.

All arguments are keyword-only so latitude/longitude order cannot be
confused. Pass an explicit frame when the fault must share local coordinates
with another object:

```python
frame = geodef.LocalFrame(lon=100.0, lat=-2.0)
fault = Fault.planar(..., frame=frame)
```

### `Fault.planar_from_corner(*, lon, lat, depth, strike, dip, length, width, n_length=1, n_width=1, medium=None)`

Same as `planar()` but the reference point is the top-left (shallowest, along-strike start) corner instead of the centroid.

### `Fault.from_mesh(mesh)`

Create a triangular fault from a `Mesh` object.

```python
from geodef.mesh import from_slab2
mesh = from_slab2("sum_slab2_dep.grd", bounds=(95, 106, -6, 6))
fault = Fault.from_mesh(mesh)
```

### `Fault.from_triangles(vertices, *, triangles=None, frame=None, frame=geodef.LocalFrame(lon=None, lat=None), medium=None)`

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

`frame` is the `LocalFrame` the vertices are expressed in, for example
`geodef.LocalFrame(lon=100.0, lat=-2.0)`.

### `Fault.load(fname, *, format=None, frame=None, medium=None)`

Load from a text file. Supported formats:

| `format` | Description |
|----------|-------------|
| `"center"` (default) | Whitespace-delimited: `id dipid strikeid lon lat depth L W strike dip` |
| `"topleft"` | Same columns but position is the top-left corner |
| `"seg"` | Unicycle segment format (local Cartesian; pass `frame=` to place it geographically) |
| `"ned"` | Unicycle `.ned` + `.tri` triangular mesh pair |

```python
fault = Fault.load("fault_model.txt")
fault = Fault.load("ramp.seg", format="seg", frame=geodef.LocalFrame(lon=100.0, lat=0.0))
fault = Fault.load("cascadia", format="ned")  # reads cascadia.ned + cascadia.tri
```

---

## Attributes and properties

| Name | Shape | Description |
|------|-------|-------------|
| `n_patches` | scalar | Number of patches |
| `lon`, `lat` | `(N,)` | Patch center coordinates in degrees |
| `depth` | `(N,)` | Patch center depth in meters, positive down |
| `strike`, `dip` | `(N,)` | Strike in degrees clockwise from the frame's North axis; dip in degrees from horizontal |
| `geographic_strike` | `(N,)` | Strike from true north at each patch (differs from `strike` by meridian convergence) |
| `length`, `width` | `(N,)` or `None` | Rectangular patch dimensions in meters (`None` for triangles) |
| `vertices` | `(N, 3, 3)` or `None` | Triangle corners in local `[east, north, up]` meters (`None` for rectangles) |
| `engine` | `str` | `"okada"` (rectangles) or `"tri"` (triangles) |
| `grid_shape` | `(nL, nW)` or `None` | Structured grid dimensions |
| `medium` | `ElasticMedium` | Shear modulus and Poisson's ratio |
| `frame` | `LocalFrame` | Frame defining every local-coordinate view |
| `centers_geo` | `(N, 3)` | Patch centers as `[lon, lat, depth_m]` |
| `centers_local` | `(N, 3)` | Patch centers as `[east_m, north_m, up_m]` |
| `corners` | `(N, 4 or 3, 3)` | Patch corners as `[east_m, north_m, up_m]`; rectangles run top-left, top-right, bottom-right, bottom-left with "top" the shallow edge |
| `corners_geo` | `(N, 4 or 3, 3)` | Patch corners as `[lon, lat, depth_m]` |
| `areas` | `(N,)` | Patch areas in m² |
| `laplacian` | `(N, N)` | Finite-difference (structured) or nearest-neighbor (unstructured) smoothing operator |

Patches of a planar grid are stored row by row: row 0 is the **shallowest**
row and each following row is one step down dip, to the right of the strike
direction. Within a row, patches advance along strike.

All geometry arrays are read-only after construction.

Use `fault.to_frame(target_frame)` to explicitly re-express local views. For
triangular faults it transforms every vertex while preserving its geographic
position; incompatible frames are never silently substituted.

---

## Forward modeling

### `fault.displacement(lon=, lat=, strike_slip=0.0, dip_slip=0.0)`

Compute surface displacements from strike-slip and dip-slip scalars or
per-patch arrays. Coordinates are keyword-only and may have any shape; the
three returned arrays have that same shape.

```python
strike_slip = np.zeros(fault.n_patches)
dip_slip = np.ones(fault.n_patches)
east, north, up = fault.displacement(
    lon=obs_lon, lat=obs_lat,
    strike_slip=strike_slip,
    dip_slip=dip_slip,
)
```

### `fault.greens_matrix(lon=, lat=, kind="displacement", depth=None)`

Build the raw Green's matrix.

```python
G = fault.greens_matrix(lon=obs_lon, lat=obs_lat)
# shape (3*n_obs, 2*N) for displacement
# Columns [:N] = strike-slip, [N:] = dip-slip
```

`kind='strain'` returns shape `(4*n_obs, 2*N)` for rectangular faults and
`(6*n_obs, 2*N)` for triangular faults. Pass `depth` in meters, positive
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
from geodef.fault import moment_to_magnitude, magnitude_to_moment
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
fault.save("output.seg", format="seg", frame=geodef.LocalFrame(lon=100.0, lat=0.0))
fault.save("triangular_fault", format="ned")  # writes .ned + .tri
fault.to_gmt("patches.gmt", values=result.slip_magnitude)  # closed polygons, -Z colored
```

---

## Checking and copying

```python
print(fault.validate())            # errors and warnings as a ValidationReport
stiffer = fault.with_medium(geodef.ElasticMedium(shear_modulus=40e9))
shared = fault.to_frame(other.frame)   # same physical fault, another local frame
```

`validate()` reports patches or vertices above the free surface, degenerate
triangles, extreme aspect ratios, and suspicious strike values. `Fault` objects
are immutable; `with_medium` and `to_frame` return new faults.

---

## Coordinate conventions

- Geographic: longitude, latitude (degrees), depth in meters (positive down).
- Local Cartesian: East, North, Up in meters.
- The `.seg` format uses local Cartesian (North, East, Depth) in a user-supplied `frame`.
