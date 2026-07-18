# `geodef.mesh` — Triangular mesh generation

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

Provides the `Mesh` dataclass and four factory functions for creating
triangular fault meshes.

## Choosing a mesh

A triangular mesh approximates a curved fault surface with planar elements.
Refine where geometry bends rapidly or observations are sensitive, but avoid
refining far beyond the data's resolving power: more triangles mean more slip
parameters, memory, and regularization dependence. Check triangle aspect
ratios, depth signs, surface continuity, and normal orientation visually before
inversion. A mesh that looks smooth in longitude/latitude can still contain
poorly shaped elements in meters.

**Optional dependencies:** `meshpy` (meshing) and `netCDF4` (slab2.0 grids),
both provided by the `mesh` extra: `uv pip install -e ".[mesh]"`.

---

## `Mesh` dataclass

Immutable triangular mesh in geographic coordinates.

```python
from geodef.mesh import Mesh

mesh.n_nodes       # number of vertices
mesh.n_triangles   # number of triangles
mesh.frame         # explicit LocalFrame, inferred from mean nodes by default
mesh.centers_geo   # (M, 3) centroids as [lon, lat, depth_m]
mesh.areas         # (M,) triangle areas in m²
mesh.vertices_enu()                  # vertices in mesh.frame
mesh.vertices_enu(frame=other_frame) # explicit alternate representation
```

Pass `frame=geodef.LocalFrame(...)` when constructing a `Mesh` to choose the
stored local representation. Legacy `vertices_enu(ref_lat, ref_lon)` remains
supported. Supplying both forms is rejected because it would make provenance
ambiguous.
`mesh.to_frame(other_frame)` returns the same geographic mesh with a different
default local representation.

### I/O

```python
mesh.save("cascadia")          # writes cascadia.ned + cascadia.tri
mesh = Mesh.load("cascadia")   # reads cascadia.ned + cascadia.tri

# coord_order: 'latlon' (default, unicycle-compatible) or 'lonlat'
mesh.save("out", coord_order="lonlat")
mesh = Mesh.load("out", coord_order="lonlat")
```

---

## Factory functions

### `from_slab2(fname, bounds, *, target_length=50_000.0, depth_growth=1.0, max_depth=None, surface_trace=None, ...)`

Generate a mesh from a slab2.0 NetCDF depth grid.

```python
from geodef.mesh import from_slab2

mesh = from_slab2("cas_slab2_dep.grd",
    bounds=(235, 245, 42, 50),   # (lon_min, lon_max, lat_min, lat_max) in degrees
    target_length=30_000.0,      # target edge length in meters (default 50 km)
    depth_growth=2.0,            # edge length ratio deep/shallow (1.0 = uniform)
    max_depth=100_000.0,         # clip slab at this depth in meters (None = no clip)
    surface_trace=None,          # optional (trace_lon, trace_lat) arrays
)
```

### `from_trace(trace_lon, trace_lat, max_depth, dip, *, dip_direction=None, n_downdip=20, target_length=None, max_area=None, ...)`

Generate a mesh from a surface trace and constant dip.

```python
from geodef.mesh import from_trace

mesh = from_trace(trace_lon, trace_lat,
    max_depth=30_000.0,      # maximum fault depth in meters
    dip=15.0,                # degrees
    dip_direction=180.0,     # azimuth of dip direction (degrees)
    n_downdip=20,            # down-dip profile resolution
    target_length=10_000.0,  # target edge length in meters
)
```

`dip` can also be a callable `dip(depth_m) -> degrees` for listric faults.

### `from_polygon(lon, lat, depth=None, *, depth_func=None, target_length=None, max_area=None, ...)`

Generate a mesh from a 3-D boundary polygon.

```python
from geodef.mesh import from_polygon

mesh = from_polygon(bound_lon, bound_lat, bound_depth,  # depth in meters
    target_length=15_000.0,  # target edge length in meters
)
```

For 2-D polygons, omit `depth` and pass `depth_func(lon, lat) -> depth_m`.

### `from_points(lon, lat, depth, *, boundary=None, target_length=None, max_area=None, ...)`

Generate a mesh from scattered 3-D points (convex hull boundary by default).

```python
from geodef.mesh import from_points

mesh = from_points(lon, lat, depth,         # depth in meters
    target_length=12_000.0,                 # target edge length in meters
    boundary=custom_boundary_lonlat,        # optional (B, 2) array; default = convex hull
)
```

---

## Converting to `Fault`

```python
import geodef

fault = geodef.Fault.from_mesh(mesh)
```

Or in one step (bypassing the explicit `Mesh` object):

```python
fault = geodef.Fault.load("cascadia", format="ned")
```

Tutorials 02, 04, and 09 explain discretization, regularization, and resolution;
use those concepts to choose and evaluate the mesh rather than selecting a
target edge length from geometry alone.
