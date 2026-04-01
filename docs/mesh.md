# `geodef.mesh` — Triangular mesh generation

Provides the `Mesh` dataclass and four factory functions for creating triangular fault meshes.

**Optional dependencies:** `meshpy` (meshing) and `netCDF4` (slab2.0 grids). Install with `pip install meshpy netCDF4`.

---

## `Mesh` dataclass

Immutable triangular mesh in geographic coordinates.

```python
from geodef.mesh import Mesh

mesh.n_nodes       # number of vertices
mesh.n_triangles   # number of triangles
mesh.centers_geo   # (M, 3) centroids as [lon, lat, depth_m]
mesh.areas         # (M,) triangle areas in m²
mesh.vertices_enu(ref_lat, ref_lon)  # (M, 3, 3) vertices in local ENU meters
```

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

### `from_slab2(fname, bounds, target_length=50.0, depth_growth=1.0, max_depth=None, ...)`

Generate a mesh from a slab2.0 NetCDF depth grid.

```python
from geodef.mesh import from_slab2

mesh = from_slab2("cas_slab2_dep.grd",
    bounds=(235, 245, 42, 50),   # (lon_min, lon_max, lat_min, lat_max) in degrees
    target_length=30.0,          # target edge length in km (default 50 km)
    depth_growth=2.0,            # edge length ratio deep/shallow (1.0 = uniform)
    max_depth=100.0,             # clip slab at this depth in km (None = no clip)
)
```

### `from_trace(trace_lon, trace_lat, max_depth, dip, dip_direction, target_length)`

Generate a mesh from a surface trace and constant dip.

```python
from geodef.mesh import from_trace

mesh = from_trace(trace_lon, trace_lat,
    max_depth=30.0,          # maximum fault depth in km
    dip=15.0,                # degrees
    dip_direction=180.0,     # azimuth of dip direction (degrees)
    target_length=10_000.0,  # target edge length in meters
)
```

`dip` can also be a callable `dip(depth_m) -> degrees` for listric faults.

### `from_polygon(lon, lat, depth, target_length)`

Generate a mesh from a 3-D boundary polygon.

```python
from geodef.mesh import from_polygon

mesh = from_polygon(bound_lon, bound_lat, bound_depth,  # depth in meters
    target_length=15_000.0,  # target edge length in meters
)
```

### `from_points(lon, lat, depth, target_length)`

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

See `examples/04_mesh_generation.ipynb` for a full demo.
