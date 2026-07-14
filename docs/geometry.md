# `geodef.geometry` — coordinate frames and array conversions

> Conventions — axes, depth sign, angles, units, array ordering,
> regularization: see [`conventions.md`](conventions.md).

`Fault` is GeoDef's fault-geometry object. The `geometry` module supplies the
smaller pieces needed to construct and transform its arrays without introducing
a second geometry hierarchy.

## `LocalFrame`

`LocalFrame` gives local East/North/Up arrays a geographic origin. It is kept as
a value object because the origin and projection must travel together whenever
local arrays are stored, cached, or combined.

```python
import geodef

frame = geodef.LocalFrame(
    origin_lat=-2.0,
    origin_lon=100.0,
    origin_alt=0.0,
)

enu = frame.to_enu(lon=lon, lat=lat, alt=alt)
geographic = frame.to_geographic(
    east=enu[..., 0],
    north=enu[..., 1],
    up=enu[..., 2],
)
```

`source.transform_enu(coordinates, target=target)` explicitly re-expresses an
array in another frame. `source.require_compatible(other)` raises rather than
silently combining coordinates defined by different origins.

`Fault.planar(..., frame=frame)`, `Fault.from_triangles(..., frame=frame)`, and
`Mesh(..., frame=frame)` attach the frame directly to the domain value that owns
the coordinates.

## Planar parameter vectors

JAX geometry kernels use the compact expert vector
`[e0, n0, depth, strike, dip, length, width]`. Deterministic and Bayesian
geometry inference also accept a mapping, so callers need not rely on order:

```python
parameters = {
    "e0": 0.0,
    "n0": 0.0,
    "depth": 15_000.0,
    "strike": 315.0,
    "dip": 25.0,
    "length": 80_000.0,
    "width": 40_000.0,
}

theta = geodef.geometry.as_planar_vector(parameters)
parameters = geodef.geometry.planar_parameter_dict(theta)
```

Every value is validated for finiteness and physical range. The associated
`LocalFrame` remains an explicit argument to geometry search or posterior
construction.

For ordinary forward models, construct the fault directly with named geographic
keywords:

```python
fault = geodef.Fault.planar(
    lat=-2.0,
    lon=100.0,
    depth=15_000.0,
    strike=315.0,
    dip=25.0,
    length=80_000.0,
    width=40_000.0,
    n_length=12,
    n_width=6,
    frame=frame,
)
```

## Triangular arrays

Expand shared nodes and connectivity only when an engine needs per-triangle
vertices:

```python
vertices = geodef.geometry.vertices_from_nodes(nodes_enu, triangles)
strike, dip = geodef.geometry.triangle_strike_dip(vertices)

fault = geodef.Fault.from_triangles(vertices, frame=frame)
# Or avoid expansion at the call site:
fault = geodef.Fault.from_triangles(
    nodes_enu,
    triangles=triangles,
    frame=frame,
)
```

`Fault.from_mesh(mesh)` is the shortest path when geographic nodes and
connectivity already live in a `Mesh`.
