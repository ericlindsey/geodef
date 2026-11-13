# `geodef.geometry` — fault-geometry array helpers

> Conventions — axes, depth sign, angles, units, array ordering,
> regularization: see [`conventions.md`](conventions.md).

`Fault` is GeoDef's fault-geometry object. The `geometry` module supplies the
smaller array helpers behind it: the planar parameter vector used by geometry
search and Bayesian sampling, and the triangle helpers used by meshes.
Coordinate frames (`LocalFrame`) and geodetic conversions live in
[`transforms`](transforms.md).

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
