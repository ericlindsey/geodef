# `geodef.geometry` — named geometry and coordinate frames

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

Geometry used by more than one calculation should carry names, units, and a
coordinate frame. This module provides immutable value objects for that role:
`LocalFrame`, `PlanarGeometry`, and `TriGeometry`.

## `LocalFrame`

```python
import geodef

frame = geodef.LocalFrame(
    origin_lat=-2.0,
    origin_lon=100.0,
    origin_alt=0.0,
    projection="wgs84-enu",
)
```

The `projection` field is always recorded, even though `"wgs84-enu"` is the
only implementation today. It means WGS84 geographic coordinates are
converted to Earth-centered Earth-fixed coordinates and rotated into the
local tangent East-North-Up frame. Future projections can therefore be added
without making existing local arrays ambiguous.

Keyword-only conversions make coordinate order explicit:

```python
enu = frame.to_enu(lon=lon, lat=lat, alt=alt)
# final axis: [east_m, north_m, up_m]

geographic = frame.to_geographic(
    east=enu[..., 0], north=enu[..., 1], up=enu[..., 2]
)
# final axis: [lon_degrees, lat_degrees, altitude_m]
```

Frames are compatible only when origin, altitude, and projection all match.
APIs reject conflicting frames. Re-expression is deliberate:

```python
coordinates_in_target = frame.transform_enu(coordinates, target=target_frame)
```

## `PlanarGeometry`

`PlanarGeometry` names the seven parameters used by planar forward models,
geometry search, and rectangular Bayesian inference.

```python
geometry = geodef.PlanarGeometry(
    center=(0.0, 0.0),       # east, north in frame (m)
    depth=25_000.0,          # positive down (m)
    strike=315.0,            # clockwise from north (degrees)
    dip=15.0,                # down from horizontal (degrees)
    length=180_000.0,
    width=90_000.0,
    frame=frame,
)

fault = geodef.Fault.planar(geometry, n_length=12, n_width=6)
geometry.to_enu()          # [east, north, up]
geometry.to_geographic()   # [lon, lat, depth]
geometry.theta             # expert/JAX seven-vector
```

Use `PlanarGeometry.from_geographic(...)` when the centroid starts in
longitude/latitude, and `PlanarGeometry.from_theta(theta, frame=frame)` at an
expert array boundary. `to_frame(target_frame)` explicitly re-expresses its
horizontal center.

The keyword-scalar `Fault.planar(lat=..., lon=..., ...)` call remains
supported and creates the equivalent named geometry automatically.

## `TriGeometry`

`TriGeometry` binds triangular vertices to the frame that defines them.
Vertices have shape `(N, 3, 3)` and final axis `[east, north, up]` in meters.

```python
tri_geometry = geodef.TriGeometry.from_nodes(nodes_enu, triangles, frame=frame)
fault = geodef.Fault.from_triangles(tri_geometry)

tri_geometry.centers_enu
tri_geometry.centers_geographic  # [lon, lat, depth]
tri_geometry.strike              # derived per patch
tri_geometry.dip                 # derived per patch
```

`from_geographic(...)` converts geographic nodes and connectivity, while
`to_frame(target_frame)` explicitly transforms all 3-D vertices.

Per-patch strike and dip on a curved mesh can vary sharply and need not define
a sensible large-scale slip basis. `TriGeometry` therefore describes physical
geometry only. A constant plate-rake direction or Euler-pole-derived plate
direction belongs to the slip representation introduced in Priority 1.2; it
will use this object's stable orientations and frame without embedding plate
kinematics in the fault surface itself.
