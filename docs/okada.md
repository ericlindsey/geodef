# `geodef.okada` — Green's function engines

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

Three modules provide displacement and strain Green's functions. For most workflows, use the `okada` dispatcher; access `okada85` or `okada92` directly only when you need specific features.

## Model assumptions

These solutions represent dislocations in a homogeneous, isotropic, linear
elastic half-space with a flat traction-free surface. They do not include
topography, elastic layering, viscoelasticity, poroelasticity, or material
heterogeneity. Agreement to machine precision is therefore numerical accuracy
within the assumed model, not proof that the Earth satisfies those assumptions.

A rectangular or triangular dislocation imposes a displacement discontinuity
(slip or opening) across an internal surface. Because linear elasticity obeys
superposition, responses from many patches add to form a Green's matrix. For
background, see the original [Okada (1985)](https://doi.org/10.1785/BSSA0750041135)
and [Okada (1992)](https://doi.org/10.1785/BSSA0820021018) papers.

---

## `geodef.okada` — Unified dispatcher

Auto-selects `okada85` (fast, surface) or `okada92` (3-D) based on observation depth.

### `okada.displacement(e, n, z, depth, strike, dip, length, width, rake, slip, opening, nu=0.25)`

```python
import geodef

# Surface observations (z=0) → uses okada85
ue, un, uz = geodef.okada.displacement(
    e=obs_east, n=obs_north, z=0.0,
    depth=20_000.0, strike=90.0, dip=15.0,
    length=50_000.0, width=20_000.0,
    rake=90.0, slip=1.0, opening=0.0,
)

# Observations at depth (z<0) → uses okada92
ue, un, uz = geodef.okada.displacement(
    e=obs_east, n=obs_north, z=-5_000.0,
    depth=20_000.0, strike=90.0, dip=15.0,
    length=50_000.0, width=20_000.0,
    rake=90.0, slip=1.0, opening=0.0,
)
```

Coordinates are in local Cartesian (east, north) relative to the fault centroid.

---

## `geodef.okada85` — Surface deformation (Okada 1985)

Observation points must be at the surface (z = 0).

### `okada85.displacement(e, n, depth, strike, dip, length, width, rake, slip, open, nu=0.25)`

Returns `(ue, un, uz)` arrays.

### `okada85.tilt(e, n, depth, strike, dip, length, width, rake, slip, open, nu=0.25)`

Returns `(duz_de, duz_dn)` — surface tilt components.

### `okada85.strain(e, n, depth, strike, dip, length, width, rake, slip, open, nu=0.25)`

Returns `(enn, ene, een, eee)` — horizontal strain tensor components.

Strain is dimensionless and is the symmetric part of the displacement
gradient. The duplicated shear entries `ene` and `een` are retained explicitly
in this interface; for a physically symmetric strain tensor they agree up to
numerical precision.

---

## `geodef.okada92` — Internal deformation (Okada 1992 / DC3D)

Observation points at any depth.

### `okada92.okada92(X, Y, Z, depth, strike, dip, length, width, strike_slip, dip_slip, opening, G, nu, allow_singular=False)`

`X`, `Y` are East and North offsets from the fault centroid in meters, `Z` is
the observation height (zero at the surface, negative below), and `G` is the
shear modulus in Pa. The argument names follow the published routine.

Returns `(disp, strain)`. For arrays of `n` observation points:
- `disp`: `[ue, un, uz]` per point, shape `(n, 3)`
- `strain`: displacement-gradient tensor per point, shape `(n, 3, 3)`

Scalar coordinates return a single `(3, 3)` strain tensor and a `(3, 1)`
displacement column.

```python
from geodef.okada92 import okada92

disp, strain = okada92(
    X=1000.0, Y=500.0, Z=-5000.0,
    depth=20_000.0, strike=90.0, dip=15.0,
    length=50_000.0, width=20_000.0,
    strike_slip=1.0, dip_slip=0.0, opening=0.0,
    G=30e9, nu=0.25,
)
```

---

## `geodef.tri` — Triangular dislocations (Nikkhoo & Walter 2015)

Half-space and full-space solutions for triangular dislocation elements.

Triangle vertex ordering determines the element orientation and local slip
basis. Preserve consistent mesh connectivity and validate a unit strike-slip
and dip-slip response before modeling a new imported mesh.

### `tri.TDdispHS(obs, tri_verts, slip, nu=0.25)`

Half-space surface displacements.

```python
from geodef.tri import TDdispHS

obs = np.column_stack([obs_e, obs_n, np.zeros(n)])  # (n, 3)
verts = np.array([[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]])  # (3, 3) ENU
slip_vec = np.array([ss, ds, 0.0])  # [strike-slip, dip-slip, tensile]

disp = TDdispHS(obs, verts, slip_vec, nu=0.25)  # (n, 3)
```

### `tri.TDstrainHS(obs, tri_verts, slip, nu=0.25)`

Half-space strain tensor.

```python
strain = TDstrainHS(obs, verts, slip_vec)  # (n, 6): [xx, yy, zz, xy, xz, yz]
```

---

## Coordinate conventions

All low-level engines use a local Cartesian frame:
- `e` / `n` — horizontal East and North offsets from fault centroid (meters)
- `z` — observation depth (≤ 0 for surface or below)
- `depth` — fault centroid depth (positive down)

The `Fault.greens_matrix()` and `greens.matrix()` functions handle the geographic-to-local conversion automatically.
