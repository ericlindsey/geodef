# `geodef.okada` — Green's function engines

Three modules provide displacement and strain Green's functions. For most workflows, use the `okada` dispatcher; access `okada85` or `okada92` directly only when you need specific features.

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

### `okada85.displacement(e, n, depth, strike, dip, length, width, rake, slip, opening, nu=0.25)`

Returns `(ue, un, uz)` arrays.

### `okada85.tilt(e, n, depth, strike, dip, length, width, rake, slip, opening, nu=0.25)`

Returns `(duz_de, duz_dn)` — surface tilt components.

### `okada85.strain(e, n, depth, strike, dip, length, width, rake, slip, opening, nu=0.25)`

Returns `(enn, ene, een, eee)` — horizontal strain tensor components.

---

## `geodef.okada92` — Internal deformation (Okada 1992 / DC3D)

Observation points at any depth.

### `okada92.okada92(e, n, z, depth, strike, dip, length, width, slip_strike, slip_dip, opening, mu, nu, allow_singular=False)`

Returns `(disp, strain)`:
- `disp`: `(3,)` array `[ue, un, uz]`
- `strain`: `(3, 3)` displacement gradient tensor

```python
from geodef.okada92 import okada92

disp, strain = okada92(
    e=1000.0, n=500.0, z=-5000.0,
    depth=20_000.0, strike=90.0, dip=15.0,
    length=50_000.0, width=20_000.0,
    slip_strike=1.0, slip_dip=0.0, opening=0.0,
    mu=30e9, nu=0.25,
)
```

---

## `geodef.tri` — Triangular dislocations (Nikkhoo & Walter 2015)

Half-space and full-space solutions for triangular dislocation elements.

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

The `Fault.greens_matrix()` and `greens.greens()` functions handle the geographic-to-local conversion automatically.
