# `geodef.greens` — Green's matrix assembly

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

Assembles projected Green's matrices from `Fault` and `DataSet` objects, and provides Laplacian regularization operators.

Most work never calls this module directly. There are three levels, from most
to least convenient:

| You want | Use |
|---|---|
| The matrices of an inversion (`G`, `d`, `W`, `L`), ready to solve or inspect | `system = geodef.LinearSystem(fault, datasets, ...)`, then `system.G` |
| `G` for a fault and datasets, in each dataset's observation layout | `geodef.greens.matrix(fault, datasets)` |
| The raw three-component `G` at arbitrary points | `fault.greens_matrix(lon=, lat=)` |

## Physical and algebraic picture

Elastic dislocation models are linear in slip. After discretizing a fault into
`N` patches, the predicted observations are

```text
d_pred = G m.
```

Column `j` of `G` is the observation pattern produced by one unit of slip on
model parameter `m_j`; row `i` describes how every slip parameter contributes
to observation `d_i`. This is why GeoDef calls `G` a Green's matrix. Geometry,
elastic constants, observation locations, and projection determine `G`; slip
does not. See [Green's functions](https://en.wikipedia.org/wiki/Green%27s_function)
for the broader linear-systems concept.

Matrix dimensions are worth checking before every custom inversion. If `G` is
`(M, P)`, then `m` must have length `P` and `G @ m` has the same length and row
ordering as the stacked observation vector.

---

## High-level assembly

### `matrix(fault, datasets, *, components='both', rake=None, slip_azimuth=None, plate_rake=None) → np.ndarray`

Build the projected Green's matrix for one or more datasets. Results are
saved to disk and reused when the cache is on (`geodef.cache.enable()`; see
[`cache.md`](cache.md)).

```python
import geodef

G = geodef.greens.matrix(fault, gnss)                # shape (n_obs, 2*N)
G = geodef.greens.matrix(fault, [gnss, insar])       # rows stacked vertically
```

By default columns are blocked: `[:N]` strike-slip, `[N:]` dip-slip.

Pass `components=` to have `matrix()` return a single-component matrix (shape
`(n_obs, N)`) directly, using the same slip basis as `geodef.solve()`:

```python
G_strike = geodef.greens.matrix(fault, gnss, components='strike')
G_dip    = geodef.greens.matrix(fault, gnss, components='dip')
G_rake   = geodef.greens.matrix(fault, gnss, components='rake', rake=90.0)
G_az     = geodef.greens.matrix(fault, gnss, components='azimuth', slip_azimuth=350.0)
G_plate  = geodef.greens.matrix(fault, gnss, components='plate', plate_rake=plate_rake)
```

For `'rake'` (a single rake for every patch) and `'azimuth'` (a fixed
geographic slip azimuth, so each patch's local rake is `slip_azimuth - strike_i`)
the two blocked column sets are combined as `cos(theta)*G_strike + sin(theta)*G_dip`.

`'plate'` keeps two blocked columns per patch, but rotates them to
`[rake_parallel | rake_perpendicular]` using a scalar or per-patch
`plate_rake`. This is the appropriate matrix basis when bounds and smoothing
should follow a large-scale tectonic direction rather than variable local
triangle orientations.

### `select_slip_columns(G_full, n_patches, components, rake=None, fault_strike=None, slip_azimuth=None, plate_rake=None) → np.ndarray`

The reduction primitive behind `matrix(components=...)`. Apply it to any
already-assembled `(M, 2*N)` matrix — a Green's matrix or a stress kernel — to
project it into a one-component slip basis. `geodef.solve()` uses it internally
for `components='strike'|'dip'|'rake'|'azimuth'|'plate'` and to project
stress-kernel regularization into the active basis.

### `stack_obs(datasets) → np.ndarray`

Concatenate observation vectors from one or more datasets.

```python
d = geodef.greens.stack_obs([gnss, insar])    # shape (total_n_obs,)
```

### `stack_weights(datasets) → np.ndarray`

Build block-diagonal inverse-covariance weight matrix.

```python
W = geodef.greens.stack_weights([gnss, insar])   # shape (total_n_obs, total_n_obs)
```

`LinearSystem` calls these internally and exposes the results as `system.d`
and `system.W`; use the functions directly when assembling `G @ m` by hand.

`W` is the inverse covariance, not a vector of standard deviations. In
derivations it is often clearer to whiten the system with a matrix `R` such
that `R.T @ R = W`, then solve with `R @ G` and `R @ d`.

### `project(data, G_raw) → np.ndarray`

Project a raw three-component displacement matrix (from
`fault.greens_matrix`) into a dataset's observation layout by applying
`data.project(east, north, up)` to every column. `matrix()` does this for you.

```python
G_raw = fault.greens_matrix(lon=insar.lon, lat=insar.lat)   # (3 * n_pixels, 2 * N)
G_los = geodef.greens.project(insar, G_raw)                 # (n_pixels, 2 * N)
```

---

## Laplacian operators

### `laplacian(fault) → np.ndarray`

Return the fault's patch-order-aware Laplacian matrix. This is the convenient
entry point for ordinary use; the builders below remain available for custom
grids.

### `build_laplacian_2d(nL, nW) → np.ndarray`

2-D finite-difference Laplacian for a structured rectangular grid. Each row sums to zero. Requires `nL >= 3` and `nW >= 3`.

```python
from geodef.greens import build_laplacian_2d
L = build_laplacian_2d(10, 5)   # shape (50, 50)
```

### `build_laplacian_2d_simple(nL, nW) → np.ndarray`

Simpler 2-D Laplacian variant with free boundary conditions: each diagonal
weight equals the number of available neighbors, so every row sums to zero
without the one-sided second-difference stencils `build_laplacian_2d` uses at
edges. Useful for teaching and for cross-checking regularization behavior.

### `build_laplacian_knn(coords, k=4) → scipy.sparse.csc_matrix`

Distance-weighted graph Laplacian for unstructured meshes (triangular or non-uniform rectangular). Finds *k* nearest neighbors, weights by inverse distance, symmetrizes.

```python
from geodef.greens import build_laplacian_knn
L = build_laplacian_knn(fault.centers_local, k=6)   # sparse (N, N)
```

`fault.laplacian` automatically chooses `build_laplacian_2d` (structured grids) or `build_laplacian_knn` (unstructured) based on whether `grid_shape` is set.

A Laplacian penalizes spatial curvature rather than slip amplitude:
`||L m||^2` is small for locally smooth slip and zero for modes in the
operator's null space (often constant slip). It therefore expresses a modeling
assumption that neighboring patches should not change abruptly. Patch size and
neighbor geometry affect its scale, so regularization strengths should not be
transferred blindly between meshes.

---

## Resolution

### `resolution_matrix(G) → np.ndarray`

Model resolution matrix `R = pinv(G) @ G` for an unregularized system — how
each true model parameter maps into the recovered one (`R = I` means perfect
recovery). For the resolution of a *regularized* inversion, use
`result.resolution()`, which accounts for the smoothing operator.

## Low-level Green's matrix functions

These are what `fault.greens_matrix` calls. They work entirely in **one local
Cartesian frame** (meters, degrees for angles) and know nothing about
longitude and latitude: `Fault` converts observation points and patches into
its frame first. All arguments are keyword-only. Rectangular patches use one
batched evaluation of the Okada kernel on both the NumPy and JAX backends.

### `displacement_greens(*, obs_east, obs_north, east, north, depth, strike, dip, length, width, nu=0.25)`

Rectangular patches (Okada 1985). `east`, `north`, `depth` locate each patch
center. Returns shape `(3*M, 2*N)`.

### `strain_greens(*, obs_east, obs_north, east, north, depth, strike, dip, length, width, nu=0.25, obs_depth=None)`

Rectangular patches, displacement gradients `[NN, NE, EN, EE]`. Shape
`(4*M, 2*N)`. Pass `obs_depth` (positive down) for internal points, which uses
Okada (1992).

### `tri_displacement_greens(*, obs_east, obs_north, vertices, nu=0.25)`

Triangular patches (Nikkhoo & Walter 2015). `vertices` are `(N, 3, 3)`
`[east, north, up]` corners in the same frame. Returns shape `(3*M, 2*N)`.

### `tri_strain_greens(*, obs_east, obs_north, vertices, nu=0.25, obs_depth=None)`

Triangular patches, strain `[xx, yy, zz, xy, xz, yz]`. Shape `(6*M, 2*N)`.

```python
frame = fault.frame
obs = frame.to_enu(lon=gnss.lon, lat=gnss.lat)
G = geodef.greens.displacement_greens(
    obs_east=obs[:, 0], obs_north=obs[:, 1],
    east=fault.centers_local[:, 0], north=fault.centers_local[:, 1],
    depth=fault.depth, strike=fault.strike, dip=fault.dip,
    length=fault.length, width=fault.width,
)
# identical to fault.greens_matrix(lon=gnss.lon, lat=gnss.lat)
```

---

## Column layout

For a fault with `N` patches and `M` observation points:

- **Displacement G:** `(3M, 2N)` — rows are `[e1, n1, u1, e2, ...]`; cols `[:N]` strike-slip, `[N:]` dip-slip
- **Strain G (rectangular):** `(4M, 2N)` — rows are `[nn, ne, en, ee]` per point
- **Strain G (triangular):** `(6M, 2N)` — rows are `[xx, yy, zz, xy, xz, yz]` per point
