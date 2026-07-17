# `geodef.greens` — Green's matrix assembly

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

Assembles projected Green's matrices from `Fault` and `DataSet` objects, and provides Laplacian regularization operators.

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

Build the projected Green's matrix for one or more datasets. Results are automatically cached.

```python
import geodef

G = geodef.greens.matrix(fault, gnss)                # shape (n_obs, 2*N)
G = geodef.greens.matrix(fault, [gnss, insar])       # rows stacked vertically
```

By default columns are blocked: `[:N]` strike-slip, `[N:]` dip-slip.

Pass `components=` to have `matrix()` return a single-component matrix (shape
`(n_obs, N)`) directly, using the same slip basis as `geodef.invert.solve()`:

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
project it into a one-component slip basis. `geodef.invert.solve()` uses it internally
for `components='strike'|'dip'|'rake'|'azimuth'|'plate'` and to project
stress-kernel regularization into the active basis.

### `stack_obs(datasets) → np.ndarray`

Concatenate observation vectors from one or more datasets.

```python
d = geodef.stack_obs([gnss, insar])    # shape (total_n_obs,)
```

### `stack_weights(datasets) → np.ndarray`

Build block-diagonal inverse-covariance weight matrix.

```python
W = geodef.stack_weights([gnss, insar])   # shape (total_n_obs, total_n_obs)
```

`geodef.invert.solve()` calls these internally; use them directly when assembling `G @ m` by hand.

`W` is the inverse covariance, not a vector of standard deviations. In
derivations it is often clearer to whiten the system with a matrix `R` such
that `R.T @ R = W`, then solve with `R @ G` and `R @ d`.

---

## Laplacian operators

### `project(data, G_raw) → np.ndarray`

Project a raw three-component displacement matrix into a dataset's observed
components. This is useful when assembling a custom forward operator.

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
`geodef.model_resolution(...)`, which accounts for the smoothing operator.

## Low-level Green's matrix functions

These operate on raw geographic arrays rather than `Fault`/`DataSet` objects. Useful for custom workflows.

### `displacement_greens(lat, lon, lat0, lon0, depth, strike, dip, L, W, nu=0.25)`

Rectangular patches (Okada85). Returns shape `(3*nobs, 2*npatch)`.

### `strain_greens(lat, lon, lat0, lon0, depth, strike, dip, L, W, nu=0.25, obs_depth=None)`

Rectangular patches, strain output. Shape `(4*nobs, 2*npatch)`. Pass `obs_depth` for internal points (uses Okada92).

### `tri_displacement_greens(lat, lon, lat0, lon0, depth, vertices, nu=0.25, *, frame=None)`

Triangular patches (Nikkhoo & Walter). Returns shape `(3*nobs, 2*npatch)`.

### `tri_strain_greens(lat, lon, lat0, lon0, depth, vertices, nu=0.25, obs_depth=None, *, frame=None)`

Triangular patches, strain output. Shape `(6*nobs, 2*npatch)`.

For direct low-level triangular calls, pass the `LocalFrame` that defines
`vertices`. Omitting it retains legacy mean-centroid frame inference.
`Fault.greens_matrix` always supplies `fault.frame`, so domain-level assembly
cannot silently reinterpret triangular vertices.

---

## Column layout

For a fault with `N` patches and `M` observation points:

- **Displacement G:** `(3M, 2N)` — rows are `[e1, n1, u1, e2, ...]`; cols `[:N]` strike-slip, `[N:]` dip-slip
- **Strain G (rectangular):** `(4M, 2N)` — rows are `[nn, ne, en, ee]` per point
- **Strain G (triangular):** `(6M, 2N)` — rows are `[xx, yy, zz, xy, xz, yz]` per point
