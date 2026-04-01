# `geodef.greens` — Green's matrix assembly

Assembles projected Green's matrices from `Fault` and `DataSet` objects, and provides Laplacian regularization operators.

---

## High-level assembly

### `greens(fault, datasets) → np.ndarray`

Build the projected Green's matrix for one or more datasets. Results are automatically cached.

```python
import geodef

G = geodef.greens.greens(fault, gnss)                # shape (n_obs, 2*N)
G = geodef.greens.greens(fault, [gnss, insar])       # rows stacked vertically
```

Columns are blocked: `[:N]` strike-slip, `[N:]` dip-slip.

`greens()` always returns both components. If you need a single-component G for a custom workflow, slice the result manually:

```python
N = fault.n_patches
G_strike = G[:, :N]   # strike-slip columns only
G_dip    = G[:, N:]   # dip-slip columns only
```

> **Planned:** a `components='both'|'strike'|'dip'` argument will be added to `greens()` for consistency with `invert()`. See `PLAN.md` §10.1.

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

`geodef.invert()` calls these internally; use them directly when assembling `G @ m` by hand.

---

## Laplacian operators

### `build_laplacian_2d(nL, nW) → np.ndarray`

2-D finite-difference Laplacian for a structured rectangular grid. Each row sums to zero. Requires `nL >= 3` and `nW >= 3`.

```python
from geodef.greens import build_laplacian_2d
L = build_laplacian_2d(n_length=10, n_width=5)   # shape (50, 50)
```

### `build_laplacian_knn(coords, k=4) → scipy.sparse.csc_matrix`

Distance-weighted graph Laplacian for unstructured meshes (triangular or non-uniform rectangular). Finds *k* nearest neighbors, weights by inverse distance, symmetrizes.

```python
from geodef.greens import build_laplacian_knn
L = build_laplacian_knn(fault.centers_local, k=6)   # sparse (N, N)
```

`fault.laplacian` automatically chooses `build_laplacian_2d` (structured grids) or `build_laplacian_knn` (unstructured) based on whether `grid_shape` is set.

---

## Low-level Green's matrix functions

These operate on raw geographic arrays rather than `Fault`/`DataSet` objects. Useful for custom workflows.

### `displacement_greens(lat, lon, lat0, lon0, depth, strike, dip, L, W, nu=0.25)`

Rectangular patches (Okada85). Returns shape `(3*nobs, 2*npatch)`.

### `strain_greens(lat, lon, lat0, lon0, depth, strike, dip, L, W, nu=0.25, obs_depth=None)`

Rectangular patches, strain output. Shape `(4*nobs, 2*npatch)`. Pass `obs_depth` for internal points (uses Okada92).

### `tri_displacement_greens(lat, lon, lat0, lon0, depth, vertices, nu=0.25)`

Triangular patches (Nikkhoo & Walter). Returns shape `(3*nobs, 2*npatch)`.

### `tri_strain_greens(lat, lon, lat0, lon0, depth, vertices, nu=0.25, obs_depth=None)`

Triangular patches, strain output. Shape `(6*nobs, 2*npatch)`.

---

## Column layout

For a fault with `N` patches and `M` observation points:

- **Displacement G:** `(3M, 2N)` — rows are `[e1, n1, u1, e2, ...]`; cols `[:N]` strike-slip, `[N:]` dip-slip
- **Strain G (rectangular):** `(4M, 2N)` — rows are `[nn, ne, en, ee]` per point
- **Strain G (triangular):** `(6M, 2N)` — rows are `[xx, yy, zz, xy, xz, yz]` per point
