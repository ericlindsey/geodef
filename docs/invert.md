# `geodef.invert` — Inversion

One-call inversion solving `d = Gm` for fault slip.

---

## `invert(fault, datasets, **kwargs) → InversionResult`

```python
import geodef

# Unregularized WLS
result = geodef.invert(fault, [gnss, insar])

# Laplacian smoothing, non-negative
result = geodef.invert(fault, [gnss, insar],
                       smoothing='laplacian',
                       smoothing_strength=1e3,
                       bounds=(0, None))

# One-parameter slip bases
result = geodef.invert(fault, gnss, components='rake', rake=90.0)
result = geodef.invert(fault, gnss,
                       components='azimuth', slip_azimuth=15.0)
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | auto | `'wls'`, `'nnls'`, `'bounded_ls'`, `'constrained'` |
| `smoothing` | `None` | `'laplacian'`, `'damping'`, `'stresskernel'`, or a custom matrix |
| `smoothing_strength` | `0.0` | Regularization weight λ, or `'abic'`/`'cv'` for auto-tuning |
| `smoothing_target` | `None` | Reference model for `(m - m_ref)` regularization |
| `bounds` | `None` | `(lower, upper)` slip bounds; each side is a scalar, a per-component array, a per-parameter array, or `None` |
| `components` | `'both'` | Slip basis: `'both'`, `'strike'`, `'dip'`, `'rake'`, or `'azimuth'` |
| `rake` | `None` | Fixed local rake angle in degrees; required for `components='rake'` |
| `slip_azimuth` | `None` | Fixed geographic slip azimuth in degrees clockwise from north; required for `components='azimuth'` |
| `cv_folds` | `5` | Number of folds for cross-validation |
| `constraints` | `None` | `(C, d)` for `C @ m <= d` (constrained solver only) |

Auto-selection of `method`: `bounds=None` → WLS; `bounds=(0, None)` →
NNLS; general bounds → `bounded_ls`.

Each side of `bounds` may be:

- a **scalar** applied to every slip parameter, e.g. `bounds=(0, None)`;
- a **per-component** array of length `n_components` (`2` for `'both'`,
  else `1`), broadcast across all patches — e.g.
  `bounds=(np.array([0.0, -1.0]), np.array([np.inf, 1.0]))` forces
  strike-slip ≥ 0 while allowing dip-slip in `[-1, 1]`;
- a **per-parameter** array of length `n_params` (`n_components * N`) giving
  an individual bound for every parameter;
- `None` for an unbounded side.

The same forms work with `method='bounded_ls'` and `method='constrained'`.

`components='rake'` solves one slip-amplitude parameter per patch using the
same local rake angle on every patch. `components='azimuth'` also solves one
amplitude per patch, but converts the geographic azimuth to a patch-local rake
using each patch's strike, so it is better for curved or variable-strike meshes.
The Green's matrix and stress-kernel regularization are projected into the
chosen slip basis automatically.

---

## `InversionResult`

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `slip` | `(N, 2)` or `(N, 1)` | Per-patch strike/dip slip for `components='both'`, or one active amplitude per patch |
| `slip_vector` | `(2N,)` or `(N,)` | Blocked `[ss_0..ss_N, ds_0..ds_N]`, or one amplitude per patch |
| `predicted` | `(M,)` | Forward-modeled observations |
| `residuals` | `(M,)` | `obs - predicted` |
| `chi2` | scalar | Reduced chi-squared |
| `rms` | scalar | RMS misfit |
| `moment` | scalar | Seismic moment in N·m |
| `Mw` | scalar | Moment magnitude |
| `smoothing` | str, ndarray, or `None` | Regularization type used |
| `smoothing_strength` | float or `None` | λ used, or `None` when no regularization was applied |
| `components` | str | Slip basis used in the inversion |
| `rake` | float or `None` | Fixed rake angle for `components='rake'` |
| `slip_azimuth` | float or `None` | Fixed geographic azimuth for `components='azimuth'` |

```python
result.save("result.npz")                   # save to disk
result.save_table("result.txt", fault)       # human-readable per-patch table
result = InversionResult.load("result.npz") # reload
```

---

## `LinearSystem`

Use `LinearSystem` directly when reusing the same fault and datasets across
multiple analyses. It precomputes and caches the projected Green's matrix,
weights, and optional smoothing matrix.

```python
system = geodef.LinearSystem(
    fault, [gnss, insar],
    smoothing='laplacian',
    components='azimuth',
    slip_azimuth=15.0,
)

lc = system.lcurve(smoothing_range=(1e-2, 1e6))
result = system.invert(smoothing_strength=lc.optimal, bounds=(0, None))
diagnostics = system.dataset_diagnostics(result)
```

---

## Hyperparameter tuning

### `lcurve(fault, datasets, smoothing, smoothing_range, n=50, **kwargs) → LCurveResult`

```python
lc = geodef.lcurve(fault, [gnss, insar], smoothing='laplacian',
                   smoothing_range=(1e-2, 1e6), n=50)
lc.plot()        # log-log misfit vs model norm; optimal marked
lc.optimal       # λ at maximum curvature
```

### `abic_curve(fault, datasets, smoothing, smoothing_range, n=50, **kwargs) → ABICCurveResult`

```python
ac = geodef.abic_curve(fault, [gnss, insar], smoothing='laplacian',
                       smoothing_range=(1e-2, 1e8), n=50)
ac.plot()        # ABIC vs λ; optimal marked
ac.optimal       # λ at minimum ABIC
```

### Auto-tuning via `smoothing_strength`

```python
result = geodef.invert(fault, data, smoothing='laplacian', smoothing_strength='abic')
result = geodef.invert(fault, data, smoothing='laplacian', smoothing_strength='cv')
```

On the JAX backend (`geodef.backend.set_backend('jax')`), `abic_curve`
evaluates all λ values in one batched computation — same API, same
results, one fused sweep instead of a Python loop.

---

## Nonlinear geometry search (JAX)

### `geometry_search(theta0, datasets, *, ref_lat, ref_lon, ...) → GeometrySearchResult`

Gradient-based inversion for planar fault geometry: the slip is solved
linearly inside a nonlinear search over selected geometry parameters
(variable projection), with exact gradients from autodiff driving
L-BFGS-B. Replaces tutorial 10's grid-then-`minimize_scalar` recipe and
handles several simultaneous parameters. Requires the JAX backend.

In symbols, the routine minimizes a reduced objective

```text
Phi(theta) = min_m [ ||W^(1/2) (d_obs - G(theta)m)||^2
                     + lambda ||L m||^2 ].
```

The inner minimization solves slip `m` for each trial geometry `theta`; the
outer optimization changes only the requested geometry parameters. This is
called variable projection
([Golub & Pereyra, 1973](https://doi.org/10.1137/0710036)), and it avoids
making the nonlinear optimizer search over every slip patch.
The weighting matrix `W` comes from the reported data covariance, while `L`
and `lambda` describe the chosen slip regularization.

```python
geodef.backend.set_backend('jax')

theta0 = [0.0, 0.0, 25e3, 315.0, 30.0, 180e3, 90e3]
#         e0   n0   depth strike dip   length width  (start; true dip 15)

result = geodef.geometry_search(
    theta0, gnss,
    ref_lat=-2.0, ref_lon=100.0,     # anchors the local frame
    free=['dip', 'depth'],           # parameters to optimize; rest fixed
    bounds={'dip': (5.0, 45.0)},
    n_length=12, n_width=6,
    components='dip',
    smoothing='laplacian', smoothing_strength=1.0,
)

result.theta          # full 7-vector at the optimum
result.slip           # inner-solve slip at the optimal geometry
result.theta_cov      # Gauss-Newton covariance of the free parameters
result.reduced_chi2
```

Notes:

- `theta0` is in the local Cartesian frame anchored at
  `(ref_lat, ref_lon)`; `e0`/`n0` are centroid offsets in meters.
- The inner solve is unconstrained WLS with fixed `smoothing_strength`;
  choose λ first (e.g. with `abic_curve` at a reasonable starting
  geometry).
- The objective is non-convex — for poorly constrained problems, run
  from several starting points. The compiled kernel is cached at module
  level, so repeated calls with the same problem shapes pay JIT
  compilation only once (~tens of seconds on the first call).
- `theta_cov` is the Gauss-Newton (Laplace) covariance scaled by the
  reduced chi-squared; it captures local curvature only. Treat it as a local
  linearized uncertainty estimate, not proof that the full geometry posterior
  is Gaussian or unimodal. Use `geodef.bayes` when those assumptions matter.

---

## Model assessment

These are computed on demand (not during `invert()`) as they require forming and inverting dense matrices.

### `model_covariance(result, fault, datasets) → np.ndarray`

```python
Cm = geodef.model_covariance(result, fault, [gnss, insar])
# shape (P, P), where P is 2N for components='both' and N otherwise
```

### `model_resolution(result, fault, datasets) → np.ndarray`

```python
R = geodef.model_resolution(result, fault, [gnss, insar])
# shape (P, P); R=I for perfect resolution, diag(R)<1 where regularization dominates
```

### `model_uncertainty(result, fault, datasets) → np.ndarray`

```python
sigma = geodef.model_uncertainty(result, fault, [gnss, insar])
# shape (P,); per-parameter 1-sigma from sqrt(diag(Cm))
```

### `dataset_diagnostics(result, fault, datasets) → list[DatasetDiagnostics]`

Per-dataset fit statistics using the hat matrix.

```python
diags = geodef.dataset_diagnostics(result, fault, [gnss, insar])
for d in diags:
    print(d.chi2, d.reduced_chi2, d.wrms, d.n_obs, d.dof)
```

### `compute_abic(result, fault, datasets) → float`

```python
abic = geodef.compute_abic(result, fault, [gnss, insar])
```
