# `geodef.invert` — Inversion

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

One-call inversion solving `d = Gm` for fault slip.

## What problem is being solved?

For observations `d`, Green's matrix `G`, slip vector `m`, and data covariance
`C_d`, weighted least squares minimizes

```text
Phi(m) = (d - Gm)^T C_d^-1 (d - Gm) + lambda ||L(m - m_ref)||^2.
```

The first term rewards agreement with data in units of their uncertainty. The
second is optional regularization: `L` may damp slip amplitude, smooth spatial
curvature, or penalize a stress measure. `lambda` controls the trade-off.
Bounds and inequality constraints encode additional assumptions such as
non-negative coupling.

An inverse solution is therefore conditional on fault geometry, covariance,
regularization, constraints, and elastic-model assumptions. A small residual
does not by itself imply a unique or physically correct slip distribution. See
[linear inverse problems](https://en.wikipedia.org/wiki/Inverse_problem) and
[Tikhonov regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization)
for general background.

---

## `solve(fault, datasets, **kwargs) → InversionResult`

```python
import geodef

# Unregularized WLS
result = geodef.invert.solve(fault, [gnss, insar])

# Laplacian smoothing, non-negative
result = geodef.invert.solve(fault, [gnss, insar],
                             regularization='laplacian',
                             regularization_strength=1e3,
                             bounds=(0, None))

# One-parameter slip bases
result = geodef.invert.solve(fault, gnss, components='rake', rake=90.0)
result = geodef.invert.solve(fault, gnss,
                             components='azimuth', slip_azimuth=15.0)

# Two plate-motion coordinates, suitable for variable-orientation meshes
plate_rake = geodef.slip.plate_rake_from_euler(
    fault, (pole_lat, pole_lon, rate)
)
result = geodef.invert.solve(fault, gnss,
                             components='plate', plate_rake=plate_rake)
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | auto | `'wls'`, `'nnls'`, `'bounded_ls'`, `'constrained'` |
| `regularization` | `None` | `'laplacian'`, `'damping'`, `'stresskernel'`, or a custom matrix |
| `regularization_strength` | `0.0` | Regularization weight λ, or `'abic'`/`'cv'` for auto-tuning |
| `regularization_target` | `None` | Vector reference for `(m - m_ref)` regularization |
| `bounds` | `None` | `(lower, upper)` slip bounds; each side is a scalar, a per-component array, a per-parameter array, or `None` |
| `components` | `'both'` | Slip basis: `'both'`, `'strike'`, `'dip'`, `'rake'`, `'azimuth'`, or `'plate'` |
| `rake` | `None` | Fixed local rake angle in degrees; required for `components='rake'` |
| `slip_azimuth` | `None` | Fixed geographic slip azimuth in degrees clockwise from north; required for `components='azimuth'` |
| `plate_rake` | `None` | Scalar or per-patch large-scale direction in local rake coordinates; required for `components='plate'` |
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

`components='plate'` retains two parameters per patch but rotates them into
large-scale rake-parallel/rake-perpendicular coordinates. Laplacian smoothing,
targets, component bounds, covariance, and resolution all operate in that
basis. This prevents abrupt triangle-local strike/dip changes from defining
the regularization coordinates. The physical strike/dip slip remains available
through `result.strike_slip` and `result.dip_slip`.

---

## `InversionResult`

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `slip` | `(N, 2)` or `(N, 1)` | Backwards-compatible per-patch array in the solved coordinates |
| `slip_vector` | `(2N,)` or `(N,)` | Backwards-compatible blocked vector in the solved coordinates |
| `strike_slip` | `(N,)` | Physical strike-slip component |
| `dip_slip` | `(N,)` | Physical dip-slip component |
| `slip_magnitude` | `(N,)` | Unsigned physical slip magnitude |
| `slip_rake` | `(N,)` | Physical local rake in degrees |
| `rake_parallel` | `(N,)` | Plate-parallel solution block (`components='plate'` only) |
| `rake_perpendicular` | `(N,)` | Plate-perpendicular solution block (`components='plate'` only) |
| `predicted` | `(M,)` | Forward-modeled observations |
| `residuals` | `(M,)` | `obs - predicted` |
| `reduced_chi2` | scalar | Reduced chi-squared, `r^T W r / (M - P)` |
| `rms` | scalar | RMS misfit |
| `moment` | scalar | Seismic moment in N·m |
| `Mw` | scalar | Moment magnitude |
| `regularization` | str, ndarray, or `None` | Regularization type used |
| `regularization_strength` | float or `None` | λ used, or `None` when no regularization was applied |
| `components` | str | Slip basis used in the inversion |
| `rake` | float or `None` | Fixed rake angle for `components='rake'` |
| `slip_azimuth` | float or `None` | Fixed geographic azimuth for `components='azimuth'` |
| `plate_rake` | `(N,)` or `None` | Per-patch large-scale direction for `components='plate'` |
| `dataset_names` | tuple of str | Stable identifiers in stacked-row order |
| `dataset_slices` | tuple of slice | Rows belonging to each named dataset |
| `dataset_diagnostics` | tuple | Solve-time fit diagnostics for each dataset |
| `solver`, `success`, `message` | scalars | Solver choice and completion status |
| `regularization_selection` | str or `None` | `abic`/`cv` when lambda was selected automatically |
| `backend`, `precision` | str | Numerical backend configuration used for the solve |
| `quantity`, `units` | str | Displacement/velocity semantics inherited from the data |
| `warnings` | tuple of str | Interpretation warnings retained with the result |
| `system_hash` | str | SHA-256 fingerprint for verifying a reconstructed system |

Use the named physical arrays for interpretation and plotting. Use
`slip_vector` when assembling linear algebra in the solved basis.

For ordinary assessment, use functions rather than slicing stacked rows:

```python
predictions = geodef.invert.prediction(result)  # {dataset_name: array}
residuals = geodef.invert.residual(result)      # {dataset_name: array}
diagnostics = geodef.invert.diagnostics(result) # {dataset_name: diagnostics}
print(geodef.invert.summary(result))

geodef.plot.prediction(result)
geodef.plot.residual(result)
geodef.plot.diagnostics(result)
geodef.plot.summary(result)
```

The mappings always follow solve order, including for one dataset. The stacked
`predicted` and `residuals` arrays remain the explicit linear-algebra views.

```python
geodef.invert.save(result, "result.npz")
geodef.invert.save_table(result, "result.txt", fault)
result = geodef.invert.load("result.npz")
```

`invert.save` writes `result.npz` and a readable `result.manifest.json`. The
same versioned manifest is embedded in the NumPy archive, so the `.npz` remains
portable by itself. The archive contains numeric arrays only and is always
loaded with pickling disabled. On load, GeoDef verifies the schema version plus
every array's declared shape, dtype, and SHA-256 checksum. Unversioned result
archives written by the earlier development API are migrated in memory and
marked with a warning; saving the loaded result writes the current schema.

---

## `LinearSystem`

Use `LinearSystem` directly when reusing the same fault and datasets across
multiple analyses. It precomputes and caches the projected Green's matrix,
weights, and optional smoothing matrix.

```python
system = geodef.LinearSystem(
    fault, [gnss, insar],
    regularization='laplacian',
    components='azimuth',
    slip_azimuth=15.0,
)

lc = system.lcurve(regularization_range=(1e-2, 1e6))
result = system.invert(regularization_strength=lc.optimal, bounds=(0, None))
diagnostics = system.dataset_diagnostics(result)
```

`LinearSystem.condition_report()` returns conditioning diagnostics for the
prepared (whitened) system: `cond_G`, its square `cond_normal_equations`
(what the unregularized normal-equations solve experiences — values near
`1/eps`, about `4.5e15` in float64, mean roundoff dominates), the numerical
`rank_G`, and, given a `regularization_strength`, `cond_H` for the
regularized matrix `H = G^T W G + lambda L^T L`.

```python
report = system.condition_report(regularization_strength=lc.optimal)
```

---

## Hyperparameter tuning

Regularization strength is part of the model and should not be chosen solely
because one map “looks smooth.” The L-curve balances residual and model norms;
ABIC balances fit and effective model complexity under Gaussian assumptions;
cross-validation tests prediction of held-out observations. Agreement among
methods is reassuring, while disagreement is useful evidence that covariance,
mesh, or prior assumptions deserve examination.

### `lcurve(fault, datasets, regularization, regularization_range, n=50, **kwargs) → LCurveResult`

```python
lc = geodef.lcurve(fault, [gnss, insar], regularization='laplacian',
                   regularization_range=(1e-2, 1e6), n=50)
lc.plot()        # log-log misfit vs model norm; optimal marked
lc.optimal       # λ at maximum curvature
```

### `abic_curve(fault, datasets, regularization, regularization_range, n=50, **kwargs) → ABICCurveResult`

```python
ac = geodef.abic_curve(fault, [gnss, insar], regularization='laplacian',
                       regularization_range=(1e-2, 1e8), n=50)
ac.plot()        # ABIC vs λ; optimal marked
ac.optimal       # λ at minimum ABIC
```

### Auto-tuning via `regularization_strength`

```python
result = geodef.invert.solve(fault, data, regularization='laplacian', regularization_strength='abic')
result = geodef.invert.solve(fault, data, regularization='laplacian', regularization_strength='cv')
```

On the JAX backend (`geodef.backend.set_backend('jax')`), `abic_curve`
evaluates all λ values in one batched computation — same API, same
results, one fused sweep instead of a Python loop.

---

## Nonlinear geometry search (JAX)

### `geometry_search(theta0, datasets, *, ...) → GeometrySearchResult`

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

frame = geodef.LocalFrame(-2.0, 100.0, projection="wgs84-enu")
geometry0 = {
    'e0': 0.0, 'n0': 0.0,
    'depth': 25e3, 'strike': 315.0, 'dip': 30.0,
    'length': 180e3, 'width': 90e3,
}

result = geodef.geometry_search(
    geometry0, gnss, frame=frame,
    free=['dip', 'depth'],           # parameters to optimize; rest fixed
    bounds={'dip': (5.0, 45.0)},
    n_length=12, n_width=6,
    components='dip',
    regularization='laplacian', regularization_strength=1.0,
)

result.fault          # concrete optimal Fault
result.frame          # frame defining the local parameter vector
result.theta          # expert/JAX seven-vector for the same geometry
result.slip           # inner-solve slip at the optimal geometry
result.theta_cov      # Gauss-Newton covariance of the free parameters
result.reduced_chi2
```

Notes:

- For expert/JAX workflows, the seven-element `theta0` array remains
  supported with either `frame=frame` or `ref_lat=..., ref_lon=...`.
- `result.fault` is the ordinary domain view. `result.theta` is the exact
  `[e0, n0, depth, strike, dip, length, width]` array view.
- The inner solve is unconstrained WLS with fixed `regularization_strength`;
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

### `model_covariance(result, fault, datasets, kind='posterior') → np.ndarray`

```python
Cm = geodef.model_covariance(result, fault, [gnss, insar])
# shape (P, P), where P is 2N for components='both' and N otherwise
```

With `H = GᵀWG + λ LᵀL` (the convention in
[`conventions.md`](conventions.md)):

- `kind='posterior'` (default) is the linear-Gaussian posterior covariance
  `H⁻¹`, treating `λ LᵀL` as a prior precision — the quantity Tutorial 09
  teaches and the one consistent with `geodef.bayes` slip draws.
- `kind='estimator'` is the frequentist covariance of the penalized
  estimator under data noise alone, `H⁻¹ GᵀWG H⁻¹` (Tarantola, 2005). It
  omits regularization bias and shrinks to zero as `λ` grows, so read it
  together with the resolution matrix.

Both reduce to `(GᵀWG)⁻¹` when the inversion is unregularized.

### `model_resolution(result, fault, datasets) → np.ndarray`

```python
R = geodef.model_resolution(result, fault, [gnss, insar])
# shape (P, P); R=I for perfect resolution, diag(R)<1 where regularization dominates
```

### `model_uncertainty(result, fault, datasets, kind='posterior') → np.ndarray`

```python
sigma = geodef.model_uncertainty(result, fault, [gnss, insar])
# shape (P,); per-parameter 1-sigma from sqrt(diag(Cm))
```

### `diagnostics(result) → dict[str, DatasetDiagnostics]`

Per-dataset fit statistics are computed during the solve and retained in the
result, so assessment does not need the live fault and datasets.

```python
diags = geodef.invert.diagnostics(result)
for name, d in diags.items():
    print(name)
    print(d.chi2, d.reduced_chi2, d.wrms, d.n_obs, d.dof)
```

### `compute_abic(result, fault, datasets) → float`

```python
abic = geodef.compute_abic(result, fault, [gnss, insar])
```
