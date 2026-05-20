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
| `bounds` | `None` | `(lower, upper)` — scalars, arrays, or `None` per side |
| `components` | `'both'` | Slip basis: `'both'`, `'strike'`, `'dip'`, `'rake'`, or `'azimuth'` |
| `rake` | `None` | Fixed local rake angle in degrees; required for `components='rake'` |
| `slip_azimuth` | `None` | Fixed geographic slip azimuth in degrees clockwise from north; required for `components='azimuth'` |
| `cv_folds` | `5` | Number of folds for cross-validation |
| `constraints` | `None` | `(C, d)` for `C @ m <= d` (constrained solver only) |

Auto-selection of `method`: `bounds=None` → WLS; `bounds=(0, None)` → NNLS; general bounds → `bounded_ls`.

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
| `slip` | `(N, 2)` or `(N, 1)` | Per-patch slip `[strike, dip]` |
| `slip_vector` | `(2N,)` or `(N,)` | Blocked `[ss_0..ss_N, ds_0..ds_N]`, or one amplitude per patch |
| `predicted` | `(M,)` | Forward-modeled observations |
| `residuals` | `(M,)` | `obs - predicted` |
| `chi2` | scalar | Reduced chi-squared |
| `rms` | scalar | RMS misfit |
| `moment` | scalar | Seismic moment in N·m |
| `Mw` | scalar | Moment magnitude |
| `smoothing` | str or ndarray | Regularization type used |
| `smoothing_strength` | float | λ used |
| `components` | str | Slip basis used in the inversion |
| `rake` | float or `None` | Fixed rake angle for `components='rake'` |
| `slip_azimuth` | float or `None` | Fixed geographic azimuth for `components='azimuth'` |

```python
result.save("result.npz")                   # save to disk
result.save_table("result.txt", fault)       # human-readable per-patch table
result = InversionResult.load("result.npz") # reload
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
